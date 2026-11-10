"""선택한 네 특징과 이미 학습한 모델을 대시보드의 분석·보고서 형식으로 연결한다.

추가 학습이나 경계 조정은 하지 않는다. 실험과 같은 추출 함수를 사용하고,
품질 때문에 빠진 창을 넘어 연속 경고를 만들지 않는다.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from modules.paderborn import load_paderborn
from modules.vibration_config import VibrationConfig
from modules.vibration_dataset import WindowedRecord, file_sha256, write_json
from modules.vibration_engine import METHOD_NAMES, find_alarm_events
from modules.vibration_feature_experiment import (
    FEATURE_SETS, ExperimentRecord, ExperimentSettings, extract_experiment_record, record_score,
)


SELECTED_FEATURES = (
    "power_ratio_0_1000", "power_ratio_12000_16000", "envelope_peak_fraction", "kurtosis_pearson",
)
SELECTED_FEATURE_LABELS = {
    "power_ratio_0_1000": "0–1 kHz 에너지 비율",
    "power_ratio_12000_16000": "12–16 kHz 에너지 비율",
    "envelope_peak_fraction": "엔벨로프 최대 에너지 비율 (20–1000 Hz)",
    "kurtosis_pearson": "첨도 (Pearson)",
}


def selected_models_dir(data_root: Path) -> Path:
    return data_root / "derived" / "deployed"


def extraction_code_hashes() -> dict:
    return {name: file_sha256(Path(__file__).with_name(name)) for name in (
        "vibration_selected_model.py", "vibration_feature_experiment.py", "paderborn.py", "vibration_engine.py")}


def runtime_versions() -> dict:
    return {"sklearn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__}


class SelectedVibrationEngine:
    """실험 모델과 경계값을 그대로 사용하는 추론 전용 엔진. fit 메서드는 없다."""

    def __init__(self, config: VibrationConfig, model: dict, training_info: dict,
                 unit: str | None, selection_id: str, source_model_hash: str):
        if tuple(model["features"]) != SELECTED_FEATURES:
            raise ValueError("대시보드 적용 대상인 선택 특징 네 개와 순서가 다릅니다.")
        self.config = config
        self.config_fingerprint = config.fingerprint
        self.feature_names = SELECTED_FEATURES
        self.feature_labels = dict(SELECTED_FEATURE_LABELS)
        self.forest = model["forest"]
        self.statistics = model["statistics"]
        self.thresholds = dict(model["thresholds"])
        self.training_info = training_info
        self.unit = unit
        self.selection_id = selection_id
        self.code_hashes = extraction_code_hashes()
        self.versions = runtime_versions()
        self.preprocessing = ExperimentSettings()
        identity = {"adapter_version": 1, "config": config.fingerprint,
                    "selection_id": selection_id, "source_model_hash": source_model_hash,
                    "features": self.feature_names, "thresholds": self.thresholds,
                    "preprocessing": asdict(self.preprocessing), "code_hashes": self.code_hashes,
                    "versions": self.versions, "training": training_info}
        self.model_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]

    @property
    def feature_settings(self) -> dict:
        return {"window_sec": self.config.window_sec, "hop_sec": self.config.hop_sec,
                "features": list(self.feature_names), "extractor_version": "selected_features_v1",
                "spectral": asdict(self.preprocessing), "code_hashes": self.code_hashes}

    def windowed_record(self, record: ExperimentRecord, digest: str) -> WindowedRecord:
        """원본 메타데이터에 품질 제외 정보와 추출 설정을 묶는다."""
        metadata = {**record.metadata, "source_sha256": digest,
                    "feature_settings": self.feature_settings,
                    "quality_summary": dict(record.time_audit),
                    "excluded_windows": list(record.excluded_windows)}
        # 원본 로더의 '주파수 분석 전 처리 필요' 안내는 실제 적용한 처리 내용으로 바꾼다.
        metadata["warnings"] = [message for message in record.metadata.get("warnings", [])
                                if not message.startswith("시간축 간격이 일정하지")]
        if record.metadata.get("time_axis_nonuniform"):
            metadata["warnings"].append(
                "원본 시간축의 간격이 일정하지 않아, 주파수 특징 계산에는 실제 시각을 기준으로 "
                "64 kHz 선형 보간을 적용했습니다. 시간 특징은 원본 샘플에서 계산했습니다.")
        if record.excluded_windows:
            metadata["warnings"].append(
                f"시간축·입력 품질 조건으로 {len(record.excluded_windows)}개 구간을 분석에서 제외했습니다. "
                "이상 이력은 사용 가능한 구간에서의 결과입니다.")
        return WindowedRecord(metadata, record.features)

    def _validate(self, record: WindowedRecord) -> np.ndarray:
        meta, frame = record.metadata, record.features
        if meta["condition"] != self.config.condition or meta.get("unit") != self.unit:
            raise ValueError("정상 기준과 검사 기록의 운전 조건 또는 진동 단위가 다릅니다.")
        if meta.get("feature_settings") != self.feature_settings:
            raise ValueError("선택 모델과 기록의 특징 추출 설정이 다릅니다.")
        required = {"window_id", "start_sec", "end_sec", *self.feature_names}
        if frame.empty or required - set(frame):
            raise ValueError("품질 검사를 통과한 분석 구간이 부족합니다.")
        values = frame[list(self.feature_names)].to_numpy(float)
        times = frame[["start_sec", "end_sec"]].to_numpy(float)
        ids = frame["window_id"].to_numpy()
        if not np.isfinite(values).all() or not np.isfinite(times).all() or not np.isfinite(ids).all():
            raise ValueError("분석 구간의 특징·시각·번호에 유효하지 않은 값이 있습니다.")
        if (np.diff(ids) <= 0).any() or (ids != ids.astype(int)).any():
            raise ValueError("구간 번호가 중복되었거나 시간순이 아닙니다.")
        if ((times[:, 0] < 0).any() or (times[:, 1] <= times[:, 0]).any()
                or (np.diff(times[:, 0]) <= 0).any() or (np.diff(times[:, 1]) <= 0).any()):
            raise ValueError("분석 구간의 시각이 올바르지 않습니다.")
        quality = meta["quality_summary"]
        count = quality["original_window_count"]
        excluded = {row["window_id"] for row in meta["excluded_windows"]}
        if (set(ids) & excluded or (set(ids) | excluded) != set(range(count))
                or quality["usable_window_count"] != len(frame)):
            raise ValueError("원래 구간과 품질 제외 구간의 목록이 일치하지 않습니다.")
        return values

    def analyze(self, record: WindowedRecord) -> dict:
        values = self._validate(record)
        methods = {}
        latest_available = bool(record.features.iloc[-1]["window_id"] ==
                                record.metadata["quality_summary"]["original_window_count"] - 1)
        breaks = np.flatnonzero(np.diff(record.features["window_id"].to_numpy()) != 1) + 1
        runs = np.split(np.arange(len(record.features)), breaks)
        for method, name in METHOD_NAMES.items():
            # 모델 입력과 보고서에는 실제 선택된 특징만 남긴다.
            unused = set(FEATURE_SETS["combined"]) - set(self.feature_names)
            frame = record.features.drop(columns=list(unused), errors="ignore").copy()
            scores = (self.statistics.score(values) if method == "statistics"
                      else -self.forest.score_samples(values))
            score = record_score(frame, scores, self.config.minimum_consecutive_windows)
            if score is None:
                raise ValueError("연속 분석 구간이 부족합니다. 품질 제외 구간을 확인해 주세요.")
            threshold = self.thresholds[method]
            frame["anomaly_score"] = scores
            confirmed = np.zeros(len(frame), dtype=bool)
            events = []
            for positions in runs:
                flags, found = find_alarm_events(frame.iloc[positions], threshold,
                                                self.config.minimum_consecutive_windows, self.feature_names)
                confirmed[positions] = flags
                events.extend(found)
            frame["exceeds_threshold"] = scores > threshold
            frame["confirmed_alert"] = confirmed
            methods[method] = {
                "name": name, "threshold": threshold, "record_score": score,
                "record_has_alert": bool(events), "last_window_available": latest_available,
                "last_window_exceeds_threshold": bool(latest_available and scores[-1] > threshold),
                "last_window_confirmed_alert": bool(latest_available and confirmed[-1]),
                "events": events, "windows": frame.to_dict("records"),
            }
        bounds = []
        for index, feature in enumerate(self.feature_names):
            center, scale = float(self.statistics.center[index]), float(self.statistics.scale[index])
            limit = self.thresholds["statistics"] * scale
            bounds.append({"feature": feature, "label": self.feature_labels[feature],
                           "center": center, "robust_scale": scale,
                           "lower_bound": center - limit, "upper_bound": center + limit})
        return {"schema_version": 1, "analysis_type": "vibration_comparison", "model_id": self.model_id,
                "model_name": "개선 모델 · 선택 특징 4개", "selection_id": self.selection_id,
                "equipment": "Paderborn 6203 베어링 시험 기록", "record": record.metadata,
                "settings": {**self.feature_settings,
                             "minimum_consecutive_windows": self.config.minimum_consecutive_windows},
                "training": self.training_info, "statistical_reference": bounds, "methods": methods,
                "notes": ["점수는 고장 확률이나 검증된 산업 위험 등급이 아닙니다.",
                          "경고 없음은 사용 가능한 구간에서 이 기준의 이상 이력을 찾지 못했다는 뜻입니다.",
                          "개발 손상 라벨을 참고해 특징을 선택했으며 모델 학습·경계 설정에는 정상만 사용했습니다.",
                          "통계 특징 경계는 Isolation Forest의 특징 중요도가 아닙니다."]}


def read_selected_record(path: Path, engine: SelectedVibrationEngine,
                          cache_dir: Path | None = None) -> WindowedRecord:
    """실험과 동일한 전처리를 수행하며, 보류한 평가 베어링은 파일을 열기 전에 차단한다."""
    if any(f"_{bearing}_" in path.name for bearing in engine.config.splits["evaluation"]):
        raise ValueError("최종 평가용 기록입니다. 대시보드에서는 분석하지 않습니다.")
    digest = file_sha256(path)
    cache_path = None
    if cache_dir is not None:
        key = hashlib.sha256(json.dumps([digest, engine.feature_settings], sort_keys=True).encode()).hexdigest()
        cache_path = cache_dir / f"{path.stem}_{key[:16]}.json"
        if cache_path.is_file():
            saved = json.loads(cache_path.read_text(encoding="utf-8"))
            return WindowedRecord(saved["metadata"], pd.DataFrame(saved["windows"]))
    raw = load_paderborn(path)
    if raw.metadata["condition"] != engine.config.condition or raw.metadata.get("unit") != engine.unit:
        raise ValueError("선택 모델과 입력의 운전 조건 또는 단위가 다릅니다.")
    extracted = extract_experiment_record(raw, engine.config, engine.preprocessing)
    record = engine.windowed_record(extracted, digest)
    engine._validate(record)
    if cache_path is not None:
        write_json(cache_path, {"metadata": record.metadata, "windows": record.features.to_dict("records")})
    return record


def load_selected_engine(data_root: Path, config: VibrationConfig) -> tuple[SelectedVibrationEngine, Path]:
    """활성화 포인터와 해시·환경·추출 코드를 확인한 뒤 모델을 읽는다."""
    root = selected_models_dir(data_root)
    pointer = root / "active_selected_model.json"
    if not pointer.is_file():
        raise ValueError("개선 모델을 먼저 활성화해 주세요. activate_selected_model.py를 실행하면 됩니다.")
    selected = json.loads(pointer.read_text(encoding="utf-8"))
    model_id = selected["model_id"]
    if not re.fullmatch(r"[0-9a-f]{20}", model_id):
        raise ValueError("개선 모델의 활성화 정보가 올바르지 않습니다.")
    folder = root / model_id
    meta = json.loads((folder / "model_metadata.json").read_text(encoding="utf-8"))
    if meta["config_fingerprint"] != config.fingerprint:
        raise ValueError("개선 모델과 현재 설정이 다릅니다. 활성화 과정을 다시 확인해 주세요.")
    if meta["code_hashes"] != extraction_code_hashes() or meta["versions"] != runtime_versions():
        raise ValueError("개선 모델의 추출 코드 또는 실행 환경이 변경됐습니다. 모델을 다시 활성화해 주세요.")
    if file_sha256(folder / "engine.joblib") != meta["model_sha256"]:
        raise ValueError("활성화한 개선 모델 파일이 변경됐습니다.")
    engine = joblib.load(folder / "engine.joblib")
    if (not isinstance(engine, SelectedVibrationEngine) or engine.model_id != model_id
            or meta["model_id"] != model_id or engine.config_fingerprint != config.fingerprint
            or engine.feature_names != SELECTED_FEATURES or engine.thresholds != meta["thresholds"]):
        raise ValueError("예상한 개선 모델이 아닙니다.")
    return engine, folder
