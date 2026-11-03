"""정상 참조 통계와 Isolation Forest를 같은 구간·경고 규칙으로 비교한다.

학습: reference 정상 기록만 사용한다.
경계 설정: tuning 정상 기록만 사용한다.
판정: 제공 라벨을 읽지 않고 특징값만 사용한다.
평가: 판정이 끝난 뒤에만 제공 라벨과 비교한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn import __version__ as SKLEARN_VERSION

from modules.vibration_config import FEATURE_LABELS, VibrationConfig
from modules.vibration_dataset import WindowedRecord
from modules.paderborn import reference_label_for_bearing


METHOD_NAMES = {"statistics": "정상 참조 통계", "isolation_forest": "Isolation Forest"}


@dataclass
class StatisticsReference:
    """중앙값과 MAD로 만든 기준. 진동 크기의 증가·감소를 모두 비교한다."""

    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> StatisticsReference:
        center = np.median(values, axis=0)
        mad = np.median(np.abs(values - center), axis=0)
        scale = 1.4826 * mad
        # 변동이 없는 특징에서도 0으로 나누지 않는다. 이 바닥값은 경고 임계값이 아니다.
        scale_floor = np.maximum(np.abs(center) * 1e-6, 1e-9)
        return cls(center, np.maximum(scale, scale_floor))

    def deviations(self, values: np.ndarray) -> np.ndarray:
        return np.abs(values - self.center) / self.scale

    def score(self, values: np.ndarray) -> np.ndarray:
        # 가장 크게 벗어난 특징을 점수로 써, 측정값과 경계의 관계를 설명할 수 있게 한다.
        return self.deviations(values).max(axis=1)


def consecutive_record_score(scores: np.ndarray, minimum_windows: int) -> float:
    """연속 k개가 모두 경계를 넘는지 판단할 수 있는 기록 대표 점수.

각 k개 묶음의 최솟값 중 최댓값이다. 이 값 > 경계이면 연속 k개 초과 구간이 존재한다.
정상 기록으로 경계를 정할 때도 실제 경고 규칙과 같은 계산을 사용한다.
"""
    scores = np.asarray(scores, dtype=float)
    if minimum_windows < 1 or len(scores) < minimum_windows or not np.isfinite(scores).all():
        raise ValueError("연속 구간 판정에 필요한 유효한 점수가 부족합니다.")
    groups = np.lib.stride_tricks.sliding_window_view(scores, minimum_windows)
    return float(groups.min(axis=1).max())


def find_alarm_events(frame: pd.DataFrame, threshold: float, minimum_windows: int,
                      feature_names: tuple[str, ...]) -> tuple[np.ndarray, list[dict]]:
    """연속 초과 구간을 이력으로 남긴다. 마지막 상태가 정상이어도 이력은 삭제하지 않는다."""
    exceeds = frame["anomaly_score"].to_numpy() > threshold
    confirmed = np.zeros(len(frame), dtype=bool)
    # False로 양끝을 감싸 True 묶음의 시작과 끝을 찾는다. 끝 인덱스는 exclusive다.
    boundaries = np.flatnonzero(np.diff(np.r_[False, exceeds, False].astype(int)))
    events = []
    for start, stop in zip(boundaries[::2], boundaries[1::2]):
        if stop - start < minimum_windows:
            continue
        confirmed[start:stop] = True
        group = frame.iloc[start:stop]
        peak = group.iloc[int(np.argmax(group["anomaly_score"].to_numpy()))]
        event = {
            "start_sec": float(group.iloc[0]["start_sec"]),
            "end_sec": float(group.iloc[-1]["end_sec"]),
            "window_count": int(stop - start),
            "peak_score": float(peak["anomaly_score"]),
            "threshold": float(threshold),
            "peak_window_id": int(peak["window_id"]),
            "peak_features": {name: float(peak[name]) for name in feature_names},
        }
        # 창이 겹치므로 시간 구간이 겹치는 이력은 합쳐 중복 지속시간을 만들지 않는다.
        if events and event["start_sec"] <= events[-1]["end_sec"]:
            previous = events[-1]
            previous["end_sec"] = max(previous["end_sec"], event["end_sec"])
            previous["window_count"] += event["window_count"]
            if event["peak_score"] > previous["peak_score"]:
                for key in ("peak_score", "peak_window_id", "peak_features"):
                    previous[key] = event[key]
        else:
            events.append(event)
    return confirmed, events


class VibrationComparisonEngine:
    """두 판정 방식을 함께 보관하되, 결과를 임의의 하나의 점수로 합치지 않는다."""

    def __init__(self, config: VibrationConfig):
        config.validate()
        self.config = config
        self.config_fingerprint = config.fingerprint
        self.statistics: StatisticsReference | None = None
        self.forest: IsolationForest | None = None
        self.thresholds: dict[str, float] = {}
        self.training_info: dict = {}
        self.unit: str | None = None
        self.model_id = ""

    def _values(self, record: WindowedRecord) -> np.ndarray:
        """라벨·베어링 ID·기록 번호는 모델 입력에 넣지 않는다."""
        meta, frame = record.metadata, record.features
        if meta["condition"] != self.config.condition:
            raise ValueError("정상 기준과 검사 기록의 운전 조건이 다릅니다.")
        if meta.get("feature_settings") != self.config.feature_settings:
            raise ValueError("정상 기준과 검사 기록의 특징 추출 설정이 다릅니다.")
        required = {"window_id", "start_sec", "end_sec", *self.config.features}
        if required - set(frame):
            raise ValueError("구간 특징의 필수 컬럼이 부족합니다.")
        if len(frame) < self.config.minimum_consecutive_windows:
            raise ValueError("기록이 연속 경고 조건을 검사하기에 너무 짧습니다.")
        values = frame[list(self.config.features)].to_numpy(dtype=float)
        times = frame[["start_sec", "end_sec"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(times).all():
            raise ValueError("구간 특징에 NaN/무한대가 있습니다. 일정 신호 등 입력 품질을 확인해 주세요.")
        if (times[:, 0] < 0).any() or (times[:, 1] <= times[:, 0]).any():
            raise ValueError("구간 시작·끝 시각이 올바르지 않습니다.")
        if (np.diff(times[:, 0]) <= 0).any() or (np.diff(times[:, 1]) <= 0).any():
            raise ValueError("구간이 시간순으로 정렬되어 있지 않습니다.")
        if frame["window_id"].duplicated().any():
            raise ValueError("중복된 구간 ID가 있습니다.")
        return values

    def fit(self, reference: list[WindowedRecord], calibration: list[WindowedRecord]) -> None:
        """정상 기준 학습과 정상 경계 설정. 손상 기록·최종 평가 기록은 받지 않는다."""
        if not reference or not calibration:
            raise ValueError("기준 생성용과 경계 설정용 정상 기록이 모두 필요합니다.")
        for records, expected in (
            (reference, set(self.config.splits["reference"])),
            (calibration, {b for b in self.config.splits["tuning"]
                           if reference_label_for_bearing(b) == "healthy"}),
        ):
            if {r.metadata["bearing_id"] for r in records} != expected:
                raise ValueError("학습·경계 설정 베어링이 설정 파일의 분리와 다릅니다.")
            if any(r.metadata.get("reference_label") != "healthy" for r in records):
                raise ValueError("학습과 경계 설정에는 정상 기록만 사용할 수 있습니다.")
        units = {r.metadata.get("unit") for r in reference + calibration}
        if len(units) != 1:
            raise ValueError("단위가 서로 다른 정상 기록을 함께 사용할 수 없습니다.")
        self.unit = units.pop()
        train_values = np.concatenate([self._values(r) for r in reference])
        self.statistics = StatisticsReference.fit(train_values)
        self.forest = IsolationForest(
            n_estimators=self.config.n_estimators,
            random_state=self.config.random_state,
            contamination="auto", n_jobs=1,
        )
        self.forest.fit(train_values)
        for method in METHOD_NAMES:
            scores = [consecutive_record_score(self._score(method, self._values(record)),
                                              self.config.minimum_consecutive_windows)
                      for record in calibration]
            # higher는 작은 표본에서 경계를 관측값 사이로 낮춰 잡지 않도록 하는 선택이다.
            self.thresholds[method] = float(np.quantile(
                scores, self.config.calibration_quantile, method="higher"))
        self.training_info = {
            "reference_bearings": sorted({r.metadata["bearing_id"] for r in reference}),
            "calibration_bearings": sorted({r.metadata["bearing_id"] for r in calibration}),
            "reference_record_count": len(reference),
            "calibration_record_count": len(calibration),
            "reference_window_count": len(train_values),
            "reference_sources": self._sources(reference),
            "calibration_sources": self._sources(calibration),
            "calibration_quantile": self.config.calibration_quantile,
            "threshold_rule": "healthy tuning record scores; higher quantile; strict greater-than",
        }
        identity = {"implementation_version": 1, "config": self.config_fingerprint,
                    "training": self.training_info, "thresholds": self.thresholds,
                    "sklearn_version": SKLEARN_VERSION}
        self.model_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]

    @staticmethod
    def _sources(records: list[WindowedRecord]) -> list[dict]:
        return [{"file": r.metadata["file"], "sha256": r.metadata["source_sha256"]} for r in records]

    def _score(self, method: str, values: np.ndarray) -> np.ndarray:
        if self.statistics is None or self.forest is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train을 실행해 주세요.")
        if method == "statistics":
            return self.statistics.score(values)
        # sklearn score_samples는 작은 값이 더 이상하다. 두 방식 모두 '클수록 이상'으로 표시한다.
        # sklearn의 기본 predict 경계 대신, 동일한 정상 경계 설정 절차를 사용한다.
        return -self.forest.score_samples(values)

    def analyze(self, record: WindowedRecord) -> dict:
        values = self._values(record)
        if not self.thresholds or self.statistics is None:
            raise ValueError("경고 기준이 없습니다. 먼저 train을 실행해 주세요.")
        if record.metadata.get("unit") != self.unit:
            raise ValueError("정상 기준과 검사 기록의 진동 단위가 다릅니다.")
        methods = {}
        for method, name in METHOD_NAMES.items():
            # 설정에서 제외한 특징은 보고서에도 넣지 않는다. 정의되지 않은 미사용 비율도 제거된다.
            unused = [feature for feature in FEATURE_LABELS if feature not in self.config.features]
            frame = record.features.drop(columns=unused, errors="ignore").copy()
            frame["anomaly_score"] = self._score(method, values)
            threshold = self.thresholds[method]
            confirmed, events = find_alarm_events(
                frame, threshold, self.config.minimum_consecutive_windows, self.config.features)
            frame["exceeds_threshold"] = frame["anomaly_score"] > threshold
            frame["confirmed_alert"] = confirmed
            methods[method] = {
                "name": name, "threshold": threshold,
                "record_score": consecutive_record_score(frame["anomaly_score"].to_numpy(),
                                                         self.config.minimum_consecutive_windows),
                "record_has_alert": bool(events),
                "last_window_exceeds_threshold": bool(frame.iloc[-1]["exceeds_threshold"]),
                "last_window_confirmed_alert": bool(confirmed[-1]),
                "events": events, "windows": frame.to_dict("records"),
            }
        bounds = []
        for index, feature in enumerate(self.config.features):
            center, scale = float(self.statistics.center[index]), float(self.statistics.scale[index])
            limit = self.thresholds["statistics"] * scale
            bounds.append({"feature": feature, "label": FEATURE_LABELS[feature],
                           "center": center, "robust_scale": scale,
                           "lower_bound": center - limit, "upper_bound": center + limit})
        return {
            "schema_version": 1, "analysis_type": "vibration_comparison", "model_id": self.model_id,
            "equipment": "Paderborn 6203 베어링 시험 기록",
            "record": record.metadata,
            "settings": {**self.config.feature_settings,
                         "minimum_consecutive_windows": self.config.minimum_consecutive_windows},
            "training": self.training_info,
            "statistical_reference": bounds,
            "methods": methods,
            "notes": [
                "점수는 고장 확률이나 검증된 산업 위험 등급이 아닙니다.",
                "경고 없음은 설정한 기준에서 이상 이력을 찾지 못했다는 뜻입니다.",
                "제공 라벨은 성능 평가용이며, 판정 입력에 사용하지 않았습니다.",
                "통계 특징 경계는 Isolation Forest의 특징 중요도나 원인 설명이 아닙니다.",
            ],
        }


def comparison_metrics(results: list[dict]) -> dict:
    """파일 단위 오경보·미탐을 계산한다. 중첩 창을 독립 평가 표본으로 세지 않는다."""
    metrics = {}
    for method, name in METHOD_NAMES.items():
        tn = fp = fn = tp = 0
        by_bearing = {}
        for result in results:
            meta = result["record"]
            label = meta["reference_label"]
            if label not in {"healthy", "outer_ring_damage", "inner_ring_damage", "combined_damage"}:
                raise ValueError("평가 정답 라벨이 없는 기록이 있습니다.")
            damaged = label != "healthy"
            predicted = result["methods"][method]["record_has_alert"]
            tn += int(not damaged and not predicted)
            fp += int(not damaged and predicted)
            fn += int(damaged and not predicted)
            tp += int(damaged and predicted)
            bearing = by_bearing.setdefault(meta["bearing_id"], {
                "provided_label": label, "record_count": 0, "alert_record_count": 0})
            bearing["record_count"] += 1
            bearing["alert_record_count"] += int(predicted)
        fpr = fp / (fp + tn) if fp + tn else None
        miss = fn / (fn + tp) if fn + tp else None
        metrics[method] = {"name": name, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                           "record_count": len(results), "bearing_count": len(by_bearing),
                           "false_positive_rate": fpr, "miss_rate": miss,
                           "by_bearing": by_bearing}
    return metrics
