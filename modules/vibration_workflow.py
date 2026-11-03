"""데이터 준비 이후의 학습·설정 조정·최종 평가 절차를 연결한다.

train은 evaluation 폴더를 읽지 않는다. evaluate는 저장된 모델과 경계를 그대로 사용한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import joblib
import sklearn

from modules.vibration_config import VibrationConfig
from modules.vibration_dataset import (
    comparison_dir, file_sha256, load_manifest, read_split, read_windowed_record, write_json,
)
from modules.vibration_engine import VibrationComparisonEngine, comparison_metrics
from modules.vibration_report import generate_vibration_template_report


def load_engine(data_root: Path, config: VibrationConfig) -> VibrationComparisonEngine:
    """이 프로젝트에서 저장한 고정 경로의 모델만 읽는다. 업로드한 모델은 받지 않는다."""
    folder = comparison_dir(data_root)
    model_path, meta_path = folder / "models.joblib", folder / "model_metadata.json"
    if not model_path.is_file() or not meta_path.is_file():
        raise ValueError("학습한 비교 모델이 없습니다. 먼저 train을 실행해 주세요.")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata["config_fingerprint"] != config.fingerprint:
        raise ValueError("설정과 저장된 모델이 다릅니다. prepare와 train을 다시 실행해 주세요.")
    if metadata["sklearn_version"] != sklearn.__version__:
        raise ValueError("모델 저장 당시와 scikit-learn 버전이 다릅니다. train을 다시 실행해 주세요.")
    if metadata["model_sha256"] != file_sha256(model_path):
        raise ValueError("저장된 모델 파일이 변경됐거나 손상됐습니다.")
    engine = joblib.load(model_path)
    if (not isinstance(engine, VibrationComparisonEngine)
            or engine.model_id != metadata["model_id"]
            or engine.config_fingerprint != config.fingerprint
            or engine.config.fingerprint != config.fingerprint):
        raise ValueError("예상한 진동 비교 모델이 아닙니다.")
    return engine


def save_analysis(result: dict, folder: Path) -> None:
    stem = Path(result["record"]["file"]).stem
    write_json(folder / f"{stem}.json", result)
    (folder / f"{stem}.md").write_text(generate_vibration_template_report(result), encoding="utf-8")


def _save_metrics(results: list[dict], split: str, data_root: Path,
                  engine: VibrationComparisonEngine) -> dict:
    report = {"schema_version": 1, "split": split, "model_id": engine.model_id,
              "config_fingerprint": engine.config_fingerprint,
              "evaluation_unit": "record", "record_count": len(results),
              "bearing_count": len({r["record"]["bearing_id"] for r in results}),
              "record_alarm_rule": "at least one event of consecutive exceeding windows",
              "metrics": comparison_metrics(results)}
    write_json(comparison_dir(data_root) / f"{split}_metrics.json", report)
    return report


def train_comparison(data_root: Path, config: VibrationConfig,
                     progress: Callable[[str], None] = print) -> dict:
    """정상 기준을 학습하고 tuning 결과를 저장한다. 최종 평가 신호는 읽지 않는다."""
    manifest = load_manifest(data_root, config)
    reference = read_split(data_root, manifest, "reference", config, progress)
    tuning = read_split(data_root, manifest, "tuning", config, progress)
    calibration = [record for record in tuning if record.metadata["reference_label"] == "healthy"]
    engine = VibrationComparisonEngine(config)
    engine.fit(reference, calibration)

    folder = comparison_dir(data_root)
    model_path = folder / "models.joblib"
    temporary = folder / "models.joblib.tmp"
    joblib.dump(engine, temporary)
    temporary.replace(model_path)
    write_json(folder / "model_metadata.json", {
        "schema_version": 1, "model_id": engine.model_id,
        "config_fingerprint": config.fingerprint, "config": config.to_dict(),
        "sklearn_version": sklearn.__version__, "model_sha256": file_sha256(model_path),
        "thresholds": engine.thresholds, "training": engine.training_info,
        "evaluation_used_for_training": False,
    })
    results = []
    for record in tuning:
        result = engine.analyze(record)
        save_analysis(result, folder / "analyses" / "tuning")
        results.append(result)
    return _save_metrics(results, "tuning", data_root, engine)


def evaluate_comparison(data_root: Path, config: VibrationConfig,
                        progress: Callable[[str], None] = print) -> dict:
    """설정 확정 후의 최종 평가. 학습이나 경계 설정 함수를 호출하지 않는다."""
    manifest = load_manifest(data_root, config)
    engine = load_engine(data_root, config)
    # 학습 원본이 바뀌었다면 예전 모델의 최종 평가로 처리하지 않는다.
    sources = engine.training_info["reference_sources"] + engine.training_info["calibration_sources"]
    entries = {entry["file"]: entry for entry in manifest["records"]}
    for source in sources:
        path = data_root / entries[source["file"]]["relative_path"]
        if file_sha256(path) != source["sha256"]:
            raise ValueError("학습·설정 조정에 사용한 원본 파일이 변경됐습니다.")
    records = read_split(data_root, manifest, "evaluation", config, progress)
    results = []
    for record in records:
        result = engine.analyze(record)
        save_analysis(result, comparison_dir(data_root) / "analyses" / "evaluation")
        results.append(result)
    return _save_metrics(results, "evaluation", data_root, engine)


def analyze_file(path: Path, data_root: Path, config: VibrationConfig) -> dict:
    """단일 기록 분석. 보류한 평가용 베어링은 evaluate 명령에서만 분석한다."""
    # 보류한 파일은 특징을 계산하기 전부터 차단한다.
    if any(f"_{bearing}_" in path.name for bearing in config.splits["evaluation"]):
        raise ValueError("최종 평가용 기록입니다. 설정 확정 후 evaluate를 실행해 주세요.")
    engine = load_engine(data_root, config)
    record = read_windowed_record(path, config)
    result = engine.analyze(record)
    save_analysis(result, comparison_dir(data_root) / "analyses" / "manual")
    return result
