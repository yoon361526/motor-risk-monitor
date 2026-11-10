"""특징 선택 실험의 모델을 보존·검증한 뒤 대시보드 기본 모델로 활성화한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import joblib
import numpy as np

from modules.vibration_config import VibrationConfig
from modules.vibration_dataset import comparison_dir, file_sha256, load_manifest, write_json
from modules.vibration_engine import comparison_metrics
from modules.vibration_feature_selection import (
    _json_hash, load_feature_records, validate_feature_source,
)
from modules.vibration_selected_model import (
    SelectedVibrationEngine, read_selected_record, selected_models_dir,
)
from modules.vibration_workflow import save_analysis


def activate_selection(selection_dir: Path, data_root: Path, config: VibrationConfig,
                       progress: Callable[[str], None] = print) -> dict:
    """재학습 없이 실험 모델을 가져오고, 개발 80개를 실제 입력 경로로 재검증한다.

    모든 검증과 결과 저장이 끝난 뒤 활성화 포인터만 바꾼다. 기존 비교 모델·실험
    모델은 덮어쓰지 않으며 evaluation 원본은 열지 않는다.
    """
    plan = json.loads((selection_dir / "analysis_plan.json").read_text(encoding="utf-8"))
    report = json.loads((selection_dir / "selection.json").read_text(encoding="utf-8"))
    plan_id = _json_hash(plan)
    if report["analysis_plan_id"] != plan_id or _json_hash(plan["config"]) != config.fingerprint:
        raise ValueError("선택 실험의 설정 또는 프로토콜이 다릅니다.")
    for name, digest in plan["source_code_hashes"].items():
        if name not in {"vibration_feature_selection.py", "vibration_feature_experiment.py"}:
            raise ValueError("예상하지 않은 실험 코드 경로입니다.")
        if file_sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError("선택 실험 코드가 변경됐습니다. 실험을 다시 확인해 주세요.")
    source = Path(plan["source"])
    validated = validate_feature_source(source, data_root, config)
    if (validated["protocol_id"] != plan["input_protocol_id"]
            or validated["feature_hashes"] != plan["input_feature_hashes"]):
        raise ValueError("선택 실험에 사용한 특징 파일이 변경됐습니다.")
    model_path = selection_dir / "selection_models.joblib"
    source_hash = file_sha256(model_path)
    package = joblib.load(model_path)
    if package["analysis_plan_id"] != plan_id:
        raise ValueError("선택 실험 결과와 모델이 다릅니다.")
    candidate = report["recommended_candidate"]
    if not candidate or candidate not in package["models"]:
        raise ValueError("활성화할 개선 후보가 없습니다.")
    model = package["models"][candidate]
    expected = report["candidates"][candidate]
    if (list(model["features"]) != expected["features"]
            or any(model["thresholds"][name] != result["threshold"]
                   for name, result in expected["methods"].items())):
        raise ValueError("선택한 특징 또는 경고 기준이 실험 결과와 다릅니다.")
    records = load_feature_records(source, validated)
    calibration = [r for r in records["tuning"] if r.metadata["reference_label"] == "healthy"]
    units = {r.metadata.get("unit") for group in records.values() for r in group}
    if len(units) != 1:
        raise ValueError("실험 기록의 단위가 서로 다릅니다.")
    def sources(group):
        return [{"file": r.metadata["file"], "sha256": r.metadata["source_sha256"]} for r in group]
    training = {"reference_bearings": list(config.splits["reference"]),
                "calibration_bearings": sorted({r.metadata["bearing_id"] for r in calibration}),
                "reference_record_count": len(records["reference"]),
                "calibration_record_count": len(calibration),
                "reference_window_count": sum(len(r.features) for r in records["reference"]),
                "reference_sources": sources(records["reference"]),
                "calibration_sources": sources(calibration),
                "calibration_quantile": config.calibration_quantile,
                "threshold_rule": "healthy tuning record scores; higher quantile; strict greater-than"}
    engine = SelectedVibrationEngine(config, model, training, units.pop(), plan_id, source_hash)
    folder = selected_models_dir(data_root) / engine.model_id
    baseline_paths = [comparison_dir(data_root) / name for name in (
        "models.joblib", "model_metadata.json", "tuning_metrics.json", "dataset_manifest.json")]
    protected = {str(path): file_sha256(path) for path in baseline_paths if path.is_file()}
    expected_predictions = {
        name: {row["file"]: row for row in result["predictions"]}
        for name, result in expected["methods"].items()}
    results = []
    entries = [e for e in load_manifest(data_root, config)["records"] if e["split"] == "tuning"]
    for index, entry in enumerate(entries, 1):
        record = read_selected_record(data_root / entry["relative_path"], engine, folder / "features")
        if record.metadata["source_sha256"] != entry["sha256"]:
            raise ValueError("준비 이후 개발 원본이 변경됐습니다.")
        result = engine.analyze(record)
        for name, method in result["methods"].items():
            wanted = expected_predictions[name][entry["file"]]
            if (not np.isclose(method["record_score"], wanted["record_score"], rtol=0, atol=1e-12)
                    or method["record_has_alert"] != wanted["record_has_alert"]):
                raise ValueError(f"실험과 실제 분석 경로의 결과가 다릅니다: {entry['file']} ({name})")
        results.append(result)
        if index % 10 == 0:
            progress(f"실제 입력 경로 재검증: {index}/{len(entries)}")
    metrics = {"schema_version": 1, "split": "tuning", "model_id": engine.model_id,
               "config_fingerprint": config.fingerprint, "evaluation_unit": "record",
               "record_count": len(results), "bearing_count": len(config.splits["tuning"]),
               "metrics": comparison_metrics(results)}
    # 실험 결과와 모든 기록의 예측이 같음을 확인한 뒤 결과물과 활성화 포인터를 저장한다.
    folder.mkdir(parents=True, exist_ok=True)
    for result in results:
        save_analysis(result, folder / "analyses/tuning")
    joblib.dump(engine, folder / "engine.joblib")
    write_json(folder / "model_metadata.json", {
        "schema_version": 1, "model_id": engine.model_id, "model_name": "개선 모델 · 선택 특징 4개",
        "config_fingerprint": config.fingerprint, "features": list(engine.feature_names),
        "feature_settings": engine.feature_settings, "thresholds": engine.thresholds,
        "selection_id": plan_id, "candidate": candidate, "source_model_sha256": source_hash,
        "code_hashes": engine.code_hashes, "versions": engine.versions,
        "model_sha256": file_sha256(folder / "engine.joblib"), "training": training,
        "evaluation_used": False, "replayed_record_count": len(results),
        "preserved_baseline_hashes": protected})
    write_json(folder / "tuning_metrics.json", metrics)
    if any(file_sha256(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError("활성화 도중 기존 모델 파일이 변경됐습니다.")
    write_json(selected_models_dir(data_root) / "active_selected_model.json", {"model_id": engine.model_id})
    return metrics
