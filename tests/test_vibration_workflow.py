"""작은 MAT 파일로 준비 목록·학습·최종 평가·보고서 연결을 검증한다."""

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.io import savemat

from modules import vibration_workflow as workflow
from modules.vibration_dataset import (
    comparison_dir, expected_records, file_sha256, load_manifest,
    read_windowed_record, write_json,
)
from modules.vibration_engine import VibrationComparisonEngine


def write_signal(path, amplitude=1.):
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(1025) / 1024
    signal = amplitude * (np.sin(2 * np.pi * 45 * time) + .15 * np.sin(2 * np.pi * 83 * time))
    savemat(path, {path.stem: {
        "X": {"Data": time, "Raster": "HostService"},
        "Y": {"Name": "vibration_1", "Data": signal, "Unit": "",
              "XIndex": 1, "Raster": "HostService", "DownSampling": 1},
    }})


@pytest.fixture
def prepared_data(tmp_path, vibration_config):
    root = tmp_path / "paderborn"
    entries = expected_records(vibration_config)
    for index, entry in enumerate(entries):
        path = root / entry["relative_path"]
        write_signal(path, 1 + index * .05)
        entry["sha256"] = file_sha256(path)
    manifest = {"config_fingerprint": vibration_config.fingerprint, "records": entries}
    write_json(comparison_dir(root) / "dataset_manifest.json", manifest)
    return root


def test_train_avoids_evaluation_and_evaluate_never_refits(prepared_data, vibration_config, monkeypatch):
    original_read = workflow.read_split
    observed = []

    def guarded_read(root, manifest, split, config, progress):
        observed.append(split)
        return original_read(root, manifest, split, config, progress)

    monkeypatch.setattr(workflow, "read_split", guarded_read)
    tuning = workflow.train_comparison(prepared_data, vibration_config, progress=lambda _: None)
    assert observed == ["reference", "tuning"]
    assert tuning["record_count"] == 2
    folder = comparison_dir(prepared_data)
    before = file_sha256(folder / "models.joblib")

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("최종 평가에서 다시 학습하면 안 됩니다.")

    monkeypatch.setattr(VibrationComparisonEngine, "fit", forbidden_fit)
    evaluation = workflow.evaluate_comparison(prepared_data, vibration_config, progress=lambda _: None)
    assert observed[-1] == "evaluation"
    assert evaluation["record_count"] == 2
    assert before == file_sha256(folder / "models.joblib")
    assert evaluation["model_id"] == tuning["model_id"]
    report_path = folder / "analyses/tuning/N09_M07_F10_KA04_1.md"
    assert "Isolation Forest" in report_path.read_text(encoding="utf-8")


def test_manifest_tampering_and_changed_settings_are_rejected(prepared_data, vibration_config):
    manifest_path = comparison_dir(prepared_data) / "dataset_manifest.json"
    manifest = load_manifest(prepared_data, vibration_config)
    manifest["records"][-1]["relative_path"] = "../other.mat"
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="용도·파일 정보"):
        load_manifest(prepared_data, vibration_config)
    with pytest.raises(ValueError, match="설정이 다릅니다"):
        load_manifest(prepared_data, replace(vibration_config, window_sec=.5))


def test_feature_cache_is_invalidated_by_file_contents(tmp_path, vibration_config):
    path = tmp_path / "N09_M07_F10_K001_1.mat"
    write_signal(path, 1.)
    first = read_windowed_record(path, vibration_config, tmp_path / "cache")
    cached = read_windowed_record(path, vibration_config, tmp_path / "cache")
    assert cached.metadata == first.metadata
    np.testing.assert_allclose(cached.features["rms_centered"], first.features["rms_centered"])
    write_signal(path, 3.)
    changed = read_windowed_record(path, vibration_config, tmp_path / "cache")
    assert changed.metadata["source_sha256"] != first.metadata["source_sha256"]
    np.testing.assert_allclose(changed.features["rms_centered"], first.features["rms_centered"] * 3)


def test_unused_undefined_features_do_not_break_cache(tmp_path, vibration_config):
    path = tmp_path / "N09_M07_F10_K001_1.mat"
    time = np.arange(1025) / 1024
    savemat(path, {path.stem: {
        "X": {"Data": time, "Raster": "HostService"},
        "Y": {"Name": "vibration_1", "Data": np.ones(len(time)), "Unit": "",
              "XIndex": 1, "Raster": "HostService", "DownSampling": 1},
    }})
    config = replace(vibration_config, features=("rms_centered",))
    first = read_windowed_record(path, config, tmp_path / "cache")
    cached = read_windowed_record(path, config, tmp_path / "cache")
    assert (cached.features["rms_centered"] == 0).all()
    assert cached.metadata == first.metadata


def test_manual_analysis_cannot_open_final_evaluation_record(prepared_data, vibration_config, monkeypatch):
    workflow.train_comparison(prepared_data, vibration_config, progress=lambda _: None)
    path = prepared_data / "raw/evaluation/N09_M07_F10_KA15_1.mat"
    def forbidden_read(*args, **kwargs):
        raise AssertionError("평가용 파일을 분석 전에 열면 안 됩니다.")
    monkeypatch.setattr(workflow, "read_windowed_record", forbidden_read)
    with pytest.raises(ValueError, match="최종 평가용"):
        workflow.analyze_file(path, prepared_data, vibration_config)


def test_modified_model_or_training_source_is_rejected(prepared_data, vibration_config):
    workflow.train_comparison(prepared_data, vibration_config, progress=lambda _: None)
    source = prepared_data / "raw/reference/N09_M07_F10_K001_1.mat"
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="원본 파일이 변경"):
        workflow.evaluate_comparison(prepared_data, vibration_config, progress=lambda _: None)
    model_path = comparison_dir(prepared_data) / "models.joblib"
    model_path.write_bytes(model_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="모델 파일이 변경"):
        workflow.load_engine(prepared_data, vibration_config)
