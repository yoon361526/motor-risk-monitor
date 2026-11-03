"""진동 비교 기능 테스트용 작은 데이터. 실제 최종 평가 파일은 사용하지 않는다."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from modules.paderborn import reference_label_for_bearing
from modules.vibration_config import VibrationConfig
from modules.vibration_dataset import WindowedRecord


@pytest.fixture
def vibration_config():
    return VibrationConfig(
        condition="N09_M07_F10", records_per_bearing=1,
        window_sec=.25, hop_sec=.125,
        features=("rms_centered", "kurtosis_pearson", "peak_to_peak", "crest_factor"),
        minimum_consecutive_windows=3, calibration_quantile=.95,
        n_estimators=20, random_state=42,
        splits={"reference": ("K001",), "tuning": ("K003", "KA04"),
                "evaluation": ("K005", "KA15")},
    )


@pytest.fixture
def make_windowed(vibration_config):
    def create(bearing="K001", values=None):
        if values is None:
            noise = .02 * np.sin(np.arange(20))[:, None]
            values = np.array([1., 3., 2., 2.]) + noise
        values = np.asarray(values, dtype=float)
        frame = pd.DataFrame(values, columns=vibration_config.features)
        frame["window_id"] = np.arange(len(frame))
        frame["start_sec"] = np.arange(len(frame)) * vibration_config.hop_sec
        frame["end_sec"] = frame["start_sec"] + vibration_config.window_sec
        name = f"N09_M07_F10_{bearing}_1.mat"
        metadata = {
            "file": name, "bearing_id": bearing, "condition": vibration_config.condition,
            "reference_label": reference_label_for_bearing(bearing),
            "reference_label_source": "provided label; not a prediction",
            "unit": None, "duration_sec": float(frame.iloc[-1]["end_sec"]),
            "warnings": ["단위 미확인"],
            "source_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "feature_settings": vibration_config.feature_settings,
        }
        return WindowedRecord(metadata, frame)
    return create
