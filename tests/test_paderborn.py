"""파일 구조·시간축·단위·진폭 보존을 검증한다. 탐지 정확도 시험은 아니다."""

import numpy as np
import pytest
from scipy.io import savemat

from modules.paderborn import VibrationRecord, extract_window_features, load_paderborn


def write_mat(tmp_path, *, values=None, times=None, unit="", x_index=2):
    path = tmp_path / "N09_M07_F10_K001_1.mat"
    if times is None:
        times = 10.0 + np.arange(129) / 128.0
    if values is None:
        values = 4.0 + np.sin(2 * np.pi * 8 * (times - times[0]))
    savemat(path, {path.stem: {
        "X": [{"Data": np.arange(5), "Raster": "other"},
              {"Data": times, "Raster": "HostService"}],
        "Y": [{"Name": "vibration_1", "Data": values, "Unit": unit,
               "XIndex": x_index, "Raster": "HostService", "DownSampling": 1},
              {"Name": "unrelated", "Data": np.zeros(5), "XIndex": 1}],
    }})
    return path


def test_resolves_channel_name_and_one_based_time_axis(tmp_path):
    record = load_paderborn(write_mat(tmp_path))
    assert record.time_sec[0] == 0
    assert record.time_sec[-1] == 1
    assert record.metadata["original_time_origin_sec"] == 10
    assert record.metadata["sample_count"] == 129
    assert record.metadata["condition"] == "N09_M07_F10"
    assert record.metadata["nominal_condition"]["rpm"] == 900
    assert record.vibration[0] == 4  # 원본 진폭을 보존한다.


def test_missing_unit_is_not_silently_assumed_to_be_g(tmp_path):
    record = load_paderborn(write_mat(tmp_path))
    assert record.metadata["unit"] is None
    assert record.metadata["warnings"]
    assert record.metadata["reference_label"] == "healthy"
    specified = load_paderborn(write_mat(tmp_path, unit="m/s^2"))
    assert specified.metadata["unit"] == "m/s^2"
    np.testing.assert_array_equal(record.vibration, specified.vibration)


@pytest.mark.parametrize("problem", ["nan", "length", "time_order", "axis_index"])
def test_invalid_signal_or_time_is_rejected(tmp_path, problem):
    times = np.arange(129) / 128.0
    values = np.ones(129)
    x_index = 2
    if problem == "nan":
        values[3] = np.nan
    elif problem == "length":
        values = values[:-1]
    elif problem == "time_order":
        times[20] = times[19]
    else:
        x_index = 0
    with pytest.raises(ValueError):
        load_paderborn(write_mat(tmp_path, values=values, times=times, x_index=x_index))


def test_features_remove_dc_but_preserve_amplitude(tmp_path):
    record = load_paderborn(write_mat(tmp_path))
    features = extract_window_features(record)
    first = features.iloc[0]
    assert first["rms_centered"] == pytest.approx(1 / np.sqrt(2))
    assert first["kurtosis_pearson"] == pytest.approx(1.5)
    assert first["peak_to_peak"] == pytest.approx(2.0)
    assert first["crest_factor"] == pytest.approx(np.sqrt(2))
    amplified = VibrationRecord(record.metadata, record.time_sec, 3 * record.vibration + 17)
    bigger = extract_window_features(amplified)
    np.testing.assert_allclose(bigger["rms_centered"], 3 * features["rms_centered"])
    np.testing.assert_allclose(bigger["kurtosis_pearson"], features["kurtosis_pearson"])
    assert features.iloc[-1]["stop_sample"] == len(record.vibration)


def test_irregular_timestamps_and_partial_tail_cover_all_samples():
    time = np.array([0, .05, .16, .24, .29, .33, .40, .49, .57, .62, .70, .77, .84, .90, .99, 1.01])
    record = VibrationRecord({"file": "example", "bearing_id": "K001", "condition": "N09_M07_F10"},
                             time, np.arange(len(time), dtype=float))
    features = extract_window_features(record)
    assert features.iloc[0]["sample_count"] == 4
    assert features.iloc[-1]["end_sec"] == time[-1]
    assert features.iloc[-1]["is_partial"]
    coverage = np.zeros(len(time), dtype=bool)
    for row in features.itertuples():
        coverage[row.start_sample:row.stop_sample] = True
    assert coverage.all()


def test_constant_signal_has_zero_rms_and_undefined_ratios(tmp_path):
    record = load_paderborn(write_mat(tmp_path, values=np.full(129, 7.0)))
    features = extract_window_features(record)
    assert (features["rms_centered"] == 0).all()
    assert (features["peak_to_peak"] == 0).all()
    assert features["kurtosis_pearson"].isna().all()
    assert features["crest_factor"].isna().all()


@pytest.mark.parametrize("window,hop", [(0, .1), (.25, 0), (.25, .5), (np.nan, .1), (.25, np.inf)])
def test_invalid_window_settings_are_rejected(tmp_path, window, hop):
    record = load_paderborn(write_mat(tmp_path))
    with pytest.raises(ValueError):
        extract_window_features(record, window, hop)
