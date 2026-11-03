"""외부 정상 기준, 중간 이상 이력, 라벨 누수, 기록 단위 평가를 검증한다."""

import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from modules.vibration_engine import (
    StatisticsReference, VibrationComparisonEngine, comparison_metrics,
    consecutive_record_score, find_alarm_events,
)
from modules.vibration_report import generate_vibration_template_report, report_payload


@pytest.fixture
def comparison_engine(vibration_config, make_windowed):
    engine = VibrationComparisonEngine(vibration_config)
    engine.fit([make_windowed("K001")], [make_windowed("K003")])
    return engine


def test_same_bearing_cannot_leak_between_splits(vibration_config):
    config = replace(vibration_config, splits={**vibration_config.splits, "evaluation": ("K001",)})
    with pytest.raises(ValueError, match="중복"):
        config.validate()


def test_damaged_reference_and_unknown_features_are_rejected(vibration_config):
    config = replace(vibration_config, splits={**vibration_config.splits, "reference": ("KA04",)})
    with pytest.raises(ValueError, match="정상 기준"):
        config.validate()
    with pytest.raises(ValueError, match="지원하지 않는 진동 특징"):
        replace(vibration_config, features=("fault_type",)).validate()


def test_abnormal_from_first_window_uses_external_reference(comparison_engine, make_windowed):
    record = make_windowed("KA04", np.tile([20., 50., 30., 30.], (20, 1)))
    result = comparison_engine.analyze(record)
    assert result["methods"]["statistics"]["record_has_alert"]
    assert result["methods"]["statistics"]["events"][0]["start_sec"] == 0


def test_isolation_forest_learns_reference_and_scores_new_observation(vibration_config, make_windowed):
    # 밀집된 정상 군집과 멀리 떨어진 관측을 사용하는 통제된 동작 시험이다.
    # 실제 손상 기록의 탐지 성능은 별도의 데이터 평가에서 측정한다.
    rng = np.random.default_rng(13)
    reference = make_windowed("K001", rng.normal(
        loc=[1., 3., 2., 2.], scale=[.01, .1, .03, .03], size=(300, 4)))
    center = np.median(reference.features[list(vibration_config.features)].to_numpy(), axis=0)
    calibration = make_windowed("K003", np.tile(center, (20, 1)))
    engine = VibrationComparisonEngine(vibration_config)
    engine.fit([reference], [calibration])
    healthy = engine.analyze(calibration)["methods"]["isolation_forest"]
    extreme = make_windowed("KA04", np.tile([20., 50., 30., 30.], (20, 1)))
    anomaly = engine.analyze(extreme)["methods"]["isolation_forest"]
    assert not healthy["record_has_alert"]
    assert anomaly["record_has_alert"]
    assert anomaly["record_score"] > healthy["record_score"]


def test_middle_anomaly_survives_normal_tail(comparison_engine, make_windowed):
    # 전후 구간은 기준 중앙값으로 고정해, 삽입한 이상 구간의 시간 전달까지 확인한다.
    record = make_windowed("KA04", np.tile(comparison_engine.statistics.center, (20, 1)))
    record.features.loc[6:10, comparison_engine.config.features] *= 20
    result = comparison_engine.analyze(record)
    method = result["methods"]["statistics"]
    assert method["record_has_alert"]
    assert not method["last_window_confirmed_alert"]
    assert method["events"][0]["start_sec"] == pytest.approx(.75)
    report = generate_vibration_template_report(result)
    assert "이상 이력 있음" in report
    assert "마지막 구간: **경고 없음**" in report
    assert "0.750000" in report


def test_short_spike_is_not_a_confirmed_event(comparison_engine, make_windowed):
    record = make_windowed("KA04")
    record.features.loc[8, comparison_engine.config.features] *= 20
    method = comparison_engine.analyze(record)["methods"]["statistics"]
    assert method["windows"][8]["exceeds_threshold"]
    assert not method["record_has_alert"]


def test_prediction_does_not_use_provided_label(comparison_engine, make_windowed):
    record = make_windowed("KA04")
    first = comparison_engine.analyze(record)
    record.metadata["reference_label"] = "healthy"
    second = comparison_engine.analyze(record)
    assert first["methods"] == second["methods"]


@pytest.mark.parametrize("problem", ["nan", "condition", "unit", "settings", "time", "short"])
def test_invalid_or_incompatible_record_is_rejected(comparison_engine, make_windowed, problem):
    record = make_windowed("KA04")
    if problem == "nan":
        record.features.loc[0, "rms_centered"] = np.nan
    elif problem == "condition":
        record.metadata["condition"] = "N15_M07_F10"
    elif problem == "unit":
        record.metadata["unit"] = "g"
    elif problem == "settings":
        record.metadata["feature_settings"] = {}
    elif problem == "time":
        record.features = record.features.iloc[::-1]
    else:
        record.features = record.features.iloc[:2]
    with pytest.raises(ValueError):
        comparison_engine.analyze(record)


def test_damage_cannot_be_used_for_threshold_calibration(vibration_config, make_windowed):
    engine = VibrationComparisonEngine(vibration_config)
    with pytest.raises(ValueError):
        engine.fit([make_windowed("K001")], [make_windowed("KA04")])


def test_zero_mad_stays_finite_and_detects_deviation():
    reference = StatisticsReference.fit(np.ones((20, 4)))
    assert np.isfinite(reference.score(np.ones((2, 4)))).all()
    assert (reference.score(np.full((2, 4), 2.)) > 0).all()


def test_record_score_exactly_matches_consecutive_event_rule():
    rng = np.random.default_rng(23)
    for _ in range(20):
        scores = rng.uniform(0, 5, 20)
        frame = pd.DataFrame({"window_id": np.arange(20), "start_sec": np.arange(20) * .125,
                              "end_sec": np.arange(20) * .125 + .25,
                              "anomaly_score": scores, "rms_centered": scores})
        confirmed, events = find_alarm_events(frame, 2.5, 3, ("rms_centered",))
        assert bool(events) == (consecutive_record_score(scores, 3) > 2.5)
        assert bool(confirmed.any()) == bool(events)


def test_file_metrics_have_correct_denominators(comparison_engine, make_windowed):
    normal = comparison_engine.analyze(make_windowed("K003"))
    damage = comparison_engine.analyze(make_windowed("KA04"))
    for method in normal["methods"]:
        normal["methods"][method]["record_has_alert"] = True
        damage["methods"][method]["record_has_alert"] = False
    metrics = comparison_metrics([normal, damage])
    for metric in metrics.values():
        assert metric["record_count"] == 2
        assert metric["bearing_count"] == 2
        assert metric["fp"] == metric["fn"] == 1
        assert metric["false_positive_rate"] == metric["miss_rate"] == 1
    assert comparison_metrics([normal])["statistics"]["miss_rate"] is None


def test_ai_payload_omits_ground_truth_and_does_not_mutate_result(comparison_engine, make_windowed):
    result = comparison_engine.analyze(make_windowed("KA04"))
    original = copy.deepcopy(result)
    payload = report_payload(result)
    assert "reference_label" not in payload["record"]
    assert all("windows" not in method for method in payload["methods"].values())
    assert result == original


def test_vibration_report_uses_its_own_ai_prompt(comparison_engine, make_windowed, monkeypatch):
    import sys
    from types import SimpleNamespace
    from modules import report_generator

    captured = {}

    class FakeAnthropic:
        def __init__(self):
            self.messages = self

        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(type="text", text="진동 비교 보고서")])

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    result = comparison_engine.analyze(make_windowed("KA04"))
    assert report_generator.generate_report(result) == "진동 비교 보고서"
    assert captured["system"] == report_generator.VIBRATION_SYSTEM_PROMPT
    assert "reference_label" not in captured["messages"][0]["content"]
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert "진동 기록 분석 보고서" in report_generator.generate_report(result)
