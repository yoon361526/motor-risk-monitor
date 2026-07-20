"""
test_risk_engine.py
===================
RiskEngine 단위 테스트.

synthetic 데이터가 아직 없으므로, 각 고장 패턴을 모사한 간단한 DataFrame을
테스트 내에서 직접 생성해 사용한다.

DataFrame 구성:
  - 총 600행(1초 간격, 0~599초)
  - 앞 20%(120행): 정상 baseline 구간
  - 뒤 구간: 각 패턴에 맞게 센서값을 변화시켜 고장을 모사
"""

import os

import numpy as np
import pandas as pd
import pytest

from modules.risk_engine import RiskEngine


CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "risk_thresholds.yaml"
)

N = 600            # 총 행 수
BASELINE_N = 120   # 초기 20% 구간


@pytest.fixture
def engine():
    return RiskEngine(CONFIG_PATH)


def _make_df(temp, vib, cur, rpm, fault_type="normal"):
    """
    각 센서에 대한 (baseline_value, final_value) 튜플을 받아
    baseline 구간은 baseline_value, 이후 구간은 baseline->final 로 선형 증감하는
    시계열 DataFrame을 만든다. 약간의 결정적 노이즈를 더한다.
    """
    t = np.arange(N)
    # baseline 이후 선형 램프(0->1) 계수
    ramp = np.clip((t - BASELINE_N) / (N - BASELINE_N), 0.0, 1.0)

    def series(base_val, final_val, noise_scale, seed):
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, noise_scale, N)
        return base_val + (final_val - base_val) * ramp + noise

    df = pd.DataFrame({
        "timestamp": t,
        "operating_time_sec": t,
        "temperature_c": series(temp[0], temp[1], 0.1, 1),
        "vibration_g": series(vib[0], vib[1], 0.002, 2),
        "current_a": series(cur[0], cur[1], 0.05, 3),
        "rpm": series(rpm[0], rpm[1], 3.0, 4),
        "fault_type": fault_type,
    })
    return df


# ---------------------------------------------------------------------------
# 개별 패턴 데이터 생성 헬퍼
# ---------------------------------------------------------------------------
def make_normal():
    # 모든 센서가 baseline 근처 유지
    return _make_df(temp=(60, 60.5), vib=(0.20, 0.205),
                    cur=(10.0, 10.1), rpm=(1500, 1498), fault_type="normal")


def make_overload():
    # 전류 크게 증가 + RPM 크게 감소
    return _make_df(temp=(60, 63), vib=(0.20, 0.22),
                    cur=(10.0, 13.0), rpm=(1500, 1350), fault_type="overload")


def make_bearing_wear():
    # 진동 크게 증가 + 온도 상승 (전류/RPM은 소폭)
    return _make_df(temp=(60, 66), vib=(0.20, 0.45),
                    cur=(10.0, 10.3), rpm=(1500, 1490), fault_type="bearing_wear")


def make_cooling_fault():
    # 온도만 지속 상승, 나머지는 거의 변화 없음
    return _make_df(temp=(60, 72), vib=(0.20, 0.205),
                    cur=(10.0, 10.15), rpm=(1500, 1495), fault_type="cooling_fault")


# ---------------------------------------------------------------------------
# 테스트
# ---------------------------------------------------------------------------
def test_baseline_uses_initial_20_percent(engine):
    """baseline이 초기 20% 구간에서 계산되는지."""
    df = make_normal()
    baseline = engine.compute_baseline(df)

    assert baseline["baseline_rows"] == BASELINE_N
    # 초기 구간 평균과 근접해야 함
    expected_temp = df["temperature_c"].iloc[:BASELINE_N].mean()
    assert baseline["baseline_temperature"] == pytest.approx(expected_temp, rel=1e-9)
    # 표준편차 키가 존재하는지
    for s in ("temperature", "vibration", "current", "rpm"):
        assert f"std_{s}" in baseline


def test_normal_pattern_status(engine):
    """정상 입력 시 status='정상' (0~39)."""
    result = engine.analyze(make_normal())
    assert result["status"] == "정상"
    assert 0 <= result["risk_score"] <= 39
    assert result["detected_patterns"] == []


def test_overload_pattern_detected(engine):
    """과부하(전류↑+RPM↓) 조합 패턴 탐지."""
    df = make_overload()
    changes = engine.compute_changes(df, engine.compute_baseline(df))
    patterns = engine.detect_patterns(changes)
    keys = [p["key"] for p in patterns]

    assert "overload" in keys
    result = engine.analyze(df)
    assert "과부하" in result["suspected_causes"]


def test_bearing_wear_pattern_detected(engine):
    """베어링 마모(진동↑+온도↑) 조합 패턴 탐지."""
    df = make_bearing_wear()
    changes = engine.compute_changes(df, engine.compute_baseline(df))
    keys = [p["key"] for p in engine.detect_patterns(changes)]

    assert "bearing_wear" in keys
    result = engine.analyze(df)
    assert "베어링 마모" in result["suspected_causes"]


def test_cooling_fault_pattern_detected(engine):
    """냉각 불량(온도만 지속 상승) 조합 패턴 탐지."""
    df = make_cooling_fault()
    changes = engine.compute_changes(df, engine.compute_baseline(df))
    keys = [p["key"] for p in engine.detect_patterns(changes)]

    assert "cooling_fault" in keys
    result = engine.analyze(df)
    assert "냉각 불량" in result["suspected_causes"]


def test_final_score_within_bounds(engine):
    """최종 점수가 항상 0~100 범위를 유지하는지 (극단 입력 포함)."""
    # 모든 센서가 극단적으로 악화된 데이터
    extreme = _make_df(temp=(60, 120), vib=(0.20, 2.0),
                       cur=(10.0, 25.0), rpm=(1500, 800), fault_type="extreme")
    for df in (make_normal(), make_overload(), make_bearing_wear(),
               make_cooling_fault(), extreme):
        result = engine.analyze(df)
        assert 0 <= result["risk_score"] <= 100


def test_analyze_output_schema(engine):
    """analyze 반환 dict가 요구된 키를 모두 포함하는지."""
    result = engine.analyze(make_overload())
    required_keys = {
        "equipment", "status", "risk_score", "baseline", "sensor_changes",
        "sensor_scores", "detected_patterns", "abnormal_sensors",
        "main_pattern", "suspected_causes", "recommended_checks",
    }
    assert required_keys.issubset(result.keys())


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
