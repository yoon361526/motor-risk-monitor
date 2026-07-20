"""
risk_engine.py
================
모터 이상징후 진단 엔진 (Risk Score Engine)

설계 원칙
---------
1. 절대 임계값이 아니라 "정상 기준(baseline) 대비 변화량"으로 판정한다.
2. 단일 센서보다 센서 조합 패턴에 더 큰 가중을 준다(조합 보너스).
3. 규칙 기반 로직만으로 상태를 판정한다(ML/생성형 AI 미사용).
4. 모든 중간 계산값을 반환하여 점수 산출 근거를 외부에서 확인할 수 있다.

모든 임계값/가중치/패턴 조건은 config/risk_thresholds.yaml 에서 관리한다.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml


# 센서 한글 표기 (출력용)
SENSOR_KR = {
    "temperature": "온도",
    "vibration": "진동",
    "current": "전류",
    "rpm": "RPM",
}

# 입력 CSV에 반드시 있어야 하는 컬럼 (fault_type은 라벨이라 필수 아님)
REQUIRED_COLUMNS = [
    "timestamp",
    "operating_time_sec",
    "temperature_c",
    "vibration_g",
    "current_a",
    "rpm",
]

# 검증에 쓰는 숫자형 컬럼
NUMERIC_COLUMNS = [
    "operating_time_sec",
    "temperature_c",
    "vibration_g",
    "current_a",
    "rpm",
]


class RiskEngine:
    """규칙 기반 모터 위험 점수 산출 엔진."""

    def __init__(self, config_path: str):
        """
        Parameters
        ----------
        config_path : str
            risk_thresholds.yaml 경로.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config 파일을 찾을 수 없습니다: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        self.equipment: str = self.config.get("equipment", "Motor")

    # ------------------------------------------------------------------
    # 0. 입력 CSV 검증
    # ------------------------------------------------------------------
    def validate_input(self, df: pd.DataFrame) -> List[str]:
        """
        입력 DataFrame의 유효성을 검사한다.
          - 치명적 문제(컬럼 누락/타입 오류/결측치/행 부족)는 ValueError로 즉시 중단.
          - 진단은 가능하지만 주의가 필요한 사항은 경고 리스트로 반환.
        반환: 경고 메시지 리스트(data_quality_warnings 로 결과에 포함).
        """
        warnings: List[str] = []

        # 1) 필수 컬럼 존재
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"필수 컬럼이 누락되었습니다: {missing}")

        # 2) 최소 행 수 (baseline 20%가 의미를 가지려면 최소한의 표본 필요)
        if len(df) < 50:
            raise ValueError(
                f"데이터 행 수가 너무 적습니다({len(df)}행). 최소 50행 이상이 필요합니다."
            )

        # 3) 숫자형 컬럼 타입
        for col in NUMERIC_COLUMNS:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"'{col}' 컬럼은 숫자형이어야 합니다.")

        # 4) 결측치
        if df[NUMERIC_COLUMNS].isna().any().any():
            raise ValueError("센서 데이터에 결측치(NaN)가 포함되어 있습니다.")

        # --- 여기부터는 경고(중단하지 않음) ---
        # 5) RPM 음수 등 비정상 값
        if (df["rpm"] < 0).any():
            warnings.append("RPM에 음수 값이 포함되어 있습니다.")

        # 6) baseline 구간이 0/음수 평균이면 변화율 계산이 왜곡됨
        ratio = self.config["baseline"]["baseline_ratio"]
        base_len = max(1, int(round(len(df) * ratio)))
        for sensor, col in (("temperature", "temperature_c"),
                            ("current", "current_a"),
                            ("rpm", "rpm")):
            base_mean = float(df[col].iloc[:base_len].mean())
            if base_mean <= 0:
                warnings.append(
                    f"{SENSOR_KR[sensor]} baseline 평균이 0 이하({base_mean:.3f})라 "
                    f"변화율 계산이 부정확할 수 있습니다."
                )

        # 7) 시간 컬럼 정렬 여부
        if not df["operating_time_sec"].is_monotonic_increasing:
            warnings.append(
                "operating_time_sec가 오름차순으로 정렬되어 있지 않습니다."
            )

        return warnings

    # ------------------------------------------------------------------
    # 1. baseline: 초기 20% 구간 평균/표준편차
    # ------------------------------------------------------------------
    def compute_baseline(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        CSV 초기 baseline_ratio(기본 20%) 구간을 '정상 기준'으로 삼아
        각 센서의 평균과 표준편차를 계산한다.

        반환값에는 baseline_<sensor> (평균) 과 std_<sensor> (표준편차)를 담는다.
        """
        ratio = self.config["baseline"]["baseline_ratio"]
        n = len(df)
        # 초기 구간 행 수 (최소 1행 보장)
        base_len = max(1, int(round(n * ratio)))
        base_df = df.iloc[:base_len]

        baseline = {}
        for sensor in ("temperature", "vibration", "current", "rpm"):
            col = f"{sensor}_c" if sensor == "temperature" else \
                  f"{sensor}_g" if sensor == "vibration" else \
                  f"{sensor}_a" if sensor == "current" else "rpm"
            baseline[f"baseline_{sensor}"] = float(base_df[col].mean())
            baseline[f"std_{sensor}"] = float(base_df[col].std(ddof=0))

        baseline["baseline_rows"] = int(base_len)
        return baseline

    # ------------------------------------------------------------------
    # 2. 변화율: 후반부 대표값 대비 baseline
    # ------------------------------------------------------------------
    def compute_changes(self, df: pd.DataFrame, baseline: Dict[str, float]) -> Dict[str, float]:
        """
        분석 대상 구간(후반부 analysis_tail_ratio)의 대표값을 산출해 변화율을 계산한다.

        대표값 산출 방식(설명 가능성):
          1) 전체 시계열에 window=moving_average_window 이동평균을 적용해 순간 노이즈 제거
          2) 그중 후반부 tail 구간의 평균을 '현재 대표값'으로 사용
        => 고장은 보통 후반부에 심화되므로, 후반부 안정 평균을 대표값으로 본다.

        변화율 정의:
          temperature_change = (현재 - 기준) / 기준
          vibration_change   = (현재 - 기준) / 기준
          current_change     = (현재 - 기준) / 기준
          rpm_drop           = (기준 - 현재) / 기준   # RPM은 '감소'가 위험
        """
        window = self.config["baseline"]["moving_average_window"]
        tail_ratio = self.config["baseline"]["analysis_tail_ratio"]
        n = len(df)
        tail_len = max(1, int(round(n * tail_ratio)))

        def representative(col: str) -> float:
            # 이동평균(min_periods=1)으로 노이즈 완화 후 후반부 평균
            smoothed = df[col].rolling(window=window, min_periods=1).mean()
            return float(smoothed.iloc[-tail_len:].mean())

        cur_temp = representative("temperature_c")
        cur_vib = representative("vibration_g")
        cur_cur = representative("current_a")
        cur_rpm = representative("rpm")

        def safe_div(numer: float, denom: float) -> float:
            # 기준값이 0에 가까우면 변화율 정의 불가 -> 0 반환
            return numer / denom if abs(denom) > 1e-9 else 0.0

        changes = {
            "temperature": safe_div(cur_temp - baseline["baseline_temperature"],
                                    baseline["baseline_temperature"]),
            "vibration": safe_div(cur_vib - baseline["baseline_vibration"],
                                  baseline["baseline_vibration"]),
            "current": safe_div(cur_cur - baseline["baseline_current"],
                                baseline["baseline_current"]),
            "rpm_drop": safe_div(baseline["baseline_rpm"] - cur_rpm,
                                 baseline["baseline_rpm"]),
        }
        return changes

    # ------------------------------------------------------------------
    # 3. 센서별 부분 점수 (0~100)
    # ------------------------------------------------------------------
    def compute_sensor_scores(self, changes: Dict[str, float]) -> Dict[str, float]:
        """
        각 센서 변화율을 config의 dead_zone/full_score_change 기준으로
        0~100 점수에 선형 매핑한다.

        score = clip((change - dead_zone) / (full - dead_zone) * 100, 0, 100)
          - dead_zone 이하 변화 : 노이즈로 간주 -> 0점
          - full_score_change 이상 : 100점
        온도/진동/전류는 '증가율', RPM은 'rpm_drop(감소율)'을 입력으로 쓴다.
        """
        mapping = self.config["sensor_score_mapping"]

        def to_score(change: float, dead_zone: float, full: float) -> float:
            if full <= dead_zone:
                return 0.0
            score = (change - dead_zone) / (full - dead_zone) * 100.0
            return float(np.clip(score, 0.0, 100.0))

        scores = {
            "temperature": to_score(changes["temperature"],
                                    mapping["temperature"]["dead_zone"],
                                    mapping["temperature"]["full_score_change"]),
            "vibration": to_score(changes["vibration"],
                                  mapping["vibration"]["dead_zone"],
                                  mapping["vibration"]["full_score_change"]),
            "current": to_score(changes["current"],
                                mapping["current"]["dead_zone"],
                                mapping["current"]["full_score_change"]),
            "rpm": to_score(changes["rpm_drop"],
                            mapping["rpm"]["dead_zone"],
                            mapping["rpm"]["full_score_change"]),
        }
        return scores

    # ------------------------------------------------------------------
    # 4. 조합 패턴 탐지
    # ------------------------------------------------------------------
    def detect_patterns(self, changes: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        config의 각 패턴 조건(변화율 임계값)을 모두 만족하는지 검사한다.
        조건 키 규칙:
          <sensor>_change_gte / rpm_drop_gte : 변화율 >= 값
          <sensor>_change_lt  / rpm_drop_lt  : 변화율 <  값 (변화가 작음)
        탐지된 패턴 목록(패턴 메타 + bonus 포함)을 반환한다.
        """
        # 조건 키에서 참조할 실제 변화율 값 매핑
        value_of = {
            "temperature_change": changes["temperature"],
            "vibration_change": changes["vibration"],
            "current_change": changes["current"],
            "rpm_drop": changes["rpm_drop"],
        }

        detected: List[Dict[str, Any]] = []
        for key, pattern in self.config["patterns"].items():
            conditions: Dict[str, float] = pattern["conditions"]
            matched = True
            for cond_key, threshold in conditions.items():
                if cond_key.endswith("_gte"):
                    metric = cond_key[:-4]           # "_gte" 제거
                    if not (value_of[metric] >= threshold):
                        matched = False
                        break
                elif cond_key.endswith("_lt"):
                    metric = cond_key[:-3]           # "_lt" 제거
                    if not (value_of[metric] < threshold):
                        matched = False
                        break
                else:
                    raise ValueError(f"알 수 없는 조건 키: {cond_key}")

            if matched:
                detected.append({
                    "key": key,
                    "label": pattern["label"],
                    "description": pattern["description"],
                    "cause": pattern["cause"],
                    "bonus": float(pattern["bonus"]),
                    # 이 패턴이 잡히면 최종 점수를 최소 이 값 이상으로 보장 (기본 0)
                    "min_score_if_detected": float(pattern.get("min_score_if_detected", 0)),
                })
        return detected

    # ------------------------------------------------------------------
    # 5. 최종 위험 점수
    # ------------------------------------------------------------------
    def compute_final_score(self, scores: Dict[str, float],
                            detected_patterns: List[Dict[str, Any]]) -> float:
        """
        Final = max( Σ(sensor_score * weight) + Σ(pattern_bonus),
                     탐지된 패턴의 최소 보장 점수 ) 를 0~100으로 clip.

        ★ 최소 보장 점수(min_score_if_detected)를 두는 이유:
          냉각 불량처럼 단일 센서(온도)만 오르는 고장은 가중합이 낮게 나와
          '정상'으로 분류될 수 있다. 명백한 고장 패턴이 탐지됐는데 정상으로
          나오는 모순을 막기 위해, 패턴이 잡히면 최소 점수를 바닥으로 깐다.
        """
        w = self.config["weights"]
        weighted = (
            scores["temperature"] * w["temperature"]
            + scores["vibration"] * w["vibration"]
            + scores["current"] * w["current"]
            + scores["rpm"] * w["rpm"]
        )
        pattern_bonus = sum(p["bonus"] for p in detected_patterns)

        # 탐지된 패턴들의 최소 보장 점수 중 가장 큰 값
        pattern_min = max(
            (p.get("min_score_if_detected", 0.0) for p in detected_patterns),
            default=0.0,
        )

        final = max(weighted + pattern_bonus, pattern_min)
        return float(np.clip(final, 0.0, 100.0))

    # ------------------------------------------------------------------
    # 6. 상태 분류
    # ------------------------------------------------------------------
    def classify_status(self, score: float) -> str:
        """점수를 정상/주의/위험으로 분류한다."""
        st = self.config["status_thresholds"]
        if score <= st["normal_max"]:
            return "정상"
        if score <= st["caution_max"]:
            return "주의"
        return "위험"

    # ------------------------------------------------------------------
    # 7. 원인 후보 추론
    # ------------------------------------------------------------------
    def infer_causes(self, detected_patterns: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        탐지된 패턴 -> suspected_causes / recommended_checks 로 변환.
        (중복 제거, 입력 순서 유지)
        """
        cause_checks = self.config["cause_checks"]
        suspected: List[str] = []
        checks: List[str] = []

        for pattern in detected_patterns:
            cause = pattern["cause"]
            if cause not in suspected:
                suspected.append(cause)
            for chk in cause_checks.get(cause, []):
                if chk not in checks:
                    checks.append(chk)

        return {"suspected_causes": suspected, "recommended_checks": checks}

    # ------------------------------------------------------------------
    # analyze: 전체 파이프라인 실행 -> 최종 JSON dict
    # ------------------------------------------------------------------
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        전체 진단 파이프라인을 실행하고 최종 결과 dict를 반환한다.
        중간 계산값(baseline, changes, scores, patterns)도 모두 포함해
        점수 산출 근거를 외부에서 검증할 수 있게 한다.
        """
        # 0) 입력 검증 (치명적 문제는 여기서 ValueError로 중단됨)
        data_quality_warnings = self.validate_input(df)

        baseline = self.compute_baseline(df)
        changes = self.compute_changes(df, baseline)
        scores = self.compute_sensor_scores(changes)
        detected = self.detect_patterns(changes)
        risk_score = self.compute_final_score(scores, detected)
        status = self.classify_status(risk_score)
        causes = self.infer_causes(detected)

        # 이상 센서: 부분 점수가 유의미(>0)한 센서를 점수 내림차순으로 나열
        abnormal_sensors = [
            SENSOR_KR[s]
            for s, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            if scores[s] > 0
        ]

        # 대표 패턴: 보너스가 가장 큰 패턴 설명
        main_pattern = (
            max(detected, key=lambda p: p["bonus"])["description"]
            if detected else "특이 조합 패턴 없음"
        )

        # 분석 구간(초): baseline 구간과 분석 대상(후반부) 구간의 시간 범위
        ot = df["operating_time_sec"].to_numpy()
        base_len = baseline["baseline_rows"]
        tail_ratio = self.config["baseline"]["analysis_tail_ratio"]
        tail_len = max(1, int(round(len(df) * tail_ratio)))
        analysis_window = {
            "baseline_range_sec": f"{int(ot[0])}-{int(ot[base_len - 1])}",
            "analysis_range_sec": f"{int(ot[-tail_len])}-{int(ot[-1])}",
        }

        # 점수 산출 근거를 자연어 문장으로 (AI 리포트 품질/설명 가능성 향상)
        score_explanation = self._build_score_explanation(changes, scores, detected)

        result = {
            "equipment": self.equipment,
            "status": status,
            "risk_score": int(round(risk_score)),
            "baseline": {
                "temperature": round(baseline["baseline_temperature"], 3),
                "vibration": round(baseline["baseline_vibration"], 4),
                "current": round(baseline["baseline_current"], 3),
                "rpm": round(baseline["baseline_rpm"], 1),
                "std": {
                    "temperature": round(baseline["std_temperature"], 4),
                    "vibration": round(baseline["std_vibration"], 5),
                    "current": round(baseline["std_current"], 4),
                    "rpm": round(baseline["std_rpm"], 2),
                },
                "rows_used": baseline["baseline_rows"],
            },
            "sensor_changes": {
                "temperature": round(changes["temperature"], 4),
                "vibration": round(changes["vibration"], 4),
                "current": round(changes["current"], 4),
                "rpm_drop": round(changes["rpm_drop"], 4),
            },
            "sensor_scores": {
                "temperature": round(scores["temperature"], 1),
                "vibration": round(scores["vibration"], 1),
                "current": round(scores["current"], 1),
                "rpm": round(scores["rpm"], 1),
            },
            "detected_patterns": [p["label"] for p in detected],
            "abnormal_sensors": abnormal_sensors,
            "main_pattern": main_pattern,
            "suspected_causes": causes["suspected_causes"],
            "recommended_checks": causes["recommended_checks"],
            # --- 근거/설명 필드 (AI 리포트 및 설명 가능성용) ---
            "analysis_window": analysis_window,
            "score_explanation": score_explanation,
            "data_quality_warnings": data_quality_warnings,
        }
        return result

    # ------------------------------------------------------------------
    # 보조: 점수 산출 근거를 자연어 문장 리스트로 생성
    # ------------------------------------------------------------------
    def _build_score_explanation(self, changes: Dict[str, float],
                                 scores: Dict[str, float],
                                 detected: List[Dict[str, Any]]) -> List[str]:
        """왜 이런 점수가 나왔는지 사람이 읽을 수 있는 근거 문장을 만든다."""
        # 센서별 변화율 키와 방향 표현
        change_key = {"temperature": "temperature", "vibration": "vibration",
                      "current": "current", "rpm": "rpm_drop"}
        direction = {"temperature": "상승", "vibration": "증가",
                     "current": "증가", "rpm": "감소"}

        lines: List[str] = []
        # 점수가 높은 센서부터 서술
        for sensor in sorted(scores, key=lambda s: scores[s], reverse=True):
            sc = scores[sensor]
            if sc <= 0:
                continue
            ch = changes[change_key[sensor]]
            lines.append(
                f"{SENSOR_KR[sensor]}이(가) 기준 대비 {ch * 100:.1f}% "
                f"{direction[sensor]}하여 {SENSOR_KR[sensor]} 점수 {sc:.1f}점"
            )

        # 탐지된 패턴별 보너스/최소 보장 설명
        for p in detected:
            note = f"'{p['label']}' 패턴 탐지 → 보너스 {p['bonus']:.0f}점"
            if p.get("min_score_if_detected", 0):
                note += f" (최소 보장 {p['min_score_if_detected']:.0f}점)"
            lines.append(note)

        if not lines:
            lines.append("기준 대비 유의미한 센서 변화가 감지되지 않았습니다.")
        return lines
