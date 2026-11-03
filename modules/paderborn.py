"""Paderborn MAT 신호 확인과 시간 구간별 진동 특징 추출.

이 모듈은 이상 판정을 수행하지 않는다. 파일명의 제공 라벨은 확인용 메타데이터이며
특징 계산에 사용하지 않는다. 단위를 확인하지 못하면 원본 수치로 보존한다.
공식 진동 샘플링 사양(64 kHz):
https://mb.uni-paderborn.de/kat/forschung/bearing-datacenter
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat


NOMINAL_VIBRATION_HZ = 64_000.0
CONDITIONS = {
    "N15_M07_F10": {"rpm": 1500, "torque_nm": 0.7, "radial_force_n": 1000},
    "N09_M07_F10": {"rpm": 900, "torque_nm": 0.7, "radial_force_n": 1000},
    "N15_M01_F10": {"rpm": 1500, "torque_nm": 0.1, "radial_force_n": 1000},
    "N15_M07_F04": {"rpm": 1500, "torque_nm": 0.7, "radial_force_n": 400},
}
LABEL_NAMES = {
    "healthy": "정상",
    "outer_ring_damage": "외륜 손상",
    "inner_ring_damage": "내륜 손상",
    "combined_damage": "복합 손상",
    "unknown": "미확인",
}
_FILENAME = re.compile(
    r"^(?P<condition>N\d{2}_M\d{2}_F\d{2})_"
    r"(?P<bearing>K(?:\d{3}|[AIB]\d{2}))_(?P<record>\d+)\.mat$"
)


@dataclass
class VibrationRecord:
    """시간축은 파일의 XIndex 연결을 따르며, 첫 시각을 0초로 옮긴다."""

    metadata: dict[str, Any]
    time_sec: np.ndarray
    vibration: np.ndarray


def _struct_list(value: Any, name: str) -> list[dict]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple, np.ndarray)):
        items = list(value)
        if items and all(isinstance(item, dict) for item in items):
            return items
    raise ValueError(f"MAT의 {name} 구조를 읽을 수 없습니다.")


def _vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name}은 실수형 신호여야 합니다.")
    array = np.asarray(array, dtype=np.float64).squeeze()
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError(f"{name}에 유효한 1차원 수치 신호가 없습니다.")
    return array


def reference_label_for_bearing(bearing: str) -> str:
    """제공 라벨을 반환한다. 분석 엔진의 예측이나 세부 손상 원인 진단이 아니다."""
    if bearing in {f"K{index:03}" for index in range(1, 7)}:
        return "healthy"
    return {
        "KA": "outer_ring_damage", "KI": "inner_ring_damage", "KB": "combined_damage"
    }.get(bearing[:2], "unknown")


def load_paderborn(path: str | Path) -> VibrationRecord:
    """고정 채널 순서나 고정 행 수 대신 채널명과 실제 시간축으로 읽는다."""
    path = Path(path)
    match = _FILENAME.fullmatch(path.name)
    if match is None or match["condition"] not in CONDITIONS:
        raise ValueError("지원하는 Paderborn 운전 조건과 원래 MAT 파일명이 필요합니다.")
    try:
        content = loadmat(path, variable_names=[path.stem], simplify_cells=True)
    except (ValueError, TypeError, OSError, NotImplementedError) as error:
        raise ValueError(f"MAT 파일을 읽지 못했습니다: {path.name}") from error
    root = content.get(path.stem)
    if not isinstance(root, dict):
        raise ValueError("MAT 내부 측정 변수와 파일명이 일치하지 않습니다.")
    channels = _struct_list(root.get("Y"), "Y")
    axes = _struct_list(root.get("X"), "X")
    candidates = [channel for channel in channels if channel.get("Name") == "vibration_1"]
    if len(candidates) != 1:
        raise ValueError("vibration_1 채널이 없거나 중복되어 있습니다.")
    channel = candidates[0]
    x_index = channel.get("XIndex")
    if not isinstance(x_index, (int, np.integer)) or not 1 <= x_index <= len(axes):
        raise ValueError("진동 채널의 XIndex가 올바르지 않습니다.")
    if channel.get("DownSampling", 1) != 1:
        raise ValueError("진동 채널의 다운샘플링 설정을 별도로 확인해야 합니다.")
    axis = axes[x_index - 1]  # MATLAB은 1부터 시작한다.
    if channel.get("Raster") != axis.get("Raster"):
        raise ValueError("진동 채널과 시간축의 Raster가 일치하지 않습니다.")
    signal = _vector(channel.get("Data"), "vibration_1")
    time = _vector(axis.get("Data"), "진동 시간축")
    if time.size != signal.size or np.any(np.diff(time) <= 0):
        raise ValueError("진동과 시간축의 길이가 다르거나 시각이 증가하지 않습니다.")
    time_origin = float(time[0])
    time = time - time_origin
    duration = float(time[-1])
    effective_fs = float((time.size - 1) / duration)
    raw_unit = channel.get("Unit")
    unit = raw_unit.strip() if isinstance(raw_unit, str) and raw_unit.strip() else None
    warnings = []
    if unit is None:
        warnings.append("MAT의 진동 단위가 비어 있어 원본 수치를 사용합니다. g로 변환하지 않았습니다.")
    nonuniform = not np.allclose(np.diff(time), 1.0 / NOMINAL_VIBRATION_HZ, rtol=0.01, atol=1e-10)
    if nonuniform:
        warnings.append("시간축 간격이 일정하지 않아 실제 시각으로 구간을 나눕니다. 주파수 분석 전 시간축 처리가 필요합니다.")
    centered = signal - signal.mean()
    return VibrationRecord(
        metadata={
            "file": path.name,
            "bearing_id": match["bearing"],
            "record_number": int(match["record"]),
            "condition": match["condition"],
            "nominal_condition": dict(CONDITIONS[match["condition"]]),
            "reference_label": reference_label_for_bearing(match["bearing"]),
            "reference_label_source": "Paderborn bearing ID; not a prediction",
            "channel": "vibration_1",
            "channel_names": [str(item.get("Name", "")) for item in channels],
            "unit": unit,
            "sample_count": int(signal.size),
            "duration_sec": duration,
            "original_time_origin_sec": time_origin,
            "nominal_sample_rate_hz": NOMINAL_VIBRATION_HZ,
            "effective_sample_rate_hz": effective_fs,
            "time_axis_nonuniform": nonuniform,
            "full_record_rms_centered": float(np.sqrt(np.mean(centered ** 2))),
            "warnings": warnings,
        },
        time_sec=time,
        vibration=signal,
    )


def extract_window_features(
    record: VibrationRecord, window_sec: float = 0.25, hop_sec: float = 0.125,
) -> pd.DataFrame:
    """원본 시각으로 창을 나누고 평균 제거 RMS·첨도·진폭 범위·파고율을 계산.

    첨도는 Pearson의 m4 / m2^2 정의다. 일정 신호의 첨도·파고율은 결측값으로
    둔다. 마지막 짧은 구간도 포함하고, 1 샘플 이내의 시간 오차는 마지막 창에
    흡수한다. 완전한 창의 끝 시각은 exclusive다.
    """
    if not np.isfinite([window_sec, hop_sec]).all() or not 0 < hop_sec <= window_sec:
        raise ValueError("0 < 이동 간격 <= 구간 길이여야 합니다.")
    if window_sec < 2.0 / NOMINAL_VIBRATION_HZ:
        raise ValueError("구간 길이가 너무 짧습니다.")
    time, signal = record.time_sec, record.vibration
    duration = float(time[-1])
    rows = []
    tolerance = 1.0 / NOMINAL_VIBRATION_HZ
    for window_id, start in enumerate(np.arange(0.0, duration, hop_sec)):
        planned_end = float(start + window_sec)
        is_last = planned_end >= duration - tolerance
        left = int(np.searchsorted(time, start, side="left"))
        right = len(time) if is_last else int(np.searchsorted(time, planned_end, side="left"))
        if right - left < 2:
            raise ValueError("선택한 구간에 샘플이 부족합니다. 구간 길이를 늘려 주세요.")
        centered = signal[left:right] - signal[left:right].mean()
        power = float(np.mean(centered ** 2))
        rms = float(np.sqrt(power))
        rows.append({
            "file": record.metadata["file"],
            "bearing_id": record.metadata["bearing_id"],
            "condition": record.metadata["condition"],
            "window_id": window_id,
            "start_sec": float(start),
            "end_sec": duration if is_last else planned_end,
            "start_sample": left,
            "stop_sample": right,
            "sample_count": right - left,
            "is_partial": planned_end > duration + tolerance,
            "rms_centered": rms,
            "kurtosis_pearson": float(np.mean(centered ** 4) / power ** 2) if power > 0 else None,
            "peak_to_peak": float(np.ptp(centered)),
            "crest_factor": float(np.max(np.abs(centered)) / rms) if rms > 0 else None,
        })
        if is_last:
            break
    return pd.DataFrame(rows)
