"""
generate_data.py
================
Risk Score Engine 시연·검증용 synthetic 데이터 생성기.

실제 산업 설비 데이터를 확보하기 어렵기 때문에, 모터 이상 유형별 센서 변화
패턴을 모사한 CSV를 만든다. 단순 랜덤이 아니라 각 고장 유형(과부하, 베어링
마모, 냉각 불량, 복합 이상)의 물리적 특성을 반영한다.

각 CSV 구성:
  - 총 600행 (1초 간격, 0~599초)
  - 앞 20%(120행) = 정상 baseline 구간
  - 이후 구간 = baseline에서 고장값으로 선형 변화 (약간의 결정적 노이즈 포함)

사용법:
    py generate_data.py            # data/ 폴더에 5개 CSV 생성

컬럼: timestamp, operating_time_sec, temperature_c, vibration_g,
      current_a, rpm, fault_type
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Windows 터미널(cp949) 한글 깨짐 방지
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

N = 600            # 총 행 수
BASELINE_N = 120   # 초기 20% 정상 구간
START_TIME = datetime(2025, 1, 15, 9, 0, 0)  # 고정 시작 시각(재현성)

DATA_DIR = Path(__file__).resolve().parent / "data"


def _make_df(temp, vib, cur, rpm, fault_type, seed):
    """
    각 센서에 (baseline_value, final_value)를 받아,
    baseline 구간은 baseline_value, 이후는 baseline→final로 선형 변화하는
    시계열 DataFrame을 만든다. 결정적 노이즈(seed 고정)를 더한다.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    # baseline 이후 선형 램프(0→1)
    ramp = np.clip((t - BASELINE_N) / (N - BASELINE_N), 0.0, 1.0)

    def series(base_val, final_val, noise_scale, s):
        noise = np.random.default_rng(s).normal(0, noise_scale, N)
        return base_val + (final_val - base_val) * ramp + noise

    timestamps = [
        (START_TIME + timedelta(seconds=int(i))).strftime("%Y-%m-%d %H:%M:%S")
        for i in t
    ]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "operating_time_sec": t,
        "temperature_c": series(temp[0], temp[1], 0.1, seed + 1).round(2),
        "vibration_g": series(vib[0], vib[1], 0.002, seed + 2).round(4),
        "current_a": series(cur[0], cur[1], 0.05, seed + 3).round(3),
        "rpm": series(rpm[0], rpm[1], 3.0, seed + 4).round(1),
        "fault_type": fault_type,
    })
    return df


# 고장 유형별 센서 (baseline → final) 정의 (테스트에서 탐지 검증된 크기)
PATTERNS = {
    "normal": dict(
        temp=(60, 60.5), vib=(0.20, 0.205), cur=(10.0, 10.1), rpm=(1500, 1498)),
    "overload": dict(          # 전류↑ + RPM↓
        temp=(60, 63), vib=(0.20, 0.22), cur=(10.0, 13.0), rpm=(1500, 1350)),
    "bearing_wear": dict(      # 진동↑ + 온도↑
        temp=(60, 66), vib=(0.20, 0.45), cur=(10.0, 10.3), rpm=(1500, 1490)),
    "cooling_fault": dict(     # 온도만 지속 상승
        temp=(60, 72), vib=(0.20, 0.205), cur=(10.0, 10.15), rpm=(1500, 1495)),
    "compound_fault": dict(    # 온도+진동+전류 동시 상승
        temp=(60, 70), vib=(0.20, 0.32), cur=(10.0, 13.0), rpm=(1500, 1495)),
}


def main():
    DATA_DIR.mkdir(exist_ok=True)

    for i, (name, params) in enumerate(PATTERNS.items()):
        df = _make_df(fault_type=name, seed=100 + i * 10, **params)
        out_path = DATA_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[생성] {out_path.relative_to(DATA_DIR.parent)}  ({len(df)}행)")

    print(f"\n총 {len(PATTERNS)}개 CSV를 {DATA_DIR}/ 에 생성했습니다.")
    print("예: py run_analysis.py data/overload.csv")


if __name__ == "__main__":
    main()
