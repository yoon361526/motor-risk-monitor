"""
run_analysis.py
===============
실제 CSV 파일로 Risk Score Engine을 실행하는 스크립트.

사용법:
    py run_analysis.py <CSV경로>

예시:
    py run_analysis.py data/motor_sample.csv

CSV는 다음 컬럼을 가져야 합니다:
    timestamp, operating_time_sec, temperature_c, vibration_g, current_a, rpm, fault_type
    (fault_type은 없어도 엔진 동작에는 지장 없음)
"""

import json
import sys
from pathlib import Path

# Windows 터미널(cp949)에서 한글이 깨지지 않도록 출력 인코딩을 UTF-8로 고정
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

import pandas as pd

from modules.risk_engine import RiskEngine

# config 경로는 실행 위치가 아니라 이 파일 위치를 기준으로 잡는다
# (어느 디렉토리에서 실행해도 동작하도록)
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = str(BASE_DIR / "config" / "risk_thresholds.yaml")


def main():
    if len(sys.argv) < 2:
        print("사용법: py run_analysis.py <CSV경로>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    print(f"[읽음] {csv_path}  ({len(df)}행)\n")

    engine = RiskEngine(CONFIG_PATH)
    result = engine.analyze(df)

    # 결과를 UTF-8 JSON 파일로 저장 (터미널 리다이렉트 없이 안 깨지게)
    out_path = "result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[저장됨] {out_path}\n")

    # 사람이 보기 좋은 요약
    print("=" * 50)
    print(f" 설비        : {result['equipment']}")
    print(f" 상태        : {result['status']}")
    print(f" 위험 점수   : {result['risk_score']} / 100")
    print(f" 대표 패턴   : {result['main_pattern']}")
    print(f" 이상 센서   : {', '.join(result['abnormal_sensors']) or '없음'}")
    print(f" 의심 원인   : {', '.join(result['suspected_causes']) or '없음'}")
    print(f" 권장 점검   : {', '.join(result['recommended_checks']) or '없음'}")
    print("=" * 50)

    # 전체 JSON (근거 확인용)
    print("\n[전체 결과 JSON]")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
