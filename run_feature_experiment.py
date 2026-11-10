"""주파수·포락선 특징 비교 실험. 기존 모델과 최종 평가 기록을 변경하지 않는다."""

from pathlib import Path
import argparse
from datetime import datetime
import sys

from modules.vibration_config import load_vibration_config
from modules.vibration_feature_experiment import run_experiment


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=BASE_DIR / "data/paderborn")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_vibration_config(BASE_DIR / "config/vibration_analysis.yaml")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = args.output or args.data_root / "derived/feature_experiments" / timestamp
    try:
        report = run_experiment(args.data_root, config, output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"오류: {error}\n")
    print(f"추천 후보: {report['recommended_candidate'] or '없음'}")
    print(f"비교 보고서: {output / 'comparison.md'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
