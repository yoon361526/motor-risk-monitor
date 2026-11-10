"""기존 특징에서 소수의 조합을 선택하는 개발 실험. 정상 기록만으로 학습한다."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

from modules.vibration_config import load_vibration_config
from modules.vibration_feature_selection import run_selection


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=BASE_DIR / "data/paderborn")
    parser.add_argument("--source", type=Path, help="이전 특징 비교 실험의 출력 폴더")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.source
    if source is None:
        completed = sorted(path.parent for path in
                           (args.data_root / "derived/feature_experiments").glob("*/comparison.json"))
        if not completed:
            parser.exit(1, "먼저 run_feature_experiment.py로 특징을 추출해 주세요.\n")
        source = completed[-1]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = args.output or args.data_root / "derived/feature_selection" / timestamp
    config = load_vibration_config(BASE_DIR / "config/vibration_analysis.yaml")
    try:
        result = run_selection(source, args.data_root, config, output)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"오류: {error}\n")
    print(f"이전 주파수 후보보다 개선된 후보: {result['recommended_candidate'] or '없음'}")
    print(f"특징 선택 보고서: {output / 'selection.md'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
