"""선택 실험의 모델을 재학습 없이 대시보드 기본 모델로 활성화한다."""

import argparse
from pathlib import Path
import sys

from modules.vibration_config import load_vibration_config
from modules.vibration_deployment import activate_selection
from run_vibration_analysis import show_metrics


BASE_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, help="선택 실험 결과 폴더")
    args = parser.parse_args()
    data_root = BASE_DIR / "data/paderborn"
    selection = args.selection
    if selection is None:
        completed = sorted(p.parent for p in (data_root / "derived/feature_selection").glob("*/selection.json"))
        if not completed:
            parser.exit(1, "먼저 run_feature_selection.py를 실행해 주세요.\n")
        selection = completed[-1]
    try:
        config = load_vibration_config(BASE_DIR / "config/vibration_analysis.yaml")
        report = activate_selection(selection, data_root, config)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"오류: {error}\n")
    show_metrics(report)
    print(f"대시보드 기본 모델 활성화 완료: {report['model_id']}")
    print("기존 모델은 보존했고 최종 평가 신호는 읽지 않았습니다.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
