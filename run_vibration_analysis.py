"""Paderborn 진동 비교 실험의 실행 진입점.

실행 순서:
    python run_vibration_analysis.py prepare
    python run_vibration_analysis.py train
    python run_vibration_analysis.py analyze data/paderborn/raw/tuning/N09_M07_F10_KA04_1.mat
    python run_vibration_analysis.py evaluate  # 분석 설정을 확정한 뒤에만 실행

prepare: 선택한 조건의 데이터만 추출
train: 정상 기준 학습 + 정상 경계 설정 + 설정 조정용 성능 확인
analyze: 새 기록이나 설정 조정용 기록 하나를 분석하고 JSON/보고서 저장
evaluate: 저장된 모델로 최종 평가 (재학습 없음)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from modules.vibration_config import load_vibration_config
from modules.vibration_dataset import comparison_dir, prepare_dataset


BASE_DIR = Path(__file__).resolve().parent


def show_metrics(report: dict) -> None:
    print(f"\n[{report['split']}] {report['record_count']}개 기록 / {report['bearing_count']}개 베어링")
    for metric in report["metrics"].values():
        fpr, miss = metric["false_positive_rate"], metric["miss_rate"]
        fpr_text = f"{fpr:.1%}" if fpr is not None else "해당 정상 기록 없음"
        miss_text = f"{miss:.1%}" if miss is not None else "해당 손상 기록 없음"
        print(f"- {metric['name']}: 오경보율 {fpr_text}, 미탐률 {miss_text}")
        print(f"  TN={metric['tn']}, FP={metric['fp']}, FN={metric['fn']}, TP={metric['tp']}")
    print("중첩된 구간 수가 아닌 기록 수로 계산한 결과입니다. 베어링별 결과도 JSON에 저장했습니다.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config/vibration_analysis.yaml")
    parser.add_argument("--data-root", type=Path, default=BASE_DIR / "data/paderborn")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="RAR에서 선택한 조건만 준비")
    prepare.add_argument("--archive-dir", type=Path, default=Path.home() / "Downloads")
    commands.add_parser("train", help="정상 기준 학습과 설정 조정용 평가")
    analyze = commands.add_parser("analyze", help="단일 MAT 분석과 보고서 저장")
    analyze.add_argument("file", type=Path)
    commands.add_parser("evaluate", help="저장된 모델로 최종 평가; 재학습하지 않음")
    args = parser.parse_args()
    try:
        config = load_vibration_config(args.config)
        if args.command == "prepare":
            manifest = prepare_dataset(args.archive_dir, args.data_root, config)
            print(f"준비 완료: {len(manifest['records'])}개 기록. 신호 분석이나 모델 학습은 수행하지 않았습니다.")
        else:
            # prepare와 기존 CSV 분석은 머신러닝 패키지 없이도 사용할 수 있다.
            from modules.vibration_workflow import analyze_file, evaluate_comparison, train_comparison

            if args.command == "train":
                show_metrics(train_comparison(args.data_root, config))
                print("최종 평가용 신호는 읽지 않았습니다.")
            elif args.command == "evaluate":
                show_metrics(evaluate_comparison(args.data_root, config))
                print("이제 최종 평가 결과를 확인했습니다. 이후 설정 변경 시 같은 자료를 새로운 미공개 평가로 취급할 수 없습니다.")
            else:
                result = analyze_file(args.file, args.data_root, config)
                for method in result["methods"].values():
                    status = "이상 이력 있음" if method["record_has_alert"] else "이상 이력 없음"
                    print(f"{method['name']}: {status}, {len(method['events'])}개 이상 구간")
        print(f"결과 폴더: {comparison_dir(args.data_root)}")
    except (ValueError, OSError, ImportError) as error:
        parser.exit(1, f"오류: {error}\n")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    main()
