"""Paderborn 원본 점검과 구간별 특징 내보내기. 이상 판정·성능 평가는 하지 않는다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from modules.paderborn import LABEL_NAMES, extract_window_features, load_paderborn

BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=BASE_DIR / "data/paderborn/raw")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "data/paderborn/derived")
    parser.add_argument("--window-sec", type=float, default=0.25)
    parser.add_argument("--hop-sec", type=float, default=0.125)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("*.mat"))
    if not paths:
        parser.error(f"MAT 파일이 없습니다: {args.input_dir}")
    records, frames = [], []
    for path in paths:
        try:
            record = load_paderborn(path)
            frame = extract_window_features(record, args.window_sec, args.hop_sec)
        except ValueError as error:
            parser.error(f"{path.name}: {error}")
        frames.append(frame)
        records.append({
            **record.metadata,
            "window_count": len(frame),
            "partial_window_count": int(frame["is_partial"].sum()),
            "feature_summary": json.loads(frame[[
                "rms_centered", "kurtosis_pearson", "peak_to_peak", "crest_factor"
            ]].agg(["mean", "median", "min", "max"]).to_json()),
        })
        print(
            f"{path.name}: {LABEL_NAMES[record.metadata['reference_label']]} (제공 라벨), "
            f"{record.metadata['sample_count']:,} samples, "
            f"{record.metadata['duration_sec']:.6f}s, {len(frame)} windows, "
            f"RMS={record.metadata['full_record_rms_centered']:.6f}"
        )
    summary = {
        "schema_version": 1,
        "purpose": "signal_and_feature_inspection_only",
        "classification_performed": False,
        "performance_evaluated": False,
        "feature_settings": {
            "window_sec": args.window_sec,
            "hop_sec": args.hop_sec,
            "centering": "subtract each window mean",
            "kurtosis_definition": "Pearson m4 / m2^2; constant signals are null",
            "amplitude_normalization": False,
            "windowing": "original timestamps; include final partial window",
        },
        "file_count": len(records),
        "conditions": sorted({record["condition"] for record in records}),
        "healthy_bearing_count": len({
            record["bearing_id"] for record in records if record["reference_label"] == "healthy"
        }),
        "records": records,
        "next_step": "Acquire separate healthy bearings for reference, tuning and evaluation before calibrating alerts.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "inspection_summary.json"
    features_path = args.out_dir / "window_features.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.concat(frames, ignore_index=True).to_csv(features_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {summary_path}")
    print(f"Saved: {features_path}")
    print("주의: 제공 라벨과 추출 특징을 확인한 결과이며, 이상 판정이나 정확도 평가 결과가 아닙니다.")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    main()
