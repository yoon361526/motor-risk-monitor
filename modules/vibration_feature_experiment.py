"""기존 진단 모델을 바꾸지 않고 진동 특징 조합을 비교하는 개발용 실험.

reference와 tuning만 읽는다. 손상 정답은 학습·경계 설정에 넣지 않고,
비교표와 후보 추천에만 사용한다. 이 결과는 최종 평가 성능이 아니다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, hilbert, sosfiltfilt, welch
import sklearn
from sklearn.ensemble import IsolationForest

from modules.paderborn import VibrationRecord, extract_window_features, load_paderborn
from modules.vibration_config import VibrationConfig
from modules.vibration_dataset import file_sha256, load_manifest, write_json
from modules.vibration_engine import StatisticsReference, consecutive_record_score


BASE_FEATURES = ("rms_centered", "kurtosis_pearson", "peak_to_peak", "crest_factor")
POWER_BANDS = ((0, 1000), (1000, 3000), (3000, 6000), (6000, 12000), (12000, 16000))
SPECTRAL_FEATURES = tuple(f"power_ratio_{lo}_{hi}" for lo, hi in POWER_BANDS) + (
    "spectral_centroid_hz", "spectral_entropy",
)
ENVELOPE_FEATURES = (
    "envelope_modulation_depth", "envelope_kurtosis",
    "envelope_ratio_20_200", "envelope_ratio_200_500", "envelope_ratio_500_1000",
    "envelope_peak_fraction",
)
FEATURE_SETS = {
    "baseline": BASE_FEATURES,
    "spectral": BASE_FEATURES + SPECTRAL_FEATURES,
    "envelope": BASE_FEATURES + ENVELOPE_FEATURES,
    "combined": BASE_FEATURES + SPECTRAL_FEATURES + ENVELOPE_FEATURES,
}


@dataclass(frozen=True)
class ExperimentSettings:
    """결과를 보기 전에 고정하는 처리 조건. 대역은 결함 주파수를 뜻하지 않는다."""

    version: int = 1
    sample_rate_hz: float = 64000.0
    maximum_gap_samples: float = 4.0
    maximum_local_rate_error: float = 0.05
    welch_samples: int = 8192
    envelope_low_hz: float = 2000.0
    envelope_high_hz: float = 12000.0
    filter_order: int = 4


@dataclass
class ExperimentRecord:
    """기록의 특징 표와 품질 제외 창. 정답 라벨은 metadata에만 보관한다."""

    metadata: dict
    features: pd.DataFrame
    excluded_windows: list[dict]
    time_audit: dict


def _power_spectrum(values: np.ndarray, settings: ExperimentSettings):
    segment = min(settings.welch_samples, len(values))
    return welch(values, fs=settings.sample_rate_hz, window="hann", nperseg=segment,
                 noverlap=segment // 2, detrend="constant", scaling="density")


def _fraction(frequencies, power, low: float, high: float, denominator: float) -> float:
    # 같은 FFT 격자에서는 df가 분자·분모에서 소거된다. 위쪽 경계는 exclusive다.
    return float(power[(frequencies >= low) & (frequencies < high)].sum() / denominator)


def spectral_features(values: np.ndarray, settings: ExperimentSettings) -> dict[str, float]:
    """고른 시간 간격의 신호에서 에너지 분포와 진폭 포락선의 변동을 계산한다.

    엔벨로프는 2~12 kHz 대역을 통과한 신호의 Hilbert 절댓값이다.
    특정 베어링의 결함 주파수나 손상 위치로 해석하지 않는다.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < settings.welch_samples or not np.isfinite(values).all():
        raise ValueError("주파수 특징에 필요한 유효한 샘플이 부족합니다.")
    values = values - values.mean()
    frequencies, power = _power_spectrum(values, settings)
    total = float(power.sum())
    if total <= 0:
        raise ValueError("에너지가 없는 신호의 주파수 특징을 계산할 수 없습니다.")
    probabilities = power / total
    positive = probabilities[probabilities > 0]
    result = {f"power_ratio_{lo}_{hi}": _fraction(frequencies, power, lo, hi, total)
              for lo, hi in POWER_BANDS}
    result["spectral_centroid_hz"] = float(np.dot(frequencies, probabilities))
    result["spectral_entropy"] = float(-(positive * np.log(positive)).sum() / np.log(len(power)))

    sos = butter(settings.filter_order, (settings.envelope_low_hz, settings.envelope_high_hz),
                 btype="bandpass", fs=settings.sample_rate_hz, output="sos")
    # 반사 패딩 후 중앙 부분만 사용해 필터·Hilbert의 창 끝 효과를 줄인다.
    padding = min(1024, len(values) // 4)
    padded = np.pad(values, padding, mode="reflect")
    envelope = np.abs(hilbert(sosfiltfilt(sos, padded)))[padding:-padding]
    mean = float(envelope.mean())
    centered = envelope - mean
    variance = float(np.mean(centered ** 2))
    if mean <= 0 or variance <= 0:
        raise ValueError("포락선의 변동이 없어 특징을 정의할 수 없습니다.")
    envelope_f, envelope_power = _power_spectrum(centered, settings)
    band = (envelope_f >= 20) & (envelope_f < 1000)
    envelope_total = float(envelope_power[band].sum())
    if envelope_total <= 0:
        raise ValueError("포락선 비교 대역에 에너지가 없습니다.")
    result.update({
        "envelope_modulation_depth": float(np.sqrt(variance) / mean),
        "envelope_kurtosis": float(np.mean(centered ** 4) / variance ** 2),
        **{f"envelope_ratio_{lo}_{hi}": _fraction(envelope_f, envelope_power, lo, hi, envelope_total)
           for lo, hi in ((20, 200), (200, 500), (500, 1000))},
        "envelope_peak_fraction": float(envelope_power[band].max() / envelope_total),
    })
    if not np.isfinite(list(result.values())).all():
        raise ValueError("주파수 특징에 비정상 수치가 있습니다.")
    return result


def extract_experiment_record(record: VibrationRecord, config: VibrationConfig,
                              settings: ExperimentSettings) -> ExperimentRecord:
    """원래 창 경계·시간 특징은 유지하고, 주파수용 신호만 실제 시각으로 보간한다.

    큰 시간 공백이나 국소 샘플 수 이상이 있는 창은 네 후보 모두에서 제외한다.
    빠진 창을 건너뛰어 연속 경고가 이어지지 않도록 window_id를 보존한다.
    """
    base = extract_window_features(record, config.window_sec, config.hop_sec)
    time, signal = record.time_sec, record.vibration
    intervals = np.diff(time)
    rows, excluded = [], []
    for row in base.to_dict("records"):
        start, end = row["start_sec"], row["end_sec"]
        left = max(0, int(np.searchsorted(time, start)) - 1)
        right = min(len(time), int(np.searchsorted(time, end, side="right")) + 1)
        local_time = time[left:right]
        gaps = np.diff(local_time)
        local_fs = (len(local_time) - 1) / (local_time[-1] - local_time[0])
        reason = None
        if gaps.max() > settings.maximum_gap_samples / settings.sample_rate_hz:
            reason = "time_gap"
        elif abs(local_fs / settings.sample_rate_hz - 1) > settings.maximum_local_rate_error:
            reason = "local_sample_rate"
        elif not np.isfinite([row[name] for name in BASE_FEATURES]).all():
            reason = "undefined_base_feature"
        # 외삽하지 않고 원래 시각 범위 안에서만 64 kHz 격자로 선형 보간한다.
        grid = np.arange(start, end, 1 / settings.sample_rate_hz)
        if len(grid) < settings.welch_samples:
            reason = reason or "short_window"
        if reason:
            excluded.append({"window_id": row["window_id"], "start_sec": start,
                             "end_sec": end, "reason": reason})
            continue
        uniform = np.interp(grid, local_time, signal[left:right])
        rows.append({**row, **spectral_features(uniform, settings)})
    frame = pd.DataFrame(rows)
    audit = {"effective_sample_rate_hz": float((len(time) - 1) / time[-1]),
             "minimum_interval_sec": float(intervals.min()),
             "median_interval_sec": float(np.median(intervals)),
             "maximum_interval_sec": float(intervals.max()),
             "original_window_count": len(base), "usable_window_count": len(frame),
             "excluded_window_count": len(excluded)}
    return ExperimentRecord(dict(record.metadata), frame, excluded, audit)


def record_score(frame: pd.DataFrame, scores: np.ndarray, minimum_windows: int) -> float | None:
    """품질 검사에서 제외한 창을 넘어 연속 경고를 만들지 않는다."""
    if len(frame) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("창과 점수 길이 또는 점수의 유효성을 확인해 주세요.")
    if frame.empty:
        return None
    breaks = np.flatnonzero(np.diff(frame["window_id"].to_numpy()) != 1) + 1
    runs = np.split(np.asarray(scores), breaks)
    eligible = [consecutive_record_score(run, minimum_windows)
                for run in runs if len(run) >= minimum_windows]
    return max(eligible) if eligible else None


def _metrics(predictions: list[dict]) -> dict:
    evaluated = [p for p in predictions if p["record_has_alert"] is not None]
    normal = [p for p in evaluated if p["provided_label"] == "healthy"]
    damaged = [p for p in evaluated if p["provided_label"] != "healthy"]
    fp = sum(p["record_has_alert"] for p in normal)
    tp = sum(p["record_has_alert"] for p in damaged)
    by_bearing = {}
    for row in predictions:
        counts = by_bearing.setdefault(row["bearing_id"], {
            "record_count": 0, "alert_record_count": 0, "unscorable_record_count": 0})
        counts["record_count"] += 1
        counts["alert_record_count"] += int(row["record_has_alert"] is True)
        counts["unscorable_record_count"] += int(row["record_has_alert"] is None)
    return {"tn": len(normal) - fp, "fp": fp, "fn": len(damaged) - tp, "tp": tp,
            "false_positive_rate": fp / len(normal) if normal else None,
            "miss_rate": (len(damaged) - tp) / len(damaged) if damaged else None,
            "evaluated_record_count": len(evaluated),
            "unscorable_record_count": len(predictions) - len(evaluated), "by_bearing": by_bearing}


def compare_feature_set(reference: list[ExperimentRecord], tuning: list[ExperimentRecord],
                        feature_names: tuple[str, ...], config: VibrationConfig) -> tuple[dict, dict]:
    """정상만 학습·경계 설정한다. 정답 라벨은 예측 이후 성능 집계에 사용한다."""
    if any(r.metadata["reference_label"] != "healthy" for r in reference):
        raise ValueError("학습에는 정상 기록만 사용할 수 있습니다.")
    train = np.concatenate([r.features[list(feature_names)].to_numpy(float)
                            for r in reference if not r.features.empty])
    forest = IsolationForest(n_estimators=config.n_estimators, random_state=config.random_state,
                             contamination="auto", n_jobs=1).fit(train)
    statistics = StatisticsReference.fit(train)
    scorers = {"statistics": statistics.score, "isolation_forest": lambda x: -forest.score_samples(x)}
    methods = {}
    for name, scorer in scorers.items():
        scores = []
        for r in tuning:
            values = r.features[list(feature_names)].to_numpy(float) if not r.features.empty else None
            score = (record_score(r.features, scorer(values), config.minimum_consecutive_windows)
                     if values is not None else None)
            scores.append(score)
        normal_scores = [s for r, s in zip(tuning, scores)
                         if r.metadata["reference_label"] == "healthy" and s is not None]
        if not normal_scores:
            raise ValueError("경계 설정용 정상 점수가 없습니다.")
        threshold = float(np.quantile(normal_scores, config.calibration_quantile, method="higher"))
        predictions = [{"file": r.metadata["file"], "bearing_id": r.metadata["bearing_id"],
                        "provided_label": r.metadata["reference_label"], "record_score": score,
                        "record_has_alert": bool(score > threshold) if score is not None else None}
                       for r, score in zip(tuning, scores)]
        methods[name] = {"threshold": threshold, "calibration_record_count": len(normal_scores),
                         "metrics": _metrics(predictions), "predictions": predictions}
    return {"features": list(feature_names), "methods": methods}, {
        "forest": forest, "statistics": statistics, "features": feature_names,
        "thresholds": {name: value["threshold"] for name, value in methods.items()},
    }


def recommend_candidate(candidates: dict) -> str | None:
    """동일 이하 FP, KA04 성능 유지, KI04 개선을 모두 만족하는 IF 후보만 추천한다."""
    baseline = candidates["baseline"]["methods"]["isolation_forest"]["metrics"]
    eligible = []
    for name, candidate in candidates.items():
        metric = candidate["methods"]["isolation_forest"]["metrics"]
        by = metric["by_bearing"]
        previous = baseline["by_bearing"]
        if (name != "baseline" and metric["unscorable_record_count"] == 0
                and metric["fp"] <= baseline["fp"]
                and by["KA04"]["alert_record_count"] >= previous["KA04"]["alert_record_count"]
                and by["KI04"]["alert_record_count"] > previous["KI04"]["alert_record_count"]):
            eligible.append(name)
    return min(eligible, key=lambda name: (
        -candidates[name]["methods"]["isolation_forest"]["metrics"]["tp"],
        candidates[name]["methods"]["isolation_forest"]["metrics"]["fp"],
        len(candidates[name]["features"]), name,
    )) if eligible else None


def _feature_summary(records: list[ExperimentRecord]) -> list[dict]:
    """중첩 창을 독립 표본으로 세지 않고, 기록별 중앙값의 분포를 저장한다."""
    rows = []
    for r in records:
        if r.features.empty:
            continue
        medians = r.features[list(FEATURE_SETS["combined"])].median()
        rows.extend({"bearing_id": r.metadata["bearing_id"], "feature": name, "value": float(value)}
                    for name, value in medians.items())
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    return [{"bearing_id": bearing, "feature": feature, "record_count": len(group),
             "p10": float(group["value"].quantile(.1)), "median": float(group["value"].median()),
             "p90": float(group["value"].quantile(.9))}
            for (bearing, feature), group in frame.groupby(["bearing_id", "feature"], sort=True)]


def render_report(report: dict) -> str:
    lines = ["# 진동 특징 보완 비교 실험", "", "개발용 결과이며 최종 평가 성능이 아닙니다.", "",
             f"정상 기준 학습 {report['reference_record_count']}개와 "
             f"설정 조정용 {report['tuning_record_count']}개를 사용합니다. 최종 평가 신호는 읽지 않습니다.", "",
             "| 특징 조합 | 방식 | 정상 오경보 | KA04 탐지 | KI04 탐지 | 미탐률 |",
             "|---|---|---:|---:|---:|---:|"]
    table_candidates = dict(report["candidates"])
    if report.get("production_baseline"):
        table_candidates = {"saved_production": report["production_baseline"], **table_candidates}
    for name, candidate in table_candidates.items():
        for method, result in candidate["methods"].items():
            m = result["metrics"]
            by = m['by_bearing']
            lines.append(f"| {name} | {method} | {m['fp']}/{m['fp'] + m['tn']} | "
                         f"{by['KA04']['alert_record_count']}/{by['KA04']['record_count']} | "
                         f"{by['KI04']['alert_record_count']}/{by['KI04']['record_count']} | {m['miss_rate']:.1%} |")
    lines += ["", f"조건을 만족하는 추천 후보: {report['recommended_candidate'] or '없음'}", "",
              f"품질 검사로 제외한 창: {sum(r['time_audit']['excluded_window_count'] for r in report['records'])}개.",
              "제외 창은 모든 후보에 동일하게 적용하며, 제외 창을 건너뛰어 연속 경고를 만들지 않습니다.",
              "분석 불가능한 기록은 정상으로 집계하지 않습니다. 기록별 제외 사유는 JSON을 확인하세요.", "",
              f"- 정상 기록만으로 학습하고 정상 tuning 기록의 {report['calibration_quantile']:.0%} 분위수로 경계를 정했습니다.",
              f"- 후보별 경계값은 다르지만 경계 설정 절차·연속 {report['minimum_consecutive_windows']}개 초과 규칙은 같습니다.",
              "- 기준 이하 오경보, KA04 탐지 유지, KI04 탐지 증가를 모두 만족해야 추천합니다.",
              "- 개발 손상 라벨을 후보 선택에 참고했으므로 독립 검증이 필요합니다.",
              "- 20개 기록은 같은 베어링의 반복 측정입니다. 독립 손상 베어링 20개가 아닙니다.",
              "- 선형 보간은 누락된 물리 신호를 복원하거나 센서 시간축의 정확성을 증명하지 않습니다.",
              "- saved_production은 기존 저장 결과, baseline은 품질 제외 조건을 맞춰 재학습한 대조군입니다.",
              "- 본 실험 모델은 기존 대시보드에 자동 적용하지 않습니다.", "",
              "계산 방법: [SciPy Welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html), "
              "[SciPy Hilbert](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html).", ""]
    return "\n".join(lines)


def run_experiment(data_root: Path, config: VibrationConfig, output: Path,
                   progress: Callable[[str], None] = print) -> dict:
    """새 실험 폴더에만 저장한다. 기존 모델·설정·최종 평가 결과를 수정하지 않는다."""
    if output.exists():
        raise ValueError("새 출력 폴더를 지정해 주세요. 이전 실험은 덮어쓰지 않습니다.")
    settings = ExperimentSettings()
    config.validate()
    if tuple(config.features) != BASE_FEATURES or set(config.splits['tuning']) != {'K003', 'K004', 'KA04', 'KI04'}:
        raise ValueError("이 실험은 현재 네 기본 특징과 K003·K004·KA04·KI04 개발 분리를 대상으로 합니다.")
    manifest = load_manifest(data_root, config)
    protected_paths = [data_root / "derived/comparison" / name for name in (
        "models.joblib", "model_metadata.json", "tuning_metrics.json", "dataset_manifest.json")]
    protected = {str(p): file_sha256(p) for p in protected_paths if p.is_file()}
    production = None
    metrics_path = data_root / "derived/comparison/tuning_metrics.json"
    if metrics_path.is_file():
        stored = json.loads(metrics_path.read_text(encoding="utf-8"))
        if stored["config_fingerprint"] != config.fingerprint:
            raise ValueError("저장된 기존 성능과 현재 설정이 다릅니다.")
        production = {"model_id": stored["model_id"], "methods": {
            name: {"metrics": value} for name, value in stored["metrics"].items()}}
    # 목록의 evaluation 정보는 분리 확인에만 사용하고 원본 경로를 열지 않는다.
    entries = [e for e in manifest["records"] if e["split"] in ("reference", "tuning")]
    protocol = {"settings": asdict(settings), "feature_sets": FEATURE_SETS,
                "config": config.to_dict(), "sources": entries,
                "source_code_hashes": {
                    name: file_sha256(Path(__file__).with_name(name)) for name in (
                        "vibration_feature_experiment.py", "vibration_engine.py", "paderborn.py")},
                "selection_rule": ("IF: FP <= control; KA04 >= control; KI04 > control; "
                                   "maximize TP, minimize FP, feature count"),
                "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}}
    protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True)
    # 특징을 읽기 전에 후보·대역·선택 조건을 저장해 시도한 실험을 남긴다.
    write_json(output / "protocol.json", {**protocol, "protocol_id": protocol_id})
    records = {"reference": [], "tuning": []}
    for index, entry in enumerate(entries, 1):
        path = data_root / entry["relative_path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"준비 이후 원본 파일이 변경됐습니다: {path.name}")
        r = extract_experiment_record(load_paderborn(path), config, settings)
        r.metadata.update({"source_sha256": entry["sha256"], "split": entry["split"]})
        records[entry["split"]].append(r)
        write_json(output / "features" / f"{path.stem}.json", {
            "metadata": r.metadata, "windows": r.features.to_dict("records"),
            "excluded_windows": r.excluded_windows, "time_audit": r.time_audit})
        if index % 10 == 0:
            progress(f"특징 계산: {index}/{len(entries)}")
    candidates, models = {}, {}
    for name, features in FEATURE_SETS.items():
        progress(f"정상 학습·비교: {name} ({len(features)}개 특징)")
        candidates[name], models[name] = compare_feature_set(records["reference"], records["tuning"], features, config)
    if any(file_sha256(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError("실험 도중 기존 비교 분석 파일이 변경됐습니다.")
    report = {"protocol_id": protocol_id, "evaluation_used": False,
              "calibration_quantile": config.calibration_quantile,
              "minimum_consecutive_windows": config.minimum_consecutive_windows,
              "protected_file_hashes": protected, "production_baseline": production,
              "reference_record_count": len(records["reference"]), "tuning_record_count": len(records["tuning"]),
              "candidates": candidates, "recommended_candidate": recommend_candidate(candidates),
              "records": [{"file": r.metadata["file"], "split": r.metadata["split"],
                           "time_audit": r.time_audit, "excluded_windows": r.excluded_windows}
                          for group in records.values() for r in group],
              "feature_distributions": _feature_summary(records["reference"] + records["tuning"])}
    joblib.dump({"protocol": protocol, "models": models}, output / "experiment_models.joblib")
    write_json(output / "comparison.json", report)
    (output / "comparison.md").write_text(render_report(report), encoding="utf-8")
    return report
