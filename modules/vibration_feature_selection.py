"""정상만 학습하는 모델의 입력 특징을 제한된 개발 실험으로 선택한다.

이전 실험의 검증된 특징 파일을 재사용한다. 개발 손상 라벨은 특징 순위와 후보
선택에만 사용하며, 모델 학습·경고 경계 설정에는 넣지 않는다. 독립 평가는 아니다.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Callable

import joblib
import matplotlib
matplotlib.use("Agg")  # 창을 열지 않고 PNG/PDF 파일만 생성한다.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

from modules.vibration_config import VibrationConfig
from modules.vibration_dataset import file_sha256, load_manifest, write_json
from modules.vibration_feature_experiment import (
    BASE_FEATURES, FEATURE_SETS, ExperimentRecord, ExperimentSettings, compare_feature_set,
)


FEATURE_POOL = FEATURE_SETS["combined"]
TOP_COUNTS = (2, 3, 4, 6)
MAXIMUM_CANDIDATES = 2 + len(FEATURE_POOL) + len(TOP_COUNTS)
MAXIMUM_CORRELATION = .90
STABILITY_SEEDS = (7, 19, 42, 73, 101)
BEARINGS = ("K001", "K002", "K003", "K004", "KA04", "KI04")
COLORS = ("#94a3b8", "#64748b", "#2563eb", "#0891b2", "#d97706", "#dc2626")


def _json_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_feature_source(source: Path, data_root: Path, config: VibrationConfig) -> dict:
    """이전 처리 조건·코드·개발 원본 목록이 같은지 확인한다. 평가 원본은 열지 않는다."""
    config.validate()
    if tuple(config.features) != BASE_FEATURES:
        raise ValueError("현재 기본 네 특징을 기준으로 하는 선택 실험입니다.")
    expected_splits = {"reference": {"K001", "K002"}, "tuning": {"K003", "K004", "KA04", "KI04"}}
    if any(set(config.splits[split]) != bearings for split, bearings in expected_splits.items()):
        raise ValueError("특징 선택 실험의 개발 베어링 분리가 달라졌습니다.")
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    identity = {key: value for key, value in protocol.items() if key != "protocol_id"}
    if _json_hash(identity) != protocol["protocol_id"]:
        raise ValueError("이전 특징 추출 프로토콜이 변경됐습니다.")
    # JSON에서는 tuple이 list가 되므로 정렬한 JSON 표현으로 비교한다.
    if _json_hash(protocol["config"]) != config.fingerprint:
        raise ValueError("이전 특징 추출과 현재 분석 설정이 다릅니다.")
    if protocol["settings"] != asdict(ExperimentSettings()):
        raise ValueError("주파수 특징 추출 조건이 달라졌습니다.")
    expected_code = {"vibration_feature_experiment.py", "vibration_engine.py", "paderborn.py"}
    if set(protocol["source_code_hashes"]) != expected_code:
        raise ValueError("이전 실험 코드의 해시 목록이 부족하거나 다릅니다.")
    for name, digest in protocol["source_code_hashes"].items():
        if Path(name).name != name:
            raise ValueError("예상하지 않은 이전 실험 코드 경로입니다.")
        if file_sha256(Path(__file__).with_name(name)) != digest:
            raise ValueError("특징 추출 코드가 변경됐습니다. 특징 비교 실험부터 다시 실행해 주세요.")
    entries = [e for e in load_manifest(data_root, config)["records"]
               if e["split"] in expected_splits]
    if entries != protocol["sources"]:
        raise ValueError("이전 특징 추출과 현재 개발 원본 목록이 다릅니다.")
    hashes = {}
    for entry in entries:
        if file_sha256(data_root / entry["relative_path"]) != entry["sha256"]:
            raise ValueError(f"개발 원본이 변경됐습니다: {entry['file']}")
        feature_path = source / "features" / f"{Path(entry['file']).stem}.json"
        hashes[feature_path.name] = file_sha256(feature_path)
    return {"protocol_id": protocol["protocol_id"], "entries": entries, "feature_hashes": hashes}


def load_feature_records(source: Path, validated: dict) -> dict[str, list[ExperimentRecord]]:
    """목록에 명시된 개발 특징만 읽고 라벨·조건·창의 유효성을 검사한다."""
    records = {"reference": [], "tuning": []}
    for entry in validated["entries"]:
        path = source / "features" / f"{Path(entry['file']).stem}.json"
        if file_sha256(path) != validated["feature_hashes"][path.name]:
            raise ValueError("검사 이후 특징 파일이 변경됐습니다.")
        cached = json.loads(path.read_text(encoding="utf-8"))
        meta = cached["metadata"]
        expected = {"file": entry["file"], "bearing_id": entry["bearing_id"],
                    "split": entry["split"], "condition": entry["condition"],
                    "source_sha256": entry["sha256"], "reference_label": entry["provided_label"]}
        if any(meta.get(key) != value for key, value in expected.items()):
            raise ValueError(f"특징 파일의 메타데이터가 원본 목록과 다릅니다: {path.name}")
        frame = pd.DataFrame(cached["windows"])
        required = {"window_id", "start_sec", "end_sec", *FEATURE_POOL}
        if not frame.empty:
            if required - set(frame) or not np.isfinite(frame[list(required)].to_numpy(float)).all():
                raise ValueError(f"특징에 필수 값이 없거나 유효하지 않습니다: {path.name}")
            ids = frame["window_id"].to_numpy()
            if (np.diff(ids) <= 0).any() or (ids != ids.astype(int)).any():
                raise ValueError("창 번호가 중복되었거나 시간순으로 정렬되지 않았습니다.")
        records[entry["split"]].append(ExperimentRecord(
            meta, frame, cached["excluded_windows"], cached["time_audit"]))
    return records


def record_medians(records: list[ExperimentRecord]) -> pd.DataFrame:
    """기록 하나를 점 하나로 요약한다. 중첩 창을 독립 표본으로 세지 않는다."""
    rows = []
    for record in records:
        if record.features.empty:
            raise ValueError("사용 가능한 특징이 없는 기록이 있어 선택 실험을 중단합니다.")
        rows.append({"file": record.metadata["file"], "bearing_id": record.metadata["bearing_id"],
                     **record.features[list(FEATURE_POOL)].median().to_dict()})
    return pd.DataFrame(rows)


def rank_features(medians: pd.DataFrame) -> list[dict]:
    """두 손상이 두 정상보다 일관되게 크거나 작은 특징을 우선한다.

    각 정상/손상 베어링 쌍의 기록 중앙값 AUC 네 개를 계산한다. 네 비교에서 모두
    같은 방향으로 구분되는 정도의 최솟값을 순위에 사용한다. 이는 개발용 순위
    지표이며 탐지율·특징 중요도·독립 검증 성능이 아니다.
    """
    ranking = []
    for feature in FEATURE_POOL:
        comparisons = {}
        for normal in ("K003", "K004"):
            for damage in ("KA04", "KI04"):
                a = medians.loc[medians["bearing_id"] == normal, feature].to_numpy()
                b = medians.loc[medians["bearing_id"] == damage, feature].to_numpy()
                if not len(a) or not len(b):
                    raise ValueError("특징 순위 계산에 필요한 개발 베어링이 부족합니다.")
                comparisons[f"{normal}/{damage}"] = float(roc_auc_score(
                    np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]))
        values = np.array(list(comparisons.values()))
        high, low = float(values.min()), float((1 - values).min())
        ranking.append({"feature": feature, "pairwise_auc": comparisons,
                        "common_direction": "higher" if high >= low else "lower",
                        "separation_score": max(0.0, 2 * max(high, low) - 1)})
    return sorted(ranking, key=lambda row: (-row["separation_score"], row["feature"]))


def build_candidates(ranking: list[dict], medians: pd.DataFrame) -> tuple[dict, list[dict]]:
    """대조군 2개 + 단일 특징 17개 + 순위 조합 최대 4개로 탐색을 제한한다."""
    candidates = {"baseline": BASE_FEATURES, "spectral": FEATURE_SETS["spectral"]}
    candidates.update({f"single_{feature}": (feature,) for feature in FEATURE_POOL})
    correlations = medians[list(FEATURE_POOL)].corr(method="spearman").abs()
    retained, redundancy = [], []
    for row in ranking:
        feature = row["feature"]
        similar = next((previous for previous in retained
                        if correlations.loc[feature, previous] >= MAXIMUM_CORRELATION), None)
        if similar is None:
            retained.append(feature)
        else:
            redundancy.append({"feature": feature, "similar_to": similar,
                               "absolute_spearman": float(correlations.loc[feature, similar])})
    for count in TOP_COUNTS:
        if len(retained) >= count:
            chosen = tuple(retained[:count])
            if set(chosen) not in [set(features) for features in candidates.values()]:
                candidates[f"ranked_{count}"] = chosen
    if len(candidates) > MAXIMUM_CANDIDATES:
        raise ValueError("미리 정한 특징 조합 실험 횟수를 초과했습니다.")
    return candidates, redundancy


def choose_candidate(candidates: dict) -> tuple[str | None, str | None]:
    """정상 FP 1개 이하·KA04 20개 탐지를 만족하고 이전 주파수 후보보다 나은지 본다."""
    baseline = candidates["baseline"]["methods"]["isolation_forest"]["metrics"]
    eligible = []
    for name, candidate in candidates.items():
        if name in ("baseline", "spectral"):
            continue
        metric = candidate["methods"]["isolation_forest"]["metrics"]
        if (metric["unscorable_record_count"] == 0 and metric["fp"] <= min(1, baseline["fp"])
                and metric["by_bearing"]["KA04"]["alert_record_count"] ==
                metric["by_bearing"]["KA04"]["record_count"]):
            eligible.append(name)
    best = min(eligible, key=lambda name: (
        -candidates[name]["methods"]["isolation_forest"]["metrics"]["tp"],
        candidates[name]["methods"]["isolation_forest"]["metrics"]["fp"],
        len(candidates[name]["features"]), name,
    )) if eligible else None
    previous = candidates["spectral"]["methods"]["isolation_forest"]["metrics"]
    improved = best if best and candidates[best]["methods"]["isolation_forest"]["metrics"]["tp"] > previous["tp"] else None
    return best, improved


def save_distribution_plot(medians: pd.DataFrame, features: list[str], path: Path,
                           bearings: tuple[str, ...] = BEARINGS) -> None:
    """점은 기록별 중앙값, 상자는 25–75%, 수염은 10–90% 범위를 표시한다."""
    columns = 2 if len(features) == 4 else min(3, len(features))
    rows = (len(features) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(5.2 * columns, 3.8 * rows), squeeze=False)
    rng = np.random.default_rng(42)  # 겹친 점을 조금 벌리는 시각화용 난수다.
    for axis, feature in zip(axes.flat, features):
        groups = [medians.loc[medians["bearing_id"] == bearing, feature].to_numpy()
                  for bearing in bearings]
        axis.boxplot(groups, positions=np.arange(len(bearings)), widths=.45, whis=(10, 90), showfliers=False,
                     medianprops={"color": "black", "linewidth": 1.2})
        for index, values in enumerate(groups):
            axis.scatter(index + rng.uniform(-.12, .12, len(values)), values,
                         color=COLORS[BEARINGS.index(bearings[index])], s=13, alpha=.65, zorder=3)
        axis.set_xticks(np.arange(len(bearings)), bearings, rotation=20)
        axis.set_title(feature, fontsize=10)
        axis.grid(axis="y", alpha=.2)
        axis.tick_params(labelsize=8)
    for axis in list(axes.flat)[len(features):]:
        axis.set_visible(False)
    subtitle = ("K001/K002 reference | K003/K004 normal tuning | KA04/KI04 damaged tuning"
                if bearings == BEARINGS else "K003/K004 normal tuning | KI04 damaged tuning")
    figure.suptitle(f"Development records: each dot is one record median\n{subtitle}", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .95 if rows <= 2 else .98))
    figure.savefig(path, dpi=160)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def render_selection_report(report: dict) -> str:
    lines = ["# 정상 학습 모델의 특징 선택 실험", "", "개발용 탐색 결과입니다. 최종 평가 성능이 아닙니다.", "",
             f"실험 조합: {len(report['candidates'])}개 / 사전 상한 {MAXIMUM_CANDIDATES}개.",
             "각 조합은 같은 정상 학습·정상 경계 설정·연속 경고 규칙을 사용합니다.", "",
             "| 특징 조합 | 특징 수 | 정상 오경보 | KA04 탐지 | KI04 탐지 | 손상 미탐률 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, candidate in report["candidates"].items():
        metric = candidate["methods"]["isolation_forest"]["metrics"]
        by = metric["by_bearing"]
        lines.append(f"| {name} | {len(candidate['features'])} | {metric['fp']}/{metric['fp'] + metric['tn']} | "
                     f"{by['KA04']['alert_record_count']}/{by['KA04']['record_count']} | "
                     f"{by['KI04']['alert_record_count']}/{by['KI04']['record_count']} | {metric['miss_rate']:.1%} |")
    lines += ["", f"조건 내 최상 후보: {report['best_feasible_candidate'] or '없음'}",
              f"이전 주파수 후보보다 개선된 후보: {report['recommended_candidate'] or '없음'}", "",
              "## 선택 후보의 특징", ""]
    best = report['best_feasible_candidate']
    if best:
        lines += [f"- {feature}" for feature in report['candidates'][best]['features']]
        lines += ["", "![정상과 KI04의 선택 특징 분포](normal_vs_ki04.png)", ""]
    lines += ["## 특징 순위", "", "| 특징 | 공통 구분 방향 | 개발용 구분 지표 |", "|---|---|---:|"]
    for row in report["ranking"]:
        lines.append(f"| {row['feature']} | {row['common_direction']} | {row['separation_score']:.3f} |")
    lines += ["", "순위는 기록 중앙값의 분포로 계산한 탐색 지표이며 모델의 특징 중요도가 아닙니다.",
              "모델은 원래 시간 창의 특징으로 학습·분석합니다. 창 중앙값만으로 판정하지 않습니다.", "",
              "![개발용 특징 분포](feature_distributions.png)", "", "## 난수 시드 민감도", "",
              "선택 후 고정한 후보만 확인합니다. 가장 좋은 시드를 골라 다시 추천하지 않습니다.", "",
              "| 조합 | 시드 | 정상 오경보 | KA04 탐지 | KI04 탐지 |", "|---|---:|---:|---:|---:|"]
    for name, results in report["stability"].items():
        for result in results:
            metric = result["metrics"]
            by = metric["by_bearing"]
            lines.append(f"| {name} | {result['seed']} | {metric['fp']} | "
                         f"{by['KA04']['alert_record_count']} | {by['KI04']['alert_record_count']} |")
    lines += ["", "## 해석 범위", "",
              "- 손상 라벨은 특징 순위·조합 선택에 사용했습니다. 모델 학습과 경계 설정에는 정상만 사용했습니다.",
              "- 같은 개발 기록으로 여러 후보를 비교했으므로 선택 후 성능에는 낙관적 편향이 있을 수 있습니다.",
              "- 같은 베어링의 20개 반복 기록은 독립 베어링 20개가 아닙니다. 새로운 손상 유형의 성능을 뜻하지 않습니다.",
              "- 시드를 바꾼 결과도 같은 데이터이므로 독립 검증이나 신뢰구간이 아닙니다.",
              "- 이전 실험과 같은 품질 제외 창을 사용하며 제외 창 양쪽을 이어 연속 경고로 만들지 않습니다.",
              "- 최종 평가 신호는 읽지 않았고, 기존 모델·설정·대시보드에는 반영하지 않았습니다.", "",
              "선택 원리 참고: [scikit-learn feature selection](https://scikit-learn.org/stable/modules/feature_selection.html).", ""]
    return "\n".join(lines)


def run_selection(source: Path, data_root: Path, config: VibrationConfig, output: Path,
                  progress: Callable[[str], None] = print) -> dict:
    """분포 확인 → 후보 고정 → 비교 → 고정 후보의 민감도 확인을 순서대로 수행한다."""
    if output.exists():
        raise ValueError("이전 결과를 덮어쓰지 않도록 새 출력 폴더를 지정해 주세요.")
    validated = validate_feature_source(source, data_root, config)
    protected_paths = [data_root / "derived/comparison" / name for name in (
        "models.joblib", "model_metadata.json", "tuning_metrics.json", "dataset_manifest.json")]
    protected = {str(path): file_sha256(path) for path in protected_paths if path.is_file()}
    plan = {"version": 1, "config": config.to_dict(), "source": str(source.resolve()),
            "input_protocol_id": validated["protocol_id"], "input_feature_hashes": validated["feature_hashes"],
            "feature_pool": FEATURE_POOL, "maximum_candidates": MAXIMUM_CANDIDATES,
            "top_counts": TOP_COUNTS, "maximum_correlation": MAXIMUM_CORRELATION,
            "ranking_rule": "max(0, 2 * max(min(pairwise_AUC), min(1-pairwise_AUC)) - 1)",
            "ranking_bearings": ["K003", "K004", "KA04", "KI04"],
            "stability_seeds": STABILITY_SEEDS,
            "selection_rule": ("FP <= min(1, baseline FP); all KA04 detected; "
                               "maximize TP then minimize FP/features; exceed spectral TP"),
            "protected_file_hashes": protected,
            "source_code_hashes": {name: file_sha256(Path(__file__).with_name(name))
                                   for name in ("vibration_feature_selection.py", "vibration_feature_experiment.py")},
            "versions": {"matplotlib": matplotlib.__version__, "pandas": pd.__version__,
                         "numpy": np.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__}}
    output.mkdir(parents=True)
    write_json(output / "analysis_plan.json", plan)
    records = load_feature_records(source, validated)
    medians = record_medians(records["reference"] + records["tuning"])
    tuning_medians = medians[medians["bearing_id"].isin(config.splits["tuning"])]
    ranking = rank_features(tuning_medians)
    feature_sets, redundancy = build_candidates(ranking, tuning_medians)
    write_json(output / "candidates.json", {"ranking": ranking, "feature_sets": feature_sets,
                                             "correlation_exclusions": redundancy})
    medians.to_csv(output / "record_feature_medians.csv", index=False)
    save_distribution_plot(medians, [row["feature"] for row in ranking], output / "feature_distributions.png")
    candidates, models = {}, {}
    for index, (name, features) in enumerate(feature_sets.items(), 1):
        progress(f"특징 조합 비교 {index}/{len(feature_sets)}: {name}")
        candidates[name], models[name] = compare_feature_set(records["reference"], records["tuning"], features, config)
    best, recommended = choose_candidate(candidates)
    stability = {}
    for name in dict.fromkeys(["baseline", "spectral", *([best] if best else [])]):
        stability[name] = []
        for seed in STABILITY_SEEDS:
            if seed == config.random_state:
                result = candidates[name]
            else:
                result, _ = compare_feature_set(records["reference"], records["tuning"],
                                                feature_sets[name], replace(config, random_state=seed))
            stability[name].append({"seed": seed, "metrics": result["methods"]["isolation_forest"]["metrics"]})
        progress(f"난수 시드 민감도 확인: {name}")
    if any(file_sha256(Path(path)) != digest for path, digest in protected.items()):
        raise ValueError("실험 도중 기존 모델이나 분석 파일이 변경됐습니다.")
    report = {"analysis_plan_id": _json_hash(plan), "evaluation_used": False,
              "input_protocol_id": validated["protocol_id"], "protected_file_hashes": protected,
              "reference_record_count": len(records["reference"]), "tuning_record_count": len(records["tuning"]),
              "ranking": ranking, "correlation_exclusions": redundancy, "candidates": candidates,
              "best_feasible_candidate": best, "recommended_candidate": recommended, "stability": stability,
              "excluded_window_count": sum(len(r.excluded_windows) for group in records.values() for r in group)}
    if best:
        save_distribution_plot(medians, list(feature_sets[best]), output / "selected_distributions.png")
        save_distribution_plot(medians, list(feature_sets[best]), output / "normal_vs_ki04.png",
                               bearings=("K003", "K004", "KI04"))
    joblib.dump({"analysis_plan_id": report["analysis_plan_id"], "models": models}, output / "selection_models.joblib")
    write_json(output / "selection.json", report)
    (output / "selection.md").write_text(render_selection_report(report), encoding="utf-8")
    return report
