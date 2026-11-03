"""Streamlit 진동 비교 화면. 한 번에 선택한 기록 하나만 읽어 메모리 사용을 제한한다."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from modules.report_generator import generate_report
from modules.vibration_config import FEATURE_LABELS, load_vibration_config
from modules.vibration_dataset import comparison_dir, load_manifest, read_windowed_record
from modules.vibration_report import generate_vibration_template_report
from modules.vibration_workflow import load_engine


def _rate(value: float | None) -> str:
    return "평가 대상 없음" if value is None else f"{value:.1%}"


def _render_metrics(folder: Path, model_id: str) -> None:
    """현재 모델과 연결된 결과만 보여 준다. 예전 모델의 평가 결과를 섞지 않는다."""
    for split, title in (("tuning", "설정 조정용 결과"), ("evaluation", "최종 평가 결과")):
        path = folder / f"{split}_metrics.json"
        if not path.is_file():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["model_id"] != model_id:
            continue
        with st.expander(title):
            rows = []
            for metric in report["metrics"].values():
                rows.append({"판정 방식": metric["name"],
                             "오경보율": _rate(metric["false_positive_rate"]),
                             "미탐률": _rate(metric["miss_rate"]),
                             "정상 무경고(TN)": metric["tn"], "정상 경고(FP)": metric["fp"],
                             "손상 미탐(FN)": metric["fn"], "손상 탐지(TP)": metric["tp"]})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption(f"기록 {report['record_count']}개 / 베어링 {report['bearing_count']}개. "
                       "연속 초과 구간이 하나 이상 있으면 해당 기록을 경고로 집계합니다.")
            st.caption("설정 조정용 결과는 최종 성능이 아닙니다. 중첩 구간은 독립 평가 표본으로 세지 않았습니다.")


def _read_upload(uploaded, data_root: Path, config):
    """업로드한 MAT만 임시 저장한다. 평가용 베어링은 분석 전에 차단한다."""
    name = uploaded.name
    if Path(name).name != name:
        raise ValueError("업로드 파일명에 폴더 경로를 포함할 수 없습니다.")
    if any(f"_{bearing}_" in name for bearing in config.splits["evaluation"]):
        raise ValueError("최종 평가용 베어링입니다. 설정 확정 후 evaluate 명령으로 평가해 주세요.")
    upload_root = (data_root / "raw" / "uploads").resolve()
    upload_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="record-", dir=upload_root) as directory:
        temporary = Path(directory).resolve()
        if not temporary.is_relative_to(upload_root):
            raise ValueError("임시 업로드 경로가 데이터 폴더를 벗어났습니다.")
        path = temporary / name
        path.write_bytes(uploaded.getvalue())
        return read_windowed_record(path, config)


def render_vibration_comparison(base_dir: Path) -> None:
    st.header("베어링 진동 비교 분석")
    st.caption("별도 정상 기록을 기준으로 통계 방식과 Isolation Forest를 비교합니다.")
    data_root = base_dir / "data" / "paderborn"
    folder = comparison_dir(data_root)
    try:
        config = load_vibration_config(base_dir / "config" / "vibration_analysis.yaml")
        manifest = load_manifest(data_root, config)
        engine = load_engine(data_root, config)
    except (ValueError, OSError) as error:
        st.info(str(error))
        st.code(".\\.venv\\Scripts\\python.exe run_vibration_analysis.py prepare\n"
                ".\\.venv\\Scripts\\python.exe run_vibration_analysis.py train", language="powershell")
        return

    _render_metrics(folder, engine.model_id)
    source = st.sidebar.radio("비교할 기록", ["설정 조정용 기록", "새 MAT 업로드"])
    try:
        if source == "설정 조정용 기록":
            entries = [entry for entry in manifest["records"] if entry["split"] == "tuning"]
            entry = st.sidebar.selectbox("MAT 기록", entries, format_func=lambda item: item["file"])
            record = read_windowed_record(data_root / entry["relative_path"], config, folder / "features")
            if record.metadata["source_sha256"] != entry["sha256"]:
                raise ValueError("준비 이후 원본 기록이 변경됐습니다. prepare를 다시 실행해 주세요.")
        else:
            uploaded = st.sidebar.file_uploader("원본 이름의 MAT 파일", type="mat")
            if uploaded is None:
                st.info("같은 운전 조건의 MAT 기록을 선택해 주세요.")
                return
            record = _read_upload(uploaded, data_root, config)
        result = engine.analyze(record)
    except (ValueError, OSError) as error:
        st.error(str(error))
        return

    meta = result["record"]
    st.caption(f"{meta['file']} · {meta['condition']} · {meta['duration_sec']:.3f}초")
    st.info("점수는 고장 확률이 아닙니다. '경고 없음'은 설정한 기준에서 이상 이력을 찾지 못했다는 뜻입니다.")
    tabs = st.tabs([method["name"] for method in result["methods"].values()])
    for tab, method in zip(tabs, result["methods"].values()):
        with tab:
            history, latest, count = st.columns(3)
            history.metric("전체 기록", "이상 이력 있음" if method["record_has_alert"] else "이상 이력 없음")
            latest.metric("마지막 구간", "경고" if method["last_window_confirmed_alert"] else "경고 없음")
            count.metric("이상 구간", len(method["events"]))
            windows = pd.DataFrame(method["windows"])
            chart = pd.DataFrame({"이상도": windows["anomaly_score"].to_numpy(),
                                  "비교 경계": method["threshold"]},
                                 index=pd.Index(windows["start_sec"], name="구간 시작 (초)"))
            st.line_chart(chart, height=240)
            st.caption(f"기록 점수 {method['record_score']:.6g} / 경계 {method['threshold']:.6g} · "
                       f"{config.minimum_consecutive_windows}개 구간 연속 초과 시 경고")
            if method["events"]:
                events = pd.DataFrame(method["events"])[["start_sec", "end_sec", "window_count", "peak_score", "threshold"]]
                st.dataframe(events.rename(columns={"start_sec": "시작 (초)", "end_sec": "끝 (초)",
                             "window_count": "구간 수", "peak_score": "최고 점수", "threshold": "경계"}), hide_index=True)
            with st.expander("구간별 수치와 판정"):
                st.dataframe(windows, hide_index=True)

    with st.expander("정상 기준과 추출 특징"):
        st.caption("이 표는 통계 방식의 특징 경계입니다. Isolation Forest의 특징 기여도는 아닙니다.")
        st.dataframe(pd.DataFrame(result["statistical_reference"]), hide_index=True)
        columns = st.columns(2)
        for index, name in enumerate(config.features):
            with columns[index % 2]:
                st.caption(FEATURE_LABELS[name])
                st.line_chart(record.features.set_index("start_sec")[[name]], height=150)
        for warning in meta["warnings"]:
            st.warning(warning)
        st.caption(f"정상 기준: {', '.join(engine.training_info['reference_bearings'])} / "
                   f"경계 설정: {', '.join(engine.training_info['calibration_bearings'])}")

    st.subheader("점검 보고서")
    use_ai = st.checkbox("생성형 AI로 문장 작성 (API 연결 필요)", value=False)
    # 파일 내용이나 모델이 달라지면 예전 보고서를 다른 결과에 표시하지 않는다.
    report_key = f"vibration_report:{engine.model_id}:{meta['file']}:{meta['source_sha256']}"
    if st.button("보고서 작성", type="primary"):
        with st.spinner("보고서 작성 중..."):
            st.session_state[report_key] = (generate_report(result) if use_ai
                                          else generate_vibration_template_report(result))
    if report_key in st.session_state:
        st.markdown(st.session_state[report_key])
        st.download_button("보고서 저장", st.session_state[report_key],
                           file_name=f"{Path(meta['file']).stem}_report.md", mime="text/markdown")
    st.download_button("분석 결과 JSON 저장", json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                       file_name=f"{Path(meta['file']).stem}_analysis.json", mime="application/json")
