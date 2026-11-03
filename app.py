"""
app.py
======
모터 이상징후 진단 대시보드 (Streamlit).

Paderborn 비교 분석: 별도 정상 기록으로 만든 통계 기준과 Isolation Forest를 비교한다.
Paderborn 진동 확인: 원신호·시간 구간별 특징을 확인한다.
기존 CSV 진단: 센서 4종의 규칙 기반 위험 점수를 표시한다.
보고서는 기본 템플릿과 선택적인 Claude 서술 기능을 제공한다.

실행:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from modules.risk_engine import RiskEngine
from modules.report_generator import generate_report

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = str(BASE_DIR / "config" / "risk_thresholds.yaml")
DATA_DIR = BASE_DIR / "data"

# 상태별 색상 (배지/강조용)
STATUS_COLOR = {"정상": "#2e7d32", "주의": "#ed6c02", "위험": "#d32f2f"}

# 센서 표시 정보 (컬럼명, 한글, 단위)
SENSORS = [
    ("temperature_c", "온도", "℃"),
    ("vibration_g", "진동", "g"),
    ("current_a", "전류", "A"),
    ("rpm", "RPM", "rpm"),
]


@st.cache_resource
def get_engine() -> RiskEngine:
    """엔진은 한 번만 로드해 재사용 (config 파싱 캐싱)."""
    return RiskEngine(CONFIG_PATH)


def load_dataframe() -> tuple[pd.DataFrame | None, str]:
    """사이드바에서 데이터 소스를 선택/업로드하고 (df, 이름)을 반환."""
    st.sidebar.header("① 데이터 선택")

    sample_files = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.exists() else []
    sample_names = [f.name for f in sample_files]

    mode = st.sidebar.radio(
        "데이터 소스",
        ["예제 데이터", "직접 업로드"],
        help="예제는 generate_data.py로 만든 CSV, 또는 내 CSV 파일 업로드",
    )

    if mode == "예제 데이터":
        if not sample_names:
            st.sidebar.warning("data/ 폴더에 예제 CSV가 없습니다.\n먼저 `py generate_data.py` 실행")
            return None, ""
        choice = st.sidebar.selectbox("예제 파일", sample_names)
        return pd.read_csv(sample_files[sample_names.index(choice)]), choice

    uploaded = st.sidebar.file_uploader("CSV 업로드", type="csv")
    if uploaded is not None:
        return pd.read_csv(uploaded), uploaded.name
    return None, ""


def render_summary(result: dict) -> None:
    """상태 배지 + 위험 점수 + 대표 패턴."""
    status = result["status"]
    color = STATUS_COLOR.get(status, "#555")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(
            f"<div style='padding:16px;border-radius:10px;background:{color};"
            f"color:white;text-align:center;'>"
            f"<div style='font-size:18px;'>설비 상태</div>"
            f"<div style='font-size:38px;font-weight:700;'>{status}</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("위험 점수", f"{result['risk_score']} / 100")
        st.progress(result["risk_score"] / 100)

    st.caption(f"대표 패턴: {result['main_pattern']}")


def render_sensor_charts(df: pd.DataFrame, result: dict) -> None:
    """센서 4개의 시계열 그래프 + 부분 점수 막대."""
    st.subheader("센서 시계열 (baseline → 분석 구간)")
    x = df["operating_time_sec"]
    cols = st.columns(2)
    for i, (col, kr, unit) in enumerate(SENSORS):
        with cols[i % 2]:
            st.caption(f"{kr} ({unit})")
            chart_df = pd.DataFrame({kr: df[col].to_numpy()}, index=x)
            st.line_chart(chart_df, height=180)

    st.subheader("센서별 부분 점수 (0~100)")
    scores = result["sensor_scores"]
    score_df = pd.DataFrame(
        {"점수": [scores["temperature"], scores["vibration"],
                 scores["current"], scores["rpm"]]},
        index=["온도", "진동", "전류", "RPM"],
    )
    st.bar_chart(score_df, height=220)


def render_diagnosis(result: dict) -> None:
    """탐지 패턴 · 의심 원인 · 권장 조치 · 근거."""
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("의심 원인")
        causes = result["suspected_causes"]
        if causes:
            for c in causes:
                st.markdown(f"- **{c}**")
        else:
            st.markdown("- 특이 원인 없음")

        st.subheader("탐지된 패턴")
        patterns = result["detected_patterns"]
        st.markdown("\n".join(f"- {p}" for p in patterns) if patterns else "- 없음")

    with c2:
        st.subheader("권장 점검")
        checks = result["recommended_checks"]
        st.markdown("\n".join(f"- {c}" for c in checks) if checks else "- 없음")

    with st.expander("점수 산출 근거 보기"):
        for line in result["score_explanation"]:
            st.markdown(f"- {line}")
        st.caption(f"분석 구간: baseline {result['analysis_window']['baseline_range_sec']}초 "
                   f"→ 분석 {result['analysis_window']['analysis_range_sec']}초")

    if result["data_quality_warnings"]:
        st.warning("데이터 품질 경고: " + "; ".join(result["data_quality_warnings"]))



@st.cache_data(show_spinner=False)
def load_paderborn_inspection(path: str, modified_ns: int, size: int,
                              window_sec: float, hop_sec: float):
    """파일 변경과 구간 설정을 캐시 키에 포함한다."""
    from modules.paderborn import extract_window_features, load_paderborn

    record = load_paderborn(path)
    return record, extract_window_features(record, window_sec, hop_sec)


def render_paderborn_inspection() -> None:
    import json
    import numpy as np
    from modules.paderborn import LABEL_NAMES

    st.header("Paderborn 진동 기록 확인")
    st.info("현재는 신호·특징 확인 단계입니다. '제공 상태'는 데이터셋의 라벨이며, 프로그램의 이상 판정이 아닙니다.")
    paths = sorted((DATA_DIR / "paderborn" / "raw").glob("*.mat"))
    if not paths:
        st.info("MAT 파일을 data/paderborn/raw/ 폴더에 원래 이름으로 넣어 주세요.")
        return
    choice = st.sidebar.selectbox("확인할 MAT 파일", paths, format_func=lambda path: path.name)
    with st.sidebar.expander("분석 구간 설정"):
        window_sec = st.number_input("구간 길이 (초)", min_value=0.05, max_value=2.0,
                                     value=0.25, step=0.05)
        hop_sec = st.number_input("이동 간격 (초)", min_value=0.025, max_value=2.0,
                                  value=0.125, step=0.025)
    if hop_sec > window_sec:
        st.error("이동 간격은 구간 길이 이하여야 합니다.")
        return
    loaded = []
    with st.spinner("진동 신호와 구간별 특징 확인 중..."):
        for path in paths:
            stat = path.stat()
            try:
                record, features = load_paderborn_inspection(
                    str(path), stat.st_mtime_ns, stat.st_size, window_sec, hop_sec)
            except ValueError as error:
                st.error(f"{path.name}: {error}")
                return
            loaded.append((path, record, features))
    _, record, features = next(item for item in loaded if item[0] == choice)
    meta = record.metadata
    conditions = meta["nominal_condition"]
    st.caption(
        f"{meta['condition']} · 설정 회전수 {conditions['rpm']:,} rpm · "
        f"설정 부하 토크 {conditions['torque_nm']} Nm · "
        f"설정 반경 방향 하중 {conditions['radial_force_n']:,} N"
    )
    cols = st.columns(4)
    cols[0].metric("제공 상태", LABEL_NAMES[meta["reference_label"]])
    cols[1].metric("기록 길이", f"{meta['duration_sec']:.3f} 초")
    cols[2].metric("진동 샘플 수", f"{meta['sample_count']:,}")
    cols[3].metric("분석 구간 수", len(features))

    st.subheader("진동 원신호")
    # 화면에서는 각 구간의 극값을 보존하고, 특징은 모든 샘플로 계산한다.
    signal = record.vibration
    edges = np.linspace(0, len(signal), min(1000, len(signal)) + 1, dtype=int)
    indices = [0, len(signal) - 1]
    for left, right in zip(edges[:-1], edges[1:]):
        part = signal[left:right]
        indices.extend((left + int(np.argmin(part)), left + int(np.argmax(part))))
    indices = np.unique(indices)
    unit_label = meta["unit"] or "원본 수치, 단위 미확인"
    st.line_chart(pd.DataFrame({f"진동 ({unit_label})": signal[indices]},
                              index=pd.Index(record.time_sec[indices], name="시간 (초)")), height=230)
    st.caption("화면에는 구간별 최솟값·최댓값을 남겼으며, 아래 특징은 전체 원신호로 계산했습니다.")

    st.subheader("구간별 진동 특징")
    feature_names = {
        "rms_centered": "RMS (구간 평균 제거)",
        "kurtosis_pearson": "첨도 (Pearson)",
        "peak_to_peak": "최대·최소 차이",
        "crest_factor": "파고율",
    }
    center = pd.Index((features["start_sec"] + features["end_sec"]) / 2, name="구간 중앙 시각 (초)")
    cols = st.columns(2)
    for index, (key, label) in enumerate(feature_names.items()):
        with cols[index % 2]:
            st.caption(label)
            st.line_chart(pd.DataFrame({label: features[key].to_numpy()}, index=center), height=170)
    st.caption(f"구간 {window_sec:g}초 · 이동 {hop_sec:g}초 · 진폭 정규화 없음 · 첨도·파고율은 무차원")

    st.subheader("확보한 기록 비교")
    condition_records = [item for item in loaded if item[1].metadata["condition"] == meta["condition"]]
    comparison = pd.DataFrame([
        {
            "베어링": item.metadata["bearing_id"],
            "제공 상태": LABEL_NAMES[item.metadata["reference_label"]],
            "파일": item.metadata["file"],
            "전체 RMS (평균 제거)": item.metadata["full_record_rms_centered"],
            "구간 첨도 중앙값": frame["kurtosis_pearson"].median(),
            "구간 파고율 중앙값": frame["crest_factor"].median(),
        }
        for _, item, frame in condition_records
    ])
    st.dataframe(comparison, hide_index=True)
    healthy_count = len({item.metadata["bearing_id"] for _, item, _ in condition_records
                         if item.metadata["reference_label"] == "healthy"})
    st.caption(f"같은 운전 조건의 기록만 비교했습니다. 확보한 정상 베어링은 {healthy_count}개입니다. "
               "진동 크기만으로 손상 여부를 결정하지 않으며, 기준 생성·설정 조정·최종 평가용 베어링을 별도로 확보해야 합니다.")
    with st.expander("단위·시간축과 상세 수치"):
        for warning in meta["warnings"]:
            st.warning(warning)
        st.json(meta)
        st.dataframe(features, hide_index=True)
    col_csv, col_json = st.columns(2)
    col_csv.download_button("구간별 특징 CSV 저장", features.to_csv(index=False).encode("utf-8-sig"),
                            file_name=f"{choice.stem}_features.csv", mime="text/csv")
    col_json.download_button("파일 확인 정보 JSON 저장", json.dumps(
        {"classification_performed": False, "feature_settings": {
            "window_sec": window_sec, "hop_sec": hop_sec,
            "centering": "subtract each window mean", "kurtosis": "Pearson m4/m2^2"},
         "record": meta}, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"),
         file_name=f"{choice.stem}_inspection.json", mime="application/json")


def main() -> None:
    st.set_page_config(page_title="모터 진단 대시보드", page_icon="🔧", layout="wide")
    st.title("🔧 모터 이상징후 진단 대시보드")
    mode = st.sidebar.radio("분석 종류", ["Paderborn 비교 분석", "Paderborn 진동 확인", "기존 CSV 진단"])
    if mode == "Paderborn 비교 분석":
        try:
            from modules.vibration_dashboard import render_vibration_comparison
        except ImportError:
            st.error("비교 분석 의존성이 필요합니다. 가상환경에서 pip install -r requirements.txt를 실행해 주세요.")
            return
        render_vibration_comparison(BASE_DIR)
        return
    if mode == "Paderborn 진동 확인":
        render_paderborn_inspection()
        return
    st.caption("규칙 기반 Risk Engine이 판정하고, AI는 결과를 리포트로 서술합니다.")

    df, name = load_dataframe()
    if df is None:
        st.info("← 왼쪽 사이드바에서 데이터를 선택하거나 업로드하세요.")
        return

    engine = get_engine()
    try:
        result = engine.analyze(df)
    except ValueError as e:
        st.error(f"입력 데이터 오류: {e}")
        return

    st.success(f"'{name}' 진단 완료 ({len(df)}행)")
    render_summary(result)
    st.divider()
    render_sensor_charts(df, result)
    st.divider()
    render_diagnosis(result)
    st.divider()

    # AI 리포트
    st.subheader("📄 AI 진단 리포트")
    st.caption("버튼을 누르면 리포트를 생성합니다.")
    # 파일 내용이 달라지면 이전 파일의 보고서를 보여 주지 않는다.
    import hashlib
    frame_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()
    report_key = f"csv_report:{name}:{frame_hash}"
    if st.button("리포트 생성", type="primary"):
        with st.spinner("리포트 작성 중..."):
            report = generate_report(result)
        st.session_state[report_key] = report
    if report_key in st.session_state:
        st.markdown(st.session_state[report_key])


if __name__ == "__main__":
    main()
