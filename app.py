"""
app.py
======
모터 이상징후 진단 대시보드 (Streamlit).

CSV를 선택/업로드하면 규칙 기반 Risk Engine이 진단하고, 위험 점수·센서 변화·
탐지 패턴·원인을 시각화한다. 버튼을 누르면 (API 키가 있으면) Claude가, 없으면
템플릿이 진단 리포트를 작성한다.

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


def main() -> None:
    st.set_page_config(page_title="모터 진단 대시보드", page_icon="🔧", layout="wide")
    st.title("🔧 모터 이상징후 진단 대시보드")
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
    if st.button("리포트 생성", type="primary"):
        with st.spinner("리포트 작성 중..."):
            report = generate_report(result)
        st.session_state["report"] = report
    if "report" in st.session_state:
        st.markdown(st.session_state["report"])


if __name__ == "__main__":
    main()
