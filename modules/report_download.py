"""진동·CSV 보고서의 공통 다운로드 UI."""

from pathlib import Path

import streamlit as st

from modules.pdf_report import generate_report_pdf


def render_report_downloads(report: str, source_name: str) -> None:
    stem = Path(source_name).stem
    pdf_column, markdown_column = st.columns(2)
    with pdf_column:
        st.download_button("보고서 저장 (PDF)", generate_report_pdf(report),
                           file_name=f"{stem}_report.pdf", mime="application/pdf")
    with markdown_column:
        st.download_button("마크다운 저장", report,
                           file_name=f"{stem}_report.md", mime="text/markdown")
