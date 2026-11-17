"""다운로드한 PDF의 한글·수치·페이지 나눔과 화면 연결을 확인한다."""

from io import BytesIO

from pypdf import PdfReader
from streamlit.testing.v1 import AppTest

from modules.pdf_report import generate_report_pdf
from modules.report_generator import generate_template_report
from modules.risk_engine import RiskEngine
from modules.vibration_config import load_vibration_config
from modules.vibration_report import generate_vibration_template_report
from modules.vibration_selected_model import load_selected_engine, read_selected_record
from test_streamlit_deployment import ROOT


def test_vibration_pdf_preserves_korean_thresholds_and_embeds_fonts():
    root = ROOT / "data/paderborn"
    engine, _ = load_selected_engine(root, load_vibration_config(ROOT / "config/vibration_analysis.yaml"))
    record = read_selected_record(root / "raw/tuning/N09_M07_F10_KI04_1.mat", engine)
    result = engine.analyze(record)
    pdf = generate_report_pdf(generate_vibration_template_report(result))
    reader = PdfReader(BytesIO(pdf))
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert pdf.startswith(b"%PDF-")
    for value in ("진동 기록 분석 보고서", "통계 방식의 비교 기준", "N09_M07_F10_KI04_1.mat"):
        assert value in text
    for method in result["methods"].values():
        assert f"{method['threshold']:.6g}" in text
        assert f"{method['record_score']:.6g}" in text
    fonts = [font.get_object() for page in reader.pages
             for font in page["/Resources"]["/Font"].values()]
    assert any("/ToUnicode" in font and "/FontFile2" in font["/FontDescriptor"] for font in fonts)


def test_long_pdf_repeats_table_headers_and_preserves_special_text():
    report = "# 점검 보고서\n\n**한글 굵게** & 1 < 2 > 0\n\n| 구간 | 측정값 |\n|---|---|\n"
    report += "\n".join(f"| 구간 {i} | {i}.123456 |" for i in range(150))
    reader = PdfReader(BytesIO(generate_report_pdf(report)))
    assert len(reader.pages) >= 3
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "1 < 2 > 0" in text
    assert "149.123456" in text
    assert all("측정값" in page.extract_text() for page in reader.pages)


def test_csv_template_exports_to_pdf():
    import pandas as pd

    result = RiskEngine(str(ROOT / "config/risk_thresholds.yaml")).analyze(pd.read_csv(ROOT / "data/normal.csv"))
    reader = PdfReader(BytesIO(generate_report_pdf(generate_template_report(result))))
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "진단 요약" in text
    assert result["status"] in text
    assert f"{result['risk_score']} / 100" in text


def test_report_download_ui_offers_pdf_and_markdown():
    app = AppTest.from_string('''
from modules.report_download import render_report_downloads
render_report_downloads("## 한글 보고서\\n점수 0.603762", "sample.mat")
''').run()
    assert not app.exception
    downloads = app.get("download_button")
    assert [button.label for button in downloads] == ["보고서 저장 (PDF)", "마크다운 저장"]
