"""화면에 작성한 마크다운 보고서를 한글 글꼴이 포함된 PDF로 변환한다."""

from __future__ import annotations

from functools import lru_cache
from html import escape
from io import BytesIO
from pathlib import Path

from markdown_it import MarkdownIt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


@lru_cache(maxsize=1)
def _register_fonts() -> None:
    folder = Path(__file__).resolve().parents[1] / "assets" / "fonts"
    for name, filename in (("ReportKorean", "NanumGothic-Regular.ttf"),
                           ("ReportKoreanBold", "NanumGothic-Bold.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(folder / filename)))
    pdfmetrics.registerFontFamily("ReportKorean", normal="ReportKorean", bold="ReportKoreanBold",
                                  italic="ReportKorean", boldItalic="ReportKoreanBold")


def _text(value: str) -> str:
    value = value.replace("⚠️", "주의: ").replace("⚠", "주의: ")
    return escape(value.replace("—", "-").replace("–", "-").replace("\u2011", "-"))


def _inline(tokens) -> str:
    """외부 HTML·이미지를 실행하거나 가져오지 않고 표시할 내용만 변환한다."""
    parts = []
    tags = {"strong_open": "<b>", "strong_close": "</b>",
            "em_open": "<i>", "em_close": "</i>",
            "softbreak": " ", "hardbreak": "<br/>"}
    for token in tokens or []:
        if token.type in tags:
            parts.append(tags[token.type])
        elif token.type == "code_inline":
            parts.append(_text(token.content))
        elif token.type == "image":
            parts.append(_text(f"[이미지: {token.content}]"))
        elif token.type not in ("link_open", "link_close"):
            parts.append(_text(token.content))
    return "".join(parts)


def generate_report_pdf(report: str, title: str = "모터 진단 보고서") -> bytes:
    """보고서를 재생성하거나 수치를 계산하지 않고 PDF 바이트를 메모리에 만든다."""
    _register_fonts()
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, title=title,
                                 author="Motor Risk Monitor", leftMargin=19 * mm,
                                 rightMargin=19 * mm, topMargin=21 * mm, bottomMargin=20 * mm)
    body = ParagraphStyle("ReportBody", fontName="ReportKorean", fontSize=9,
                          leading=14, spaceAfter=5, wordWrap="CJK", alignment=TA_LEFT,
                          textColor=colors.HexColor("#253348"))
    cell = ParagraphStyle("ReportCell", parent=body, fontSize=8, leading=12, spaceAfter=0)
    headings = {level: ParagraphStyle(
        f"ReportHeading{level}", parent=body, fontName="ReportKoreanBold",
        fontSize={1: 19, 2: 13, 3: 11}.get(level, 10),
        leading={1: 26, 2: 20, 3: 17}.get(level, 15),
        spaceBefore=12, spaceAfter=7, keepWithNext=True,
    ) for level in range(1, 7)}
    story = []
    tokens = MarkdownIt("commonmark", {"html": False}).enable("table").parse(report)
    lists = []
    list_prefix = None
    heading = None
    rows = None
    row = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        kind = token.type
        if kind == "heading_open":
            heading = int(token.tag[1])
        elif kind in ("bullet_list_open", "ordered_list_open"):
            lists.append({"ordered": kind == "ordered_list_open", "next": int(token.attrGet("start") or 1)})
        elif kind in ("bullet_list_close", "ordered_list_close"):
            lists.pop()
            list_prefix = None
        elif kind == "list_item_open":
            item = lists[-1]
            list_prefix = f"{item['next']}. " if item["ordered"] else "- "
            item["next"] += 1
        elif kind == "list_item_close":
            list_prefix = None
        elif kind == "table_open":
            rows = []
        elif kind == "tr_open":
            row = []
        elif kind == "tr_close":
            rows.append(row)
        elif kind == "table_close":
            if rows:
                table = Table(rows, colWidths=[document.width / len(rows[0])] * len(rows[0]),
                              repeatRows=1, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef5")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#c9d3df")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                story.extend([table, Spacer(1, 8)])
            rows = None
        elif kind == "inline":
            markup = _inline(token.children)
            if rows is not None:
                row.append(Paragraph(markup, cell))
            else:
                style = headings[heading] if heading else body
                if lists:
                    style = ParagraphStyle("ReportList", parent=style, leftIndent=12 * len(lists))
                story.append(Paragraph((list_prefix or "") + markup, style))
                list_prefix = None
                heading = None
        elif kind in ("fence", "code_block"):
            for line in token.content.splitlines():
                story.append(Paragraph(_text(line) or "&#160;", body))
        elif kind == "hr":
            story.extend([Spacer(1, 5), HRFlowable(width="100%", color=colors.HexColor("#c9d3df")),
                          Spacer(1, 8)])
        index += 1

    def page_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("ReportKorean", 8)
        canvas.setFillColor(colors.HexColor("#66758a"))
        canvas.drawString(document.leftMargin, 12 * mm, title)
        canvas.drawRightString(A4[0] - document.rightMargin, 12 * mm, f"{doc.page}")
        canvas.restoreState()

    document.build(story or [Paragraph("보고서 내용이 없습니다.", body)],
                   onFirstPage=page_footer, onLaterPages=page_footer)
    return buffer.getvalue()
