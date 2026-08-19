"""Claude 리포트 생성기의 fallback과 API 응답 처리를 검증한다."""

import sys
from types import SimpleNamespace

from modules import report_generator


RESULT = {
    "equipment": "DC Motor",
    "status": "주의",
    "risk_score": 55,
    "abnormal_sensors": ["전류"],
    "score_explanation": ["전류 증가"],
    "main_pattern": "전류 증가",
    "suspected_causes": ["과부하"],
    "recommended_checks": ["부하 상태 확인"],
    "data_quality_warnings": [],
}


def test_missing_key_uses_template(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    report = report_generator.generate_report(RESULT)

    assert "55 / 100" in report
    assert "템플릿 기반" in report


def test_claude_text_response_is_returned(monkeypatch):
    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="Claude 진단 보고서")]
            )

    class FakeAnthropic:
        def __init__(self):
            self.messages = FakeMessages()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setitem(
        sys.modules,
        "anthropic",
        SimpleNamespace(Anthropic=FakeAnthropic),
    )

    report = report_generator.generate_report(RESULT)

    assert report == "Claude 진단 보고서"
    assert captured["system"] == report_generator.SYSTEM_PROMPT
    assert captured["model"] == report_generator.MODEL
    assert captured["messages"][0]["role"] == "user"
