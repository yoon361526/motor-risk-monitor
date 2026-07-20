"""
report_generator.py
===================
Risk Engine가 산출한 진단 결과(result.json)를 바탕으로,
Claude(생성형 AI)가 사람이 읽기 좋은 설비 진단 보고서를 자연어로 작성한다.

★ 역할 분담 (설계 원칙)
  - 판정(위험 점수/상태/패턴/원인)은 이미 규칙 기반 Risk Engine이 끝냈다.
  - AI는 판정을 다시 하지 않는다. 주어진 근거(result.json)를 '해석·서술'만 한다.
    => 이렇게 하면 설명 가능성(왜 그 결론인지)이 유지된다.

인증
  - 환경변수 ANTHROPIC_API_KEY 가 있으면 그대로 동작한다.
  - 키가 없으면 안내 메시지를 출력하고 종료한다(엔진/테스트에는 영향 없음).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

# Windows 터미널(cp949)에서 한글이 깨지지 않도록 출력 인코딩을 UTF-8로 고정
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# Claude 모델 ID (Anthropic 최신 Opus)
MODEL = "claude-opus-4-8"

# -----------------------------------------------------------------------------
# 시스템 프롬프트: AI의 역할과 제약을 못박는다.
#   - 판정 금지, 근거 밖 추측 금지, 한국어, 구조화된 보고서.
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """당신은 산업 설비(모터) 진단 보고서를 작성하는 정비 엔지니어입니다.

중요한 규칙:
1. 위험 점수, 상태(정상/주의/위험), 탐지된 패턴, 의심 원인은 이미 규칙 기반 진단
   엔진이 계산해 당신에게 제공됩니다. 당신은 이 판정을 절대 바꾸거나 재계산하지 않습니다.
2. 제공된 데이터(JSON)에 근거해서만 서술합니다. 데이터에 없는 수치나 원인을 지어내지 마세요.
3. 센서 변화율(sensor_changes)과 부분 점수(sensor_scores)를 인용해, 왜 이런 진단이
   나왔는지 근거를 설명합니다. (설명 가능성)
4. 한국어로, 현장 담당자가 바로 이해하고 조치할 수 있게 작성합니다.

보고서 구성(이 순서, 마크다운):
## 1. 진단 요약
   - 설비, 최종 상태, 위험 점수(0~100)를 한두 문장으로.
## 2. 근거 분석
   - baseline 대비 각 센서가 얼마나 변했는지, 어떤 센서가 이상 신호를 냈는지.
   - 변화율(%)과 부분 점수를 근거로 제시.
## 3. 의심 원인 및 대표 패턴
   - 탐지된 조합 패턴과 그로부터 추론된 원인.
## 4. 권장 조치
   - recommended_checks를 우선순위 있게, 실행 가능한 문장으로.
"""


def build_user_prompt(result: Dict[str, Any]) -> str:
    """진단 결과 JSON을 AI에게 전달할 사용자 메시지로 구성한다."""
    result_json = json.dumps(result, ensure_ascii=False, indent=2)
    return (
        "아래는 규칙 기반 진단 엔진이 산출한 모터 진단 결과입니다. "
        "이 결과를 근거로 진단 보고서를 작성해 주세요.\n\n"
        f"```json\n{result_json}\n```"
    )


def generate_report(result: Dict[str, Any]) -> str:
    """
    result(dict)를 받아 Claude로 진단 보고서(마크다운 문자열)를 생성해 반환한다.
    ANTHROPIC_API_KEY 미설정 시 RuntimeError를 발생시킨다.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY 환경변수가 설정되어 있지 않습니다.\n"
            "  PowerShell 예:  $env:ANTHROPIC_API_KEY = 'sk-ant-...'\n"
            "  설정 후 다시 실행하세요."
        )

    import anthropic  # 키가 있을 때만 import (없어도 파일 로드는 되게)

    client = anthropic.Anthropic()  # 키는 환경변수에서 자동으로 읽음

    # 보고서 생성은 구조화된 데이터로부터의 서술 작업 → adaptive thinking 사용
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(result)}],
    )

    # 응답에서 텍스트 블록만 추출 (thinking 블록은 건너뜀)
    text = "".join(
        block.text for block in response.content if block.type == "text"
    )
    return text.strip()


def main():
    # 입력: result.json 경로 (기본값 result.json)
    result_path = sys.argv[1] if len(sys.argv) > 1 else "result.json"

    if not os.path.exists(result_path):
        print(f"[오류] 결과 파일을 찾을 수 없습니다: {result_path}")
        print("       먼저 run_analysis.py를 실행해 result.json을 생성하세요.")
        sys.exit(1)

    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    try:
        report = generate_report(result)
    except RuntimeError as e:
        print(f"[안내] {e}")
        sys.exit(1)

    # 화면 출력
    print(report)

    # 보고서를 UTF-8 마크다운 파일로도 저장 (터미널 한글 깨짐 방지)
    out_path = "report.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[저장됨] {out_path}")


if __name__ == "__main__":
    main()
