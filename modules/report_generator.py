"""
report_generator.py
===================
Risk Engine가 산출한 진단 결과(result.json)를 바탕으로,
생성형 AI(Anthropic Claude)가 사람이 읽기 좋은 설비 진단 보고서를 자연어로 작성한다.

★ 역할 분담 (설계 원칙)
  - 판정(위험 점수/상태/패턴/원인)은 이미 규칙 기반 Risk Engine이 끝냈다.
  - AI는 판정을 다시 하지 않는다. 주어진 근거(result.json)를 '해석·서술'만 한다.
    => 이렇게 하면 설명 가능성(왜 그 결론인지)이 유지된다.

인증
  - 환경변수 ANTHROPIC_API_KEY 가 있으면 그대로 동작한다(.env에 넣어두면 자동 인식).
  - 키가 없거나 호출이 실패하면 템플릿 기반 보고서로 자동 fallback한다.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Windows 터미널(cp949)에서 한글이 깨지지 않도록 출력 인코딩을 UTF-8로 고정
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


def _load_dotenv() -> None:
    """
    프로젝트 루트의 .env 파일을 읽어 환경변수로 로드한다 (외부 패키지 불필요).
    - 이미 설정된 환경변수는 덮어쓰지 않는다(터미널에서 직접 지정한 값 우선).
    - .env는 .gitignore로 깃허브에 올라가지 않으므로 키를 안전하게 보관할 수 있다.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        # 값이 비어있지 않고, 아직 환경변수에 없을 때만 설정
        if key and val and key not in os.environ:
            os.environ[key] = val


# import 시점에 .env 로드 (키를 .env에 넣어두면 자동 인식)
_load_dotenv()

# Claude 모델 ID — 환경변수 ANTHROPIC_MODEL로 재정의 가능.
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

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


def generate_template_report(result: Dict[str, Any]) -> str:
    """
    API 키 없이도 동작하는 템플릿 기반 보고서(마크다운).
    LLM 없이 result.json의 값을 그대로 채워 넣어 최소한의 보고서를 만든다.
    => 어떤 환경에서도 시스템이 끝까지 동작하도록 하는 fallback.
    """
    checks = result.get("recommended_checks", [])
    checks_text = "\n".join(f"- {c}" for c in checks) if checks else "- 즉각적인 조치 필요 없음"

    explanation = result.get("score_explanation", [])
    expl_text = "\n".join(f"- {e}" for e in explanation) if explanation else "- 근거 정보 없음"

    warnings = result.get("data_quality_warnings", [])
    warn_text = ("\n\n> ⚠️ 데이터 품질 경고: " + "; ".join(warnings)) if warnings else ""

    return f"""## 1. 진단 요약
설비 **{result.get('equipment', 'Motor')}**의 상태는 **{result['status']}**이며,
위험 점수는 **{result['risk_score']} / 100**입니다.

## 2. 근거 분석
주요 이상 센서: **{', '.join(result.get('abnormal_sensors', [])) or '없음'}**

점수 산출 근거:
{expl_text}

## 3. 의심 원인 및 대표 패턴
- 대표 패턴: **{result.get('main_pattern', '없음')}**
- 의심 원인: **{', '.join(result.get('suspected_causes', [])) or '특이 원인 없음'}**

## 4. 권장 조치
{checks_text}{warn_text}

---
_이 보고서는 API 키가 없어 템플릿 기반으로 자동 생성되었습니다._
_AI 서술 보고서를 원하면 ANTHROPIC_API_KEY를 설정하고 다시 실행하세요._
""".strip()


def generate_report(result: Dict[str, Any]) -> str:
    """
    result(dict)를 받아 진단 보고서(마크다운 문자열)를 생성해 반환한다.
      - ANTHROPIC_API_KEY가 있으면 Claude로 자연어 보고서를 생성.
      - 없으면 템플릿 기반 보고서로 fallback (시스템이 끝까지 동작하도록).
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        # 'LLM API + 템플릿 fallback' 구조: 키가 없어도 보고서는 나온다.
        return generate_template_report(result)

    try:
        from anthropic import Anthropic  # 키가 있을 때만 import

        client = Anthropic()  # 키는 ANTHROPIC_API_KEY에서 자동으로 읽음

        # Claude Messages API에서 system은 최상위 매개변수로 전달한다.
        response = client.messages.create(
            model=MODEL,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": build_user_prompt(result)},
            ],
        )

        text = "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()
        if not text:
            raise ValueError("Claude 응답에 텍스트가 없습니다.")
        return text

    except Exception as e:
        # 잘못된 키/네트워크/요금 문제 등으로 API 호출이 실패해도
        # 크래시 대신 안내 후 템플릿 리포트로 대체 (시스템은 끝까지 동작)
        print(f"[경고] AI 리포트 생성 실패({type(e).__name__}: {e}).\n"
              f"       템플릿 기반 보고서로 대체합니다.")
        return generate_template_report(result)


def main():
    # 입력: result.json 경로 (기본값 result.json)
    result_path = sys.argv[1] if len(sys.argv) > 1 else "result.json"

    if not os.path.exists(result_path):
        print(f"[오류] 결과 파일을 찾을 수 없습니다: {result_path}")
        print("       먼저 run_analysis.py를 실행해 result.json을 생성하세요.")
        sys.exit(1)

    with open(result_path, "r", encoding="utf-8") as f:
        result = json.load(f)

    # 키가 있으면 Claude, 없으면 템플릿으로 자동 fallback
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("[안내] ANTHROPIC_API_KEY가 없어 템플릿 기반 보고서로 생성합니다.\n")
    report = generate_report(result)

    # 화면 출력
    print(report)

    # 보고서를 UTF-8 마크다운 파일로도 저장 (터미널 한글 깨짐 방지)
    out_path = "report.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[저장됨] {out_path}")


if __name__ == "__main__":
    main()
