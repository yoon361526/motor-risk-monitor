"""진동 비교 결과를 수치 그대로 설명하는 보고서. 진단 원인을 새로 추론하지 않는다."""

from __future__ import annotations

import copy


def report_payload(result: dict) -> dict:
    """LLM에는 핵심 근거만 전달한다. 평가 정답 라벨과 전체 구간 표는 제외한다."""
    payload = copy.deepcopy(result)
    payload["record"].pop("reference_label", None)
    payload["record"].pop("reference_label_source", None)
    for method in payload["methods"].values():
        method.pop("windows", None)
    for key in ("reference_sources", "calibration_sources"):
        payload["training"].pop(key, None)
    return payload


def generate_vibration_template_report(result: dict) -> str:
    """AI를 호출하지 않고 이상 이력·비교 기준·측정값을 마크다운으로 작성한다."""
    record = result["record"]
    settings = result["settings"]
    training = result["training"]
    lines = [
        "## 진동 기록 분석 보고서",
        f"- 기록: `{record['file']}`",
        f"- 운전 조건: `{record['condition']}`",
        f"- 측정 길이: {record['duration_sec']:.6f}초",
        f"- 진동 단위: {record.get('unit') or '미확인 — 원본 수치 사용'}",
        f"- 정상 기준 베어링: {', '.join(training['reference_bearings'])}",
        f"- 경계 설정용 정상 베어링: {', '.join(training['calibration_bearings'])}",
        f"- 분석 구간: {settings['window_sec']:g}초 / 이동: {settings['hop_sec']:g}초",
        f"- 경고 조건: {settings['minimum_consecutive_windows']}개 구간 연속 초과",
        "",
        "## 방식별 결과와 이상 이력",
    ]
    for method in result["methods"].values():
        history = "이상 이력 있음" if method["record_has_alert"] else "이상 이력 없음"
        latest = "경고" if method["last_window_confirmed_alert"] else "경고 없음"
        lines += [f"### {method['name']}", f"- 전체 기록: **{history}**",
                  f"- 마지막 구간: **{latest}**",
                  f"- 기록 점수: {method['record_score']:.6g} / 비교 경계: {method['threshold']:.6g}"]
        for index, event in enumerate(method["events"], 1):
            lines += [
                f"- 이상 구간 {index}: **{event['start_sec']:.6f}~{event['end_sec']:.6f}초**",
                f"  - 최고 구간 점수: {event['peak_score']:.6g} / 경계: {event['threshold']:.6g}",
                "  - 최고 점수 구간의 측정값: " + ", ".join(
                    f"{name}={value:.6g}" for name, value in event["peak_features"].items()),
            ]
        lines.append("")
    lines += ["## 통계 방식의 비교 기준",
              "| 특징 | 정상 기준 중앙값 | 하한 | 상한 |",
              "|---|---:|---:|---:|"]
    for feature in result["statistical_reference"]:
        lines.append(f"| {feature['label']} | {feature['center']:.6g} | "
                     f"{feature['lower_bound']:.6g} | {feature['upper_bound']:.6g} |")
    lines += ["", "이 표는 통계 방식의 경계입니다. Isolation Forest의 특징 기여도는 아닙니다.",
              "", "## 점검 시 확인할 사항",
              "- 이상 이력이 있으면 해당 시간의 원신호와 운전 조건을 함께 확인하세요.",
              "- 이 결과만으로 손상 위치·원인이나 장비 정지 여부를 확정하지 않습니다.",
              "", "## 분석 범위"]
    lines += [f"- {note}" for note in result.get("notes", [])]
    lines += [f"- {warning}" for warning in record.get("warnings", [])]
    lines += ["", "_분석 엔진의 결과를 그대로 옮긴 템플릿 보고서입니다._"]
    return "\n".join(lines)
