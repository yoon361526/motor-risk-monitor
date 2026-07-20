# 모터 이상징후 진단 엔진 (Motor Risk Score Engine)

모터 센서 데이터(온도·진동·전류·RPM)를 분석해 **위험 점수**를 계산하고,
설비 상태를 **정상 / 주의 / 위험**으로 분류하는 **규칙 기반(rule-based) 진단 엔진**입니다.
진단 결과를 바탕으로 Claude(생성형 AI)가 사람이 읽기 좋은 **진단 보고서**를 자연어로 작성합니다.

```
data.csv ──▶ [Risk Engine] ──▶ result.json ──▶ [Claude AI] ──▶ report.md
             (규칙 기반 판정)     (근거 데이터)      (자연어 서술)     (진단 보고서)
```

---

## 핵심 설계 원칙

1. **절대 임계값이 아니라 "정상 기준(baseline) 대비 변화량"으로 판정한다.**
   - CSV 초기 20% 구간을 정상 기준으로 삼고, 이후 구간의 변화율로 이상을 감지.
2. **단일 센서보다 센서 조합 패턴을 중요하게 본다.**
   - 예: 전류↑ + RPM↓ = 과부하, 진동↑ + 온도↑ = 베어링 마모.
3. **판정은 전적으로 규칙 기반 로직이 수행한다.** (ML / 생성형 AI 미사용)
   - AI는 판정을 하지 않고, 엔진이 산출한 근거를 **해석·서술**만 한다.
4. **점수 산출 근거를 외부에서 확인할 수 있다. (설명 가능성)**
   - 모든 중간 계산값(baseline, 변화율, 부분 점수, 탐지 패턴)을 결과에 포함.

---

## 디렉토리 구조

```
motor-monitor/
├── modules/
│   ├── risk_engine.py          # 규칙 기반 위험 점수 엔진 (RiskEngine 클래스)
│   └── report_generator.py     # Claude로 진단 보고서 생성
├── config/
│   └── risk_thresholds.yaml    # 임계값·가중치·패턴 조건 (하드코딩 없음)
├── tests/
│   └── test_risk_engine.py     # pytest 단위 테스트
├── run_analysis.py             # CSV → result.json 실행 스크립트
├── .gitignore
└── README.md
```

---

## 설치

Python 3.10+ 필요.

```bash
pip install pandas numpy pyyaml pytest anthropic
```

> Windows에서 `python` 명령이 없으면 `py`를 사용하세요. (예: `py -m pip install ...`)

---

## 사용법

### 1단계 — 위험 점수 계산 (CSV → result.json)

```bash
py run_analysis.py <CSV경로>
# 예: py run_analysis.py data/motor_sample.csv
```

- 터미널에 진단 요약이 출력되고, 전체 결과가 `result.json`(UTF-8)에 저장됩니다.

**입력 CSV 컬럼**

| 컬럼 | 설명 |
|------|------|
| `timestamp` | 측정 시각 |
| `operating_time_sec` | 가동 시간(초) |
| `temperature_c` | 온도(℃) |
| `vibration_g` | 진동(g) |
| `current_a` | 전류(A) |
| `rpm` | 회전수 |
| `fault_type` | 라벨 (엔진은 사용하지 않음, 검증용) |

> 행 수는 자유이며 앞 20%가 자동으로 baseline으로 잡힙니다 (600행 기준 설계).

### 2단계 — AI 진단 보고서 생성 (result.json → report.md)

```bash
# 먼저 Anthropic API 키를 환경변수로 설정
# PowerShell:  $env:ANTHROPIC_API_KEY = "sk-ant-..."
# bash:        export ANTHROPIC_API_KEY="sk-ant-..."

py modules/report_generator.py result.json
```

- 보고서가 화면에 출력되고 `report.md`로 저장됩니다.
- 키가 없으면 안내 메시지 후 종료하며, 엔진/테스트에는 영향이 없습니다.

> Windows 터미널에서 한글이 깨지면 실행 전 아래를 한 번 입력하세요:
> `$OutputEncoding = [Console]::OutputEncoding = [Text.Encoding]::UTF8`

---

## 진단 결과(result.json) 형식

```json
{
  "equipment": "DC Motor",
  "status": "위험",
  "risk_score": 84,
  "baseline": { "temperature": ..., "vibration": ..., "current": ..., "rpm": ..., "std": {...} },
  "sensor_changes": { "temperature": ..., "vibration": ..., "current": ..., "rpm_drop": ... },
  "sensor_scores": { "temperature": ..., "vibration": ..., "current": ..., "rpm": ... },
  "detected_patterns": ["전류 증가 + RPM 감소"],
  "abnormal_sensors": ["전류", "RPM", "온도"],
  "main_pattern": "전류 증가와 RPM 감소가 동시에 발생",
  "suspected_causes": ["과부하"],
  "recommended_checks": ["부하 상태 확인", "회전부 간섭 확인", "전원 공급 상태 확인"]
}
```

---

## 설정 (config/risk_thresholds.yaml)

판정에 쓰이는 모든 값을 코드에서 분리해 YAML로 관리합니다. 수정하면 재배포 없이 민감도가 바뀝니다.

| 항목 | 내용 |
|------|------|
| `baseline` | baseline 비율(20%), 분석 구간, 이동평균 창 |
| `weights` | 센서 가중치 (온도 0.25 / 진동 0.30 / 전류 0.25 / RPM 0.20) |
| `sensor_score_mapping` | 변화율 → 0~100 점수 매핑 (dead_zone, full_score_change) |
| `patterns` | 조합 패턴 탐지 조건 + 보너스 (과부하/베어링마모/냉각불량/복합이상) |
| `status_thresholds` | 상태 경계값 (0~39 정상 / 40~69 주의 / 70~100 위험) |
| `cause_checks` | 원인별 권장 점검 항목 |

---

## 조합 패턴

| 패턴 | 의심 원인 | 보너스 강도 |
|------|-----------|------------|
| 전류 증가 + RPM 감소 | 과부하 | 높음 |
| 진동 증가 + 온도 상승 | 베어링 마모 | 높음 |
| 온도만 지속 상승 | 냉각 불량 | 중간~높음 |
| 온도 + 진동 + 전류 동시 증가 | 복합 이상 | 매우 높음 |

---

## 테스트

```bash
py -m pytest -q
```

- baseline이 초기 20% 구간에서 계산되는지
- 정상 패턴 → `status="정상"`
- 과부하 / 베어링 마모 / 냉각 불량 패턴 탐지
- 최종 점수 0~100 범위 보장
- 출력 스키마 검증

---

## 기술 스택

- Python, Pandas, NumPy, PyYAML
- 진단 보고서: Anthropic Claude API (`claude-opus-4-8`)
- 외부 판정 API 없음 · ML 없음 (판정은 순수 규칙 기반)
