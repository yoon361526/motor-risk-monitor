# 모터 이상징후 진단 엔진 (Motor Risk Score Engine)

모터 센서 데이터(온도·진동·전류·RPM)를 분석해 **위험 점수**를 계산하고,
설비 상태를 **정상 / 주의 / 위험**으로 분류하는 **규칙 기반(rule-based) 진단 엔진**입니다.
진단 결과를 바탕으로 생성형 AI(Anthropic Claude)가 사람이 읽기 좋은 **진단 보고서**를 자연어로 작성합니다.

```
data.csv ──▶ [Risk Engine] ──▶ result.json ──▶ [Claude] ──▶ report.md
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
│   ├── __init__.py             # 패키지 초기화
│   ├── risk_engine.py          # 규칙 기반 위험 점수 엔진 (RiskEngine 클래스)
│   └── report_generator.py     # Claude(또는 템플릿)로 진단 보고서 생성
├── config/
│   └── risk_thresholds.yaml    # 임계값·가중치·패턴 조건 (하드코딩 없음)
├── tests/
│   └── test_risk_engine.py     # pytest 단위 테스트
├── data/                       # generate_data.py로 만든 synthetic CSV
├── app.py                      # Streamlit 진단 대시보드 (웹 UI)
├── generate_data.py            # 고장 유형별 synthetic 데이터 생성기
├── run_analysis.py             # CSV → result.json 실행 스크립트
├── .env.example                # API 키 템플릿 (.env는 gitignore)
├── requirements.txt
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

## 대시보드 (웹 UI) — 권장 실행 방법

CSV를 올리면 진단 결과·점수·센서 그래프·AI 리포트를 한 화면에서 보여주는
Streamlit 대시보드입니다. 성과공유회/시연에 적합합니다.

```bash
py -m pip install -r requirements.txt   # 최초 1회 (streamlit 포함)
py -m streamlit run app.py              # 로컬 서버 실행 → 브라우저 자동 열림
```

> Windows에서 `streamlit run ...`이 "명령을 찾을 수 없음"으로 실패하면
> (스크립트 폴더가 PATH에 없어서), 위처럼 **`py -m streamlit run app.py`**로 실행하세요.

- 사이드바에서 **예제 데이터 선택** 또는 **CSV 업로드** → 즉시 진단
- 위험 점수 게이지, 센서 4종 시계열, 탐지 패턴/원인/조치, 점수 근거 표시
- "리포트 생성" 버튼 → AI(또는 템플릿) 진단 리포트
- 로컬 서버라 **인터넷 없이도** 동작 (`.env`에 키가 있으면 AI 리포트까지)

> 종료: 터미널에서 `Ctrl + C`

아래 CLI 방식(1~2단계)은 대시보드 없이 터미널에서 돌리고 싶을 때 사용합니다.

## 사용법 (CLI)

### 0단계 — synthetic 데이터 생성 (선택)

실제 센서 데이터가 없을 때, 고장 유형별 예제 CSV를 `data/`에 생성합니다.

```bash
py generate_data.py
```

생성되는 파일 (각 600행):

| 파일 | 모사한 고장 | 엔진 진단 결과 |
|------|-------------|----------------|
| `data/normal.csv` | 정상 | 정상 |
| `data/overload.csv` | 과부하 (전류↑+RPM↓) | 주의 |
| `data/bearing_wear.csv` | 베어링 마모 (진동↑+온도↑) | 주의 |
| `data/cooling_fault.csv` | 냉각 불량 (온도만↑) | 주의 |
| `data/compound_fault.csv` | 복합 이상 (온도+진동+전류↑) | 위험 |

> 단순 랜덤이 아니라 각 고장 유형의 센서 변화 특성을 반영하며, 재현성을 위해
> 난수 시드를 고정했습니다.

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
# 키는 프로젝트 루트의 .env 파일에 넣어두면 자동 인식됩니다 (.env는 gitignore).
#   ANTHROPIC_API_KEY=sk-ant-...
# 또는 환경변수로 직접 지정도 가능:
# PowerShell:  $env:ANTHROPIC_API_KEY = "sk-ant-..."
# bash:        export ANTHROPIC_API_KEY="sk-ant-..."

py modules/report_generator.py result.json
```

- 보고서가 화면에 출력되고 `report.md`로 저장됩니다.
- **API 키가 없어도 동작합니다.** 키가 있으면 Claude가 자연어 보고서를 쓰고,
  없으면 `result.json` 값을 채운 **템플릿 기반 보고서로 자동 fallback**합니다.
- 모델을 바꾸려면 `ANTHROPIC_MODEL` 환경변수를 지정하세요
  (기본값 `claude-sonnet-4-6`).

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
  "recommended_checks": ["부하 상태 확인", "회전부 간섭 확인", "전원 공급 상태 확인"],
  "analysis_window": { "baseline_range_sec": "0-119", "analysis_range_sec": "480-599" },
  "score_explanation": [
    "전류이(가) 기준 대비 26.0% 증가하여 전류 점수 62.2점",
    "'전류 증가 + RPM 감소' 패턴 탐지 → 보너스 20점 (최소 보장 45점)"
  ],
  "data_quality_warnings": []
}
```

**근거/설명 필드**
- `analysis_window` — baseline 구간과 분석 대상(후반부) 구간의 시간 범위(초).
- `score_explanation` — 왜 이 점수가 나왔는지 사람이 읽을 수 있는 근거 문장.
- `data_quality_warnings` — 입력 데이터 품질 경고(RPM 음수, 미정렬 등). 치명적 문제는 예외 발생.

---

## 설정 (config/risk_thresholds.yaml)

판정에 쓰이는 모든 값을 코드에서 분리해 YAML로 관리합니다. 수정하면 재배포 없이 민감도가 바뀝니다.

| 항목 | 내용 |
|------|------|
| `baseline` | baseline 비율(20%), 분석 구간, 이동평균 창 |
| `weights` | 센서 가중치 (온도 0.25 / 진동 0.30 / 전류 0.25 / RPM 0.20) |
| `sensor_score_mapping` | 변화율 → 0~100 점수 매핑 (dead_zone, full_score_change) |
| `patterns` | 조합 패턴 탐지 조건 + 보너스 + `min_score_if_detected`(패턴 감지 시 최소 보장 점수) |
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
- 진단 보고서: Anthropic Claude API (`claude-sonnet-4-6`) — 키 없으면 템플릿 fallback
- 외부 판정 API 없음 · ML 없음 (판정은 순수 규칙 기반)

---

## 설계 노트 및 한계

### 이 프로젝트의 역할 분담
> 생성형 AI는 **이상 여부를 판단하지 않습니다.** 이상 판단은 baseline 변화율과
> 조합 패턴을 기반으로 한 **Risk Score Engine**이 수행하고, AI는 그 결과를
> 사람이 읽기 쉬운 자연어 리포트로 변환하는 역할만 합니다.

### threshold 값의 성격
> 본 프로젝트의 임계값·가중치는 실제 산업 표준값이 아니라, **synthetic data 기반
> MVP 검증을 위한 초기 경험적 기준값**입니다. 추후 실제 DC 모터 센서 데이터를
> 확보하면 baseline·threshold를 재보정(calibration)할 수 있도록 config 파일로
> 분리해 두었습니다.

### 현재 한계
> 현재 버전은 synthetic data 기반 MVP이며 실제 산업 설비 진단 정확도를 보장하지
> 않습니다. 목적은 설비 이상징후 분석 시스템의 구조를 학부생 수준에서 구현하고,
> 생성형 AI를 리포트 자동화 컴포넌트로 통합하는 것입니다.

향후 확장 방향:
- 후반부 평균 대신 **rolling window 기반 risk score 시계열** 산출 (중간 구간 이상도 포착)
- 실제 센서 데이터 확보 후 threshold calibration
