# 모터 이상징후 진단 엔진 (Motor Risk Score Engine)

## 진동 비교 분석: 정상 참조 통계 + Isolation Forest

다운로드한 Paderborn 자료를 사용해, **외부 정상 기록 기반 통계 방식**과
**정상 기록으로 학습한 Isolation Forest**를 같은 조건에서 비교합니다.
두 방식의 결과를 각각 표시하고, 이상 구간 이력과 점검 보고서를 저장합니다.
현재 기능은 측정 기록의 오프라인 분석이며, 산업 설비의 실시간 보호 기능이 아닙니다.

### 실행

```powershell
# 의존성 설치
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Downloads의 베어링별 RAR에서 N09_M07_F10 조건만 추출
.\.venv\Scripts\python.exe run_vibration_analysis.py prepare

# 정상 기준 학습, 정상 경계 설정, 설정 조정용 결과 저장
.\.venv\Scripts\python.exe run_vibration_analysis.py train

# 대시보드에서 'Paderborn 비교 분석' 선택
.\.venv\Scripts\python.exe -m streamlit run app.py
```

다른 다운로드 폴더는 `prepare --archive-dir "D:\\datasets"`로 지정할 수 있습니다.
Windows 기본 `tar`로 RAR을 읽습니다. 다른 OS에서는 RAR을 지원하는 bsdtar가 필요합니다.
기존 원본은 그대로 두며, 다른 내용의 파일이 이미 있으면 덮어쓰지 않고 중단합니다.

### 데이터 분리와 평가

초기 분리와 분석 설정은 `config/vibration_analysis.yaml`에 있습니다.

| 용도 | 베어링 | 기록 수 |
|---|---|---:|
| 기준 생성(reference) | 정상 K001, K002 | 40 |
| 설정 조정(tuning) | 정상 K003, K004 / 손상 KA04, KI04 | 80 |
| 최종 평가(evaluation) | 정상 K005, K006 / 손상 KA15, KI14 | 80 |

필요한 압축파일은 위 베어링 10개입니다. 네 운전 조건 전체를 풀지 않고,
한 조건의 200개 MAT와 설명 PDF·측정 로그만 `data/paderborn/raw/` 아래에 추출합니다.
같은 베어링의 기록이나 겹치는 시간 구간을 서로 다른 집합으로 나누지 않습니다.
같은 정상 베어링의 여러 기록은 독립 베어링 표본이 아니므로, 평가에는 기록 수와 베어링 수를 함께 표시합니다.

`train`은 reference와 tuning만 읽습니다. 두 모델은 reference 정상 기록으로 학습하고,
경고 경계는 tuning 중 **정상 기록만** 사용해 정합니다. 손상 라벨은 학습 입력에 넣지 않습니다.
이후 tuning의 정상·손상 기록으로 오경보율·미탐률을 보여 줍니다. 이 결과는 최종 성능이 아닙니다.

분석 방식과 설정을 확정한 뒤 다음 명령으로 최종 평가합니다.

```powershell
.\.venv\Scripts\python.exe run_vibration_analysis.py evaluate
```

이 명령은 저장된 모델과 경계를 그대로 사용하며 재학습하지 않습니다.
최종 평가 결과를 확인한 뒤 설정을 변경하면 같은 자료를 새로운 미공개 평가로 취급할 수 없습니다.
대시보드의 기본 파일 목록은 tuning만 포함하며, 평가용 베어링의 개별 업로드·수동 분석은 제한합니다.

단일 기록 분석과 JSON·템플릿 보고서 저장은 다음과 같습니다.

```powershell
.\.venv\Scripts\python.exe run_vibration_analysis.py analyze data/paderborn/raw/tuning/N09_M07_F10_KA04_1.mat
```

### 점수와 이상 이력의 의미

- 특징: 구간 평균을 제거한 RMS, Pearson 첨도, 최대·최소 차이, 파고율.
- 통계 방식: 정상 특징의 중앙값과 `1.4826 × MAD`를 사용한 절대 편차 중 최댓값.
  MAD가 0에 가까우면 작은 양의 바닥값을 사용해 0 나눗셈을 막습니다.
- 머신러닝 방식: Isolation Forest의 `-score_samples`. 큰 값이 더 이상한 방향입니다.
  기본 설정은 트리 200개, 난수 시드 42입니다. 기본 `predict` 경계는 사용하지 않습니다.
- 경계 설정: 정상 tuning 기록마다 '연속 3개 점수의 최솟값 중 최댓값'을 계산하고,
  그 기록 점수의 95백분위수(`higher`)를 각 방식의 경계로 정합니다.
- 실제 경고: 경계보다 **엄격히 큰** 점수가 3개 구간 연속 나타날 때 확정합니다.
  경계 설정과 실제 판정에 같은 연속 조건을 적용합니다.
- 기본 창 0.25초·이동 0.125초에서 연속 3개 창의 시간 범위는 약 0.5초입니다.
  겹치는 이상 시간 구간은 합치고, 마지막 구간의 상태와 전체 이상 이력을 따로 보관합니다.
- 점수는 고장 확률이나 검증된 위험 등급이 아닙니다. 두 방식의 점수 크기를 직접 비교하지 않습니다.
  경고 없음은 이 기준에서 이상 이력을 찾지 못했다는 뜻이며, 정상 상태를 보장하지 않습니다.
- 진동 단위가 없는 MAT는 원본 수치로 보존합니다. 다른 단위나 운전 조건의 기준을 섞지 않습니다.
  일정 신호처럼 첨도·파고율이 정의되지 않는 입력은 정상값으로 채우지 않고 분석을 중단합니다.
- 통계 특징 경계는 Isolation Forest의 특징 중요도가 아닙니다. 손상 위치나 원인도 단정하지 않습니다.

### 결과와 코드 위치

결과는 `data/paderborn/derived/comparison/`에 저장됩니다.

- `dataset_manifest.json`: 용도별 원본 목록과 SHA-256. 특징 캐시도 원본 해시와 추출 설정을 사용합니다.
- `models.joblib`, `model_metadata.json`: 이 프로젝트에서 학습한 두 모델, 설정, 기준·경계 설정 원본 정보.
  출처를 모르는 joblib 모델을 이 경로에 넣지 마세요.
- `tuning_metrics.json`, 명시적으로 평가한 경우 `evaluation_metrics.json`: 파일 단위 혼동행렬·오경보율·미탐률·베어링별 결과.
- `analyses/tuning/`, `analyses/manual/`, `analyses/evaluation/`: 기록별 JSON과 마크다운 보고서.

`modules/vibration_config.py`는 설정 검사, `vibration_dataset.py`는 데이터 준비·특징 캐시,
`vibration_engine.py`는 통계·Isolation Forest·이상 이력, `vibration_workflow.py`는 학습·평가 절차,
`vibration_dashboard.py`는 화면, `vibration_report.py`는 수치를 그대로 옮기는 보고서를 담당합니다.
각 파일의 docstring과 주석에 계산 목적과 주의할 데이터 분리를 설명했습니다.

기본 보고서는 API 없이 작성합니다. 화면에서 생성형 AI 옵션을 선택하고 보고서 버튼을 누르면
기존 Claude 기능으로 문장을 작성합니다. AI에는 평가 정답 라벨을 전달하지 않으며,
통계·머신러닝 엔진이 계산한 판정과 근거를 그대로 설명하도록 합니다.

테스트는 `python -m pytest -q`로 실행합니다. 외부 정상 기준, 처음부터 높은 특징값,
중간 이상 후 정상 복귀, 순간 초과 제외, 라벨 누수 방지, 최종 평가의 재학습 금지,
캐시 무효화, 보고서 시간·측정값 전달 등을 검사합니다.

---


## Paderborn 진동 확인 (추가된 기능)

대시보드의 **Paderborn 진동 확인**에서 원본 신호, 구간별 RMS·첨도·최대/최소 차이·파고율,
같은 운전 조건의 기록 비교표와 CSV/JSON 다운로드를 확인할 수 있습니다.
**기존 CSV 진단**은 사이드바의 분석 종류에서 선택합니다.

1. 원래 이름을 유지한 MAT 파일과 설명 PDF를 `data/paderborn/raw/`에 둡니다.
2. 의존성을 설치하고 대시보드를 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

터미널에서 실제 파일 구조와 특징을 확인하려면:

```powershell
.\.venv\Scripts\python.exe inspect_paderborn.py
```

출력은 `data/paderborn/derived/inspection_summary.json`과 `window_features.csv`입니다.
기본 구간은 0.25초, 이동 간격은 0.125초이며 검증으로 확정한 탐지 설정이 아닙니다.
실제 시간축을 사용하고 마지막 짧은 구간도 포함합니다. 각 구간의 평균을 제거한 뒤
RMS를 계산하며, 첨도는 Pearson 정의(m4/m2²)를 사용합니다. 진폭 정규화는 하지 않습니다.

원신호 확인 화면에 먼저 등록한 3개 파일은 `N09_M07_F10`(900rpm, 0.7Nm, 반경 방향 하중 1,000N) 조건의
K001·KA04·KI04입니다. 진동 채널은 `vibration_1`이며, 채널의 `XIndex`로 시간축을 찾습니다.
이 파일들은 진동 단위 정보가 비어 있어 원본 수치를 보존하고, g로 간주하거나 변환하지 않습니다.

이 원신호 확인 화면은 **신호·특징 확인 단계**입니다. 파일명의 제공 라벨은 프로그램의 판정이 아니며,
원인 진단·경고 임계값·탐지 정확도를 산출하지 않습니다. 이번 샘플은 정상 K001의 RMS가
두 손상 파일보다 높으므로 'RMS가 높으면 손상'이라는 규칙의 타당성을 보여주지 않습니다.
새 비교 분석 기능에서는 별도 정상 기준과 데이터 분리로 경고 기준을 실험합니다.
같은 베어링의 기록이나 구간을 서로 다른 평가 집합에 섞지 않습니다.

데이터 원본과 재생성 가능한 추출 결과는 Git에서 제외합니다. 공식 자료:
https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter

---

## 기존 CSV 진단 기능

아래는 기존 CSV 전용 엔진의 설명입니다. 위의 Paderborn 비교 분석과는 별도입니다.
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
