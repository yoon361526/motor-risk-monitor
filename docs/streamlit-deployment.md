# Streamlit Community Cloud 배포

저장소 `yoon361526/motor-risk-monitor`, 브랜치 `main`, 진입 파일 `app.py`를 사용한다.
Advanced settings의 Python 버전은 모델을 검증한 **3.12**로 지정한다.
루트 `requirements.txt`가 의존성을 설치하며, 모델이 검증하는 NumPy·SciPy·scikit-learn
버전과 모델을 저장할 때 사용한 pandas·joblib 버전을 고정했다.

## Git에 포함할 파일

기존 Python 코드·설정·CSV 예제에 더해 아래 9개 파일만 `.gitignore`에서 예외로 허용한다.
Paderborn 추가 파일의 총 크기는 약 36 MiB다.

| 파일 | 용도 |
|---|---|
| `data/paderborn/derived/comparison/dataset_manifest.json` | 전체 200개 원본의 용도·해시·설정 검증 |
| `data/paderborn/derived/deployed/active_selected_model.json` | 활성 모델 식별자 |
| `data/paderborn/derived/deployed/d006850046ef19c3afe6/engine.joblib` | 이미 학습한 모델 |
| 같은 폴더의 `model_metadata.json` | 모델·추출 코드·환경 무결성 확인 |
| 같은 폴더의 `tuning_metrics.json` | 전체 개발 기록 80개의 비교 결과 |
| `data/paderborn/raw/tuning/N09_M07_F10_K003_1.mat` | 정상 예제 |
| `data/paderborn/raw/tuning/N09_M07_F10_K004_1.mat` | 정상 예제 |
| `data/paderborn/raw/tuning/N09_M07_F10_KA04_1.mat` | 외륜 손상 예제 |
| `data/paderborn/raw/tuning/N09_M07_F10_KI04_1.mat` | 내륜 손상 예제 |

전체 manifest는 실험의 데이터 분리를 보존하기 위해 줄이지 않는다. 화면은 실제 존재하는
개발 기록만 선택 목록에 표시한다. 예제 4개를 전체 평가 표본으로 취급하지 않으며,
설정 조정용 지표는 원래의 80개 기록 결과임을 화면에 표시한다.
원신호 확인 화면도 같은 예제를 사용한다. 호환되는 새 MAT 업로드도 가능하다.

## 서버에서 하지 않는 작업

서버 시작 시 `prepare`, 학습, 특징 선택, `activate_selected_model.py`를 실행하지 않는다.
학습용·최종 평가용 원본, 다른 모델, 실험 캐시, 개별 분석 보고서는 계속 Git에서 제외한다.
업로드 파일과 실행 중 생성한 특징 캐시도 제외한다. 서버 재시작으로 캐시가 사라져도
예제 원본과 활성 모델만으로 다시 계산할 수 있다.

데이터 출처: [Paderborn Bearing DataCenter](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter).
파일명의 제공 라벨은 정답 메타데이터이며 모델의 판정 결과가 아니다.

## 변경 반영

이 변경과 위 파일을 커밋해 GitHub에 push한 뒤 Streamlit 앱을 재배포한다.
`.gitignore` 변경만으로는 파일이 GitHub에 업로드되지 않는다. `.env`, 가상환경,
실험 데이터 전체를 추가하지 말고 아래처럼 변경 파일과 허용된 데이터만 추가한다.

```powershell
git add .gitignore .gitattributes requirements.txt app.py
git add modules/vibration_dataset.py modules/vibration_dashboard.py
git add tests/test_streamlit_deployment.py docs/streamlit-deployment.md
git add data/paderborn
git diff --cached --stat
```

모델을 교체할 때는 새 모델 ID에 해당하는 세 파일만 허용하도록 `.gitignore`와 배포 테스트의
모델 ID를 함께 갱신하고, 메타데이터에 맞는 의존성 버전을 지정한다. `.gitattributes`는
운영체제가 달라도 추출 코드의 LF 줄바꿈과 모델·MAT의 원본 바이트를 유지한다.

AI 서술 보고서는 선택 사항이다. 사용할 경우 Streamlit Secrets에
`ANTHROPIC_API_KEY = "실제 키"`를 등록한다. 키가 없으면 템플릿 보고서를 사용한다.
키를 저장소에 커밋하지 않는다.

## PDF 보고서 저장

보고서 작성 후 `보고서 저장 (PDF)`로 화면의 보고서 내용을 내려받는다.
진동 비교 분석과 CSV 진단 모두 지원하며, 마크다운 저장도 제공한다.
`assets/fonts/`의 나눔고딕 Regular·Bold와 `OFL.txt`를 코드와 함께 배포해야 한다.
글꼴은 PDF에 포함되므로 서버의 시스템 글꼴을 설치할 필요가 없다.
PDF 생성은 외부 API를 호출하지 않는다.

공식 안내: [배포](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy),
[Secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).
