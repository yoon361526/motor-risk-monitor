"""로컬 원본·실험 캐시 없이 배포 파일만으로 세 화면을 실행한다."""

from pathlib import Path
import shutil

import pytest
from streamlit.testing.v1 import AppTest

from modules.vibration_dataset import available_tuning_records


ROOT = Path(__file__).resolve().parents[1]
MODEL = "d006850046ef19c3afe6"
EXAMPLES = [f"N09_M07_F10_{bearing}_1.mat" for bearing in ("K003", "K004", "KA04", "KI04")]


@pytest.fixture
def deployment(tmp_path):
    files = ["app.py", "config/risk_thresholds.yaml", "config/vibration_analysis.yaml",
             "data/normal.csv", "data/paderborn/derived/comparison/dataset_manifest.json",
             "data/paderborn/derived/deployed/active_selected_model.json"]
    files += [f"data/paderborn/derived/deployed/{MODEL}/{name}" for name in (
        "engine.joblib", "model_metadata.json", "tuning_metrics.json")]
    files += [f"data/paderborn/raw/tuning/{name}" for name in EXAMPLES]
    for name in files:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


def test_only_existing_tuning_files_are_offered(tmp_path):
    for name in ("healthy.mat", "evaluation.mat"):
        (tmp_path / name).touch()
    manifest = {"records": [
        {"split": "tuning", "relative_path": "healthy.mat"},
        {"split": "tuning", "relative_path": "missing.mat"},
        {"split": "evaluation", "relative_path": "evaluation.mat"},
    ]}
    assert available_tuning_records(tmp_path, manifest) == manifest["records"][:1]


def test_deployment_without_training_data_or_caches_runs_all_screens(deployment):
    app = AppTest.from_file(str(deployment / "app.py"), default_timeout=45).run()
    assert not app.exception
    assert not app.error
    assert sorted(app.selectbox[0].options) == EXAMPLES
    for name in EXAMPLES:
        app.selectbox[0].set_value(name).run()
        assert not app.exception
        assert not app.error
        assert any(name in caption.value for caption in app.caption)
    app.radio[0].set_value("Paderborn 진동 확인").run()
    assert not app.exception
    assert not app.error
    assert sorted(app.selectbox[0].options) == EXAMPLES
    app.radio[0].set_value("기존 CSV 진단").run()
    assert not app.exception
    assert not app.error
    assert any("normal.csv" in message.value for message in app.success)
