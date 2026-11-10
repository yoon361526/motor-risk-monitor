"""선택 모델의 추론, 품질 제외, 보고서 일치, 활성화 절차를 검증한다."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat
from sklearn.ensemble import IsolationForest

from modules.vibration_engine import StatisticsReference
from modules.vibration_feature_experiment import ExperimentRecord, compare_feature_set
from modules.vibration_feature_selection import (
    _json_hash, load_feature_records, validate_feature_source,
)
from modules.vibration_dataset import file_sha256, write_json
from modules.vibration_report import generate_vibration_template_report, report_payload
from modules.vibration_selected_model import (
    SELECTED_FEATURES, SelectedVibrationEngine, load_selected_engine,
    read_selected_record, selected_models_dir,
)
from test_vibration_feature_selection import cached_development


class KnownScoreForest:
    """알려진 점수로 경고 이력의 조건을 검사하는 테스트용 스코어러."""

    def score_samples(self, values):
        return -values[:, 0]


@pytest.fixture
def selected_engine(vibration_config):
    model = {"forest": KnownScoreForest(),
             "statistics": StatisticsReference(np.ones(4), np.ones(4)),
             "features": SELECTED_FEATURES,
             "thresholds": {"statistics": 2., "isolation_forest": 2.}}
    training = {"reference_bearings": ['K001'], "calibration_bearings": ['K003']}
    return SelectedVibrationEngine(vibration_config, model, training, None, 'selection-test', 'model-test')


def make_record(engine, scores, ids=None, total=None):
    ids = np.arange(len(scores)) if ids is None else np.array(ids)
    total = int(ids[-1] + 1) if total is None else total
    frame = pd.DataFrame(np.repeat(np.array(scores)[:, None], 4, axis=1), columns=SELECTED_FEATURES)
    frame['window_id'] = ids
    frame['start_sec'] = ids * .125
    frame['end_sec'] = ids * .125 + .25
    excluded = [{'window_id': n, 'start_sec': n * .125, 'end_sec': n * .125 + .25,
                 'reason': 'time_gap'} for n in range(total) if n not in ids]
    metadata = {'file': 'N09_M07_F10_KI04_1.mat', 'bearing_id': 'KI04',
                'condition': engine.config.condition, 'unit': None, 'duration_sec': total * .125 + .125,
                'reference_label': 'inner_ring_damage', 'warnings': []}
    quality = {'original_window_count': total, 'usable_window_count': len(frame),
               'excluded_window_count': len(excluded)}
    return engine.windowed_record(ExperimentRecord(metadata, frame, excluded, quality), 'raw-test')


def test_selected_model_preserves_middle_history_and_report_values(selected_engine):
    record = make_record(selected_engine, [1, 1, 6, 6, 6, 1, 1])
    result = selected_engine.analyze(record)
    for method in result['methods'].values():
        assert method['record_has_alert']
        assert not method['last_window_confirmed_alert']
        assert method['events'][0]['start_sec'] == .25
        assert set(method['events'][0]['peak_features']) == set(SELECTED_FEATURES)
    report = generate_vibration_template_report(result)
    assert '개선 모델' in report
    assert '경고 기준값: 2' in report
    assert '0.250000' in report
    assert '0–1 kHz 에너지 비율' in report
    assert 'reference_label' not in report_payload(result)['record']
    record.metadata['reference_label'] = 'healthy'
    assert selected_engine.analyze(record)['methods'] == result['methods']


def test_missing_window_does_not_join_three_high_scores(selected_engine):
    record = make_record(selected_engine, [1, 6, 6, 6, 1, 1, 1], ids=[0, 1, 2, 4, 5, 6, 7])
    result = selected_engine.analyze(record)
    assert all(not m['record_has_alert'] for m in result['methods'].values())


def test_excluded_tail_is_not_reported_as_no_alarm(selected_engine):
    record = make_record(selected_engine, [1, 1, 1, 1], total=5)
    result = selected_engine.analyze(record)
    assert all(not m['last_window_available'] for m in result['methods'].values())
    report = generate_vibration_template_report(result)
    assert '마지막 구간: **분석 제외**' in report
    assert '품질 검사로 제외한 구간' in report
    assert '시간 공백' in report


def test_nonuniform_time_notice_describes_applied_preprocessing(selected_engine):
    windowed = make_record(selected_engine, [1, 1, 1, 1])
    metadata = {**windowed.metadata, 'time_axis_nonuniform': True,
                'warnings': ['시간축 간격이 일정하지 않아 주파수 분석 전 시간축 처리가 필요합니다.']}
    record = selected_engine.windowed_record(ExperimentRecord(
        metadata, windowed.features, [], windowed.metadata['quality_summary']), 'source')
    assert any('64 kHz 선형 보간을 적용했습니다' in message for message in record.metadata['warnings'])
    assert not any('처리가 필요합니다' in message for message in record.metadata['warnings'])


@pytest.mark.parametrize('problem', ['unit', 'condition', 'settings', 'nan', 'ids'])
def test_incompatible_selected_input_is_rejected(selected_engine, problem):
    record = make_record(selected_engine, [1, 1, 1, 1])
    if problem == 'unit':
        record.metadata['unit'] = 'g'
    elif problem == 'condition':
        record.metadata['condition'] = 'N15_M07_F10'
    elif problem == 'settings':
        record.metadata['feature_settings'] = {}
    elif problem == 'nan':
        record.features.loc[0, SELECTED_FEATURES[0]] = np.nan
    else:
        record.features.loc[1, 'window_id'] = 0
    with pytest.raises(ValueError):
        selected_engine.analyze(record)


def test_final_file_is_blocked_before_hash_or_signal_read(selected_engine, tmp_path, monkeypatch):
    import modules.vibration_selected_model as selected

    def forbidden(*args, **kwargs):
        raise AssertionError('평가 원본을 열면 안 됩니다.')
    monkeypatch.setattr(selected, 'file_sha256', forbidden)
    monkeypatch.setattr(selected, 'load_paderborn', forbidden)
    with pytest.raises(ValueError, match='최종 평가용'):
        read_selected_record(tmp_path / 'N09_M07_F10_KA15_1.mat', selected_engine)


def test_fresh_mat_and_cache_have_identical_selected_features(selected_engine, tmp_path):
    path = tmp_path / 'N09_M07_F10_KI04_1.mat'
    time = np.arange(48001) / 64000
    def write(frequency):
        signal = (1 + .4 * np.sin(2 * np.pi * 80 * time)) * np.sin(2 * np.pi * frequency * time)
        savemat(path, {path.stem: {
            'X': {'Data': time, 'Raster': 'HostService'},
            'Y': {'Name': 'vibration_1', 'Data': signal, 'Unit': '', 'XIndex': 1,
                  'Raster': 'HostService', 'DownSampling': 1},
        }})
    write(5000)
    first = read_selected_record(path, selected_engine, tmp_path / 'cache')
    cached = read_selected_record(path, selected_engine, tmp_path / 'cache')
    pd.testing.assert_frame_equal(first.features, cached.features)
    assert first.metadata == cached.metadata
    write(8000)
    changed = read_selected_record(path, selected_engine, tmp_path / 'cache')
    assert first.metadata['source_sha256'] != changed.metadata['source_sha256']
    assert len(list((tmp_path / 'cache').glob('*.json'))) == 2


def test_activation_never_refits_or_reads_final_and_switches_only_after_validation(cached_development, monkeypatch):
    import modules.vibration_deployment as deployment

    root, source, config = cached_development
    validated = validate_feature_source(source, root, config)
    records = load_feature_records(source, validated)
    result, model = compare_feature_set(records['reference'], records['tuning'], SELECTED_FEATURES, config)
    module_dir = Path(deployment.__file__).parent
    plan = {'config': config.to_dict(), 'source': str(source),
            'input_protocol_id': validated['protocol_id'], 'input_feature_hashes': validated['feature_hashes'],
            'source_code_hashes': {name: file_sha256(module_dir / name) for name in (
                'vibration_feature_selection.py', 'vibration_feature_experiment.py')}}
    plan_id = _json_hash(plan)
    selection = root / 'selection'
    write_json(selection / 'analysis_plan.json', plan)
    write_json(selection / 'selection.json', {
        'analysis_plan_id': plan_id, 'recommended_candidate': 'ranked_4', 'candidates': {'ranked_4': result},
    })
    joblib.dump({'analysis_plan_id': plan_id, 'models': {'ranked_4': model}}, selection / 'selection_models.joblib')
    original = root / 'derived/comparison/models.joblib'
    original.write_bytes(b'preserve-baseline')
    reads = []
    wrong_score = False
    by_file = {r.metadata['file']: r for r in records['tuning']}

    def reader(path, engine, cache_dir):
        assert path.parent.name == 'tuning'
        reads.append(path)
        r = by_file[path.name]
        r.metadata.update({'unit': None, 'duration_sec': 1.375, 'warnings': []})
        r.time_audit.update({'original_window_count': 10, 'excluded_window_count': 0})
        record = engine.windowed_record(r, file_sha256(path))
        if wrong_score:
            record.features = record.features.copy()
            record.features[list(SELECTED_FEATURES)] = 1000
        return record

    def forbidden_fit(*args, **kwargs):
        raise AssertionError('활성화에서 다시 학습하면 안 됩니다.')

    monkeypatch.setattr(deployment, 'read_selected_record', reader)
    monkeypatch.setattr(IsolationForest, 'fit', forbidden_fit)
    report = deployment.activate_selection(selection, root, config, progress=lambda _: None)
    assert len(reads) == 4
    assert original.read_bytes() == b'preserve-baseline'
    engine, folder = load_selected_engine(root, config)
    assert engine.model_id == report['model_id']
    assert engine.thresholds == model['thresholds']
    assert (folder / 'analyses/tuning/N09_M07_F10_KI04_1.md').is_file()
    pointer = selected_models_dir(root) / 'active_selected_model.json'
    before = file_sha256(pointer)
    wrong_score = True
    with pytest.raises(ValueError, match='결과가 다릅니다'):
        deployment.activate_selection(selection, root, config, progress=lambda _: None)
    assert file_sha256(pointer) == before
    (folder / 'engine.joblib').write_bytes(b'changed')
    with pytest.raises(ValueError, match='모델 파일이 변경'):
        load_selected_engine(root, config)
