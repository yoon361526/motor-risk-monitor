"""특징 선택의 방향성, 탐색 상한, 데이터 분리와 원본 보존을 검증한다."""

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules.vibration_config import load_vibration_config
from modules.vibration_dataset import expected_records, file_sha256, write_json
from modules.vibration_feature_experiment import ExperimentSettings
from modules.vibration_feature_selection import (
    FEATURE_POOL, MAXIMUM_CANDIDATES, _json_hash, build_candidates, choose_candidate,
    load_feature_records, rank_features, run_selection, validate_feature_source,
)


def test_rank_requires_same_direction_for_both_normal_bearings_and_damage_types():
    # 손상이 정상 두 군집의 가운데 있으면 쌍별 구분은 쉬워도 공통 방향 점수는 낮아야 한다.
    groups = {'K003': 0., 'K004': 10., 'KA04': 5., 'KI04': 6.}
    rows = []
    for bearing, value in groups.items():
        for n in range(5):
            row = {'bearing_id': bearing, **{f: value + n * .001 for f in FEATURE_POOL}}
            row[FEATURE_POOL[0]] = (0 if bearing.startswith('K00') else 20) + n * .001
            row[FEATURE_POOL[1]] = (0 if bearing.startswith('K00') else -20) + n * .001
            rows.append(row)
    ranking = {r['feature']: r for r in rank_features(pd.DataFrame(rows))}
    assert ranking[FEATURE_POOL[0]]['separation_score'] == 1
    assert ranking[FEATURE_POOL[0]]['common_direction'] == 'higher'
    assert ranking[FEATURE_POOL[1]]['separation_score'] == 1
    assert ranking[FEATURE_POOL[1]]['common_direction'] == 'lower'
    assert ranking[FEATURE_POOL[2]]['separation_score'] == 0


def test_candidate_budget_and_correlated_features():
    rng = np.random.default_rng(9)
    medians = pd.DataFrame(rng.normal(size=(80, len(FEATURE_POOL))), columns=FEATURE_POOL)
    medians[FEATURE_POOL[1]] = -3 * medians[FEATURE_POOL[0]]
    ranking = [{'feature': feature} for feature in FEATURE_POOL]
    candidates, excluded = build_candidates(ranking, medians)
    assert len(candidates) <= MAXIMUM_CANDIDATES == 23
    assert len([name for name in candidates if name.startswith('single_')]) == 17
    assert any(r['feature'] == FEATURE_POOL[1] for r in excluded)
    assert FEATURE_POOL[1] not in candidates['ranked_2']


@pytest.mark.parametrize('fp,ka,ki,recommended', [(1, 20, 4, 'test'), (1, 20, 3, None),
                                               (2, 20, 20, None), (1, 19, 20, None)])
def test_selection_must_improve_previous_spectral_candidate(fp, ka, ki, recommended):
    def candidate(false_alarms, outer, inner):
        return {'features': ['x'], 'methods': {'isolation_forest': {'metrics': {
            'fp': false_alarms, 'tp': outer + inner, 'unscorable_record_count': 0,
            'by_bearing': {'KA04': {'alert_record_count': outer, 'record_count': 20},
                           'KI04': {'alert_record_count': inner, 'record_count': 20}},
        }}}}
    candidates = {'baseline': candidate(1, 20, 0), 'spectral': candidate(1, 20, 3),
                  'test': candidate(fp, ka, ki)}
    _, result = choose_candidate(candidates)
    assert result == recommended


@pytest.fixture
def cached_development(tmp_path):
    config = replace(load_vibration_config('config/vibration_analysis.yaml'),
                     records_per_bearing=1, n_estimators=5)
    source = tmp_path / 'previous_experiment'
    entries = expected_records(config)
    rng = np.random.default_rng(12)
    for entry in entries:
        path = tmp_path / entry['relative_path']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode())
        entry['sha256'] = file_sha256(path)
        if entry['split'] == 'evaluation':
            continue
        offset = 0 if entry['provided_label'] == 'healthy' else 3
        frame = pd.DataFrame(rng.normal(offset, 1, (10, 17)), columns=FEATURE_POOL)
        frame['window_id'] = np.arange(10)
        frame['start_sec'] = np.arange(10) * .125
        frame['end_sec'] = frame['start_sec'] + .25
        metadata = {'file': entry['file'], 'bearing_id': entry['bearing_id'],
                    'condition': entry['condition'], 'split': entry['split'],
                    'source_sha256': entry['sha256'], 'reference_label': entry['provided_label']}
        write_json(source / 'features' / f'{path.stem}.json', {
            'metadata': metadata, 'windows': frame.to_dict('records'),
            'time_audit': {'usable_window_count': 10}, 'excluded_windows': [],
        })
    write_json(tmp_path / 'derived/comparison/dataset_manifest.json', {
        'config_fingerprint': config.fingerprint, 'records': entries,
    })
    module_dir = Path(__file__).resolve().parents[1] / 'modules'
    protocol = {'settings': asdict(ExperimentSettings()), 'config': config.to_dict(),
                'sources': [e for e in entries if e['split'] != 'evaluation'],
                'source_code_hashes': {name: file_sha256(module_dir / name) for name in (
                    'vibration_feature_experiment.py', 'vibration_engine.py', 'paderborn.py')}}
    write_json(source / 'protocol.json', {**protocol, 'protocol_id': _json_hash(protocol)})
    return tmp_path, source, config


def test_cache_metadata_and_content_changes_are_rejected(cached_development):
    root, source, config = cached_development
    validated = validate_feature_source(source, root, config)
    path = next((source / 'features').glob('*.json'))
    content = json.loads(path.read_text(encoding='utf-8'))
    content['metadata']['reference_label'] = 'wrong'
    write_json(path, content)
    with pytest.raises(ValueError, match='검사 이후'):
        load_feature_records(source, validated)
    refreshed = validate_feature_source(source, root, config)
    with pytest.raises(ValueError, match='메타데이터'):
        load_feature_records(source, refreshed)


def test_pipeline_does_not_open_evaluation_or_change_production(cached_development, monkeypatch):
    import modules.vibration_feature_selection as selection

    root, source, config = cached_development
    protected = root / 'derived/comparison/models.joblib'
    protected.write_bytes(b'original-model')
    real_hash = selection.file_sha256
    reads = []

    def guarded_hash(path):
        assert path.parent.name != 'evaluation'
        reads.append(path)
        return real_hash(path)

    monkeypatch.setattr(selection, 'file_sha256', guarded_hash)
    # 이 검사는 데이터 흐름을 검증한다. 그래프는 실제 실행 결과를 별도로 확인한다.
    monkeypatch.setattr(selection, 'save_distribution_plot', lambda *args, **kwargs: None)
    output = root / 'selection'
    result = run_selection(source, root, config, output, progress=lambda _: None)
    assert result['reference_record_count'] == 2
    assert result['tuning_record_count'] == 4
    assert len(result['candidates']) <= 23
    assert not result['evaluation_used']
    assert protected.read_bytes() == b'original-model'
    assert (output / 'analysis_plan.json').is_file()
    assert (output / 'candidates.json').is_file()
    assert (output / 'selection.md').is_file()
    assert reads
    with pytest.raises(ValueError, match='덮어쓰지'):
        run_selection(source, root, config, output)


def test_changed_original_is_rejected(cached_development):
    root, source, config = cached_development
    path = next((root / 'raw/reference').glob('*.mat'))
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='원본이 변경'):
        validate_feature_source(source, root, config)
