"""주파수 계산의 의미와 실험 데이터 분리를 검증한다."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from modules.paderborn import VibrationRecord
from modules.vibration_config import load_vibration_config
from modules.vibration_dataset import expected_records, file_sha256, write_json
from modules.vibration_feature_experiment import (
    BASE_FEATURES, ExperimentRecord, ExperimentSettings, compare_feature_set,
    extract_experiment_record, recommend_candidate, record_score, run_experiment, spectral_features,
)


def make_signal(frequency=5000, modulation=.2, duration=1.0, jitter=False):
    time = np.arange(int(64000 * duration)) / 64000
    if jitter:
        time[1:-1] += .15 / 64000 * np.sin(np.arange(1, len(time) - 1))
    values = (1 + modulation * np.sin(2 * np.pi * 80 * time)) * np.sin(2 * np.pi * frequency * time)
    return VibrationRecord({"file": "N09_M07_F10_K001_1.mat", "bearing_id": "K001",
                            "condition": "N09_M07_F10", "reference_label": "healthy"}, time, values)


def test_frequency_bands_and_amplitude_invariance():
    settings = ExperimentSettings()
    low = spectral_features(make_signal(1500).vibration, settings)
    high = spectral_features(make_signal(8000).vibration, settings)
    assert low['power_ratio_1000_3000'] > .99
    assert high['power_ratio_6000_12000'] > .99
    scaled = spectral_features(3 * make_signal(8000).vibration, settings)
    for name in high:
        assert scaled[name] == pytest.approx(high[name], rel=1e-8, abs=1e-8)


def test_envelope_detects_amplitude_modulation():
    settings = ExperimentSettings()
    quiet = spectral_features(make_signal(modulation=.1).vibration, settings)
    modulated = spectral_features(make_signal(modulation=.8).vibration, settings)
    assert modulated['envelope_modulation_depth'] > 5 * quiet['envelope_modulation_depth']
    assert modulated['envelope_ratio_20_200'] > .95


def test_jitter_is_resampled_using_actual_time():
    config = load_vibration_config('config/vibration_analysis.yaml')
    settings = ExperimentSettings()
    regular = extract_experiment_record(make_signal(), config, settings)
    jittered = extract_experiment_record(make_signal(jitter=True), config, settings)
    assert not jittered.excluded_windows
    np.testing.assert_allclose(regular.features['spectral_centroid_hz'],
                               jittered.features['spectral_centroid_hz'], rtol=.01)


def test_large_gap_is_excluded_and_never_bridged():
    config = load_vibration_config('config/vibration_analysis.yaml')
    r = make_signal()
    keep = (r.time_sec < .49) | (r.time_sec > .495)
    r.time_sec, r.vibration = r.time_sec[keep], r.vibration[keep]
    result = extract_experiment_record(r, config, ExperimentSettings())
    assert any(w['reason'] == 'time_gap' for w in result.excluded_windows)
    assert all(not (w.start_sec < .495 and w.end_sec > .49) for w in result.features.itertuples())
    assert record_score(pd.DataFrame({'window_id': [0, 1, 3]}), np.array([9., 9., 9.]), 3) is None
    assert record_score(pd.DataFrame({'window_id': [0, 1, 2]}), np.array([9., 9., 9.]), 3) == 9


def test_experiment_never_reads_evaluation_or_changes_existing_model(tmp_path, monkeypatch):
    import modules.vibration_feature_experiment as experiment

    config = replace(load_vibration_config('config/vibration_analysis.yaml'), records_per_bearing=1,
                     n_estimators=10)
    entries = expected_records(config)
    for entry in entries:
        path = tmp_path / entry['relative_path']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(entry['file'].encode())
        entry['sha256'] = file_sha256(path)
    write_json(tmp_path / 'derived/comparison/dataset_manifest.json',
               {'config_fingerprint': config.fingerprint, 'records': entries})
    protected = tmp_path / 'derived/comparison/models.joblib'
    protected.write_bytes(b'keep-existing-model')
    reads = []

    def guarded_load(path):
        assert path.parent.name != 'evaluation'
        reads.append(path)
        entry = next(e for e in entries if e['file'] == path.name)
        r = make_signal(modulation=.2 if entry['provided_label'] == 'healthy' else .8)
        r.metadata.update({'file': path.name, 'bearing_id': entry['bearing_id'],
                           'reference_label': entry['provided_label']})
        return r

    monkeypatch.setattr(experiment, 'load_paderborn', guarded_load)
    report = run_experiment(tmp_path, config, tmp_path / 'experiment', progress=lambda _: None)
    assert len(reads) == 6
    assert not report['evaluation_used']
    assert protected.read_bytes() == b'keep-existing-model'
    assert (tmp_path / 'experiment/comparison.md').is_file()
    assert set(report['candidates']) == {'baseline', 'spectral', 'envelope', 'combined'}
    with pytest.raises(ValueError, match='덮어쓰지'):
        run_experiment(tmp_path, config, tmp_path / 'experiment')


def test_zero_energy_is_rejected():
    with pytest.raises(ValueError, match='에너지가 없는'):
        spectral_features(np.zeros(16000), ExperimentSettings())


def test_damage_values_and_damage_name_do_not_change_training_or_thresholds():
    config = replace(load_vibration_config('config/vibration_analysis.yaml'), n_estimators=10)
    rng = np.random.default_rng(42)

    def record(bearing, label):
        frame = pd.DataFrame(rng.normal(size=(20, 4)), columns=BASE_FEATURES)
        frame['window_id'] = np.arange(20)
        return ExperimentRecord({'file': bearing, 'bearing_id': bearing, 'reference_label': label},
                                frame, [], {})

    reference = [record('K001', 'healthy'), record('K002', 'healthy')]
    normal = record('K003', 'healthy')
    damage = record('KA04', 'outer_ring_damage')
    first, first_model = compare_feature_set(reference, [normal, damage], BASE_FEATURES, config)
    damage.features[list(BASE_FEATURES)] *= 100
    damage.metadata.update({'bearing_id': 'KI04', 'reference_label': 'inner_ring_damage'})
    second, second_model = compare_feature_set(reference, [normal, damage], BASE_FEATURES, config)
    for method in first['methods']:
        assert first['methods'][method]['threshold'] == second['methods'][method]['threshold']
    probe = normal.features[list(BASE_FEATURES)].to_numpy()
    np.testing.assert_array_equal(first_model['forest'].score_samples(probe),
                                  second_model['forest'].score_samples(probe))


@pytest.mark.parametrize('fp,ka,ki,expected', [(1, 20, 3, 'spectral'), (2, 20, 20, None),
                                            (1, 19, 20, None), (1, 20, 0, None)])
def test_recommendation_requires_both_damage_types_and_no_more_false_alarms(fp, ka, ki, expected):
    def candidate(false_alarms, outer, inner):
        return {'features': BASE_FEATURES, 'methods': {'isolation_forest': {'metrics': {
            'fp': false_alarms, 'tp': outer + inner, 'unscorable_record_count': 0,
            'by_bearing': {'KA04': {'alert_record_count': outer}, 'KI04': {'alert_record_count': inner}},
        }}}}
    candidates = {'baseline': candidate(1, 20, 0), 'spectral': candidate(fp, ka, ki)}
    assert recommend_candidate(candidates) == expected
