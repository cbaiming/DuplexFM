"""Identity binding and matched frozen-readout checks for the miRNA control."""
import copy
import importlib.util
import json
from pathlib import Path
import statistics

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('mirna_control', ROOT / 'scripts/run_mirna_supervision_control.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


@pytest.fixture
def manifest_fixture(tmp_path):
    rows = [{'sample_id': 'first'}, {'sample_id': 'second'}]
    data = {'rows': rows}
    registry = m.original.read(ROOT / 'MODEL_REGISTRY.json')
    manifest = dict(schema_version=1, input_sha256=m.original.INPUT_SHA, rows=2,
                    row_ids_sha256=m.row_ids_sha256(rows), features=[])
    for state, filename in [('trained', 'terminal.pt'), ('initial', 'initial.pt')]:
        path = tmp_path / (state + '.npy')
        np.save(path, np.arange(384, dtype=np.float32).reshape(2, 192))
        checkpoint = f'runs/seed42/{filename}'
        manifest['features'].append(dict(state=state, seed=42, path=path.name,
            sha256=m.original.sha(path), checkpoint_path=checkpoint,
            checkpoint_sha256=registry['model_checkpoint_sha256_from_existing_verified_reference'][checkpoint]))
    path = tmp_path / 'features.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    return path, data, registry, manifest


def save_manifest(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def test_manifest_loads_exact_matched_pairs_and_records_identity(manifest_fixture):
    path, data, registry, _ = manifest_fixture
    features, receipts = m.load_features(path, data, registry, [42])
    assert set(features) == {('trained', 42), ('initial', 42)}
    assert receipts['trained', 42]['row_ids_sha256'] == m.row_ids_sha256(data['rows'])
    assert receipts['trained', 42]['checkpoint_sha256'] != receipts['initial', 42]['checkpoint_sha256']


@pytest.mark.parametrize('change,error', [
    ('row_order', 'row identity'), ('feature_sha', 'file SHA'),
    ('checkpoint_state', 'source checkpoint'), ('checkpoint_sha', 'source checkpoint'),
    ('missing_pair', 'Missing trained/initial'), ('duplicate', 'Duplicate'),
    ('per_entry_row_order', 'row identity'), ('bad_dtype', 'float32'),
])
def test_manifest_rejects_unbound_or_unmatched_features(manifest_fixture, change, error):
    path, data, registry, manifest = manifest_fixture
    if change == 'row_order':
        data['rows'].reverse()
    elif change == 'feature_sha':
        manifest['features'][0]['sha256'] = '0' * 64
    elif change == 'checkpoint_state':
        manifest['features'][0]['checkpoint_path'] = 'runs/seed42/initial.pt'
    elif change == 'checkpoint_sha':
        manifest['features'][0]['checkpoint_sha256'] = '0' * 64
    elif change == 'missing_pair':
        manifest['features'].pop()
    elif change == 'duplicate':
        manifest['features'].append(copy.deepcopy(manifest['features'][0]))
    elif change == 'per_entry_row_order':
        manifest['features'][0]['row_ids_sha256'] = '0' * 64
    else:
        target = path.parent / manifest['features'][0]['path']
        np.save(target, np.zeros((2, 192), dtype=np.float64))
        manifest['features'][0]['sha256'] = m.original.sha(target)
    save_manifest(path, manifest)
    with pytest.raises(ValueError, match=error):
        m.load_features(path, data, registry, [42])


def test_source_only_zeros_td_and_td24_has_no_source():
    td = np.arange(48, dtype=float).reshape(2, 24)
    source = np.ones((2, 192), dtype=np.float32)
    data = {'rows': [{}, {}], 'td': td}
    plus = m.input_matrix(data, source, 'source_plus_td')
    only = m.input_matrix(data, source, 'source_only')
    baseline = m.input_matrix(data, None, 'TD24')
    np.testing.assert_array_equal(plus[:, :24], td)
    np.testing.assert_array_equal(only[:, :24], np.zeros((2, 24)))
    np.testing.assert_array_equal(plus[:, 24:], only[:, 24:])
    np.testing.assert_array_equal(baseline[:, :24], td)
    np.testing.assert_array_equal(baseline[:, 24:], np.zeros((2, 192)))


def reporting_fixture():
    rows, folds = [], []
    for setting in m.SETTINGS:
        if setting == 'Inter_Hu_to_Mix':
            continue
        rows.extend(dict(sample_id=f'{setting}_{i}', cohort=setting) for i in range(6))
    for setting in m.SETTINGS:
        cohort = 'Mixset' if setting == 'Inter_Hu_to_Mix' else setting
        for fold in range(2):
            folds.append(dict(setting=setting, fold=fold, test_ids=[f'{cohort}_{i}' for i in range(3 * fold, 3 * fold + 3)]))
    predictions = []
    for seed in (42, 43, 44):
        for state in m.STATES:
            for mode in m.MODES:
                for j, item in enumerate(folds):
                    for sid in item['test_ids']:
                        i = int(sid.rsplit('_', 1)[1])
                        y = i / 5
                        noise = (i % 3 - 1) * (.08 + .02 * (seed - 42))
                        p = y + noise if state == 'trained' else y + 3 * noise + .03 * (j % 2)
                        predictions.append(dict(state=state, seed=seed, mode=mode, setting=item['setting'],
                                                fold=item['fold'], sample_id=sid, label=y, prediction=p))
    for item in folds:
        for sid in item['test_ids']:
            i = int(sid.rsplit('_', 1)[1])
            predictions.append(dict(state='TD24', seed=None, mode='TD24', setting=item['setting'], fold=item['fold'],
                                    sample_id=sid, label=i / 5, prediction=i / 5 + .1 * (i % 2)))
    return {'rows': rows}, folds, predictions


def test_reporting_pools_settings_then_uses_matched_seed_sample_sd():
    data, folds, predictions = reporting_fixture()
    metrics, summary, deltas, delta_summary = m.summarize(predictions, data, folds)
    assert len(metrics) == 13 * 8  # 3 seeds x 2 states x 2 modes, plus TD24 once.
    assert len(deltas) == 3 * 2 * 8
    for row in summary:
        assert row['source_seed_count'] == (1 if row['state'] == 'TD24' else 3)
        if row['state'] == 'TD24':
            assert row['PCC_sample_SD'] is None
    main = next(r for r in metrics if (r['state'], r['seed'], r['mode'], r['setting']) == ('trained', 42, 'source_only', 'macro_main'))
    members = [r['PCC'] for r in metrics if (r['state'], r['seed'], r['mode']) == ('trained', 42, 'source_only') and r['setting'] in m.SETTINGS[:4]]
    assert main['PCC'] == pytest.approx(statistics.mean(members))
    lookup = {(r['state'], r['seed'], r['mode'], r['setting']): r for r in metrics}
    for row in deltas:
        assert row['PCC'] == pytest.approx(lookup['trained', row['seed'], row['mode'], row['setting']]['PCC'] - lookup['initial', row['seed'], row['mode'], row['setting']]['PCC'])
    for row in delta_summary:
        values = [d['PCC'] for d in deltas if (d['mode'], d['setting']) == (row['mode'], row['setting'])]
        assert row['source_seed_count'] == 3
        assert row['PCC_sample_SD'] == pytest.approx(statistics.stdev(values))


def test_reporting_does_not_publish_macro_for_partial_outer_coverage():
    data, folds, predictions = reporting_fixture()
    subset = [item for item in folds if item['fold'] == 0]
    records = [p for p in predictions if p['fold'] == 0]
    metrics, _, deltas, _ = m.summarize(records, data, subset)
    assert {r['aggregation'] for r in metrics} == {'partial_outer_tests'}
    assert not any(r['setting'].startswith('macro') for r in metrics + deltas)
    with pytest.raises(ValueError, match='Duplicate held-out'):
        m.summarize(records + [records[0]], data, subset)


@pytest.fixture(scope='module')
def actual_inputs():
    base = ROOT / 'data/sirna_oligo_random'
    dataset = m.original.load_data(base)
    math = m.original.load_math(ROOT)
    archives = m.load_archives(base)
    source = np.load(base / 'features/full_seed42.npy', allow_pickle=False)
    item = next(f for f in dataset['plan']['folds'] if (f['setting'], f['fold']) == ('wang2009', 0))
    return dataset, math, archives, source, item


@pytest.mark.parametrize('mode', ['source_plus_td', 'source_only', 'TD24'])
def test_numerical_readout_reproduces_archived_predictions(actual_inputs, tmp_path, mode):
    data, math, archives, source, item = actual_inputs
    x = m.input_matrix(data, source, mode)
    records, report = m.refit_fold(data, x, math, 'TD24' if mode == 'TD24' else 'trained',
                                  None if mode == 'TD24' else 42, mode, item, archives, tmp_path / 'fold')
    assert report['archive_checked']
    assert report['max_archived_prediction_error'] <= 1e-7
    assert report['fits'] == 26
    assert len(records) == len(item['test_ids'])
    assert report['outer_test_used_for_model_selection'] is False


def test_outer_test_labels_and_values_cannot_change_fit_or_scaling(actual_inputs, tmp_path):
    data, math, archives, source, item = actual_inputs
    x = m.input_matrix(data, source, 'source_only')
    _, before = m.refit_fold(data, x, math, 'initial', 42, 'source_only', item, archives, tmp_path / 'before')
    changed = copy.deepcopy(data)
    indices = [data['lookup'][sid] for sid in item['test_ids']]
    changed['y'][indices] = 1000
    for i in indices:
        changed['rows'][i]['label'] = '1000'
    altered = x.copy()
    altered[indices, 24:] += 100
    _, after = m.refit_fold(changed, altered, math, 'initial', 42, 'source_only', item, archives, tmp_path / 'after')
    assert before['alpha'] == after['alpha']
    assert before['inner_PCC'] == after['inner_PCC']
    assert before['inner_MSE'] == after['inner_MSE']
    assert (tmp_path / 'before/transforms.json').read_bytes() == (tmp_path / 'after/transforms.json').read_bytes()
    with np.load(tmp_path / 'before/arrays.npz') as a, np.load(tmp_path / 'after/arrays.npz') as b:
        for name in ('inner_coef', 'inner_intercept', 'inner_oof', 'final_coef', 'final_intercept'):
            np.testing.assert_array_equal(a[name], b[name])


def test_existing_output_is_refused_before_reading_manifest(tmp_path):
    with pytest.raises(SystemExit) as result:
        m.main(['--features-manifest', str(tmp_path / 'absent.json'), '--output', str(tmp_path)])
    assert result.value.code == 2
