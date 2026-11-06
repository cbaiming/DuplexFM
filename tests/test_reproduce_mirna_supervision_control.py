"""Saved-result verification rejects identity errors and changed statistics."""
import copy
import csv
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('reproduce_mirna_control', ROOT / 'scripts/reproduce_mirna_supervision_control.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


@pytest.fixture
def saved_fixture():
    data = {'rows': [], 'plan': {'folds': []}}
    for setting in m.control.SETTINGS:
        if setting != 'Inter_Hu_to_Mix':
            data['rows'].extend(dict(sample_id=f'{setting}_{i}', cohort=setting,
                                     label=str(i / 9), y=str(int(i >= 5))) for i in range(10))
    data['lookup'] = {row['sample_id']: i for i, row in enumerate(data['rows'])}
    for setting in m.control.SETTINGS:
        cohort = 'Mixset' if setting == 'Inter_Hu_to_Mix' else setting
        for fold in range(1 if setting == 'Inter_Hu_to_Mix' else 5):
            indices = range(10) if setting == 'Inter_Hu_to_Mix' else range(2 * fold, 2 * fold + 2)
            data['plan']['folds'].append(dict(setting=setting, fold=fold, test_ids=[f'{cohort}_{i}' for i in indices]))
    rows = []
    for state, seed, mode in sorted(m.VARIANTS, key=repr):
        for item in data['plan']['folds']:
            for sample in item['test_ids']:
                source = data['rows'][data['lookup'][sample]]
                i = int(sample.rsplit('_', 1)[1])
                noise = (i % 3 - 1) * (0.025 * (1 + (seed or 42) - 42))
                prediction = float(source['label']) + noise * (3 if state == 'initial' else 1)
                rows.append(dict(state=state, seed='' if seed is None else str(seed), mode=mode,
                                 setting=item['setting'], fold=str(item['fold']), sample_id=sample,
                                 label=source['label'], binary=source['y'], prediction=str(prediction)))
    return data, rows


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_exact_inventory_pooling_macros_and_paired_tables_recalculate(saved_fixture, tmp_path):
    dataset, rows = saved_fixture
    predictions = m.validate_predictions(rows, dataset)
    assert len(predictions) == 13 * 60
    tables = m.control.summarize(predictions, dataset, dataset['plan']['folds'])
    assert [len(table) for table in tables] == [104, 40, 48, 16]
    for (filename, keys), table in zip(m.TABLE_KEYS.items(), tables):
        path = tmp_path / filename
        save_csv(path, table)
        report = m.compare_table(path, table, keys)
        assert report['max_numeric_error'] == 0


@pytest.mark.parametrize('change,error', [
    ('duplicate', 'Duplicate saved'), ('wrong_fold', 'wrong outer fold'),
    ('label', 'label or binary'), ('binary', 'label or binary'),
    ('missing_group', 'Missing prediction'), ('missing_sample', 'exact-once'),
    ('seeded_td', 'TD24 must'), ('nonfinite', 'Nonfinite'),
])
def test_saved_predictions_reject_incomplete_or_changed_identities(saved_fixture, change, error):
    dataset, rows = saved_fixture
    rows = copy.deepcopy(rows)
    if change == 'duplicate':
        rows.append(copy.deepcopy(rows[0]))
    elif change == 'wrong_fold':
        rows[0]['fold'] = '4'
    elif change == 'label':
        rows[0]['label'] = '0.1234'
    elif change == 'binary':
        rows[0]['binary'] = '1' if rows[0]['binary'] == '0' else '0'
    elif change == 'missing_group':
        rows = [row for row in rows if (row['state'], row['seed'], row['mode']) != ('initial', '42', 'source_only')]
    elif change == 'missing_sample':
        rows.pop(0)
    elif change == 'seeded_td':
        next(row for row in rows if row['state'] == 'TD24')['seed'] = '42'
    else:
        rows[0]['prediction'] = 'nan'
    with pytest.raises(ValueError, match=error):
        m.validate_predictions(rows, dataset)


@pytest.mark.parametrize('filename', tuple(m.TABLE_KEYS))
def test_saved_statistics_must_match_recomputed_values(saved_fixture, tmp_path, filename):
    data, rows = saved_fixture
    tables = dict(zip(m.TABLE_KEYS, m.control.summarize(m.validate_predictions(rows, data), data, data['plan']['folds'])))
    expected = tables[filename]
    changed = copy.deepcopy(expected)
    column = 'PCC' if 'PCC' in changed[0] else 'PCC_mean'
    changed[0][column] += 0.0001
    path = tmp_path / filename
    save_csv(path, changed)
    with pytest.raises(ValueError, match='metric differs'):
        m.compare_table(path, expected, m.TABLE_KEYS[filename])


def test_no_seed_sd_for_td24_and_no_duplicate_table_group(saved_fixture, tmp_path):
    data, rows = saved_fixture
    _, summary, _, _ = m.control.summarize(m.validate_predictions(rows, data), data, data['plan']['folds'])
    changed = copy.deepcopy(summary)
    next(row for row in changed if row['state'] == 'TD24')['PCC_sample_SD'] = 0
    path = tmp_path / 'summary.csv'
    save_csv(path, changed)
    with pytest.raises(ValueError, match='SD fabricated'):
        m.compare_table(path, summary, m.TABLE_KEYS['summary.csv'])
    save_csv(path, summary + [summary[0]])
    with pytest.raises(ValueError, match='Duplicate saved table'):
        m.compare_table(path, summary, m.TABLE_KEYS['summary.csv'])


def test_csv_rejects_duplicate_columns_or_extra_fields(tmp_path):
    path = tmp_path / 'predictions.csv'
    path.write_text('state,state\ntrained,initial\n', encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate CSV columns'):
        m.read_csv(path)
    path.write_text('state,seed\ntrained,42,unexpected\n', encoding='utf-8')
    with pytest.raises(ValueError, match='Malformed CSV row'):
        m.read_csv(path)


def test_run_uses_saved_predictions_and_tables_without_fitting(saved_fixture, tmp_path, monkeypatch):
    dataset, rows = saved_fixture
    base = tmp_path / 'data/sirna_oligo_random'
    results = tmp_path / 'results/siRNA_oligo_random/mirna_supervision_control'
    save_csv(base / 'predictions/mirna_supervision_control.csv', rows)
    tables = m.control.summarize(m.validate_predictions(rows, dataset), dataset, dataset['plan']['folds'])
    for filename, table in zip(m.TABLE_KEYS, tables):
        save_csv(results / filename, table)
    monkeypatch.setattr(m.control.original, 'load_data', lambda _: dataset)
    monkeypatch.setattr(m.control.original, 'load_math', lambda _: pytest.fail('saved-result verification must never refit'))
    report = m.run(tmp_path)
    assert report['passed'] and report['variants'] == 13 and report['fold_groups'] == 338
    assert report['new_model_fits'] == 0 and report['GPU_used'] is False
    assert len(report['files_sha256']) == 5
