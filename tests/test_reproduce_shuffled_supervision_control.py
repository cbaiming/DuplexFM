"""The added source state must not change references or invent seed replicates."""
import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


m = load('verify_shuffled_control', ROOT / 'scripts/reproduce_shuffled_supervision_control.py')
existing = load('existing_prediction_fixtures', ROOT / 'tests/test_reproduce_mirna_supervision_control.py')


@pytest.fixture
def fixture():
    dataset, original = existing.saved_fixture.__wrapped__()
    records = [copy.deepcopy(row) for row in original if row['seed'] in ('', '42')]
    records.extend(dict(row, state='shuffled') for row in list(records) if row['state'] == 'initial')
    return dataset, original, records


def test_complete_single_seed_comparison_has_no_source_sd(fixture):
    dataset, original, records = fixture
    rows = m.validate_predictions(records, dataset)
    assert len(rows) == 7 * 60
    assert m.reference_identity(rows, original) == 5 * 60
    metrics, summary, _, _ = m.control.summarize(rows, dataset, dataset['plan']['folds'])
    assert len(metrics) == len(summary) == 56
    assert len(m.paired_deltas(metrics)) == 32
    assert all(r['source_seed_count'] == 1 and r['PCC_sample_SD'] is None for r in summary)


def test_wrong_shuffled_seed_and_modified_reference_are_rejected(fixture):
    dataset, original, records = fixture
    next(r for r in records if r['state'] == 'shuffled')['seed'] = '43'
    with pytest.raises(ValueError, match='Unknown state/seed/mode'):
        m.validate_predictions(records, dataset)
    next(r for r in records if r['state'] == 'shuffled')['seed'] = '42'
    next(r for r in records if r['state'] == 'trained')['prediction'] = '0.1234567'
    with pytest.raises(ValueError, match='reference predictions changed'):
        m.reference_identity(m.validate_predictions(records, dataset), original)
