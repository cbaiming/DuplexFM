"""Portable input/split validation; numerical solver itself is preserved upstream."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/refit_oligo.py'
SPEC = importlib.util.spec_from_file_location('portable_oligo_refit', PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def tiny_plan():
    lookup = {}
    for cohort in m.SETTINGS:
        if cohort != 'Inter_Hu_to_Mix':
            lookup.update({f'{cohort}_{i}': {'cohort': cohort} for i in range(10)})
    plan = {'folds': []}
    for setting in m.SETTINGS:
        cohort = 'Mixset' if setting == 'Inter_Hu_to_Mix' else setting
        for fold in range(1 if setting == 'Inter_Hu_to_Mix' else 5):
            test = [f'{cohort}_{i}' for i in range(10)] if setting == 'Inter_Hu_to_Mix' else [f'{cohort}_{2 * fold + i}' for i in range(2)]
            fit = [f'Huesken_{i}' for i in range(10)] if setting == 'Inter_Hu_to_Mix' else [sid for sid, row in lookup.items() if row['cohort'] == cohort and sid not in test]
            groups = np.array_split(np.arange(len(fit)), 5)
            inner = [{'train': [i for i in range(len(fit)) if i not in set(valid)], 'valid': valid.tolist()} for valid in groups]
            plan['folds'].append(dict(setting=setting, fold=fold, fit_ids=fit, test_ids=test, inner=inner))
    return plan, lookup


def test_valid_fixed_fold_inventory_and_no_outer_test_in_inner():
    plan, lookup = tiny_plan()
    m.validate_plan(plan, lookup, dict.fromkeys(m.SETTINGS, 10))
    changed = copy.deepcopy(plan)
    changed['folds'][0]['inner'][0]['valid'].append(len(changed['folds'][0]['fit_ids']))
    with pytest.raises(ValueError, match='Invalid inner indices'):
        m.validate_plan(changed, lookup, dict.fromkeys(m.SETTINGS, 10))


def test_train_test_overlap_and_duplicate_inner_rejected():
    plan, lookup = tiny_plan()
    changed = copy.deepcopy(plan)
    changed['folds'][0]['fit_ids'][0] = changed['folds'][0]['test_ids'][0]
    with pytest.raises(ValueError, match='Invalid outer'):
        m.validate_plan(changed, lookup, dict.fromkeys(m.SETTINGS, 10))
    changed = copy.deepcopy(plan)
    changed['folds'][0]['inner'][0]['valid'].append(changed['folds'][0]['inner'][0]['valid'][0])
    with pytest.raises(ValueError, match='Invalid inner indices'):
        m.validate_plan(changed, lookup, dict.fromkeys(m.SETTINGS, 10))


def test_fold_requests_are_explicit_and_fail_closed():
    plan, _ = tiny_plan()
    assert len(m.selected_folds(plan, ['Huesken'], [0])) == 1
    assert len(m.selected_folds(plan, ['Huesken'], [0], True)) == 26
    with pytest.raises(ValueError, match='absent'):
        m.selected_folds(plan, ['Inter_Hu_to_Mix'], [4])


def test_td_control_never_loads_source_features(tmp_path):
    td = np.arange(48).reshape(2, 24).astype(float)
    dataset = {'rows': [{}, {}], 'td': td}
    x, identity = m.input_matrix(tmp_path, dataset, 42, 'TD24')
    np.testing.assert_array_equal(x[:, :24], td)
    np.testing.assert_array_equal(x[:, 24:], np.zeros((2, 192)))
    assert identity is None
    with pytest.raises(ValueError, match='Unknown frozen'):
        m.input_matrix(tmp_path, dataset, 99, 'source')


def test_existing_output_is_never_overwritten(tmp_path):
    with pytest.raises(SystemExit) as result:
        m.main(['--output', str(tmp_path)])
    assert result.value.code == 2
