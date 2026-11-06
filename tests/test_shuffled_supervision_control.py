"""Fail-closed checks for the equal-update source-label control."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('shuffled_control', ROOT / 'scripts/export_shuffled_supervision_control.py')
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


def test_train_only_permutation_rejects_validation_contamination():
    labels = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.float32)
    fit, valid = np.asarray([0, 1, 2, 3]), np.asarray([4, 5])
    order = np.random.default_rng(42).permutation(len(fit))
    shuffled = np.full(len(labels), -1, dtype=np.float32)
    shuffled[fit] = labels[fit[order]]
    report = control.check_permutation(labels, fit, valid, order, shuffled)
    assert report['positives'] == report['shuffled_positives'] == 2
    shuffled[4] = labels[4]
    with pytest.raises(ValueError, match='only cover source training'):
        control.check_permutation(labels, fit, valid, order, shuffled)


def test_equal_budget_rejects_one_missing_update_or_changed_learning_rate():
    options = dict(seed=42, batch_size=512, micro_batch_size=512, max_updates=32728,
                   reference_rows=2094107, amp='bf16', learning_rate=3e-4,
                   lora_learning_rate=2e-5, weight_decay=1e-4, clip_grad_norm=1.0)
    reference = dict(options=options)
    shuffled = dict(inputs=dict(original_signature=reference), options=dict(options))
    complete = dict(status='complete', updates=32728, seen_rows=8 * 2094107,
                    reference_epochs=8, selection='terminal_update')
    history = [dict(step=4091 * n, seen_rows=2094107 * n) for n in range(1, 9)]
    control.check_budget(reference, shuffled, complete, history)
    complete['updates'] -= 1
    with pytest.raises(ValueError, match='Incomplete source budget'):
        control.check_budget(reference, shuffled, complete, history)
    complete['updates'] += 1
    shuffled['options']['learning_rate'] = 1e-3
    with pytest.raises(ValueError, match='options differ'):
        control.check_budget(reference, shuffled, complete, history)
