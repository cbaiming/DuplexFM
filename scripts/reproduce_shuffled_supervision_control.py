#!/usr/bin/env python3
"""Verify the saved equal-budget, single-source-seed shuffled-label comparison.

Checks exact sample/fold coverage, labels, feature identity, source-budget audit,
unchanged seed42 reference predictions and every saved result-table cell.
No model weights, training or GPU are needed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


ROOT = Path(__file__).resolve().parents[1]
saved = module('_saved_equal_budget_reference', ROOT / 'scripts/reproduce_mirna_supervision_control.py')
shuffle = module('_equal_budget_source', ROOT / 'scripts/export_shuffled_supervision_control.py')
control = saved.control
VARIANTS = {(state, 42, mode) for state in ('trained', 'initial', 'shuffled') for mode in control.MODES}
VARIANTS.add(('TD24', None, 'TD24'))


def validate_predictions(rows, dataset):
    previous = saved.VARIANTS
    try:
        saved.VARIANTS = VARIANTS
        return saved.validate_predictions(rows, dataset)
    finally:
        saved.VARIANTS = previous


def reference_identity(rows, reference_rows):
    def key(row):
        return row['state'], str(row['seed'] or ''), row['mode'], row['setting'], int(row['fold']), row['sample_id']

    expected = {key(r): float(r['prediction']) for r in reference_rows
                if r['state'] == 'TD24' or r['seed'] == '42'}
    observed = {key(r): float(r['prediction']) for r in rows if r['state'] != 'shuffled'}
    if observed != expected:
        raise ValueError('Real-label, initial or TD24 reference predictions changed')
    return len(expected)


def paired_deltas(metrics):
    lookup = {(r['state'], r['mode'], r['setting']): r for r in metrics}
    result = []
    for real in metrics:
        if real['state'] != 'trained':
            continue
        for other in ('initial', 'shuffled'):
            baseline = lookup[other, real['mode'], real['setting']]
            result.append(dict(comparison='trained_minus_' + other, seed=42, mode=real['mode'],
                               setting=real['setting'], n=real['n'], aggregation=real['aggregation'],
                               **{name: real[name] - baseline[name] for name in control.METRICS}))
    return result


def verify(root):
    root = Path(root).resolve()
    data = root / 'data/sirna_oligo_random'
    results = root / 'results/siRNA_oligo_random/shuffled_supervision_control'
    dataset = control.original.load_data(data)
    manifest_path = data / 'shuffled_supervision_features.json'
    _, manifest = shuffle.load_shuffled_features(manifest_path, dataset)
    audit = manifest['source_audit']
    required = dict(passed=True, source_seed=42, shuffled_source_seed_count=1,
                    original_initial_sha256=shuffle.INITIAL_SHA,
                    real_terminal_sha256=shuffle.REAL_SHA,
                    shuffled_terminal_sha256=shuffle.SHUFFLE_SHA,
                    updates=32728, reference_epochs=8, source_seen_rows=8 * 2094107,
                    exact_initial_parameters_buffers_rng=True)
    for key, value in required.items():
        if audit.get(key) != value:
            raise ValueError('Source-budget audit differs: ' + key)
    labels = audit['labels']
    if (labels['train_rows'] != 2094107 or labels['validation_rows'] != 401428
            or labels['positives'] != labels['shuffled_positives']
            or labels['changed_labels'] <= 0 or not labels['validation_and_test_labels_unchanged']):
        raise ValueError('Source-label permutation audit differs')
    completion = control.original.read(results / 'complete.json')
    public_audit = root / 'audit/shuffled_supervision_control.json'
    if (control.original.read(public_audit) != audit
            or control.original.sha(public_audit) != manifest['source_audit_sha256']):
        raise ValueError('Public source-audit identity differs')
    if (completion.get('source_audit') != audit
            or completion.get('feature_manifest_sha256') != manifest.get('raw_export_manifest_sha256')
            or completion.get('raw_export_manifest_sha256') != manifest.get('raw_export_manifest_sha256')):
        raise ValueError('Source-export and readout provenance disagree')
    if (completion.get('passed') is not True or completion.get('source_seed') != 42
            or completion.get('shuffled_source_seed_count') != 1
            or completion.get('new_readout_fits') != 1352
            or completion.get('prediction_rows') != 29099
            or completion.get('complete_26_outer_folds') is not True):
        raise ValueError('Single-seed shuffled readout is incomplete')
    for key, expected in [('input_sha256', control.original.INPUT_SHA),
                          ('split_sha256', control.original.PLAN_SHA)]:
        if completion.get(key) != expected:
            raise ValueError('Readout input identity differs: ' + key)
    previous_path = data / 'predictions/mirna_supervision_control.csv'
    if control.original.sha(previous_path) != completion['previous_predictions_sha256']:
        raise ValueError('Original matched-control prediction bank changed')
    rows = validate_predictions(saved.read_csv(data / 'predictions/shuffled_supervision_control.csv',
                                               saved.PREDICTION_COLUMNS), dataset)
    reused = reference_identity(rows, saved.read_csv(previous_path, saved.PREDICTION_COLUMNS))
    metrics, summary, _, _ = control.summarize(rows, dataset, dataset['plan']['folds'])
    checks = {
        'per_seed.csv': saved.compare_table(results / 'per_seed.csv', metrics, saved.TABLE_KEYS['per_seed.csv']),
        'summary.csv': saved.compare_table(results / 'summary.csv', summary, saved.TABLE_KEYS['summary.csv']),
        'paired_deltas.csv': saved.compare_table(results / 'paired_deltas.csv', paired_deltas(metrics),
                              ('comparison', 'seed', 'mode', 'setting', 'aggregation')),
    }
    return dict(passed=True, source_seed=42, prediction_rows=len(rows),
                unchanged_reference_predictions=reused, outer_groups=26,
                feature_sha256=manifest['sha256'], tables=checks,
                uncertainty='One shuffled source seed; no source-seed SD or significance claim',
                model_execution=False, GPU_used=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
