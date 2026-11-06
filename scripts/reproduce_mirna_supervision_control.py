#!/usr/bin/env python3
"""Verify saved matched miRNA-supervision control predictions and all tables.

This command checks the fixed input/split identities and recomputes efficacy
metrics, source-seed sample SD, and trained-minus-initial paired differences.
It does not load source features or checkpoints, fit a model, or use a GPU.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

_SPEC = importlib.util.spec_from_file_location('_saved_mirna_control', Path(__file__).with_name('run_mirna_supervision_control.py'))
control = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(control)

SEEDS = (42, 43, 44)
VARIANTS = {(state, seed, mode) for state in control.STATES for seed in SEEDS for mode in control.MODES}
VARIANTS.add(('TD24', None, 'TD24'))
PREDICTION_COLUMNS = ('state', 'seed', 'mode', 'setting', 'fold', 'sample_id', 'label', 'binary', 'prediction')
TABLE_KEYS = {
    'per_seed.csv': ('state', 'seed', 'mode', 'setting', 'aggregation'),
    'summary.csv': ('state', 'mode', 'setting', 'aggregation'),
    'paired_deltas_per_seed.csv': ('seed', 'mode', 'setting', 'aggregation'),
    'paired_deltas_summary.csv': ('mode', 'setting', 'aggregation'),
}


def read_csv(path, columns=None):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        if not header or len(header) != len(set(header)) or (columns is not None and set(header) != set(columns)):
            raise ValueError('Unexpected or duplicate CSV columns: ' + str(path))
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError('Malformed CSV row: ' + str(path))
    return rows


def validate_predictions(rows, dataset):
    folds = {(item['setting'], item['fold']): set(item['test_ids']) for item in dataset['plan']['folds']}
    expected_folds = {(setting, fold) for setting in control.SETTINGS
                      for fold in range(1 if setting == 'Inter_Hu_to_Mix' else 5)}
    if set(folds) != expected_folds or len(folds) != len(dataset['plan']['folds']):
        raise ValueError('Fixed26 fold inventory differs')
    expected_groups = {(*variant, setting, fold) for variant in VARIANTS for setting, fold in folds}
    seen, groups, predictions = set(), {}, []
    for row in rows:
        state, mode = row['state'], row['mode']
        try:
            seed = None if row['seed'] == '' else int(row['seed'])
            fold = int(row['fold'])
            label, prediction = float(row['label']), float(row['prediction'])
        except (TypeError, ValueError) as error:
            raise ValueError('Invalid saved prediction number') from error
        if not math.isfinite(label) or not math.isfinite(prediction):
            raise ValueError('Nonfinite saved prediction or label')
        variant = state, seed, mode
        if variant not in VARIANTS:
            raise ValueError('Unknown state/seed/mode; TD24 must appear once without a seed')
        setting, sample = row['setting'], row['sample_id']
        key = (*variant, setting, fold)
        identity = (*variant, setting, sample)
        if identity in seen:
            raise ValueError('Duplicate saved prediction sample')
        seen.add(identity)
        if key not in expected_groups or sample not in folds[setting, fold]:
            raise ValueError('Saved prediction has wrong outer fold membership')
        if sample not in dataset['lookup']:
            raise ValueError('Unknown prediction sample identity')
        source = dataset['rows'][dataset['lookup'][sample]]
        if label != float(source['label']) or row['binary'] != source['y']:
            raise ValueError('Saved prediction label or binary identity differs')
        groups.setdefault(key, set()).add(sample)
        predictions.append(dict(row, seed=seed, fold=fold, prediction=prediction))
    if set(groups) != expected_groups:
        raise ValueError('Missing prediction state/seed/mode/fold group')
    if any(ids != folds[key[-2:]] for key, ids in groups.items()):
        raise ValueError('Incomplete exact-once test sample coverage')
    return predictions


def compare_table(path, expected, keys, tolerance=1e-10):
    rows = read_csv(path, expected[0].keys())
    normalize = lambda value: '' if value is None else str(value)
    saved, recomputed = {}, {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        if key in saved:
            raise ValueError('Duplicate saved table group: ' + str(path))
        saved[key] = row
    for row in expected:
        key = tuple(normalize(row[name]) for name in keys)
        if key in recomputed:
            raise ValueError('Duplicate recomputed table group')
        recomputed[key] = row
    if set(saved) != set(recomputed):
        raise ValueError('Saved table group inventory differs: ' + str(path))
    maximum = 0.0
    for key, expected_row in recomputed.items():
        for name, value in expected_row.items():
            actual = saved[key][name]
            if value is None:
                if actual != '':
                    raise ValueError('SD fabricated for a single source: ' + str(path))
            elif isinstance(value, int):
                try:
                    equal = int(actual) == value
                except ValueError:
                    equal = False
                if not equal:
                    raise ValueError('Saved table integer differs: ' + str(path) + '/' + name)
            elif isinstance(value, float):
                try:
                    difference = abs(float(actual) - value)
                except ValueError as error:
                    raise ValueError('Invalid saved table number: ' + str(path)) from error
                if not math.isfinite(difference) or difference > tolerance:
                    raise ValueError('Saved table metric differs: ' + str(path) + '/' + name)
                maximum = max(maximum, difference)
            elif actual != value:
                raise ValueError('Saved table text differs: ' + str(path) + '/' + name)
    return dict(rows=len(rows), max_numeric_error=maximum, atol=tolerance)


def run(root, data_root=None, results_root=None):
    root = Path(root).resolve()
    base = Path(data_root).resolve() if data_root else root / 'data/sirna_oligo_random'
    results = Path(results_root).resolve() if results_root else root / 'results/siRNA_oligo_random/mirna_supervision_control'
    dataset = control.original.load_data(base)
    path = base / 'predictions/mirna_supervision_control.csv'
    predictions = validate_predictions(read_csv(path, PREDICTION_COLUMNS), dataset)
    tables = control.summarize(predictions, dataset, dataset['plan']['folds'])
    reports = {}
    for (filename, keys), expected in zip(TABLE_KEYS.items(), tables):
        reports[filename] = compare_table(results / filename, expected, keys)
    files = [path, *(results / name for name in TABLE_KEYS)]
    return dict(passed=True, prediction_rows=len(predictions), source_seeds=list(SEEDS),
                variants=len(VARIANTS), outer_folds=26, fold_groups=len(VARIANTS) * 26,
                deterministic_TD24_repetitions=1, tables=reports,
                input_sha256=control.original.INPUT_SHA, split_sha256=control.original.PLAN_SHA,
                files_sha256={str(file): control.original.sha(file) for file in files},
                macro_main_settings=list(control.MACROS['macro_main']),
                macro_extension_settings=list(control.MACROS['macro_extension']),
                uncertainty='sample SD across matched source seeds; no independent-fold significance',
                new_model_fits=0, source_features_loaded=False, GPU_used=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path, help='Optional data/sirna_oligo_random supplement directory')
    parser.add_argument('--results-root', type=Path, help='Optional directory containing the four saved control tables')
    args = parser.parse_args(argv)
    try:
        report = run(args.root, args.data_root, args.results_root)
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
