#!/usr/bin/env python3
"""Matched frozen-source control for the contribution of miRNA supervision.

Compare each exact archived terminal checkpoint with its own exact initial
checkpoint, with the unchanged R55 head and fixed Oligo splits. A manifest binds
each feature array to its checkpoint and ordered input identities. Neither the
source encoder nor its accessibility backbone is trained here. The default is
a seed42/Huesken/fold0 smoke run; --all requests all 26 outer folds.
"""
from __future__ import annotations

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr

_SPEC = importlib.util.spec_from_file_location('_supervision_original_refit', Path(__file__).with_name('refit_oligo.py'))
original = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(original)

SETTINGS = original.SETTINGS
STATES = ('trained', 'initial')
MODES = ('source_plus_td', 'source_only')
MACROS = {'macro_main': SETTINGS[:4], 'macro_extension': SETTINGS[4:]}
METRICS = ('PCC', 'Spearman', 'MSE', 'MAE')
MODULE_ARCHIVE_SHA = 'c3927602d7c915c544141fa1c299e176b1c196ea1a525e7fd8e6023990a41a9b'


def row_ids_sha256(rows):
    """Ordered identity digest; no labels or TD values enter this digest."""
    return hashlib.sha256(('\n'.join(row['sample_id'] for row in rows) + '\n').encode('utf-8')).hexdigest()


def load_features(path, dataset, registry, seeds):
    """Fail closed on missing pairs, mismatched row order, or checkpoint state."""
    path = Path(path).resolve()
    manifest = original.read(path)
    digest = row_ids_sha256(dataset['rows'])
    if manifest.get('schema_version') != 1:
        raise ValueError('Feature manifest requires schema_version=1')
    if manifest.get('input_sha256') != original.INPUT_SHA:
        raise ValueError('Feature manifest input SHA differs')
    if manifest.get('rows') != len(dataset['rows']) or manifest.get('row_ids_sha256') != digest:
        raise ValueError('Feature manifest ordered row identity differs')
    checkpoint_shas = registry['model_checkpoint_sha256_from_existing_verified_reference']
    features = {}
    receipts = {}
    seen = set()
    for item in manifest.get('features', []):
        key = item.get('state'), item.get('seed')
        if key in seen or key[0] not in STATES or type(key[1]) is not int or key[1] not in (42, 43, 44):
            raise ValueError('Duplicate or invalid feature state/seed')
        seen.add(key)
        state, seed = key
        checkpoint_path = f'runs/seed{seed}/' + ('terminal.pt' if state == 'trained' else 'initial.pt')
        if item.get('checkpoint_path') != checkpoint_path or item.get('checkpoint_sha256') != checkpoint_shas[checkpoint_path]:
            raise ValueError('Feature source checkpoint differs from exact MODEL_REGISTRY state/seed')
        if item.get('row_ids_sha256', digest) != digest:
            raise ValueError('Feature entry ordered row identity differs')
        feature_path = Path(item['path'])
        if not feature_path.is_absolute():
            feature_path = path.parent / feature_path
        feature_path = feature_path.resolve()
        if seed not in seeds:
            continue
        actual_sha = original.sha(feature_path)
        if actual_sha != item.get('sha256'):
            raise ValueError('Feature file SHA differs: ' + str(feature_path))
        values = np.load(feature_path, allow_pickle=False)
        if values.shape != (len(dataset['rows']), 192) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError('Expected finite aligned float32 source Nx192')
        features[key] = values
        receipts[key] = dict(state=state, seed=seed, path=str(feature_path), sha256=actual_sha,
                             checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_shas[checkpoint_path],
                             row_ids_sha256=digest)
    required = {(state, seed) for state in STATES for seed in seeds}
    if set(features) != required:
        raise ValueError('Missing trained/initial matched feature pairs: ' + repr(required - set(features)))
    return features, receipts


def input_matrix(dataset, source, mode):
    n = len(dataset['rows'])
    if mode == 'TD24':
        return np.column_stack((dataset['td'], np.zeros((n, 192))))
    if mode not in MODES:
        raise ValueError('Unknown readout mode')
    td = dataset['td'] if mode == 'source_plus_td' else np.zeros((n, 24))
    return np.column_stack((td, np.asarray(source, dtype=np.float64)))


def load_archives(base):
    cpu = original.archived_predictions(base)
    path = base / 'predictions/pairac_modules.csv'
    if original.sha(path) != MODULE_ARCHIVE_SHA:
        raise ValueError('Archived source-only predictions SHA differs')
    modules = {}
    for row in original.table(path):
        if row['source'] not in {f'full_seed{s}' for s in (42, 43, 44)} or row['mode'] != 'source':
            continue
        key = int(row['source'][-2:]), row['setting'], int(row['fold']), row['sample_id']
        if key in modules:
            raise ValueError('Duplicate archived source-only prediction')
        modules[key] = row
    if len(modules) != 3 * sum(original.COUNTS.values()):
        raise ValueError('Incomplete archived source-only prediction inventory')
    return cpu, modules


def expected_predictions(dataset, state, seed, mode, item, archives):
    if state == 'initial':
        return None
    cpu, modules = archives
    expected = []
    for sid in item['test_ids']:
        if mode == 'source_only':
            saved = modules[seed, item['setting'], item['fold'], sid]
        else:
            model = f'PairAC{seed}' if state == 'trained' else 'TD24_R55'
            saved = cpu[model, item['setting'], item['fold'], sid]
        row = dataset['rows'][dataset['lookup'][sid]]
        if float(saved['label']) != float(row['label']) or saved['binary'] != row['y']:
            raise ValueError('Archived prediction labels differ')
        expected.append(float(saved['prediction']))
    return np.asarray(expected)


def write_csv(path, rows, fieldnames=None):
    if not rows:
        return
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def refit_fold(dataset, x, math, state, seed, mode, item, archives, output):
    """Reuse original transformations, alpha selection, solver, and audits."""
    head, scale, audit = math['head'], math['scale'], math['audit']
    source = mode != 'TD24'
    fi = np.asarray([dataset['lookup'][sid] for sid in item['fit_ids']], dtype=int)
    ti = np.asarray([dataset['lookup'][sid] for sid in item['test_ids']], dtype=int)
    y = dataset['y'][fi]
    output.mkdir(parents=True, exist_ok=False)
    oof = np.full((len(fi), 5), np.nan)
    transforms, group_records, certificates, errors = [], [], [], []
    coef = np.empty((5, 5, 216 if source else 24))
    intercept = np.empty((5, 5))
    for stage, split in enumerate(item['inner']):
        train, valid = np.asarray(split['train']), np.asarray(split['valid'])
        t = original.fit_scaler(x[fi[train]], scale)
        transforms.append(t)
        z, zv = head.transform(x[fi[train]], t), head.transform(x[fi[valid]], t)
        targets = [dataset['rows'][fi[i]]['target57'] for i in train]
        groups, metadata = original.training_groups(targets, math)
        group_records.append(dict(labels=groups, metadata=metadata))
        profile = audit.profile(z if source else z[:, :24], y[train], groups)
        stage_certs = []
        for candidate, alpha in enumerate(head.ALPHAS):
            b, center, cert = head.fit(z, y[train], groups, alpha, source)
            if not cert['certified']:
                raise ValueError('Inner R55 numerical certificate failed')
            check = audit.check_source if source else audit.audit_TD
            errors.append(check(z if source else z[:, :24], y[train], groups, b, center, cert, alpha, cached=profile))
            oof[valid, candidate] = (zv if source else zv[:, :24]) @ b + center
            coef[stage, candidate], intercept[stage, candidate] = b, center
            stage_certs.append(cert)
        certificates.append(stage_certs)
    if not np.isfinite(oof).all():
        raise ValueError('Incomplete inner out-of-fold predictions')
    pc, mse = scale.score(y, oof)
    selected = head.choose(pc, mse)
    alpha = head.ALPHAS[selected]
    t = original.fit_scaler(x[fi], scale)
    z, zt = head.transform(x[fi], t), head.transform(x[ti], t)
    groups, metadata = original.training_groups([dataset['rows'][i]['target57'] for i in fi], math)
    group_records.append(dict(labels=groups, metadata=metadata))
    b, center, cert = head.fit(z, y, groups, alpha, source)
    if not cert['certified']:
        raise ValueError('Final R55 numerical certificate failed')
    check = audit.check_source if source else audit.audit_TD
    errors.append(check(z if source else z[:, :24], y, groups, b, center, cert, alpha))
    prediction = (zt if source else zt[:, :24]) @ b + center
    if mode == 'source_only':
        np.testing.assert_array_equal(x[:, :24], np.zeros_like(x[:, :24]))
        np.testing.assert_array_equal(b[:24], np.zeros(24))
        np.testing.assert_array_equal(t['td_mean'], np.zeros(24))
        np.testing.assert_array_equal(t['td_scale'], np.ones(24))
    full_b = b if source else np.r_[b, np.zeros(192)]
    raw_b, raw_center = head.deployment(t, full_b, center)
    np.testing.assert_allclose(prediction, x[ti] @ raw_b + raw_center, atol=1e-10, rtol=1e-10)
    expected = expected_predictions(dataset, state, seed, mode, item, archives)
    error = None
    if expected is not None:
        np.testing.assert_allclose(prediction, expected, atol=1e-7, rtol=0,
                                   err_msg='Trained/TD24 refit differs from archived result')
        error = float(np.max(np.abs(prediction - expected)))
    np.savez_compressed(output / 'arrays.npz', inner_coef=coef, inner_intercept=intercept, inner_oof=oof,
                        final_coef=b, final_intercept=center, deployment_coef=raw_b,
                        deployment_intercept=raw_center, prediction=prediction)
    original.write(output / 'certificates.json', dict(inner=certificates, final=cert))
    original.write(output / 'transforms.json', dict(inner=transforms, final=t))
    original.write(output / 'training_groups.json', group_records)
    records = []
    for sid, value in zip(item['test_ids'], prediction):
        row = dataset['rows'][dataset['lookup'][sid]]
        records.append(dict(state=state, seed=seed, mode=mode, setting=item['setting'], fold=item['fold'],
                            sample_id=sid, label=row['label'], binary=row['y'], prediction=float(value)))
    write_csv(output / 'predictions.csv', records)
    report = dict(passed=True, state=state, seed=seed, mode=mode, setting=item['setting'], fold=item['fold'],
                  fits=26, alpha=alpha, candidate=selected, inner_PCC=pc, inner_MSE=mse,
                  fit_ids=item['fit_ids'], test_ids=item['test_ids'], test_rows=len(ti),
                  archive_checked=expected is not None, max_archived_prediction_error=error,
                  archived_prediction_atol=1e-7 if expected is not None else None,
                  max_independent_normal_error=float(np.max(np.asarray(errors)[:, 0])),
                  inner_selection_uses_only_outer_fit=True, outer_test_used_for_model_selection=False,
                  source_model_training=False, GPU_used=False)
    original.write(output / 'complete.json', report)
    return records, original.plain(report)


def scores(rows):
    values = np.asarray([[float(r['label']), float(r['prediction'])] for r in rows])
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('At least two finite held-out predictions required')
    y, p = values.T
    result = dict(PCC=float(pearsonr(y, p).statistic), Spearman=float(spearmanr(y, p).statistic),
                  MSE=float(np.mean((y - p) ** 2)), MAE=float(np.mean(np.abs(y - p))))
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('Undefined held-out efficacy correlation')
    return result


def summarize(predictions, dataset, requested):
    """Pool outer predictions once per setting, then average setting metrics.

    Source-seed SD and trained-minus-initial deltas use matched source seeds.
    Correlated outer folds are never treated as independent replicates.
    """
    expected = {}
    for item in requested:
        expected.setdefault(item['setting'], set()).update(item['test_ids'])
    complete_settings = set()
    for setting, ids in expected.items():
        cohort = 'Mixset' if setting == 'Inter_Hu_to_Mix' else setting
        all_ids = {r['sample_id'] for r in dataset['rows'] if r['cohort'] == cohort}
        if ids == all_ids:
            complete_settings.add(setting)
    groups, seen = {}, set()
    for row in predictions:
        key = row['state'], row['seed'], row['mode'], row['setting']
        identity = (*key, row['sample_id'])
        if identity in seen:
            raise ValueError('Duplicate held-out sample in prediction group')
        seen.add(identity)
        groups.setdefault(key, []).append(row)
    metrics = []
    for (state, seed, mode, setting), rows in sorted(groups.items(), key=lambda x: repr(x[0])):
        if {r['sample_id'] for r in rows} != expected[setting]:
            raise ValueError('Incomplete held-out prediction coverage')
        metrics.append(dict(state=state, seed=seed, mode=mode, setting=setting,
                            aggregation='pooled' if setting in complete_settings else 'partial_outer_tests',
                            n=len(rows), **scores(rows)))
    source_groups = {(r['state'], r['seed'], r['mode']) for r in metrics}
    for state, seed, mode in sorted(source_groups, key=repr):
        by_setting = {r['setting']: r for r in metrics if (r['state'], r['seed'], r['mode']) == (state, seed, mode)}
        for name, settings in MACROS.items():
            if not set(settings) <= complete_settings:
                continue
            metrics.append(dict(state=state, seed=seed, mode=mode, setting=name, aggregation='setting_mean',
                                n=sum(by_setting[s]['n'] for s in settings),
                                **{metric: statistics.mean(by_setting[s][metric] for s in settings) for metric in METRICS}))
    summary = aggregate_seed_metrics(metrics, paired=False)
    lookup = {(r['state'], r['seed'], r['mode'], r['setting']): r for r in metrics}
    deltas = []
    for row in metrics:
        if row['state'] != 'trained':
            continue
        initial = lookup['initial', row['seed'], row['mode'], row['setting']]
        if initial['n'] != row['n'] or initial['aggregation'] != row['aggregation']:
            raise ValueError('Trained/initial prediction groups differ')
        deltas.append(dict(seed=row['seed'], mode=row['mode'], setting=row['setting'], n=row['n'],
                           aggregation=row['aggregation'],
                           **{metric: row[metric] - initial[metric] for metric in METRICS}))
    return metrics, summary, deltas, aggregate_seed_metrics(deltas, paired=True)


def aggregate_seed_metrics(rows, paired):
    groups = {}
    for row in rows:
        key = (row['mode'], row['setting'], row['aggregation']) if paired else (row['state'], row['mode'], row['setting'], row['aggregation'])
        groups.setdefault(key, []).append(row)
    result = []
    for key, values in sorted(groups.items()):
        names = ('mode', 'setting', 'aggregation') if paired else ('state', 'mode', 'setting', 'aggregation')
        entry = dict(zip(names, key))
        entry.update(source_seed_count=len(values), n=values[0]['n'])
        for metric in METRICS:
            numbers = [r[metric] for r in values]
            entry[metric + '_mean'] = statistics.mean(numbers)
            entry[metric + '_sample_SD'] = statistics.stdev(numbers) if len(numbers) > 1 else None
        result.append(entry)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--features-manifest', type=Path, required=True)
    parser.add_argument('--seed', nargs='+', type=int, choices=(42, 43, 44), default=[42])
    parser.add_argument('--mode', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--setting', nargs='+', choices=SETTINGS, default=['Huesken'])
    parser.add_argument('--fold', nargs='+', type=int, default=[0])
    parser.add_argument('--all', action='store_true', help='Run all 26 fixed outer folds; specify --seed 42 43 44 for three source seeds')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing output is refused')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    base = args.data_root.resolve() if args.data_root else root / 'data/sirna_oligo_random'
    output = args.output.resolve()
    protected = (base, root / 'data', root / 'reference', root / 'workflows')
    if output.exists() or output == root or root.is_relative_to(output) or any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        parser.error('--output must be new and outside data, reference, workflows, or checkout ancestors')
    seeds, modes = list(dict.fromkeys(args.seed)), list(dict.fromkeys(args.mode))
    started = time.monotonic()
    try:
        dataset = original.load_data(base)
        features, receipts = load_features(args.features_manifest, dataset, original.read(root / 'MODEL_REGISTRY.json'), seeds)
        archives = load_archives(base)
        math = original.load_math(root)
        requested = original.selected_folds(dataset['plan'], args.setting, args.fold, args.all)
        output.mkdir(parents=True, exist_ok=False)
        original.write(output / 'run_inputs.json', dict(protocol='matched_exact_initial_vs_miRNA_trained_frozen_R55',
                       input_sha256=original.INPUT_SHA, split_sha256=original.PLAN_SHA,
                       feature_manifest_sha256=original.sha(args.features_manifest),
                       model_registry_sha256=original.sha(root / 'MODEL_REGISTRY.json'),
                       cpu_reference_sha256=original.ARCHIVE_SHA, source_only_reference_sha256=MODULE_ARCHIVE_SHA,
                       features=list(receipts.values()), seeds=seeds, modes=modes,
                       requested_outer_folds=len(requested), alphas=list(math['head'].ALPHAS),
                       source_model_training=False, CPU_only=True,
                       inference='descriptive matched source-seed differences; no independent-fold significance',
                       interpretation='initial removes miRNA updates but retains the same pretrained accessibility backbone'))
        predictions, reports = [], []
        runs = [(state, seed, mode) for seed in seeds for state in STATES for mode in modes] + [('TD24', None, 'TD24')]
        for state, seed, mode in runs:
            x = input_matrix(dataset, features.get((state, seed)), mode)
            model = f'{state}_seed{seed}' if seed is not None else 'TD24_R55'
            for item in requested:
                folder = output / model / mode / item['setting'] / f"fold{item['fold']}"
                part, report = refit_fold(dataset, x, math, state, seed, mode, item, archives, folder)
                predictions.extend(part)
                reports.append(report)
                print('FOLD_COMPLETE', state, seed, mode, item['setting'], item['fold'],
                      round(time.monotonic() - started, 2), flush=True)
        metrics, summary, deltas, delta_summary = summarize(predictions, dataset, requested)
        write_csv(output / 'predictions.csv', predictions)
        write_csv(output / 'per_seed.csv', metrics)
        write_csv(output / 'summary.csv', summary)
        write_csv(output / 'paired_deltas_per_seed.csv', deltas)
        write_csv(output / 'paired_deltas_summary.csv', delta_summary)
        report = dict(passed=True, complete_26_outer_folds=args.all, source_seeds=seeds, modes=modes,
                      fold_results=reports, fits=26 * len(reports), prediction_rows=len(predictions),
                      deterministic_TD24_repetitions=1, source_model_training=False, GPU_used=False,
                      archived_prediction_atol=1e-7,
                      max_archived_prediction_error=max(r['max_archived_prediction_error'] for r in reports if r['archive_checked']),
                      macro_main_settings=list(MACROS['macro_main']), macro_extension_settings=list(MACROS['macro_extension']),
                      deltas='trained minus exact initial; positive PCC/Spearman and negative MSE/MAE favor supervision',
                      uncertainty='sample SD across matched source seeds; no independent-fold significance',
                      elapsed_CPU_wall_seconds=time.monotonic() - started)
        original.write(output / 'complete.json', report)
        print(json.dumps({k: v for k, v in report.items() if k != 'fold_results'}, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        if output.exists() and not (output / 'complete.json').exists() and not (output / 'failure.json').exists():
            original.write(output / 'failure.json', dict(passed=False, error_type=type(error).__name__, error=str(error),
                           elapsed_CPU_wall_seconds=time.monotonic() - started))
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
