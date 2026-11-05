#!/usr/bin/env python3
"""Refit the fixed Oligo-random R55 readout on cached, frozen PairAC features.

Only a single Huesken outer fold and source seed42 are run by default.  Use
--all explicitly for all26 outer folds.  `source` means source192 plus TD24,
not a zero-shot score or a source-only reader.  The source encoder is never
loaded or updated.  Original scientific solvers are imported without edits.
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
import sys
import time

import numpy as np

SETTINGS = ('Huesken', 'Mixset', 'Takayuki', 'Inter_Hu_to_Mix', 'wang2009', 'alas1_patent')
COUNTS = dict(zip(SETTINGS, (2196, 469, 664, 469, 131, 228)))
INPUT_SHA = 'a928975aa974a20bf3303ee46257336390a1e1a90b643a1d92960539c38c4980'
PLAN_SHA = 'd4e2ca306f7074ff643d5ec60dc0f6c72fe692e6ca60ec0211e701c7c0c4de5b'
FEATURE_SHA = {42: '652418cadaa9c7b7e5cc49f6d6f31f2a3b0a626662ae1245ce72c2fc87ce38dd',
               43: 'f1186b843f66887ac5a5504187a471e99f9ca05cfe1c1d9192630000279eb8b7',
               44: '5c799e22178599e5249b1b21ef730b5e42d7aacbb94334a70875147d33f78767'}
ARCHIVE_SHA = '1f97700044941f67ebdd7fbd4f6d9caaf8c8808c2237da06d6c78af3196360d4'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(plain(value), handle, indent=2, allow_nan=False)
        handle.write('\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def load_math(root):
    if not __debug__:
        raise ValueError('Run without Python -O: original numerical audits use assertions')
    base = root / 'workflows/sirna_oligo_random/frozen_readout/math'
    inc = base / 'sirna_increment_20261014'
    sys.path[:0] = [str(inc / 'round4'), str(base / 'expert_first_20261014')]
    result = {
        'head': module('_portable_oligo_head55', inc / 'round55/head.py'),
        'scale': module('_portable_oligo_scale4', inc / 'round4/core.py'),
        'groups': module('_portable_oligo_groups49', inc / 'round49/group_nuisance.py'),
        'group_audit': module('_portable_oligo_groups_verify49', inc / 'round49/verify.py'),
        'audit': module('_portable_oligo_audit55', inc / 'round55/audit_math.py'),
    }
    if list(result['head'].ALPHAS) != [.1, 1., 10., 100., 1000.]:
        raise ValueError('Original five-alpha contract changed')
    return result


def validate_plan(plan, lookup, counts=COUNTS):
    seen = set()
    coverage = {setting: [] for setting in SETTINGS}
    for item in plan['folds']:
        key = item['setting'], item['fold']
        if key in seen or key[0] not in SETTINGS or type(key[1]) is not int:
            raise ValueError('Duplicate or invalid outer fold')
        seen.add(key)
        fit, test = item['fit_ids'], item['test_ids']
        if not fit or not test or len(set(fit)) != len(fit) or len(set(test)) != len(test):
            raise ValueError('Empty or duplicate sample identities')
        if set(fit) & set(test) or not set(fit + test) <= set(lookup):
            raise ValueError('Invalid outer train/test identities')
        coverage[key[0]].extend(test)
        n = len(fit)
        valid_coverage = np.zeros(n, dtype=int)
        if len(item['inner']) != 5:
            raise ValueError('Five training-only inner folds required')
        for split in item['inner']:
            train, valid = split['train'], split['valid']
            for indices in (train, valid):
                if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or i < 0 or i >= n for i in indices):
                    raise ValueError('Invalid inner indices')
            if set(train) & set(valid) or set(train + valid) != set(range(n)):
                raise ValueError('Inner partition must split only outer-fit rows')
            valid_coverage[valid] += 1
        if not np.all(valid_coverage == 1):
            raise ValueError('Inner validation must cover fit rows exactly once')
    expected = {(s, i) for s in SETTINGS for i in range(1 if s == 'Inter_Hu_to_Mix' else 5)}
    if seen != expected:
        raise ValueError('Fixed26 outer-fold inventory differs')
    for setting, ids in coverage.items():
        cohort = 'Mixset' if setting == 'Inter_Hu_to_Mix' else setting
        expected_ids = {sid for sid, row in lookup.items() if row['cohort'] == cohort}
        if len(ids) != counts[setting] or len(set(ids)) != len(ids) or set(ids) != expected_ids:
            raise ValueError('Outer tests do not cover the setting exactly once')


def load_data(base):
    for filename, expected in (('input_rows.json', INPUT_SHA), ('native_split_plan.json', PLAN_SHA)):
        if sha(base / filename) != expected:
            raise ValueError('Fixed Oligo input/split SHA differs: ' + filename)
    rows = read(base / 'input_rows.json')
    lookup = {row['sample_id']: row for row in rows}
    if len(rows) != 3688 or len(lookup) != len(rows):
        raise ValueError('Expected3688 unique aligned input rows')
    td = np.asarray([[float(v) for v in row['td'].split(',')] for row in rows], dtype=np.float64)
    y = np.asarray([float(row['label']) for row in rows], dtype=np.float64)
    if td.shape != (3688, 24) or not np.isfinite(td).all() or not np.isfinite(y).all():
        raise ValueError('Expected finite aligned TD24 and labels')
    for row in rows:
        if len(row['guide']) != 19 or len(row['target57']) != 57 or set(row['guide'] + row['target57']) - set('ACGU'):
            raise ValueError('Expected canonical guide19 and target57')
        if row['target57'][19:38] != row['guide'].translate(str.maketrans('ACGU', 'UGCA'))[::-1]:
            raise ValueError('Target designed site differs from guide complement')
        if row['y'] not in ('0', '1'):
            raise ValueError('Invalid binary label')
    plan = read(base / 'native_split_plan.json')
    validate_plan(plan, lookup)
    return {'rows': rows, 'td': td, 'y': y, 'lookup': {row['sample_id']: i for i, row in enumerate(rows)}, 'plan': plan}


def input_matrix(base, dataset, seed, mode):
    if mode == 'TD24':
        return np.column_stack((dataset['td'], np.zeros((len(dataset['rows']), 192)))), None
    if mode != 'source' or seed not in FEATURE_SHA:
        raise ValueError('Unknown frozen source seed/mode')
    path = base / 'features' / f'full_seed{seed}.npy'
    if sha(path) != FEATURE_SHA[seed]:
        raise ValueError('Frozen source feature SHA differs')
    values = np.load(path, allow_pickle=False)
    if values.shape != (3688, 192) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError('Expected original finite float32 source3688x192')
    return np.column_stack((dataset['td'], values.astype(np.float64))), FEATURE_SHA[seed]


def archived_predictions(base):
    path = base / 'predictions/cpu_reference.csv'
    if sha(path) != ARCHIVE_SHA:
        raise ValueError('Archived main-reference predictions SHA differs')
    saved = {}
    for row in table(path):
        key = row['model'], row['setting'], int(row['fold']), row['sample_id']
        if key in saved:
            raise ValueError('Duplicate archived prediction')
        saved[key] = row
    if len(saved) != 16628:
        raise ValueError('Expected16628 main-reference predictions')
    return saved


def fit_scaler(x, scale):
    tm, ts = scale.scaler(x[:, :24])
    sm, ss = scale.scaler(x[:, 24:], True)
    return dict(td_mean=tm, td_scale=ts, src_mean=sm, src_scale=ss)


def training_groups(targets, math):
    labels, metadata = math['groups'].groups(targets)
    independent, independent_metadata = math['group_audit'].independent_groups(targets)
    np.testing.assert_array_equal(labels, independent)
    if metadata != independent_metadata:
        raise ValueError('Independent target25 grouping differs')
    return labels, metadata


def refit_fold(dataset, x, math, seed, mode, item, archive, output):
    head, scale, audit = math['head'], math['scale'], math['audit']
    source = mode == 'source'
    model = f'PairAC{seed}' if source else 'TD24_R55'
    fi = np.asarray([dataset['lookup'][sid] for sid in item['fit_ids']], dtype=int)
    ti = np.asarray([dataset['lookup'][sid] for sid in item['test_ids']], dtype=int)
    y = dataset['y'][fi]
    expected = []
    for sid in item['test_ids']:
        saved = archive[model, item['setting'], item['fold'], sid]
        row = dataset['rows'][dataset['lookup'][sid]]
        if float(saved['label']) != float(row['label']) or saved['binary'] != row['y']:
            raise ValueError('Archived prediction label changed')
        expected.append(float(saved['prediction']))
    output.mkdir(parents=True, exist_ok=False)
    oof = np.full((len(fi), 5), np.nan)
    transforms, group_records, certificates, errors = [], [], [], []
    coef = np.empty((5, 5, 216 if source else 24))
    intercept = np.empty((5, 5))
    for stage, split in enumerate(item['inner']):
        train, valid = np.asarray(split['train']), np.asarray(split['valid'])
        t = fit_scaler(x[fi[train]], scale)
        transforms.append(t)
        z, zv = head.transform(x[fi[train]], t), head.transform(x[fi[valid]], t)
        targets = [dataset['rows'][fi[i]]['target57'] for i in train]
        groups, metadata = training_groups(targets, math)
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
    t = fit_scaler(x[fi], scale)
    z, zt = head.transform(x[fi], t), head.transform(x[ti], t)
    groups, metadata = training_groups([dataset['rows'][i]['target57'] for i in fi], math)
    group_records.append(dict(labels=groups, metadata=metadata))
    b, center, cert = head.fit(z, y, groups, alpha, source)
    if not cert['certified']:
        raise ValueError('Final R55 numerical certificate failed')
    check = audit.check_source if source else audit.audit_TD
    errors.append(check(z if source else z[:, :24], y, groups, b, center, cert, alpha))
    prediction = (zt if source else zt[:, :24]) @ b + center
    full_b = b if source else np.r_[b, np.zeros(192)]
    raw_b, raw_center = head.deployment(t, full_b, center)
    np.testing.assert_allclose(prediction, x[ti] @ raw_b + raw_center, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(prediction, expected, atol=1e-7, rtol=0, err_msg='Refit differs from archived main result')
    np.savez_compressed(output / 'arrays.npz', inner_coef=coef, inner_intercept=intercept, inner_oof=oof,
                        final_coef=b, final_intercept=center, deployment_coef=raw_b, deployment_intercept=raw_center, prediction=prediction)
    write(output / 'certificates.json', dict(inner=certificates, final=cert))
    write(output / 'transforms.json', dict(inner=transforms, final=t))
    write(output / 'training_groups.json', group_records)
    with (output / 'predictions.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['model', 'setting', 'fold', 'sample_id', 'label', 'binary', 'prediction'])
        writer.writeheader()
        for sid, value in zip(item['test_ids'], prediction):
            row = dataset['rows'][dataset['lookup'][sid]]
            writer.writerow(dict(model=model, setting=item['setting'], fold=item['fold'], sample_id=sid,
                                 label=row['label'], binary=row['y'], prediction=float(value)))
    report = dict(passed=True, model=model, mode=mode, seed=seed if source else None, setting=item['setting'], fold=item['fold'],
                  fits=26, inner_fits=25, final_fits=1, alpha=alpha, candidate=selected, inner_PCC=pc, inner_MSE=mse,
                  test_rows=len(ti), max_archived_prediction_error=float(np.max(np.abs(prediction - expected))), archived_prediction_atol=1e-7,
                  max_independent_normal_error=float(np.max(np.asarray(errors)[:, 0])),
                  inner_selection_uses_only_outer_fit=True, outer_test_used_for_model_selection=False, source_model_training=False, GPU_used=False)
    write(output / 'complete.json', report)
    return plain(report)


def selected_folds(plan, settings, folds, all_folds=False):
    wanted = set(SETTINGS if all_folds else settings)
    subset = [item for item in plan['folds'] if item['setting'] in wanted and (all_folds or item['fold'] in folds)]
    expected = {(s, f) for s in wanted for f in folds} if not all_folds else {(s, i) for s in wanted for i in range(1 if s == 'Inter_Hu_to_Mix' else 5)}
    if not subset or {(item['setting'], item['fold']) for item in subset} != expected:
        raise ValueError('Requested setting/fold absent from fixed plan')
    return subset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path, help='Optional data/sirna_oligo_random directory from a separate data supplement')
    parser.add_argument('--seed', nargs='+', type=int, choices=(42, 43, 44), default=[42])
    parser.add_argument('--mode', choices=('TD24', 'source'), default='source', help='source: frozen source192 plus TD24; TD24: deterministic control')
    parser.add_argument('--setting', nargs='+', choices=SETTINGS, default=['Huesken'])
    parser.add_argument('--fold', nargs='+', type=int, default=[0])
    parser.add_argument('--all', action='store_true', help='Explicitly refit all26 outer folds for requested source seed(s)')
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing directories are refused')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    base = args.data_root.resolve() if args.data_root else root / 'data/sirna_oligo_random'
    output = args.output.resolve()
    if output.exists() or output == root or root.is_relative_to(output) or output.is_relative_to(base) or base.is_relative_to(output) or output.is_relative_to(root / 'data') or output.is_relative_to(root / 'reference') or output.is_relative_to(root / 'workflows'):
        parser.error('--output must be new and must not overwrite data, reference, workflows, or a checkout ancestor')
    if args.mode == 'TD24' and args.seed != [42]:
        parser.error('TD24 is deterministic; do not fabricate source-seed repeats')
    started = time.monotonic()
    try:
        data = load_data(base)
        saved = archived_predictions(base)
        math = load_math(root)
        requested = selected_folds(data['plan'], args.setting, args.fold, args.all)
        output.mkdir(parents=True, exist_ok=False)
        write(output / 'run_inputs.json', dict(protocol='fixed_Oligo_random_26_outer_folds_R55', input_sha256=INPUT_SHA,
              split_sha256=PLAN_SHA, archived_predictions_sha256=ARCHIVE_SHA, source_features_sha256={str(seed): FEATURE_SHA[seed] for seed in args.seed} if args.mode == 'source' else {},
              CPU_only=True, source_model_training=False, requested_outer_folds=len(requested)))
        reports = []
        for seed in dict.fromkeys(args.seed):
            x, _ = input_matrix(base, data, seed, args.mode)
            for item in requested:
                model = f'PairAC{seed}' if args.mode == 'source' else 'TD24_R55'
                folder = output / model / item['setting'] / f"fold{item['fold']}"
                reports.append(refit_fold(data, x, math, seed, args.mode, item, saved, folder))
        report = dict(passed=True, mode=args.mode, fold_results=reports, fits=26 * len(reports), elapsed_CPU_wall_seconds=time.monotonic() - started,
                      max_archived_prediction_error=max(r['max_archived_prediction_error'] for r in reports), source_model_training=False, GPU_used=False)
        write(output / 'complete.json', report)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        if output.exists() and not (output / 'complete.json').exists() and not (output / 'failure.json').exists():
            write(output / 'failure.json', dict(passed=False, error_type=type(error).__name__, elapsed_CPU_wall_seconds=time.monotonic() - started))
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
