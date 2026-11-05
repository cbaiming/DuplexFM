"""One model/outer-fold HPO group, seed42 search then fixed three-seed fits.

Only current outer-fit labels may enter HPO. Operational rung/phase caps are not
fit identities. No test prediction is requested until winner.json is immutable.
Persistence is delegated to a byte-pinned copy of the reviewed original helper;
its old fixed-job runner is never called. This module imports no torch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import time

import numpy as np

MODELS = ('OligoFormer-corrected', 'BERT-siRNA', 'RNAiSpline')
SEEDS = (42, 43, 44)
GRID_IDS = ('base', 'lr_half', 'lr_double', 'wd_zero', 'wd_ten',
            'batch_half', 'batch_double', 'lr_double_batch_double')
RUNG1 = {'OligoFormer-corrected': 30, 'BERT-siRNA': 8, 'RNAiSpline': 13}
_STOP = False


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')


def value_sha(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def confined(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if path != root and root not in path.parents:
        raise ValueError('Path escapes HPO group: ' + str(name))
    return path


def descriptor(base, value):
    path = Path(value['path'])
    if not path.is_absolute():
        path = Path(base) / path
    path = path.resolve()
    if path.is_symlink() or sha(path) != value['sha256']:
        raise ValueError('Pinned HPO file differs: ' + str(path))
    return path


def module(path, name):
    spec = importlib.util.spec_from_file_location(name + '_' + sha(path)[:16], path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def load_persistence(config_path, expected_config_sha):
    if sha(config_path) != expected_config_sha:
        raise ValueError('HPO worker config SHA differs')
    cfg = read(config_path)
    path = descriptor(Path(config_path).resolve().parent, cfg['persistence'])
    return module(path, 'hpo_locked_persistence')


def load_group(config_path, expected_config_sha, group_id):
    config_path = Path(config_path).resolve()
    if sha(config_path) != expected_config_sha:
        raise ValueError('HPO worker config SHA differs')
    cfg, base = read(config_path), config_path.parent
    if cfg.get('schema') != 1 or cfg.get('task') != 'oligo_random_six_hpo4_v1':
        raise ValueError('Unregistered HPO configuration schema')
    keys = ('input_rows', 'split_plan', 'input_manifest', 'backend', 'execution_lock',
            'persistence', 'oligo_features')
    paths = {key: descriptor(base, cfg[key]) for key in keys}
    manifest = read(paths['input_manifest'])
    for key, name in (('input_rows', 'input_rows.json'), ('split_plan', 'native_split_plan.json')):
        if manifest['generated'][name]['sha256'] != cfg[key]['sha256']:
            raise ValueError('HPO inputs differ from the registered input manifest')
    groups, plan = cfg['groups'], read(paths['split_plan'])
    if len(groups) != 78 or len({g['id'] for g in groups}) != 78:
        raise ValueError('Exactly 78 unique model/outer-fold groups are required')
    expected = {(m, x['setting'], x['fold']) for m in MODELS for x in plan['folds']}
    actual = {(g['model'], g['setting'], g['fold']) for g in groups}
    if len(plan['folds']) != 26 or actual != expected:
        raise ValueError('HPO groups must cover each registered model and all 26 outer folds')
    matches = [g for g in groups if g['id'] == group_id]
    if len(matches) != 1 or not re.fullmatch(r'[A-Za-z0-9_.-]{1,160}', group_id) or group_id in ('.', '..'):
        raise ValueError('Unknown or unsafe HPO group ID')
    group = {key: matches[0][key] for key in ('id', 'model', 'setting', 'fold')}
    item = next(x for x in plan['folds'] if x['setting'] == group['setting'] and x['fold'] == group['fold'])
    fit, test = item['fit_ids'], item['test_ids']
    if len(set(fit)) != len(fit) or len(set(test)) != len(test) or set(fit) & set(test):
        raise ValueError('Outer fit/test identities repeat or overlap')
    coverage, universe = np.zeros(len(fit), dtype=np.int8), set(range(len(fit)))
    if len(item['inner']) != 5:
        raise ValueError('Exactly five common inner folds are required')
    for inner in item['inner']:
        train, valid = inner['train'], inner['valid']
        if len(set(train)) != len(train) or len(set(valid)) != len(valid) or set(train) & set(valid) or set(train) | set(valid) != universe:
            raise ValueError('Inner folds must be disjoint full partitions of current outer fit')
        coverage[valid] += 1
    if not np.all(coverage == 1):
        raise ValueError('Every outer-fit row must be inner validation exactly once')
    if cfg.get('source_seeds', list(SEEDS)) != list(SEEDS) or cfg['rung1_supervised_epochs'] != RUNG1:
        raise ValueError('Registered seeds or rung1 caps differ')
    identity = dict(schema=1, group=group, config_sha256=expected_config_sha,
                    worker_source_sha256=sha(__file__), backend_sha256=cfg['backend']['sha256'],
                    input_rows_sha256=cfg['input_rows']['sha256'], split_plan_sha256=cfg['split_plan']['sha256'],
                    input_manifest_sha256=cfg['input_manifest']['sha256'],
                    execution_lock_sha256=cfg['execution_lock']['sha256'],
                    persistence=dict(path=str(paths['persistence']), sha256=cfg['persistence']['sha256']),
                    fit_ids_sha256=value_sha(fit), test_ids_sha256=value_sha(test),
                    inner_sha256=value_sha(item['inner']),
                    candidates_sha256=value_sha(cfg['candidates'][group['model']]), source_seeds=list(SEEDS),
                    protocol='seed42_8to3to1_inner_only_then_fixed_winner_three_seed_v1',
                    outer_test_used_for_selection=False)
    return cfg, paths, group, item, identity


def rank_candidates(records, keep):
    if len({r['candidate_id'] for r in records}) != len(records):
        raise ValueError('Candidate scores repeat')
    if any(r['candidate_id'] not in GRID_IDS or not math.isfinite(float(r['mse'])) or r['mse'] < 0 for r in records):
        raise ValueError('Candidate ranking needs finite nonnegative validation MSE')
    # Fixed original candidate order breaks exact ties; no downstream data enters.
    return [r['candidate_id'] for r in sorted(records, key=lambda r: (r['mse'], GRID_IDS.index(r['candidate_id'])))[:keep]]


def canonical_candidates(values):
    result = []
    for value in values:
        hp = value.get('hyperparameters', {k: value[k] for k in ('lr', 'wd', 'batch')} if 'lr' in value else None)
        if hp is None or set(hp) != {'lr', 'wd', 'batch'}:
            raise ValueError('Each candidate needs exactly lr/wd/batch')
        result.append(dict(id=value['id'], hyperparameters=dict(hp)))
    if tuple(v['id'] for v in result) != GRID_IDS:
        raise ValueError('The eight registered candidates must preserve fixed order')
    return result


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def check_trace(root, value):
    for name, digest in value.get('files', {}).items():
        if sha(confined(root, name)) != digest:
            raise ValueError('Immutable search trace file differs: ' + name)
    if value.get('outer_test_used_for_selection') is not False:
        raise ValueError('Search trace lacks explicit test exclusion')
    return value


def verify_validation_score(root, record):
    check_trace(root, record)
    paths = [confined(root, n) for n in record['files'] if n.endswith('.csv')]
    receipts = [confined(root, n) for n in record['files'] if n.endswith('.json')]
    if len(paths) != 1:
        raise ValueError('Each inner score requires one immutable prediction CSV')
    if len(receipts)!=1:
        raise ValueError('Each inner score requires one immutable checkpoint receipt')
    receipt=read(receipts[0])
    fit_root=Path(root)/'search/fits'/record['candidate_id']/('inner'+str(record['inner']))
    for name,digest in receipt['files'].items():
        if sha(confined(fit_root,name))!=digest:
            raise ValueError('Immutable inner selected-checkpoint artifact differs')
    if receipt['selected_epochs']!=record['selected_epochs']:
        raise ValueError('Immutable score selected epoch differs from its checkpoint receipt')
    with paths[0].open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != record['validation_n'] or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Inner score row count or identities repeat')
    mse = float(np.mean([(float(r['prediction']) - float(r['label']))**2 for r in rows]))
    if not math.isfinite(mse) or abs(mse - record['mse']) > 1e-12:
        raise ValueError('Inner score does not reproduce from its immutable predictions')
    return mse


def verify_rungs(root, rung1, rung2=None):
    check_trace(root, rung1)
    if tuple(r['candidate_id'] for r in rung1['scores']) != GRID_IDS or rung1['seed'] != 42:
        raise ValueError('Rung1 must score exactly the fixed eight seed42 candidates')
    for record in rung1['scores']:
        verify_validation_score(root, record)
    if rung1['promoted'] != rank_candidates(rung1['scores'], 3):
        raise ValueError('Rung1 promotion differs from fixed MSE ranking')
    if rung2 is not None:
        check_trace(root, rung2)
        if [r['candidate_id'] for r in rung2['scores']] != rung1['promoted'] or rung2['seed'] != 42:
            raise ValueError('Rung2 must score exactly the promoted three seed42 candidates')
        for record in rung2['scores']:
            if [p['inner'] for p in record['parts']] != [0, 1]:
                raise ValueError('Rung2 requires the first two inner folds')
            for part in record['parts']:
                verify_validation_score(root, part)
            count = sum(p['validation_n'] for p in record['parts'])
            mean = sum(p['mse'] * p['validation_n'] for p in record['parts']) / count
            if count != record['validation_n'] or abs(mean - record['mse']) > 1e-12:
                raise ValueError('Rung2 sample-weighted MSE differs')
        if rung2['winner'] != rank_candidates(rung2['scores'], 1)[0]:
            raise ValueError('Rung2 winner differs from the fixed MSE ranking')


def prune_completed_resume(base, store, fit_root, receipt, reason):
    """Only redundant last state after complete selected payload is durable."""
    fit_root = Path(fit_root).resolve()
    if store.job_root not in fit_root.parents:
        raise ValueError('Completed-state pruning escapes the current HPO group')
    last, pointer = fit_root / 'last.pt', fit_root / 'last.sha256.json'
    plan_path = fit_root / 'completed_resume_pruning_plan.json'
    if not last.exists() and not pointer.exists() and not plan_path.exists():
        return
    if any(name in receipt.get('files', {}) for name in ('last.pt', 'last.sha256.json')):
        raise ValueError('Complete receipt still requires the resume checkpoint')
    if plan_path.exists():
        plan = read(plan_path)
        if plan['complete_sha256'] != sha(fit_root / 'complete.json'):
            raise ValueError('Interrupted pruning receipt differs')
        manifest = base.confined(store.job_backup, plan['durable_backup']['snapshot'])
        if sha(manifest) != plan['durable_backup']['sha256']:
            raise ValueError('Interrupted pruning backup manifest differs')
    else:
        if not last.exists() or not pointer.exists():
            raise ValueError('Complete fit has an incomplete last/pointer pair; preserve for review')
        meta, digest = read(pointer), sha(last)
        if meta.get('identity_sha256') != sha(fit_root / 'identity.json') or meta.get('sha256') != digest:
            raise ValueError('Completed resume payload/pointer differs')
        expected = receipt.get('resume_checkpoint_sha256_at_endpoint')
        if expected is not None and digest != expected:
            raise ValueError('Completed resume payload differs from its endpoint receipt')
        durable = store.snapshot(reason + '_before_completed_prune', force=True)
        plan = dict(complete_sha256=sha(fit_root / 'complete.json'), durable_backup=durable,
                    removed={p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in (last, pointer)},
                    active_or_prefix_state_removed=False)
        base.immutable_json(plan_path, plan)
    for name, value in plan['removed'].items():
        if name not in ('last.pt', 'last.sha256.json'):
            raise ValueError('Pruning plan contains an unregistered target')
        path = fit_root / name
        if path.exists():
            if sha(path) != value['sha256']:
                raise ValueError('Pruning target changed; preserved')
            path.unlink()
    base.immutable_json(fit_root / 'completed_resume_pruning.json', dict(plan, completed=True))


def prune_eliminated_candidate(base, store, group_root, rung1, candidate_id, durable=None):
    """Local-only removal after immutable promotion; older AFS objects survive."""
    group_root = Path(group_root).resolve()
    if candidate_id in rung1['promoted'] or candidate_id not in GRID_IDS:
        raise ValueError('An active/promoted/unregistered candidate cannot be pruned')
    if read(group_root / 'search/rung1.json') != rung1:
        raise ValueError('Elimination must be bound to the immutable rung1 decision')
    fit_root = group_root / 'search/fits' / candidate_id / 'inner0'
    if store.job_root != group_root or (fit_root / 'complete.json').exists():
        raise ValueError('Eliminated prefix is not the current group or was incorrectly fully fitted')
    prefix = fit_root / 'rungs' / ('epoch_' + str(rung1['cap'])) / 'receipt.json'
    receipt = read(prefix)
    saved = group_root / 'search/receipts' / f'rung1_{candidate_id}_inner0.json'
    if (receipt.get('status') != 'rung_complete' or receipt.get('full_fit_complete') is not False or
            receipt.get('rung_epoch_cap') != rung1['cap'] or read(saved) != receipt):
        raise ValueError('Eliminated prefix does not have its original immutable score receipt')
    for name,digest in receipt['files'].items():
        if sha(confined(fit_root,name))!=digest:
            raise ValueError('Eliminated prefix selected weights/history differ')
    last, pointer = fit_root / 'last.pt', fit_root / 'last.sha256.json'
    plan_path = fit_root / 'eliminated_prefix_pruning_plan.json'
    if plan_path.exists():
        plan = read(plan_path)
        if (plan['rung1_sha256'] != sha(group_root / 'search/rung1.json') or
                plan['candidate_id'] != candidate_id or plan['prefix_receipt_sha256'] != sha(prefix) or
                plan['promoted_candidates'] != rung1['promoted']):
            raise ValueError('Interrupted elimination pruning plan differs')
        manifest = base.confined(store.job_backup, plan['durable_backup']['snapshot'])
        if sha(manifest) != plan['durable_backup']['sha256']:
            raise ValueError('Eliminated prefix archive manifest differs')
    else:
        if not last.is_file() or not pointer.is_file():
            raise ValueError('Unarchived eliminated prefix lacks its last/pointer pair')
        meta, digest = read(pointer), sha(last)
        if (meta.get('identity_sha256') != sha(fit_root / 'identity.json') or
                meta.get('sha256') != digest or receipt.get('resume_checkpoint_sha256_at_endpoint') != digest):
            raise ValueError('Eliminated prefix resume state differs from its registered endpoint')
        if durable is None:
            durable = store.snapshot('rung1_eliminated_prefix_before_prune', force=True)
        manifest = base.confined(store.job_backup, durable['snapshot'])
        if sha(manifest) != durable['sha256']:
            raise ValueError('Eliminated prefix backup did not pass manifest SHA read-back')
        plan = dict(schema=1, candidate_id=candidate_id, promoted_candidates=rung1['promoted'],
            rung1_sha256=sha(group_root / 'search/rung1.json'), prefix_receipt_sha256=sha(prefix),
            durable_backup=durable,
            removed={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in (last,pointer)},
            local_only=True, older_AFS_objects_removed=False, prefix_marked_full_complete=False,
            retained='immutable_prefix_best_history_receipt_score_and_source_initial_reference')
        base.immutable_json(plan_path,plan)
    for name,value in plan['removed'].items():
        if name not in ('last.pt','last.sha256.json'):
            raise ValueError('Elimination pruning plan contains an unexpected target')
        path=fit_root/name
        if path.exists():
            if sha(path)!=value['sha256']:
                raise ValueError('Eliminated prefix pruning target changed; preserved')
            path.unlink()
    base.immutable_json(fit_root/'eliminated_prefix_pruning.json',dict(plan,completed=True))


def prune_eliminated_prefixes(base, store, group_root, rung1):
    verify_rungs(group_root,rung1)
    eliminated=[cid for cid in GRID_IDS if cid not in rung1['promoted']]
    if len(eliminated)!=5:
        raise ValueError('Exactly five fixed candidates must be eliminated before pruning')
    # One independent full-group snapshot covers all newly eliminated states.
    # Subsequent plans reference that read-back snapshot; content objects are not
    # deleted from AFS. Already registered interrupted cleanup reuses its archive.
    new=[cid for cid in eliminated if not (Path(group_root)/'search/fits'/cid/'inner0/eliminated_prefix_pruning_plan.json').exists()]
    durable=store.snapshot('rung1_five_eliminated_prefix_states_before_local_prune',force=True) if new else None
    for cid in eliminated:
        prune_eliminated_candidate(base,store,group_root,rung1,cid,durable)


def verify_group_complete(group_root, expected_identity, initial_root):
    root = Path(group_root)
    receipt = read(root / 'complete.json')
    if receipt.get('status') != 'complete' or receipt.get('identity_sha256') != value_sha(expected_identity):
        raise ValueError('Completed HPO group identity differs')
    for name, digest in receipt['files'].items():
        if sha(confined(root, name)) != digest:
            raise ValueError('Completed HPO group file differs: ' + name)
    persistence = descriptor(root, expected_identity['persistence'])
    base = module(persistence, 'hpo_verify_persistence')
    for name, execution in receipt['fit_execution_identities'].items():
        base.verify_fit(confined(root, name), execution, initial_root)
    if len(receipt['fit_execution_identities']) != 22:
        raise ValueError('A complete group requires six promoted search fits and sixteen remaining/final seed fits')
    rung1, rung2, winner = [read(root / 'search' / stage) for stage in ('rung1.json', 'rung2.json', 'winner.json')]
    verify_rungs(root, rung1, rung2)
    check_trace(root, winner)
    if (winner['candidate']['id'] != rung2['winner'] or winner['rung1_sha256'] != sha(root / 'search/rung1.json') or
            winner['rung2_sha256'] != sha(root / 'search/rung2.json')):
        raise ValueError('Frozen winner does not bind the exact inner-only search traces')
    if receipt['source_seeds'] != list(SEEDS) or set(receipt['per_seed_predictions']) != {'42', '43', '44'}:
        raise ValueError('Completed group does not contain all three fixed seeds')
    if receipt['predictions'] != 3 * receipt['test_n'] or any(n != receipt['test_n'] for n in receipt['per_seed_predictions'].values()):
        raise ValueError('Completed group predictions do not have three-seed exact-once coverage')
    test_ids = receipt['test_sample_ids']
    if value_sha(test_ids) != expected_identity['test_ids_sha256'] or len(test_ids) != receipt['test_n']:
        raise ValueError('Completed group test identities differ')
    with (root / 'predictions.csv').open(encoding='utf-8', newline='') as handle:
        predictions = list(csv.DictReader(handle))
    for seed in SEEDS:
        selected = [row for row in predictions if int(row['seed']) == seed]
        if [row['sample_id'] for row in selected] != test_ids or not all(math.isfinite(float(row['prediction'])) for row in selected):
            raise ValueError('Completed group prediction rows do not have exact finite per-seed coverage')
    if len(predictions) != receipt['predictions']:
        raise ValueError('Completed group prediction count differs')
    return receipt


class ControlledStop(Exception):
    pass


def run_group(config_path, expected_config_sha, group_id, root, afs, device='cuda:0',
              worker_id='worker', restore_latest=False, backend_override=None, persistence_override=None,
              stop_override=None):
    cfg, paths, group, item, identity = load_group(config_path, expected_config_sha, group_id)
    root, afs = Path(root).resolve(), Path(afs).resolve()
    out, initial = root / 'run/groups' / group_id, root / 'run/initial_objects'
    base = persistence_override or module(paths['persistence'], 'hpo_runtime_persistence')
    store = base.BackupStore(afs / 'backups', out, value_sha(identity), initial,
                            cfg.get('backup_interval_seconds', 3600))
    # Restore verifies all source objects and exact paths before its first write.
    if restore_latest:
        if out.exists():
            raise ValueError('Explicit restore requires absent local group; divergent state is preserved')
        store.restore_latest()
    stop = stop_override or (lambda: _STOP or (root / 'run/STOP').exists() or (out / 'STOP').exists())
    with base.JobLease(out, value_sha(identity), worker_id):
        base.immutable_json(out / 'group_identity.json', identity)
        if (out / 'complete.json').exists():
            result = verify_group_complete(out, identity, initial)
            store.snapshot('complete_reuse', force=True)
            return dict(status='complete', group_id=group_id, reused=True, predictions=result['predictions'])
        backend = backend_override or module(paths['backend'], 'hpo_locked_backend')
        candidates = canonical_candidates(cfg['candidates'][group['model']])
        if candidates != backend.candidate_grid(group['model']):
            raise ValueError('Configured grid differs from the locked native-policy grid')
        by_id = {v['id']: v for v in candidates}
        rows = read(paths['input_rows'])
        ids = [r['sample_id'] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError('Input sample identifiers repeat')
        index = {value: i for i, value in enumerate(ids)}
        outer_fit = np.asarray([index[v] for v in item['fit_ids']], dtype=np.int64)
        outer_test = np.asarray([index[v] for v in item['test_ids']], dtype=np.int64)
        features = np.load(paths['oligo_features'], mmap_mode='r', allow_pickle=False)
        if features.shape != (len(rows), 280) or not np.isfinite(features).all():
            raise ValueError('Expected finite n by 280 Oligo feature input')
        arrays = backend.prepare_arrays(rows, oligo_features=features)
        assets = dict(cfg['assets'], initial_objects_dir=str(initial), device=device)
        fit_map, cache_refs = {}, {}
        search = out / 'search'

        def checkpoint():
            if stop():
                raise ControlledStop()

        def partition(j):
            if j == 5:
                return outer_fit, None
            inner = item['inner'][j]
            return outer_fit[np.asarray(inner['train'], dtype=int)], outer_fit[np.asarray(inner['valid'], dtype=int)]

        def fit(candidate_id, seed, j, *, finalize=True, cap=None, fixed_epochs=None):
            checkpoint()
            candidate = by_id[candidate_id]
            train, valid = partition(j)
            if seed == 42 and j in (0, 1):
                relative = f'search/fits/{candidate_id}/inner{j}'
            else:
                relative = f'seeds/{seed}/fits/' + (f'inner{j}' if j < 5 else 'final')
            fit_out = out / relative
            execution = dict(group=identity, candidate_id=candidate_id,
                hyperparameters=candidate['hyperparameters'], seed=seed, stage=f'inner{j}' if j < 5 else 'final',
                train_sample_ids_sha256=value_sha([ids[int(i)] for i in train]),
                validation_sample_ids_sha256=value_sha([] if valid is None else [ids[int(i)] for i in valid]),
                fixed_epochs=fixed_epochs)
            cache = None
            cache_execution = None
            if group['model'] == 'RNAiSpline':
                cache_execution = dict(group=identity, seed=seed, stage=execution['stage'],
                    train_sample_ids_sha256=execution['train_sample_ids_sha256'],
                    SSL_policy='native_fit_only_60_epochs_batch16_lr0.0001')
                cache = backend.ensure_ssl_cache(group['model'], arrays, train, seed,
                    out / 'ssl_cache' / str(seed) / execution['stage'], assets,
                    execution_identity=cache_execution, stop_requested=stop,
                    checkpoint_callback=store.checkpoint_callback, device=device)
                if not cache or cache.get('status') == 'stopped':
                    raise ControlledStop()
                cache_refs[str(seed) + '/' + execution['stage']] = cache
                cache_path = Path(cache['path'])
                prune_completed_resume(base, store, cache_path.parent, read(cache_path), 'SSL_cache_complete')
            receipt = backend.run_fit(group['model'], arrays, train, seed, fit_out, assets,
                validation_indices=valid, fixed_epochs=fixed_epochs, execution_identity=execution,
                stop_requested=stop, checkpoint_callback=store.checkpoint_callback, device=device,
                hyperparameters=candidate['hyperparameters'], candidate_id=candidate_id,
                max_supervised_epochs=cap, finalize=finalize, ssl_cache=cache,
                ssl_execution_identity=cache_execution)
            if receipt.get('status') == 'stopped':
                raise ControlledStop()
            if receipt.get('status') != ('complete' if finalize else 'rung_complete'):
                raise ValueError('Backend did not finish the requested rung/full fit')
            if finalize:
                base.verify_fit(fit_out, execution, initial)
                fit_map[relative] = execution
                prune_completed_resume(base, store, fit_out, receipt, 'full_fit_complete')
            base.write(out / 'progress.json', dict(group_id=group_id, model=group['model'], candidate_id=candidate_id,
                seed=seed, stage=execution['stage'], finalize=finalize, last_fit_status=receipt['status'], time=time.time()))
            store.snapshot('full_fit_complete' if finalize else 'rung_complete', force=True)
            return fit_out, execution, receipt, valid

        def score(candidate_id, rung, j, result):
            fit_out, execution, receipt, valid = result
            pred = np.asarray(backend.predict_validation(group['model'], arrays, valid, fit_out, assets,
                              expected_identity=execution, device=device,
                              rung_epoch_cap=RUNG1[group['model']] if rung == 'rung1' else None), dtype=np.float64)
            truth = np.asarray(arrays['labels'], dtype=np.float64)[valid]
            if pred.shape != truth.shape or not len(truth) or not np.isfinite(pred).all():
                raise ValueError('Validation scoring needs exact finite inner validation predictions')
            mse = float(np.mean((pred - truth)**2))
            recorded = float(receipt['best_validation_mse'])
            if not math.isfinite(recorded) or abs(mse - recorded) > max(1e-6, 1e-5 * abs(recorded)):
                raise ValueError('Independent inner MSE disagrees with the selected checkpoint receipt')
            label = f'{rung}_{candidate_id}_inner{j}'
            score_path = search / 'scores' / (label + '.csv')
            base.atomic_bytes(score_path, csv_bytes([dict(sample_id=ids[int(i)], prediction=float(p), label=float(y))
                                                   for i, p, y in zip(valid, pred, truth)]))
            receipt_path = search / 'receipts' / (label + '.json')
            base.immutable_json(receipt_path, receipt)
            immutable_files = {p.relative_to(out).as_posix(): sha(p) for p in (score_path, receipt_path)}
            backup = store.snapshot(label, force=True)
            return dict(candidate_id=candidate_id, inner=j, validation_n=len(truth), mse=mse,
                        selected_epochs=int(receipt['selected_epochs']), files=immutable_files,
                        backup=backup, outer_test_used_for_selection=False)

        try:
            checkpoint()
            if (search / 'rung1.json').exists():
                rung1 = check_trace(out, read(search / 'rung1.json'))
                verify_rungs(out, rung1)
            else:
                scores = [score(c['id'], 'rung1', 0, fit(c['id'], 42, 0, finalize=False,
                            cap=RUNG1[group['model']])) for c in candidates]
                promoted = rank_candidates(scores, 3)
                rung1 = dict(schema=1, status='rung_complete', seed=42, cap=RUNG1[group['model']],
                    scores=scores, promoted=promoted, tie_rule='fixed_candidate_grid_order',
                    outer_test_used_for_selection=False,
                    files={k: v for record in scores for k, v in record['files'].items()})
                base.immutable_json(search / 'rung1.json', rung1)
                store.snapshot('rung1_selection_complete', force=True)
            prune_eliminated_prefixes(base,store,out,rung1)
            checkpoint()
            if (search / 'rung2.json').exists():
                rung2 = check_trace(out, read(search / 'rung2.json'))
                verify_rungs(out, rung1, rung2)
                # Existing full receipts still need to enter complete-group accounting.
                for cid in rung1['promoted']:
                    for j in (0, 1):
                        fit(cid, 42, j)
            else:
                scores = []
                for cid in rung1['promoted']:
                    parts = [score(cid, 'rung2', j, fit(cid, 42, j)) for j in (0, 1)]
                    count = sum(v['validation_n'] for v in parts)
                    scores.append(dict(candidate_id=cid, mse=sum(v['mse'] * v['validation_n'] for v in parts) / count,
                        validation_n=count, parts=parts, outer_test_used_for_selection=False,
                        files={k: v for v in parts for k, v in v['files'].items()}))
                winner = rank_candidates(scores, 1)[0]
                rung2 = dict(schema=1, status='rung_complete', seed=42, scores=scores, winner=winner,
                    score_rule='sample_weighted_MSE_of_two_selected_inner_checkpoints',
                    outer_test_used_for_selection=False,
                    files={k: v for record in scores for k, v in record['files'].items()})
                base.immutable_json(search / 'rung2.json', rung2)
                store.snapshot('rung2_selection_complete', force=True)
            winner = rung2['winner']
            winner_value = dict(schema=1, status='winner_frozen', candidate=by_id[winner], search_seed=42,
                rung1_sha256=sha(search / 'rung1.json'), rung2_sha256=sha(search / 'rung2.json'),
                outer_test_used_for_selection=False, no_further_HPO_for_seeds=[43, 44], files={})
            base.immutable_json(search / 'winner.json', winner_value)
            store.snapshot('winner_frozen_before_any_outer_test', force=True)
            all_predictions, seed_counts, selected = [], {}, {}
            for seed in SEEDS:
                epochs = []
                for j in range(5):
                    _, _, receipt, _ = fit(winner, seed, j)
                    epochs.append(int(receipt['selected_epochs']))
                median = int(statistics.median(epochs))
                final_path, execution, final_receipt, _ = fit(winner, seed, 5, fixed_epochs=median)
                if final_receipt.get('fixed_epochs') != median:
                    raise ValueError('Final fit did not use the median of five selected inner epochs')
                selected[str(seed)] = dict(inner_epochs=epochs, final_epochs=median)
                prediction_path = out / 'seeds' / str(seed) / 'predictions.csv'
                evidence_path = prediction_path.with_suffix('.receipt.json')
                if evidence_path.exists():
                    evidence = read(evidence_path)
                    if evidence['winner_sha256'] != sha(search / 'winner.json') or evidence['predictions_sha256'] != sha(prediction_path):
                        raise ValueError('Previously published fixed-winner seed predictions differ')
                    with prediction_path.open(encoding='utf-8', newline='') as handle:
                        prediction_rows = list(csv.DictReader(handle))
                else:
                    checkpoint()
                    if not (search / 'winner.json').exists():
                        raise ValueError('Winner must be immutable before any outer-test prediction')
                    prediction = np.asarray(backend.predict_fit(group['model'], arrays, outer_test, final_path, assets,
                        expected_identity=execution, device=device), dtype=np.float64)
                    if prediction.shape != (len(outer_test),) or not np.isfinite(prediction).all():
                        raise ValueError('Expected exactly one finite final prediction per outer test input')
                    prediction_rows = [dict(model=group['model'], seed=seed, setting=group['setting'], fold=group['fold'],
                        sample_id=ids[int(i)], label=rows[int(i)]['label'], binary=rows[int(i)]['y'], prediction=float(p))
                        for i, p in zip(outer_test, prediction)]
                    base.atomic_bytes(prediction_path, csv_bytes(prediction_rows))
                    base.immutable_json(evidence_path, dict(seed=seed, winner_sha256=sha(search / 'winner.json'),
                        predictions_sha256=sha(prediction_path), final_complete_sha256=sha(final_path / 'complete.json'),
                        outer_test_used_for_selection=False))
                if [r['sample_id'] for r in prediction_rows] != item['test_ids']:
                    raise ValueError('Outer test prediction IDs/order differ')
                seed_counts[str(seed)] = len(prediction_rows)
                all_predictions.extend(prediction_rows)
                store.snapshot(f'seed{seed}_complete', force=True)
            base.atomic_bytes(out / 'predictions.csv', csv_bytes(all_predictions))
            base.immutable_json(out / 'selection.json', dict(winner=by_id[winner], seed_selection=selected,
                search_seed=42, fixed_other_seeds=[43, 44], outer_test_used_for_selection=False))
            if len(fit_map) != 22:
                raise ValueError('Expected 22 unique full fit identities after HPO and three-seed finalization')
            files = {p.relative_to(out).as_posix(): sha(p) for p in base.regular_files(out)
                     if (p.name != 'complete.json' or p.parent != out) and
                     not (p.name in ('last.pt', 'last.sha256.json') and (p.parent / 'complete.json').exists())}
            receipt = dict(schema=1, status='complete', group=group, identity_sha256=value_sha(identity),
                source_seeds=list(SEEDS), test_n=len(outer_test), predictions=len(all_predictions),
                test_sample_ids=item['test_ids'],
                per_seed_predictions=seed_counts, full_fit_count=len(fit_map), fit_execution_identities=fit_map,
                winner=by_id[winner], winner_sha256=sha(search / 'winner.json'),
                SSL_cache_references=cache_refs, files=files, outer_test_used_for_selection=False)
            base.immutable_json(out / 'complete.json', receipt)
            if persistence_override is None:
                verify_group_complete(out, identity, initial)
            store.snapshot('group_complete', force=True)
            return dict(status='complete', group_id=group_id, reused=False, predictions=len(all_predictions),
                        winner=winner, source_seeds=list(SEEDS))
        except ControlledStop:
            store.snapshot('stopped', force=True)
            return dict(status='stopped', group_id=group_id)
        except BaseException as error:
            base.write(out / 'failed.json', dict(type=type(error).__name__, error=str(error), time=time.time()))
            store.snapshot('failed', force=True)
            if stop():
                return dict(status='stopped', group_id=group_id)
            raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--expected-config-sha', required=True)
    p.add_argument('--group-id', required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--afs', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--worker-id', default='worker')
    p.add_argument('--restore-latest', action='store_true')
    args = p.parse_args()

    def stop(_signum, _frame):
        global _STOP
        _STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    result = run_group(args.config, args.expected_config_sha, args.group_id, args.root, args.afs,
        args.device, args.worker_id, args.restore_latest)
    print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'complete' else 3


if __name__ == '__main__':
    raise SystemExit(main())
