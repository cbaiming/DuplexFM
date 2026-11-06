#!/usr/bin/env python3
"""Audit and export an archived equal-update shuffled-miRNA-label control.

The archived source fit is not repeated. ``export`` checks its original
execution lock, fit-only label permutation, initial tensors and terminal update
count, then exports the canonical 3688 rows using the existing FP32 exporter.
``readout`` uses the unchanged R55 solver and fixed outer/inner splits, and
compares its predictions with the existing seed42 trained/initial readout.
Only one shuffled source seed is available; source-seed uncertainty is absent.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
INITIAL_SHA = '94bd542f156045f6e73062197672298c955d4fd7ec0e938720966d9ff23bd870'
REAL_SHA = 'c65f522bbf6e99779601ddf024fcb68ebf7740b9b6289b6dd1f7258b25eb8baf'
SHUFFLE_SHA = '27fc770ccfde6e01899c581e50a6546ce65f6166ba8ff91e7f5f0bf3d9910781'
UPDATES = 32728
TRAIN_ROWS = 2094107
VAL_ROWS = 401428


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


exporter = module('_shuffle_existing_exporter', REPO / 'scripts/export_mirna_supervision_control.py')
read, sha, write, require_hash = exporter.read, exporter.sha, exporter.write, exporter.require_hash


def require_equal(left, right, message):
    if left != right:
        raise ValueError(message)


def check_budget(reference_signature, shuffled_signature, completion, history):
    """Reject truncated or otherwise unmatched archived source fits."""
    require_equal(shuffled_signature['inputs']['original_signature'], reference_signature,
                  'Shuffled source does not bind the original training signature')
    options = reference_signature['options']
    require_equal(shuffled_signature['options'], options, 'Source optimizer/batch options differ')
    expected = dict(seed=42, batch_size=512, micro_batch_size=512, max_updates=UPDATES,
                    reference_rows=TRAIN_ROWS, amp='bf16', learning_rate=3e-4,
                    lora_learning_rate=2e-5, weight_decay=1e-4, clip_grad_norm=1.0)
    for key, value in expected.items():
        require_equal(options.get(key), value, 'Unmatched source option: ' + key)
    for key, value in dict(status='complete', updates=UPDATES, seen_rows=8 * TRAIN_ROWS,
                           reference_epochs=8, selection='terminal_update').items():
        require_equal(completion.get(key), value, 'Incomplete source budget: ' + key)
    require_equal(len(history), 8, 'Expected eight source validation records')
    for epoch, record in enumerate(history, 1):
        require_equal(record['step'], epoch * (UPDATES // 8), 'Source epoch updates differ')
        require_equal(record['seen_rows'], epoch * TRAIN_ROWS, 'Source epoch row budget differs')
    return expected


def check_permutation(original, fit, valid, order, shuffled):
    """A fixed train-only permutation must preserve every class count."""
    original, fit, valid = np.asarray(original), np.asarray(fit), np.asarray(valid)
    if (len(np.unique(fit)) != len(fit) or len(np.unique(valid)) != len(valid)
            or np.intersect1d(fit, valid).size):
        raise ValueError('Non-unique or overlapping source fit/validation indices')
    np.testing.assert_array_equal(order, np.random.default_rng(42).permutation(len(fit)))
    np.testing.assert_array_equal(shuffled[fit], original[fit[order]])
    outside = np.ones(len(original), dtype=bool)
    outside[fit] = False
    if np.any(shuffled[outside] != -1) or not np.isin(shuffled[fit], [0, 1]).all():
        raise ValueError('Shuffled labels must only cover source training rows')
    positives = int(original[fit].sum())
    require_equal(int(shuffled[fit].sum()), positives, 'Source class balance changed')
    return dict(train_rows=len(fit), validation_rows=len(valid), positives=positives,
                shuffled_positives=int(shuffled[fit].sum()), seed=42,
                changed_labels=int(np.count_nonzero(original[fit] != shuffled[fit])),
                outside_training_sentinel=-1, validation_and_test_labels_unchanged=True)


def tree_equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(tree_equal(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(tree_equal(a, b) for a, b in zip(left, right))
    return left == right


def audit_source(source_root, assets, registry_path):
    """Audit trusted local checkpoints; unpickling is limited to hash-bound files."""
    import torch
    source_root = Path(source_root)
    lock = read(source_root / 'execution_lock.json')
    for name, digest in lock['files_sha256'].items():
        candidate = (source_root / name).resolve()
        if not candidate.is_relative_to(source_root.resolve()):
            raise ValueError('Execution-lock path escapes source root')
        require_hash(candidate, digest)
    require_equal(lock['original_terminal_sha256'], REAL_SHA, 'Wrong original trained source')
    original_signature = read(source_root / 'reference/signature.json')
    signature = read(source_root / 'runs/shuffle42/signature.json')
    complete = read(source_root / 'runs/shuffle42/complete.json')
    history = read(source_root / 'runs/shuffle42/history.json')
    options = check_budget(original_signature, signature, complete, history)
    require_hash(source_root / 'runs/shuffle42/history.json', complete['history_sha256'])
    require_hash(source_root / 'reference/initial.pt', INITIAL_SHA)
    require_hash(source_root / 'runs/shuffle42/initial.pt', complete['initial_sha256'])
    require_hash(source_root / 'runs/shuffle42/terminal.pt', SHUFFLE_SHA)
    require_equal(complete['terminal_sha256'], SHUFFLE_SHA, 'Wrong shuffled source terminal')
    require_equal(signature['inputs']['reference_initial_sha256'], INITIAL_SHA, 'Wrong original initialization')
    require_equal(signature['inputs']['execution_lock_sha256'], sha(source_root / 'execution_lock.json'),
                  'Source execution-lock hash differs')
    require_hash(source_root / 'preparation/input_audit.json', signature['inputs']['input_audit_sha256'])
    preparation = read(source_root / 'preparation/input_audit.json')
    labels_path = Path(assets) / 'mirbench/data/labels.npy'
    require_hash(labels_path, preparation['data']['files_sha256']['data/labels.npy'])
    fit, valid = (np.load(source_root / 'preparation' / name, allow_pickle=False) for name in ('fit.npy', 'val.npy'))
    require_equal(len(fit), TRAIN_ROWS, 'Source fit row count differs')
    require_equal(len(valid), VAL_ROWS, 'Source validation row count differs')
    for split, values in (('train', fit), ('val', valid)):
        path = Path(assets) / f'mirbench/data/indices_{split}.npy'
        expected_sha = original_signature['inputs']['fit_sha256' if split == 'train' else 'validation_sha256']
        require_hash(path, expected_sha)
        np.testing.assert_array_equal(values, np.load(path, allow_pickle=False))
    order_path, shuffled_path = (source_root / 'preparation' / name for name in ('permutation.npy', 'shuffled_labels.npy'))
    require_hash(order_path, preparation['labels']['permutation_sha256'])
    require_hash(shuffled_path, preparation['labels']['shuffled_labels_sha256'])
    label_audit = check_permutation(np.load(labels_path, allow_pickle=False), fit, valid,
                                    np.load(order_path, allow_pickle=False), np.load(shuffled_path, allow_pickle=False))
    # These exact archived checkpoint hashes were checked above before unpickling.
    initial = torch.load(source_root / 'reference/initial.pt', map_location='cpu', weights_only=False)
    shuffled_initial = torch.load(source_root / 'runs/shuffle42/initial.pt', map_location='cpu', weights_only=False)
    terminal = torch.load(source_root / 'runs/shuffle42/terminal.pt', map_location='cpu', weights_only=False)
    for checkpoint in (initial, shuffled_initial):
        require_equal(checkpoint['state']['step'], 0, 'Initial checkpoint already updated')
        require_equal(checkpoint['state']['seen_rows'], 0, 'Initial checkpoint has seen rows')
        require_equal(checkpoint['optimizer']['state'], {}, 'Initial optimizer already updated')
    for key in ('trainable_parameters', 'fusion_buffers', 'rng'):
        if not tree_equal(initial[key], shuffled_initial[key]):
            raise ValueError('Original/shuffled initial ' + key + ' differ')
    require_equal(terminal['state']['step'], UPDATES, 'Shuffled terminal update differs')
    require_equal(terminal['state']['seen_rows'], 8 * TRAIN_ROWS, 'Shuffled terminal seen-row budget differs')
    require_equal(terminal['signature'], signature, 'Terminal source signature differs')
    require_equal(terminal['scientific_checkpoint_selection'], 'fixed_terminal_not_best_validation',
                  'Source checkpoint was selected using validation')
    models = {item['id']: item for item in read(registry_path)['models']}
    for name, expected, step in (('initial42', INITIAL_SHA, 0), ('full42', REAL_SHA, UPDATES)):
        entry = models[name]
        require_equal(entry['original_sha256'], expected, 'Original source registry differs')
        require_equal(entry['step'], step, 'Original source update count differs')
        if not entry['available'] or not entry['tensor_bitwise_restored']:
            raise ValueError('Original source checkpoint unavailable/unverified')
    result = dict(passed=True, source_seed=42, shuffled_source_seed_count=1,
                  original_initial_sha256=INITIAL_SHA, real_terminal_sha256=REAL_SHA,
                  shuffled_terminal_sha256=SHUFFLE_SHA, options=options,
                  updates=UPDATES, reference_epochs=8, source_seen_rows=8 * TRAIN_ROWS,
                  exact_initial_parameters_buffers_rng=True, labels=label_audit,
                  permutation_sha256=sha(order_path), shuffled_labels_sha256=sha(shuffled_path),
                  signature_sha256=sha(source_root / 'runs/shuffle42/signature.json'),
                  history_sha256=sha(source_root / 'runs/shuffle42/history.json'),
                  execution_lock_sha256=sha(source_root / 'execution_lock.json'),
                  selection='fixed_terminal_update_without_downstream_selection',
                  archived_source_training_seconds=complete['elapsed_seconds'],
                  limitation='One shuffled source seed; matched updates/options do not imply bitwise cross-hardware training trajectories')
    return result, terminal


def export(config_path, data_root, output, device_name='cuda:0', batch_size=32):
    config_path, data_root, output = map(lambda p: Path(p).resolve(), (config_path, data_root, output))
    if output.exists():
        raise FileExistsError(output)
    config = read(config_path)
    audit, saved = audit_source(config['shuffle_source_root'], config['assets'], config['checkpoint_registry'])
    rows = exporter.input_rows(data_root)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import torch
    source_code = REPO / 'reference/pairac_concat_v1/source_full/code'
    sys.path.insert(0, str(source_code))
    from duplexfm_joint.data import PairStore
    from duplexfm_joint import io_utils as utilities
    from duplexfm_mechanism.model import PairACModel
    from duplexfm_mechanism.protocol import source_config
    device = torch.device(device_name)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Registered export requires CUDA on A6000')
    torch.cuda.set_device(device)
    hardware = torch.cuda.get_device_name(device)
    if config.get('require_device_name', 'A6000') not in hardware:
        raise ValueError('Unexpected GPU: ' + hardware)
    torch.set_num_threads(int(config.get('cpu_threads', 4)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    stores, aligned, store_audit = exporter.open_stores(config['stores'], rows, PairStore)
    start = time.monotonic()
    try:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'features').mkdir()
        write(output / 'source_audit.json', audit)
        compact_path = output / 'shuffled42_compact.pt'
        torch.save(dict(trainable_parameters=saved['trainable_parameters'], fusion_buffers=saved['fusion_buffers'],
                        original_sha256=SHUFFLE_SHA, step=UPDATES), compact_path)
        del saved
        identity = dict(state='shuffled', seed=42, path=str(compact_path),
                        compact_sha256=sha(compact_path), checkpoint_sha256=SHUFFLE_SHA, step=UPDATES)
        model = exporter.load_model(identity, config['assets'], device, PairACModel, source_config, utilities)
        values = exporter.extract(model, stores, aligned, batch_size, device, utilities.to_device)
        feature_path = output / 'features/shuffled_seed42.npy'
        exporter.save_array(feature_path, values)
        code_hashes = {p.relative_to(source_code).as_posix(): sha(p) for p in sorted(source_code.rglob('*'))
                       if p.is_file() and p.suffix in ('.py', '.json')}
        manifest = dict(schema_version=1, state='shuffled', seed=42, rows=len(rows), input_sha256=exporter.INPUT_SHA,
                        row_ids_sha256=exporter.row_ids_sha(rows), path='features/shuffled_seed42.npy',
                        sha256=sha(feature_path), checkpoint_sha256=SHUFFLE_SHA,
                        original_initial_sha256=INITIAL_SHA, real_terminal_sha256=REAL_SHA,
                        source_audit_sha256=sha(output / 'source_audit.json'), source_audit=audit,
                        config_sha256=sha(config_path), exporter_sha256=sha(__file__),
                        reused_exporter_sha256=sha(REPO / 'scripts/export_mirna_supervision_control.py'),
                        source_files_sha256=code_hashes, stores=store_audit,
                        public_backbone_provenance=model.base_provenance,
                        hardware=hardware, precision='FP32', tf32=False, batch_size=batch_size,
                        labels_forwarded=False, source_model_training=False,
                        wall_seconds=time.monotonic() - start)
        # Fail if inputs or source tensors changed during extraction.
        require_hash(data_root / 'input_rows.json', exporter.INPUT_SHA)
        require_hash(Path(config['shuffle_source_root']) / 'runs/shuffle42/terminal.pt', SHUFFLE_SHA)
        for store, previous in zip(stores, store_audit):
            current = store.verify_integrity()
            require_equal(current, {key: previous[key] for key in current}, 'PairStore changed during export')
        write(output / 'features_manifest.json', manifest)
        print(json.dumps(dict(status='complete', rows=len(rows), wall_seconds=manifest['wall_seconds'])), flush=True)
        return manifest
    finally:
        for store in stores:
            store.close()


def load_shuffled_features(manifest_path, dataset):
    manifest_path = Path(manifest_path)
    manifest = read(manifest_path)
    for key, expected in dict(schema_version=1, state='shuffled', seed=42, rows=len(dataset['rows']),
                               input_sha256=exporter.INPUT_SHA, row_ids_sha256=exporter.row_ids_sha(dataset['rows']),
                               checkpoint_sha256=SHUFFLE_SHA).items():
        require_equal(manifest.get(key), expected, 'Invalid shuffled feature identity: ' + key)
    if not manifest.get('source_audit', {}).get('passed'):
        raise ValueError('Source budget audit missing')
    path = manifest_path.parent / manifest['path']
    require_hash(path, manifest['sha256'])
    features = np.load(path, allow_pickle=False)
    exporter.validate_features(features, len(dataset['rows']))
    return features, manifest


def readout(root, feature_manifest, previous_predictions, output):
    """Add the shuffled source to the established complete seed42 comparison."""
    control = module('_shuffle_existing_readout', root / 'scripts/run_mirna_supervision_control.py')
    original = control.original
    dataset = original.load_data(root / 'data/sirna_oligo_random')
    features, manifest = load_shuffled_features(feature_manifest, dataset)
    math = original.load_math(root)
    archives = control.load_archives(root / 'data/sirna_oligo_random')
    requested = original.selected_folds(dataset['plan'], [], [], True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    # Extend only the archive-identity lookup; the source solver, transformations,
    # alpha selection, group penalty and numerical certificates are unchanged.
    old_expected = control.expected_predictions
    control.expected_predictions = lambda data, state, seed, mode, item, saved: (
        None if state == 'shuffled' else old_expected(data, state, seed, mode, item, saved))
    with Path(previous_predictions).open(encoding='utf-8-sig', newline='') as handle:
        previous = list(csv.DictReader(handle))
    records = []
    expected_ids = {(item['setting'], int(item['fold']), sid) for item in requested for sid in item['test_ids']}
    for state in ('trained', 'initial', 'TD24'):
        for mode in (control.MODES if state != 'TD24' else ['TD24']):
            part = [dict(row) for row in previous if row['state'] == state and row['mode'] == mode
                    and (state == 'TD24' or row['seed'] == '42')]
            ids = {(row['setting'], int(row['fold']), row['sample_id']) for row in part}
            if ids != expected_ids or len(part) != len(expected_ids):
                raise ValueError('Archived real/initial/TD24 prediction coverage differs')
            for row in part:
                row['seed'] = None if state == 'TD24' else 42
                row['fold'] = int(row['fold'])
                observed = dataset['rows'][dataset['lookup'][row['sample_id']]]
                require_equal(float(row['label']), float(observed['label']), 'Archived labels differ')
                require_equal(row['binary'], observed['y'], 'Archived binary labels differ')
                row['prediction'] = float(row['prediction'])
            records.extend(part)
    output.mkdir(parents=True, exist_ok=False)
    start, reports = time.monotonic(), []
    for mode in control.MODES:
        x = control.input_matrix(dataset, features, mode)
        for item in requested:
            folder = output / 'shuffled_seed42' / mode / item['setting'] / f"fold{item['fold']}"
            part, report = control.refit_fold(dataset, x, math, 'shuffled', 42, mode, item, archives, folder)
            records.extend(part)
            reports.append(report)
            print('FOLD_COMPLETE shuffled 42', mode, item['setting'], item['fold'], round(time.monotonic() - start, 2), flush=True)
    metrics, summary, _, _ = control.summarize(records, dataset, requested)
    lookup = {(r['state'], r['mode'], r['setting']): r for r in metrics}
    deltas = []
    for real in metrics:
        if real['state'] != 'trained':
            continue
        for other_state in ('initial', 'shuffled'):
            other = lookup[other_state, real['mode'], real['setting']]
            deltas.append(dict(comparison='trained_minus_' + other_state, seed=42, mode=real['mode'],
                               setting=real['setting'], n=real['n'], aggregation=real['aggregation'],
                               **{key: real[key] - other[key] for key in control.METRICS}))
    control.write_csv(output / 'predictions.csv', records)
    control.write_csv(output / 'per_seed.csv', metrics)
    control.write_csv(output / 'summary.csv', summary)
    control.write_csv(output / 'paired_deltas.csv', deltas)
    write(output / 'complete.json', dict(passed=True, complete_26_outer_folds=True,
          source_seed=42, shuffled_source_seed_count=1, modes=list(control.MODES),
          feature_manifest_sha256=sha(feature_manifest), previous_predictions_sha256=sha(previous_predictions),
          input_sha256=original.INPUT_SHA, split_sha256=original.PLAN_SHA,
          script_sha256=sha(__file__), reused_readout_sha256=sha(root / 'scripts/run_mirna_supervision_control.py'),
          source_audit=manifest['source_audit'], fold_results=reports, new_readout_fits=26 * len(reports),
          prediction_rows=len(records), source_training=False, GPU_used=False,
          uncertainty='Single source seed: descriptive comparison, no source-seed SD or independent-fold significance',
          elapsed_CPU_wall_seconds=time.monotonic() - start))
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    exp = commands.add_parser('export')
    exp.add_argument('--config', type=Path, required=True)
    exp.add_argument('--data-root', type=Path, default=REPO / 'data/sirna_oligo_random')
    exp.add_argument('--output', type=Path, required=True)
    exp.add_argument('--device', default='cuda:0')
    exp.add_argument('--batch-size', type=int, default=32)
    run = commands.add_parser('readout')
    run.add_argument('--root', type=Path, default=REPO)
    run.add_argument('--features-manifest', type=Path, required=True)
    run.add_argument('--previous-predictions', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'export':
        export(args.config, args.data_root, args.output, args.device, args.batch_size)
    else:
        readout(args.root.resolve(), args.features_manifest.resolve(), args.previous_predictions.resolve(), args.output)


if __name__ == '__main__':
    main()
