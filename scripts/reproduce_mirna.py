"""Portable PairAC source train/evaluate/predict entry points using exact local assets.

No model or project weight is downloaded. New training uses the preserved engine
and fixed update endpoint, without the historical H100 lease/scheduler. It is a
new execution, not a claim of bitwise identity across environments or hardware.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'reference/pairac_concat_v1/source_full/code'
sys.path.insert(0, str(CODE))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def safe_partition(value):
    if not value or Path(value).name != value or any(character in value for character in '/\\:') or value in ('.', '..'):
        raise ValueError('Partition must be one safe filename component')
    return value


def qualified_store(path):
    from duplexfm_joint.data import PairStore
    historical = read(ROOT / 'data/mirbench_v7/historical_inputs.json')
    store = PairStore(path)
    try:
        integrity = store.verify_integrity()
        if integrity['files_sha256'] != historical['files_sha256']:
            raise ValueError('PairStore arrays differ from the historical source input contract')
        return store, integrity
    except BaseException:
        store.close()
        raise


def validate_checkpoint_contract(checkpoint, args, digest):
    """A caller-supplied hash establishes bytes, not source seed or protocol."""
    from duplexfm_mechanism.protocol import contract, source_config
    recorded = read(ROOT / 'data/mirbench_v7/preparation_contract.json')
    compact = read(ROOT / 'MODEL_REGISTRY.json').get('compact_inference_checkpoints', {})
    matches = [(int(seed), item) for seed, item in compact.items() if item['sha256'] == digest]
    if matches:
        seed, item = matches[0]
        if seed != args.seed:
            raise ValueError('Registered compact checkpoint belongs to a different source seed')
        if (checkpoint.get('step') != 32728
                or checkpoint.get('original_sha256') != recorded['source_terminal_sha256'][str(seed)]
                or checkpoint.get('original_sha256') != item['original_sha256']
                or not isinstance(checkpoint.get('trainable_parameters'), dict)
                or not isinstance(checkpoint.get('fusion_buffers'), dict)
                or len(checkpoint['trainable_parameters']) != 132):
            raise ValueError('Registered compact checkpoint metadata or tensor inventory differs')
        return {'sha256': digest, 'source_seed': seed, 'registered_compact_identity': True,
                'original_sha256': item['original_sha256'], 'training_resume_supported': False}
    if checkpoint.get('state', {}).get('step') != recorded['source_terminal_updates']:
        raise ValueError('Source inference requires the fixed 32728-update terminal checkpoint')
    historical = recorded['source_terminal_sha256']
    matched = [int(seed) for seed, value in historical.items() if digest == value]
    if matched:
        if matched != [args.seed]:
            raise ValueError('Historical checkpoint belongs to a different source seed')
        return {'sha256': digest, 'source_seed': args.seed, 'historical_terminal_identity': True}
    signature = checkpoint.get('signature', {})
    inputs, options = signature.get('inputs', {}), signature.get('options', {})
    spec = contract()
    if inputs.get('version') != 'portable_PairAC_source_execution_v1':
        raise ValueError('Unqualified new checkpoint: missing portable source execution signature')
    if inputs.get('source_seed') != args.seed or options.get('seed') != args.seed:
        raise ValueError('New checkpoint source seed differs from --seed')
    if inputs.get('original_contract') != spec:
        raise ValueError('New checkpoint source protocol differs from the fixed PairAC contract')
    state = checkpoint['state']
    epochs = spec['reference_epochs']
    steps_per_epoch = spec['updates'] // epochs
    history = state.get('history', [])
    if (state.get('seen_rows') != epochs * spec['reference_train_rows']
            or state.get('last_validation_step') != spec['updates']
            or len(history) != epochs
            or any(record.get('reference_epoch') != index or record.get('step') != index * steps_per_epoch
                   for index, record in enumerate(history, 1))
            or checkpoint.get('scientific_checkpoint_selection') != 'fixed_terminal_not_best_validation'):
        raise ValueError('New checkpoint terminal epoch/validation cursor differs from the fixed endpoint')
    expected_config = {key: value for key, value in source_config(args.assets).items() if not key.endswith('_model')}
    if inputs.get('model_config') != expected_config:
        raise ValueError('New checkpoint architecture/model configuration differs')
    expected_inputs = read(ROOT / 'data/mirbench_v7/historical_inputs.json')['files_sha256']
    if inputs.get('input_arrays_sha256') != expected_inputs:
        raise ValueError('New checkpoint source input array identities differ')
    expected_options = {'reference_rows': spec['reference_train_rows'], 'batch_size': spec['batch'],
                        'eval_batch_size': spec['eval_batch'], 'max_updates': spec['updates'],
                        'learning_rate': spec['learning_rate'], 'lora_learning_rate': spec['lora_learning_rate'],
                        'weight_decay': spec['weight_decay'], 'clip_grad_norm': spec['clip_grad_norm'],
                        'checkpoint_steps': 500, 'backup_seconds': 3600}
    if any(options.get(key) != value for key, value in expected_options.items()):
        raise ValueError('New checkpoint training options differ from the fixed source objective')
    if options.get('amp') not in ('bf16', 'fp32') or options.get('micro_batch_size') not in (64, 128, 256, 512):
        raise ValueError('New checkpoint training precision/micro-batch contract is invalid')
    expected_code = {path.relative_to(CODE).as_posix(): sha(path) for path in CODE.rglob('*.py')}
    if inputs.get('source_code_sha256') != expected_code:
        raise ValueError('New checkpoint source-code identities differ')
    foundation = inputs.get('foundation', {})
    expected_foundation = read(CODE / 'foundation_reference.json')
    for kind in ('target', 'guide'):
        actual = foundation.get(kind, {}).get('sha256', {})
        if any(actual.get(name) != value for name, value in expected_foundation[kind]['sha256'].items()):
            raise ValueError('New checkpoint foundation identities differ')
    if not foundation.get('foundation_reference_verified'):
        raise ValueError('New checkpoint foundation verification was not recorded')
    return {'sha256': digest, 'source_seed': args.seed, 'historical_terminal_identity': False,
            'new_execution_contract_verified': True, 'training_precision': options['amp'],
            'training_micro_batch': options['micro_batch_size']}


def load_model(args, require_checkpoint=True):
    import torch
    from duplexfm_mechanism.model import PairACModel
    from duplexfm_mechanism.protocol import source_config
    from duplexfm_joint.io_utils import seed_all, load_checkpoint, load_trainable_state
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested CUDA device is unavailable')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if args.amp == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('Historical BF16 inference/training requires supported CUDA; use explicit --amp fp32 on CPU')
    checkpoint = None
    checkpoint_identity = None
    if require_checkpoint:
        if args.checkpoint is None:
            raise ValueError('An explicit project-trained --checkpoint is required')
        contract = read(ROOT / 'data/mirbench_v7/preparation_contract.json')
        actual = sha(args.checkpoint)
        compact = read(ROOT / 'MODEL_REGISTRY.json').get('compact_inference_checkpoints', {}).get(str(args.seed), {})
        default = compact['sha256'] if actual == compact.get('sha256') else contract['source_terminal_sha256'][str(args.seed)]
        expected = args.checkpoint_sha256 or default
        if actual != expected:
            raise ValueError('Checkpoint SHA-256 differs from the supplied/registered identity')
        checkpoint = load_checkpoint(args.checkpoint)
        checkpoint_identity = validate_checkpoint_contract(checkpoint, args, actual)
    seed_all(args.seed)
    model = PairACModel(source_config(args.assets), 'seed' + str(args.seed))
    if checkpoint is not None:
        load_trainable_state(model, checkpoint)
    model.to(device)
    return model, device, checkpoint_identity


def evaluate(args):
    import numpy as np
    import torch
    from duplexfm_joint.io_utils import to_device, autocast_context, binary_metrics
    from duplexfm_mechanism.protocol import OFFICIAL_TESTS
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory; predictions are not overwritten')
    store, integrity = qualified_store(args.store.resolve())
    try:
        partitions = list(OFFICIAL_TESTS) if args.command == 'evaluate' and not args.partition else args.partition
        if not partitions or len(partitions) != len(set(partitions)):
            raise ValueError('Specify one or more unique partitions')
        selections = {safe_partition(name): store.indices(name) for name in partitions}
        if args.limit is not None:
            if args.limit <= 0:
                raise ValueError('--limit must be positive')
            selections = {name: rows[:args.limit] for name, rows in selections.items()}
        if args.compare_ap and (args.command != 'evaluate' or args.limit is not None or set(partitions) != set(OFFICIAL_TESTS)):
            raise ValueError('AP comparison requires complete evaluation of all four official tests')
        model, device, checkpoint_identity = load_model(args)
        model.requires_grad_(False).eval()
        output.mkdir(parents=True, exist_ok=False)
        records = []
        for name, indices in selections.items():
            destination = output / name
            destination.mkdir()
            truth, scores = [], []
            for lo in range(0, len(indices), 8192):
                selected = indices[lo:lo + 8192]
                labels, values = [], []
                with torch.inference_mode():
                    for begin in range(0, len(selected), args.batch_size):
                        ids = selected[begin:begin + args.batch_size]
                        batch = store.collate(ids)
                        if not np.array_equal(batch['row_indices'].numpy(), ids):
                            raise ValueError('Prediction row identity changed')
                        batch = to_device(batch, device)
                        with autocast_context(device, args.amp):
                            probability = model(batch).float().sigmoid()
                        labels.append(batch['labels'].cpu().numpy())
                        values.append(probability.cpu().numpy())
                y, p = np.concatenate(labels), np.concatenate(values)
                np.savez(destination / f'{lo:08d}.npz', rows=selected, labels=y, scores=p)
                truth.append(y)
                scores.append(p)
            metric = binary_metrics(np.concatenate(truth), np.concatenate(scores))
            record = {'partition': name, 'metrics': metric, 'rows': len(indices)}
            write(destination / 'metrics.json', record)
            records.append(record)
        report = {'status': 'complete', 'source_seed': args.seed, 'checkpoint': checkpoint_identity,
                  'precision': args.amp, 'device_type': device.type,
                  'historical_precision': args.amp == 'bf16', 'partial_qualification': args.limit is not None,
                  'source_keep_overlap': True, 'training_started': False,
                  'PairStore_manifest_sha256': integrity['manifest_sha256'], 'records': records,
                  'note': 'miRNA interaction probability, not zero-shot siRNA efficacy'}
        if args.compare_ap:
            expected = read(args.compare_ap)
            values = expected.get('per_test_AP', expected)
            errors = {}
            for record in records:
                name = record['partition']
                value = values[name]
                if isinstance(value, dict):
                    value = value.get(str(args.seed), value.get('seed' + str(args.seed)))
                errors[name] = abs(record['metrics']['aps'] - float(value))
            report['AP_comparison'] = {'atol': args.ap_atol, 'absolute_errors': errors,
                                       'passed': all(error <= args.ap_atol for error in errors.values())}
            if not report['AP_comparison']['passed']:
                write(output / 'comparison_failed.json', report)
                raise ValueError('Regenerated AP differs from the explicit reference beyond tolerance')
        write(output / 'complete.json', report)
        return report
    finally:
        store.close()


def train(args):
    import numpy as np
    import torch
    from duplexfm_mechanism.protocol import contract, source_config
    from duplexfm_mechanism.training_hourly import run_engine, StopGuard
    from duplexfm_joint.io_utils import provenance_content
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError('Training output exists; use --resume only for this exact saved execution')
    if args.resume and not (output / 'last.pt').is_file():
        raise ValueError('--resume requires this execution\'s existing last.pt')
    store, integrity = qualified_store(args.store.resolve())
    try:
        spec = contract()
        fit, val = store.indices('train'), store.indices('val')
        if (len(fit), len(val)) != (spec['reference_train_rows'], spec['reference_val_rows']) or np.intersect1d(fit, val).size:
            raise ValueError('Historical fixed training/validation indices differ')
        model, device, _ = load_model(args, require_checkpoint=False)
        options = {'seed': args.seed, 'reference_rows': spec['reference_train_rows'], 'batch_size': spec['batch'],
                   'eval_batch_size': spec['eval_batch'], 'max_updates': spec['updates'], 'amp': args.amp,
                   'learning_rate': spec['learning_rate'], 'lora_learning_rate': spec['lora_learning_rate'],
                   'weight_decay': spec['weight_decay'], 'clip_grad_norm': spec['clip_grad_norm'],
                   'checkpoint_steps': 500, 'backup_seconds': 3600, 'micro_batch_size': args.micro_batch}
        signature = {'version': 'portable_PairAC_source_execution_v1', 'source_seed': args.seed,
                     'runtime': {'torch': str(torch.__version__), 'cuda_build': torch.version.cuda,
                                 'device_type': device.type,
                                 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'},
                     'foundation': provenance_content(model.base_provenance),
                     'input_arrays_sha256': integrity['files_sha256'],
                     'model_config': {key: value for key, value in source_config(args.assets).items() if not key.endswith('_model')},
                     'original_contract': spec, 'source_code_sha256': {path.relative_to(CODE).as_posix(): sha(path) for path in CODE.rglob('*.py')},
                     'interpretation': 'New execution of original fixed objective; no cross-hardware bitwise identity claim'}
        result = run_engine(model, store, fit, val, output, options, signature, resume=args.resume,
                            guard=StopGuard(stop_file=output / 'STOP_REQUESTED'))
        return {'status': result.get('status', 'finished'), 'engine_result': result, 'source_seed': args.seed,
                'new_execution': True, 'historical_checkpoint_reconstructed': False}
    finally:
        store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('train', 'evaluate', 'predict'):
        sub = commands.add_parser(name)
        sub.add_argument('--assets', type=Path, required=True, help='Local directory containing exact mrnabert/ and rnafm/ snapshots')
        sub.add_argument('--store', type=Path, required=True, help='Complete SHA-qualified historical PairStore')
        sub.add_argument('--seed', type=int, choices=(42, 43, 44), default=42)
        sub.add_argument('--device', default='cuda:0')
        sub.add_argument('--amp', choices=('bf16', 'fp32'), default='bf16')
        sub.add_argument('--output', type=Path, required=True)
        if name == 'train':
            sub.add_argument('--micro-batch', type=int, choices=(64, 128, 256, 512), default=512)
            sub.add_argument('--resume', action='store_true')
        else:
            sub.add_argument('--checkpoint', type=Path, required=True, help='Explicit user-supplied PairAC terminal task state')
            sub.add_argument('--checkpoint-sha256', help='Exact new checkpoint identity; defaults to registered historical seed terminal')
            sub.add_argument('--partition', nargs='+', required=name == 'predict')
            sub.add_argument('--batch-size', type=int, default=512)
            sub.add_argument('--limit', type=int, help='Small explicit qualification; never a complete benchmark result')
            sub.add_argument('--compare-ap', type=Path, help='Optional JSON per_test_AP {test: value or {seed:value}}')
            sub.add_argument('--ap-atol', type=float, default=1e-6)
    args = parser.parse_args(argv)
    try:
        if not __debug__:
            raise ValueError('Run without Python -O; original scientific code uses assertions')
        if getattr(args, 'batch_size', 1) <= 0 or getattr(args, 'ap_atol', 0) < 0:
            raise ValueError('Invalid batch size or AP tolerance')
        report = train(args) if args.command == 'train' else evaluate(args)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, AssertionError, RuntimeError, TypeError) as error:
        print('PairAC source workflow failed: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
