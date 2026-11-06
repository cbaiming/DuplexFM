#!/usr/bin/env python3
"""Export exact pre-miRNA states and audit their matched terminal states.

This command requires local public backbone weights, the independently frozen
accessibility PairStores and the registered task checkpoints. It never trains a
model. All six exports use the same device and FP32 inference; trained exports
must reproduce the released feature bank before an initial-state bank is accepted.

Configuration JSON::

    {"assets": "/path/to/assets", "checkpoint_registry": "/path/source_registry.json",
     "stores": [{"root": "/path/to/store", "rows_csv": "/path/to/store/rows.csv",
                 "manifest_sha256": "...", "rows_sha256": "..."}]}

Each registry model must bind id (initial42/full42, etc.), seed, step, path,
sha256, original_sha256, available and tensor_bitwise_restored. Registry paths
are relative to the registry directory. Stores can optionally specify cohorts
to select their membership. Output directories must be new, including on retry.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

INPUT_SHA = 'a928975aa974a20bf3303ee46257336390a1e1a90b643a1d92960539c38c4980'
TRAINED_SHA = {
    42: '652418cadaa9c7b7e5cc49f6d6f31f2a3b0a626662ae1245ce72c2fc87ce38dd',
    43: 'f1186b843f66887ac5a5504187a471e99f9ca05cfe1c1d9192630000279eb8b7',
    44: '5c799e22178599e5249b1b21ef730b5e42d7aacbb94334a70875147d33f78767',
}
SEEDS = (42, 43, 44)
ROW_COUNT = 3688
REPO = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=unique)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def require_hash(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or sha(path) != expected:
        raise ValueError('File SHA256 differs: ' + str(path))


def row_ids_sha(rows):
    return hashlib.sha256(('\n'.join(row['sample_id'] for row in rows) + '\n').encode('utf-8')).hexdigest()


def input_rows(base):
    path = Path(base) / 'input_rows.json'
    require_hash(path, INPUT_SHA)
    # Efficacy and binary labels are deliberately absent from the inference rows.
    rows = [{key: item[key] for key in ('sample_id', 'cohort', 'guide', 'target57')}
            for item in read(path)]
    if len(rows) != ROW_COUNT or len({row['sample_id'] for row in rows}) != ROW_COUNT:
        raise ValueError('Expected 3688 unique, ordered input rows')
    complement = str.maketrans('ACGU', 'UGCA')
    for row in rows:
        guide, target = row['guide'], row['target57']
        if (len(guide) != 19 or len(target) != 57 or set(guide + target) - set('ACGU')
                or target[19:38] != guide.translate(complement)[::-1]):
            raise ValueError('Canonical guide19/design-site target57 required')
    return rows


def checkpoint_sources(registry_path, model_registry):
    registry_path = Path(registry_path).resolve()
    registry = read(registry_path)
    models = registry.get('models')
    if not isinstance(models, list):
        raise ValueError('Source registry must contain a models list')
    identities = {}
    for item in models:
        name = item.get('id')
        if name in identities:
            raise ValueError('Duplicate source model id')
        identities[name] = item
    result = []
    references = model_registry['model_checkpoint_sha256_from_existing_verified_reference']
    for seed in SEEDS:
        for state, prefix, endpoint, step in (('trained', 'full', 'terminal.pt', 32728),
                                               ('initial', 'initial', 'initial.pt', 0)):
            name = f'{prefix}{seed}'
            item = identities.get(name, {})
            original_path = f'runs/seed{seed}/{endpoint}'
            if (item.get('available') is not True or item.get('tensor_bitwise_restored') is not True
                    or type(item.get('seed')) is not int or item['seed'] != seed
                    or type(item.get('step')) is not int or item['step'] != step
                    or item.get('original_sha256') != references[original_path]):
                raise ValueError('Unverified source checkpoint identity: ' + name)
            checkpoint = Path(item['path'])
            if not checkpoint.is_absolute():
                checkpoint = registry_path.parent / checkpoint
            checkpoint = checkpoint.resolve()
            require_hash(checkpoint, item.get('sha256'))
            result.append(dict(state=state, seed=seed, path=str(checkpoint),
                               compact_sha256=item['sha256'], checkpoint_path=original_path,
                               checkpoint_sha256=item['original_sha256'], step=step,
                               tensor_bitwise_restored=True))
    return result


def validate_checkpoint(saved, identity):
    """Check a compact endpoint or its original complete checkpoint."""
    if not isinstance(saved, dict) or not all(isinstance(saved.get(k), dict)
                                             for k in ('trainable_parameters', 'fusion_buffers')):
        raise ValueError('Task checkpoint tensor dictionaries are missing')
    if 'original_sha256' in saved:
        if saved['original_sha256'] != identity['checkpoint_sha256']:
            raise ValueError('Compact original checkpoint SHA differs')
        step = saved.get('step')
    else:
        step = saved.get('state', {}).get('step')
        if identity['compact_sha256'] != identity['checkpoint_sha256']:
            raise ValueError('Original checkpoint bytes do not match registered original SHA')
        if identity['state'] == 'initial':
            if saved.get('state', {}).get('seen_rows') != 0 or saved.get('optimizer', {}).get('state', {}):
                raise ValueError('Initial checkpoint has observed rows or optimizer updates')
    if type(step) is not int or step != identity['step']:
        raise ValueError('Checkpoint update count differs')
    if not saved['trainable_parameters']:
        raise ValueError('Empty checkpoint parameters')


def align_store(store, row_path, rows):
    with Path(row_path).open(encoding='utf-8-sig', newline='') as handle:
        stored = list(csv.DictReader(handle))
    if len(stored) != len(store):
        raise ValueError('PairStore CSV/array row count differs')
    lookup = {row['sample_id']: i for i, row in enumerate(stored)}
    if len(lookup) != len(stored):
        raise ValueError('Duplicate PairStore sample id')
    alphabet = np.asarray(list('ACGU'))
    aligned = []
    for row in rows:
        if row['sample_id'] not in lookup:
            continue
        index = lookup[row['sample_id']]
        for name, expected in (('guide', row['guide']), ('target', row['target57'])):
            token_id = int(getattr(store, name + '_ids')[index])
            length = int(getattr(store, name + '_lengths')[token_id])
            tokens = getattr(store, name + '_tokens')[token_id, :length]
            if np.any(tokens < 0) or np.any(tokens > 3) or ''.join(alphabet[tokens]) != expected:
                raise ValueError('PairStore sequence differs for ' + row['sample_id'])
        aligned.append((row['sample_id'], index))
    return aligned


def open_stores(specs, rows, store_type):
    stores, audits, selection = [], [], {}
    try:
        for item in specs:
            root = Path(item['root']).resolve()
            row_path = Path(item.get('rows_csv', root / 'rows.csv')).resolve()
            require_hash(root / 'manifest.json', item.get('manifest_sha256'))
            require_hash(row_path, item.get('rows_sha256'))
            store = store_type(root)
            stores.append(store)
            integrity = store.verify_integrity()
            candidates = [row for row in rows if 'cohorts' not in item or row['cohort'] in item['cohorts']]
            aligned = align_store(store, row_path, candidates)
            if not aligned:
                raise ValueError('PairStore contributes no benchmark rows')
            for sample_id, index in aligned:
                if sample_id in selection:
                    raise ValueError('Benchmark row appears in multiple PairStores: ' + sample_id)
                selection[sample_id] = (len(stores) - 1, index)
            audits.append(dict(root=str(root), rows_csv=str(row_path), rows_sha256=sha(row_path),
                               selected_rows=len(aligned), **integrity))
        if set(selection) != {row['sample_id'] for row in rows}:
            raise ValueError('PairStores must cover benchmark rows exactly once')
        return stores, [selection[row['sample_id']] for row in rows], audits
    except BaseException:
        for store in stores:
            store.close()
        raise


def feature_batch(store, indices):
    batch = store.collate(np.asarray(indices, dtype=np.int64))
    # Only sequence, coordinate and independently frozen accessibility inputs
    # cross the model boundary. PairStore's efficacy labels are never forwarded.
    return {key: value for key, value in batch.items() if key not in ('labels', 'row_indices')}


def validate_features(values, rows):
    if values.shape != (rows, 192) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError('Expected finite float32 rows x 192 features')


def parity_report(actual, reference, atol=1e-4, rtol=1e-5):
    if not (0 <= atol <= 1e-4 and 0 <= rtol <= 1e-5):
        raise ValueError('Parity tolerances may only tighten the audited limits')
    validate_features(actual, len(reference))
    validate_features(reference, len(actual))
    difference = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    passed = bool(np.allclose(actual, reference, atol=atol, rtol=rtol))
    report = dict(passed=passed, atol=atol, rtol=rtol, compared_values=int(actual.size),
                  max_abs_difference=float(difference.max()), mean_abs_difference=float(difference.mean()))
    if not passed:
        raise ValueError('GPU FP32 trained features fail released-bank parity: ' + json.dumps(report))
    return report


def load_model(identity, assets, device, model_type, source_config, utilities):
    import torch
    utilities.seed_all(identity['seed'])
    require_hash(identity['path'], identity['compact_sha256'])
    saved = torch.load(identity['path'], map_location='cpu', weights_only=True)
    validate_checkpoint(saved, identity)
    model = model_type(source_config(assets), f"seed{identity['seed']}")
    utilities.load_trainable_state(model, saved)
    parameters = dict(model.named_parameters())
    buffers = dict(model.fusion.named_buffers())
    for entries, current in ((saved['trainable_parameters'], parameters), (saved['fusion_buffers'], buffers)):
        for name, tensor in entries.items():
            if not torch.equal(current[name].detach().cpu(), tensor):
                raise ValueError('Checkpoint tensor restoration differs: ' + name)
    # Initial LoRA must preserve the public foundations exactly.
    if identity['state'] == 'initial':
        adapters = [tensor for name, tensor in saved['trainable_parameters'].items() if name.endswith('lora_B')]
        if not adapters or any(torch.count_nonzero(tensor).item() for tensor in adapters):
            raise ValueError('Initial LoRA B is missing or nonzero')
    model.float().to(device).requires_grad_(False).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError('Feature model was not frozen')
    del saved
    return model


def extract(model, stores, aligned, batch_size, device, to_device):
    import torch
    result = np.empty((len(aligned), 192), dtype=np.float32)
    groups = {}
    for position, (store_index, row_index) in enumerate(aligned):
        groups.setdefault(store_index, []).append((position, row_index))
    with torch.inference_mode():
        for store_index, items in groups.items():
            for offset in range(0, len(items), batch_size):
                part = items[offset:offset + batch_size]
                batch = to_device(feature_batch(stores[store_index], [item[1] for item in part]), device)
                values = model.source192(batch, 19).float().cpu().numpy()
                validate_features(values, len(part))
                result[[item[0] for item in part]] = values
    validate_features(result, len(aligned))
    return result


def save_array(path, values):
    with Path(path).open('xb') as handle:
        np.save(handle, values, allow_pickle=False)


def run(config_path, data_root, output, device_name='cuda:0', batch_size=32):
    if batch_size < 1:
        raise ValueError('Positive batch size required')
    config_path, data_root, output = Path(config_path).resolve(), Path(data_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Output directory already exists: ' + str(output))
    config = read(config_path)
    if 'checkpoint_registry_sha256' in config:
        require_hash(config['checkpoint_registry'], config['checkpoint_registry_sha256'])
    rows = input_rows(data_root)
    registry_path = REPO / 'MODEL_REGISTRY.json'
    source_registry_sha = sha(config['checkpoint_registry'])
    identities = checkpoint_sources(config['checkpoint_registry'], read(registry_path))
    reference_paths = {seed: data_root / 'features' / f'full_seed{seed}.npy' for seed in SEEDS}
    references = {}
    for seed, path in reference_paths.items():
        require_hash(path, TRAINED_SHA[seed])
        references[seed] = np.load(path, allow_pickle=False)
        validate_features(references[seed], len(rows))
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
        raise ValueError('This registered export requires the authorized CUDA device')
    torch.cuda.set_device(device)
    hardware = torch.cuda.get_device_name(device)
    required = config.get('require_device_name', 'A6000')
    if required not in hardware:
        raise ValueError('Unexpected GPU: ' + hardware)
    torch.set_num_threads(int(config.get('cpu_threads', 4)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    stores, aligned, store_audit = open_stores(config['stores'], rows, PairStore)
    start = time.monotonic()
    try:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'features').mkdir()
        (output / 'parity').mkdir()
        provenance = dict(schema_version=1, config_sha256=sha(config_path),
                          exporter_sha256=sha(__file__), input_sha256=INPUT_SHA,
                          model_registry_sha256=sha(registry_path),
                          checkpoint_registry_sha256=source_registry_sha,
                          source_files_sha256={path.relative_to(source_code).as_posix(): sha(path)
                                               for path in sorted(source_code.rglob('*'))
                                               if path.is_file() and path.suffix in ('.py', '.json')},
                          row_ids_sha256=row_ids_sha(rows), rows=len(rows), stores=store_audit,
                          models=identities, device=str(device), hardware=hardware,
                          precision='FP32', torch_version=torch.__version__, batch_size=batch_size,
                          tf32=False, labels_forwarded=False, encoder_training=False)
        write(output / 'provenance.json', provenance)
        features, parity, backbone_provenance = [], [], None
        for identity in identities:
            state, seed = identity['state'], identity['seed']
            print(json.dumps(dict(phase='export', state=state, seed=seed)), flush=True)
            model = load_model(identity, config['assets'], device, PairACModel, source_config, utilities)
            if backbone_provenance is None:
                backbone_provenance = model.base_provenance
                write(output / 'backbone_provenance.json', backbone_provenance)
            elif model.base_provenance != backbone_provenance:
                raise ValueError('Public backbone identities differ between source endpoints')
            values = extract(model, stores, aligned, batch_size, device, utilities.to_device)
            relative = f'features/{state}_seed{seed}.npy'
            if state == 'trained':
                report = parity_report(values, references[seed], config.get('parity_atol', 1e-4),
                                       config.get('parity_rtol', 1e-5))
                report.update(seed=seed, archived_sha256=TRAINED_SHA[seed])
                gpu_path = output / 'parity' / f'trained_seed{seed}_gpu_fp32.npy'
                save_array(gpu_path, values)
                report['gpu_features_sha256'] = sha(gpu_path)
                with reference_paths[seed].open('rb') as source, (output / relative).open('xb') as target:
                    shutil.copyfileobj(source, target)
                require_hash(output / relative, TRAINED_SHA[seed])
                parity.append(report)
                write(output / 'parity' / f'seed{seed}.json', report)
            else:
                save_array(output / relative, values)
            features.append(dict(state=state, seed=seed, path=relative, sha256=sha(output / relative),
                                 checkpoint_path=identity['checkpoint_path'],
                                 checkpoint_sha256=identity['checkpoint_sha256'],
                                 compact_checkpoint_path=identity['path'],
                                 compact_checkpoint_sha256=identity['compact_sha256']))
            del model, values
            gc.collect()
            torch.cuda.empty_cache()
        # Refuse publication if any input or checkpoint changed during extraction.
        require_hash(data_root / 'input_rows.json', INPUT_SHA)
        require_hash(config_path, provenance['config_sha256'])
        require_hash(__file__, provenance['exporter_sha256'])
        require_hash(registry_path, provenance['model_registry_sha256'])
        require_hash(config['checkpoint_registry'], provenance['checkpoint_registry_sha256'])
        for name, digest in provenance['source_files_sha256'].items():
            require_hash(source_code / name, digest)
        for identity in identities:
            require_hash(identity['path'], identity['compact_sha256'])
        for store, audit in zip(stores, store_audit):
            integrity = store.verify_integrity()
            if integrity != {key: audit[key] for key in integrity}:
                raise ValueError('PairStore changed during extraction')
            require_hash(audit['rows_csv'], audit['rows_sha256'])
        manifest = dict(schema_version=1, input_sha256=INPUT_SHA, row_ids_sha256=row_ids_sha(rows),
                        rows=len(rows), features=features, parity=parity,
                        interpretation='No miRNA task updates; public foundation and frozen structural supervision retained',
                        provenance_sha256=sha(output / 'provenance.json'),
                        backbone_provenance_sha256=sha(output / 'backbone_provenance.json'),
                        wall_seconds=time.monotonic() - start)
        write(output / 'features_manifest.json', manifest)
        return manifest
    finally:
        for store in stores:
            store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=REPO / 'data/sirna_oligo_random')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args(argv)
    try:
        manifest = run(args.config, args.data_root, args.output, args.device, args.batch_size)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))
    print(json.dumps(dict(status='complete', rows=manifest['rows'], manifest=str(args.output / 'features_manifest.json'))))


if __name__ == '__main__':
    main()
