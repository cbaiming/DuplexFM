"""Rebuild the exact PairAC miRBench token/maps/splits from pinned official gzip TSVs.

No download, source training, GPU or physics cache is used. A complete PairStore
is published only after the separately supplied historical reactivity cache
matches its exact archived SHA-256. Tokens-only output is explicitly incomplete.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'workflows/mirbench'))
import cleaning


def sha(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
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


def encode(sequences, width):
    """Original uint8 token encoding and right tail PAD, without normalization."""
    lut = np.full(256, 4, dtype=np.uint8)
    for index, base in enumerate('ACGU'):
        lut[ord(base)] = index
    values = np.full((len(sequences), width), 4, dtype=np.uint8)
    for index, sequence in enumerate(sequences):
        if not sequence or len(sequence) > width or set(sequence) - set('ACGU'):
            raise ValueError('Noncanonical or oversized real sequence')
        values[index, :len(sequence)] = lut[np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)]
    return values


def load_frame(path, name, expected=None):
    if expected is not None:
        if sha(path, 'md5') != expected['md5'] or sha(path) != expected['sha256']:
            raise ValueError('Official compressed TSV fingerprint differs: ' + name)
    frame, audit = cleaning.load_clean_pair_dataframe(path, 'all', 'noncodingRNA_fam', 200000, False, name)
    if expected is not None and (audit['raw_rows'] != expected['rows'] or len(frame) != expected['rows']):
        raise ValueError('Official row count/cleaning differs: ' + name)
    return frame, audit


def build_arrays(frames, order, split, destination):
    """Keep original row order and first-occurrence factorization exactly."""
    train_name = order[0]
    merged, merge_audit = cleaning.merge_training_dataframes([frames[train_name]], 'noncodingRNA_fam')
    assignment = {row['noncodingRNA_fam']: row['split'] for row in split['group_assignments']}
    if len(assignment) != len(split['group_assignments']) or set(assignment.values()) - {'train', 'val'}:
        raise ValueError('Invalid fixed family assignments')
    merged['partition'] = merged.noncodingRNA_fam.map(assignment)
    if merged.partition.isna().any():
        raise ValueError('Official training family missing from fixed assignments')
    merged['dataset'] = train_name
    merged['row_id'] = np.arange(len(merged))
    all_frames = [merged]
    for name in order[1:]:
        frame = frames[name].copy()
        frame['dataset'], frame['partition'], frame['row_id'] = name, name, np.arange(len(frame))
        all_frames.append(frame)
    table = pd.concat(all_frames, ignore_index=True)
    if not table.gene.str.fullmatch('[ACGU]+').all() or not table.noncodingRNA.str.fullmatch('[ACGU]+').all():
        raise ValueError('Official cleaned sequences must be canonical RNA')
    if set(table.gene.str.len()) != {50} or not table.label.isin((0, 1)).all():
        raise ValueError('Historical source requires target50 and binary labels')
    target_ids, targets = pd.factorize(table.gene, sort=False)
    guide_ids, guides = pd.factorize(table.noncodingRNA, sort=False)
    data = destination / 'data'
    data.mkdir(parents=True, exist_ok=False)
    arrays = {'target_tokens': encode(targets, 50), 'guide_tokens': encode(guides, int(table.noncodingRNA.str.len().max())),
              'guide_lengths': np.asarray([len(sequence) for sequence in guides], dtype=np.int16),
              'target_ids': target_ids.astype(np.int32), 'guide_ids': guide_ids.astype(np.int32),
              'labels': table.label.to_numpy(np.float32)}
    for name, value in arrays.items():
        np.save(data / (name + '.npy'), value, allow_pickle=False)
    partitions = {}
    for name in ['train', 'val', *order[1:]]:
        indices = np.flatnonzero(table.partition.to_numpy() == name)
        np.save(data / ('indices_' + name + '.npy'), indices, allow_pickle=False)
        partitions[name] = {'n': len(indices), 'positive': int(table.iloc[indices].label.sum())}
    # These files support provenance and future reactivity regeneration. No physics.
    pd.Series(targets).to_csv(destination / 'target_sequences.txt', index=False, header=False)
    pd.Series(guides).to_csv(destination / 'guide_sequences.txt', index=False, header=False)
    files = {path.relative_to(destination).as_posix(): sha(path) for path in data.glob('*.npy')}
    return {'rows': len(table), 'unique_targets': len(targets), 'unique_guides': len(guides),
            'partitions': partitions, 'files_sha256': files, 'merge_audit': merge_audit,
            'target_sequences_sha256': sha(destination / 'target_sequences.txt'),
            'guide_sequences_sha256': sha(destination / 'guide_sequences.txt')}


def verify_arrays(output, expected):
    for name, digest in expected['files_sha256'].items():
        if name == 'accessibility.npy':
            continue
        path = output / name
        if not path.is_file() or sha(path) != digest:
            raise ValueError('Rebuilt historical array SHA-256 differs: ' + name)


def attach_accessibility(output, cache, historical):
    receipt = read(output / 'tokens_complete.json')
    verify_arrays(output, historical)
    if receipt['historical_array_verification'] != 'all_twelve_token_map_split_arrays_exact':
        raise ValueError('Token preparation was not qualified for the historical dataset')
    expected = historical['files_sha256']['accessibility.npy']
    if sha(cache) != expected:
        raise ValueError('Historical reactivity cache SHA-256 differs; no zero/substitute cache is accepted')
    target = np.load(output / 'data/target_tokens.npy', mmap_mode='r', allow_pickle=False)
    accessibility = np.load(cache, mmap_mode='r', allow_pickle=False)
    if accessibility.shape != target.shape or accessibility.dtype != np.float32:
        raise ValueError('Reactivity cache must be float32 aligned to unique target50 rows')
    for start in range(0, len(accessibility), 32768):
        block = accessibility[start:start + 32768]
        if not np.isfinite(block).all() or np.any((block < 0) | (block > 1)):
            raise ValueError('Invalid reactivity cache values')
    del accessibility, target
    shutil.copyfile(cache, output / 'accessibility.npy')
    if sha(output / 'accessibility.npy') != expected:
        raise ValueError('Reactivity cache copy verification failed')
    manifest = {'format_version': 1, 'target_padding': 'collate_only_center', 'target_width': 60,
                'token_encoding': {'A': 0, 'C': 1, 'G': 2, 'U': 3, 'PAD': 4},
                'rows': receipt['rows'], 'source_keep_overlap': True,
                'sample_order_and_split_policy': 'Original cleaner, first-occurrence factorization and fixed family assignments',
                'files_sha256': historical['files_sha256'], 'original_manifest_sha256': historical['manifest_sha256'],
                'public_manifest_is_new_container': True}
    write(output / 'manifest.json', manifest)
    sys.path.insert(0, str(ROOT / 'reference/pairac_concat_v1/source_full/code'))
    from duplexfm_joint.data import PairStore
    with PairStore(output) as store:
        result = store.verify_integrity()
    write(output / 'complete.json', {'status': 'complete', 'PairStore_ready': True, 'verified_files': result['verified_files'],
                                    'manifest_sha256': sha(output / 'manifest.json'), 'source_keep_overlap': True})
    return {'status': 'complete', 'PairStore_ready': True, 'verified_files': result['verified_files']}


def prepare(raw, output, root=ROOT, tokens_only=False, accessibility=None):
    if output.exists():
        raise FileExistsError('Output already exists; use a new directory or --complete-existing')
    if not tokens_only and accessibility is None:
        raise ValueError('Supply the exact --accessibility-cache, or explicitly choose --tokens-only')
    contracts = root / 'data/mirbench_v7'
    contract, historical = read(contracts / 'preparation_contract.json'), read(contracts / 'historical_inputs.json')
    split_path = contracts / 'fixed_split_manifest.json'
    if sha(split_path) != contract['fixed_split_sha256']:
        raise ValueError('Original family-assignment file changed')
    frames, audits = {}, {}
    for name in contract['official_file_order']:
        path = raw / (name + '.tsv.gz')
        frames[name], audits[name] = load_frame(path, name, contract['files'][name])
    output.mkdir(parents=True, exist_ok=False)
    result = build_arrays(frames, contract['official_file_order'], read(split_path), output)
    for key in ('rows', 'unique_targets', 'unique_guides'):
        if result[key] != contract[key]:
            raise ValueError('Rebuilt historical data dimensions differ: ' + key)
    for name, record in contract['partitions'].items():
        if result['partitions'][name] != {key: record[key] for key in ('n', 'positive')}:
            raise ValueError('Rebuilt partition membership/count differs: ' + name)
    verify_arrays(output, historical)
    write(output / 'tokens_complete.json', {**result, 'status': 'tokens_only_incomplete', 'PairStore_ready': False,
          'historical_array_verification': 'all_twelve_token_map_split_arrays_exact',
          'fixed_split_sha256': sha(split_path), 'cleaner_original_module_sha256': contract['canonical_cleaner_sha256'],
          'missing': ['exact separately frozen accessibility.npy'], 'no_physics': True})
    if accessibility is not None:
        return attach_accessibility(output, accessibility, historical)
    return {'status': 'tokens_only_incomplete', 'PairStore_ready': False, 'verified_token_arrays': 12}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, help='Directory containing the five pinned official .tsv.gz files')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens-only', action='store_true', help='Explicitly produce incomplete token/maps/splits without a reactivity cache')
    parser.add_argument('--accessibility-cache', type=Path, help='Exact historical cache supplied by user; never downloaded')
    parser.add_argument('--complete-existing', action='store_true', help='Attach exact cache to a previously qualified tokens-only directory')
    args = parser.parse_args(argv)
    if args.tokens_only and args.accessibility_cache is not None:
        parser.error('--tokens-only and --accessibility-cache are mutually exclusive')
    if args.complete_existing:
        if args.accessibility_cache is None or args.tokens_only:
            parser.error('--complete-existing requires --accessibility-cache')
    elif args.raw_dir is None:
        parser.error('--raw-dir is required for reconstruction')
    try:
        output = args.output.resolve()
        if args.complete_existing:
            if (output / 'manifest.json').exists() or (output / 'complete.json').exists() or (output / 'accessibility.npy').exists():
                raise FileExistsError('Existing complete/cache files will not be overwritten')
            result = attach_accessibility(output, args.accessibility_cache.resolve(), read(ROOT / 'data/mirbench_v7/historical_inputs.json'))
        else:
            result = prepare(args.raw_dir.resolve(), output, tokens_only=args.tokens_only,
                             accessibility=args.accessibility_cache.resolve() if args.accessibility_cache else None)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, AssertionError) as error:
        print('miRBench preparation failed: ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
