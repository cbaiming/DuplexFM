"""Durable, checksummed checkpoints with one independent previous generation."""
from __future__ import annotations

import errno
import hashlib
import json
import os
import pickle
from pathlib import Path
import shutil
import time

from .common import canonical, read, sha


def sync_directory(path):
    if os.name == 'nt':
        return
    fd = os.open(str(path), os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        try:
            os.fsync(fd)
        except OSError as exc:
            if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise
    finally:
        os.close(fd)


def durable_json(path, value, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value)
    if immutable and path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f'Immutable execution artifact differs: {path}')
        return
    temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    with temp.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    sync_directory(path.parent)


def independent_copy(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    digest = sha(source)
    with source.open('rb') as src, temp.open('xb') as dst:
        shutil.copyfileobj(src, dst, 8 << 20)
        dst.flush()
        os.fsync(dst.fileno())
    if sha(temp) != digest or sha(source) != digest:
        raise ValueError('Unstable independent checkpoint copy')
    os.replace(temp, target)
    sync_directory(target.parent)
    return digest


def receipt_path(path):
    return Path(path).with_suffix('.commit.json')


def valid_checkpoint(path, expected_signature=None):
    import torch
    path = Path(path)
    receipt = read(receipt_path(path))
    if path.is_symlink() or path.stat().st_nlink != 1 or sha(path) != receipt['sha256']:
        raise ValueError(f'Checkpoint is not an independently committed object: {path}')
    value = torch.load(path, map_location='cpu', weights_only=True)
    signature_hash = hashlib.sha256(canonical(value['signature'])).hexdigest()
    if (signature_hash != receipt['signature_sha256'] or value['state']['step'] != receipt['step']
            or value['state']['last_validation_step'] != receipt['last_validation_step']):
        raise ValueError('Checkpoint receipt and state disagree')
    if expected_signature is not None and value['signature'] != expected_signature:
        raise ValueError('Checkpoint source/settings/input signature changed')
    if not all(k in value for k in ('optimizer', 'rng', 'trainable_parameters', 'fusion_buffers', 'state')):
        raise ValueError('Checkpoint is not a complete training state')
    return value


def save_checkpoint(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Rotate only a previously committed current generation. A torn publication
    # never replaces the last good fallback with unchecked bytes.
    if path.name == 'last.pt' and path.exists():
        committed = read(receipt_path(path))
        if sha(path) != committed['sha256']:
            raise ValueError('Current checkpoint changed; recover before continuing')
        previous = path.with_name('last.previous.pt')
        independent_copy(path, previous)
        durable_json(receipt_path(previous), committed)
    temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    with temp.open('xb') as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
    digest = sha(temp)
    os.replace(temp, path)
    sync_directory(path.parent)
    durable_json(receipt_path(path), {
        'sha256': digest,
        'signature_sha256': hashlib.sha256(canonical(value['signature'])).hexdigest(),
        'step': value['state']['step'], 'last_validation_step': value['state']['last_validation_step'],
        'published_unix': time.time(), 'file_fsynced': True,
        'includes': ['parameters', 'fusion_buffers', 'optimizer', 'python_numpy_torch_cuda_rng', 'data_cursor'],
    })


def backup_checkpoint(folder):
    folder = Path(folder)
    root = folder.parent.parent
    latest = read(folder / 'latest_backup.json')
    manifest_path = root / latest['manifest']
    if not manifest_path.resolve().is_relative_to(root.resolve()) or sha(manifest_path) != latest['sha256']:
        raise ValueError('Hourly backup manifest changed')
    manifest = read(manifest_path)['files']
    prefix = folder.relative_to(root).as_posix()
    paths = []
    for name in ('last.pt', 'last.commit.json'):
        entry = manifest[prefix + '/' + name]
        digest = entry['sha256']
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Unsafe backup object digest')
        obj = root / 'backups/objects' / digest
        if obj.is_symlink() or obj.stat().st_nlink != 1 or obj.stat().st_size != entry['size'] or sha(obj) != digest:
            raise ValueError('Hourly backup object changed')
        paths.append(obj)
    # Restore into a separate evidence directory first, without touching last.
    temp = folder / ('verified_hourly_' + str(time.time_ns()))
    temp.mkdir()
    independent_copy(paths[0], temp / 'last.pt')
    independent_copy(paths[1], temp / 'last.commit.json')
    valid_checkpoint(temp / 'last.pt', read(folder / 'signature.json'))
    return temp / 'last.pt'


def recover_checkpoint(folder):
    """Select a committed checkpoint; preserve damaged current bytes and receipt."""
    folder = Path(folder)
    expected = read(folder / 'signature.json')
    # A stop between initial.pt commit and its small metadata publication is
    # recoverable without regenerating or guessing any initialization bytes.
    initial = valid_checkpoint(folder / 'initial.pt', expected)
    if initial['state']['step'] != 0 or initial['optimizer'].get('state', {}):
        raise ValueError('Initial checkpoint was not saved before the first update')
    initial_record = {'sha256': sha(folder / 'initial.pt'), 'before_update': 0, 'exact_for_this_run': True}
    if (folder / 'initial.json').exists():
        if read(folder / 'initial.json') != initial_record:
            raise ValueError('Initial checkpoint identity metadata changed')
    else:
        durable_json(folder / 'initial.json', initial_record, immutable=True)
    errors, chosen = [], None
    for name in ('last.pt', 'last.previous.pt', 'hourly_backup', 'initial.pt'):
        if name == 'hourly_backup':
            if not (folder / 'latest_backup.json').exists():
                continue
            try:
                path = backup_checkpoint(folder)
                value = valid_checkpoint(path, expected)
                chosen = (path, value)
                break
            except (ValueError, OSError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError, json.JSONDecodeError) as exc:
                errors.append({'path': name, 'error': f'{type(exc).__name__}: {exc}'})
                continue
        path = folder / name
        if not path.exists():
            continue
        try:
            value = valid_checkpoint(path, expected)
            chosen = (path, value)
            break
        except (ValueError, OSError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError, json.JSONDecodeError) as exc:
            errors.append({'path': name, 'error': f'{type(exc).__name__}: {exc}'})
    if chosen is None:
        raise ValueError(f'No verified complete checkpoint; retained all evidence: {errors}')
    source, value = chosen
    if source != folder / 'last.pt':
        stamp = str(time.time_ns())
        for path in (folder / 'last.pt', receipt_path(folder / 'last.pt')):
            if path.exists():
                os.replace(path, path.with_name(path.name + '.retained_' + stamp))
        independent_copy(source, folder / 'last.pt')
        durable_json(receipt_path(folder / 'last.pt'), read(receipt_path(source)))
        sync_directory(folder)
    result = {'selected': source.relative_to(folder).as_posix(), 'step': value['state']['step'],
              'last_validation_step': value['state']['last_validation_step'],
              'sha256': sha(folder / 'last.pt'), 'prior_errors': errors}
    durable_json(folder / ('recovery_' + str(time.time_ns()) + '.json'), result, immutable=True)
    return result
