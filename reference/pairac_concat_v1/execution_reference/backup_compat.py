"""Execution-only stable backup snapshots; original scientific files stay locked."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import time

VERSION = 'stable-backup-v1-20261019'
MAX_ATTEMPTS = 3
_THREAD_LOCK = threading.RLock()
_EXTRA_PATHS = ()


class SnapshotChanged(ValueError):
    def __init__(self, path, detail):
        self.path, self.detail = str(path), detail
        super().__init__(f'Unstable backup input {path}: {detail}')


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


@contextmanager
def _backup_lock(root):
    """Serialize backup publication only, not training or checkpoint updates."""
    path = root / 'backups/backup_compat.lock'
    if path.is_symlink():
        raise ValueError('Backup lock may not be a symlink')
    with _THREAD_LOCK, path.open('a+b') as f:
        if os.name == 'nt':
            import msvcrt
            if f.seek(0, 2) == 0:
                f.write(b'0'); f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == 'nt':
                f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _copy_open_file(source, target):
    digest, size = hashlib.sha256(), 0
    for block in iter(lambda: source.read(8 << 20), b''):
        target.write(block); digest.update(block); size += len(block)
    return digest.hexdigest(), size


def _capture(path, temp, relative):
    if path.is_symlink():
        raise ValueError(f'Symlink backup input: {relative}')
    with path.open('rb') as source, temp.open('xb') as target:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f'Nonregular backup input: {relative}')
        digest, size = _copy_open_file(source, target)
        target.flush(); os.fsync(target.fileno())
        after_fd = os.fstat(source.fileno())
    try:
        after_path = path.stat()
    except FileNotFoundError as exc:
        raise SnapshotChanged(relative, {'reason': 'source disappeared'}) from exc
    if path.is_symlink():
        raise ValueError(f'Input replaced by symlink: {relative}')
    identities = [_identity(s) for s in (before, after_fd, after_path)]
    if len(set(identities)) != 1 or size != before.st_size:
        raise SnapshotChanged(relative, {'reason': 'metadata changed', 'identities': identities, 'copied_bytes': size})
    # A second source read catches same-size changes; do not weaken SHA checks.
    source_digest = _sha(path)
    last = path.stat()
    if path.is_symlink() or _identity(last) != identities[0] or source_digest != digest:
        raise SnapshotChanged(relative, {'reason': 'source reread changed', 'before': identities[0],
                                        'after': _identity(last), 'first_sha': digest, 'second_sha': source_digest})
    if _sha(temp) != digest:
        raise ValueError(f'Independent snapshot readback mismatch: {relative}')
    return digest, size


def _verify_object(path, digest, size):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Invalid backup object {path.name}')
    before = path.stat()
    if before.st_nlink != 1 or before.st_size != size or _sha(path) != digest:
        raise ValueError(f'Backup object integrity mismatch: {path.name}')
    if _identity(before) != _identity(path.stat()):
        raise ValueError(f'Backup object changed during readback: {path.name}')


def _verify_checkpoint_pairs(entries, objects):
    """Never publish an inconsistent checkpoint/commit/signature combination."""
    known = {'initial.pt', 'last.pt', 'last.previous.pt', 'terminal.pt'}
    for relative, entry in entries.items():
        p = Path(relative)
        if p.name not in known:
            continue
        commit = p.with_suffix('.commit.json').as_posix()
        signature = (p.parent / 'signature.json').as_posix()
        if commit not in entries or signature not in entries:
            raise ValueError(f'Missing checkpoint commit/signature in backup: {relative}')
        record = json.loads((objects / entries[commit]['sha256']).read_text())
        if record['sha256'] != entry['sha256'] or record['signature_sha256'] != entries[signature]['sha256']:
            raise ValueError(f'Checkpoint commit/signature mismatch in backup: {relative}')


def backup(root, paths, label):
    from duplexfm_mechanism.common import check_disk, write
    from duplexfm_mechanism.durable_io import sync_directory
    root = Path(root).resolve(strict=True)
    paths = list(paths)
    existing = {Path(p).absolute() for p in paths}
    paths += [p for p in _EXTRA_PATHS if p.absolute() not in existing]
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', label):
        raise ValueError('Unsafe backup label')
    check_disk(root)
    objects = root / 'backups/objects'; objects.mkdir(parents=True, exist_ok=True)
    if objects.is_symlink() or not objects.resolve().is_relative_to(root):
        raise ValueError('Unsafe backup object directory')
    entries, retries = {}, []
    with _backup_lock(root):
        for input_path in paths:
            raw = Path(input_path)
            if raw.is_symlink():
                raise ValueError(f'Symlink backup input: {raw}')
            path = raw.resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file() or 'backups' in path.relative_to(root).parts:
                raise ValueError('Only regular task inputs outside backups may be snapshotted')
            relative = path.relative_to(root).as_posix()
            if relative in entries:
                raise ValueError(f'Duplicate backup input: {relative}')
            for attempt in range(1, MAX_ATTEMPTS + 1):
                temp = objects / f'.snapshot.{os.getpid()}.{time.time_ns()}.tmp'
                captured = False
                try:
                    digest, size = _capture(path, temp, relative)
                    captured = True
                    break
                except SnapshotChanged as exc:
                    row = {'path': relative, 'attempt': attempt, 'detail': exc.detail}
                    retries.append(row)
                    if attempt == MAX_ATTEMPTS:
                        failure = root / 'backups/failures' / f'{time.time_ns()}_{label}.json'
                        write(failure, {'compat_version': VERSION, 'status': 'failed', 'retries': retries}, immutable=True)
                        raise SnapshotChanged(relative, {'reason': 'retry limit reached', 'attempts': MAX_ATTEMPTS,
                                                         'evidence': str(failure.relative_to(root))}) from exc
                finally:
                    # Only this invocation's unpublished temporary file is cleaned.
                    if temp.exists() and not captured:
                        temp.unlink()
            dest = objects / digest
            try:
                if dest.exists():
                    _verify_object(dest, digest, size)
                else:
                    os.replace(temp, dest); sync_directory(objects)
                _verify_object(dest, digest, size)
                entries[relative] = {'sha256': digest, 'size': size}
            finally:
                if temp.exists():temp.unlink()
                del digest, size
        _verify_checkpoint_pairs(entries, objects)
        manifest = root / 'backups/manifests' / f'{time.time_ns()}_{label}.json'
        write(manifest, {'files': entries, 'scope': 'task files only; large assets referenced, not copied',
                         'independent_readback': True, 'compat_version': VERSION,
                         'exclusive_backup_publication': True, 'snapshot_retries': retries}, immutable=True)
        receipt = {'manifest': manifest.relative_to(root).as_posix(), 'sha256': _sha(manifest),
                   'objects': len(entries), 'time_unix': time.time(), 'compat_version': VERSION,
                   'snapshot_retry_count': len(retries)}
        write(root / 'backups/latest.json', receipt)
        return receipt


def install(extra_paths=()):
    """Patch only execution-time backup aliases; keep original source files intact."""
    global _EXTRA_PATHS
    _EXTRA_PATHS = tuple(Path(p) for p in extra_paths)
    from duplexfm_mechanism import common, training_hourly, parallel, evaluation
    for module in (common, training_hourly, parallel, evaluation):
        module.backup = backup
