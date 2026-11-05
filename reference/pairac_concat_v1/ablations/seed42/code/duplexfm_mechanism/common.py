from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def read(path):
    def unique(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"Duplicate JSON key: {key}")
            out[key] = value
        return out
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique)


def write(path, value, *, immutable=False):
    path = Path(path)
    payload = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"Immutable artifact differs: {path}")
        return
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    from .durable_io import sync_directory
    sync_directory(path.parent)


def safe_relative(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or "\x00" in name or name.startswith("/")
            or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError(f"Unsafe relative name: {name!r}")
    return Path(name)


def verify_files(root, files):
    root = Path(root).resolve()
    if not files:
        raise ValueError("Empty file manifest")
    for name, expected in files.items():
        path = root / safe_relative(name)
        if path.is_symlink() or not path.resolve().is_relative_to(root) or sha(path) != expected:
            raise ValueError(f"File identity differs: {name}")
    return len(files)


def check_disk(root, minimum_gib=10):
    if shutil.disk_usage(root).free < minimum_gib * 2**30:
        raise RuntimeError(f"Less than {minimum_gib} GiB free; refusing further work")


class FileLock:
    """OS-backed nonblocking lock; released by the OS if the owner dies."""
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                if self.path.stat().st_size == 0:
                    self.handle.write(b"0")
                    self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"Another controller holds {self.path}") from None
        return self

    def __exit__(self, *args):
        if self.handle is not None:
            if os.name == "nt":
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            self.handle.close()


def backup(root, paths, label):
    """Independent copies, content-addressed deduplication, second-read SHA check."""
    root = Path(root).resolve()
    check_disk(root)
    objects = root / "backups/objects"
    objects.mkdir(parents=True, exist_ok=True)
    entries = {}
    for path in paths:
        path = Path(path).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Only this experiment's regular files may be backed up")
        before = path.stat()
        digest = sha(path)
        dest = objects / digest
        if not dest.exists():
            tmp = objects / (digest + f".{os.getpid()}.{time.time_ns()}.tmp")
            with path.open('rb') as source, tmp.open('xb') as target:
                shutil.copyfileobj(source,target,8 << 20)
                target.flush()
                os.fsync(target.fileno())
            if sha(tmp) != digest:
                raise ValueError("Source changed during backup")
            os.replace(tmp, dest)
            from .durable_io import sync_directory
            sync_directory(objects)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Unstable backup input")
        if dest.is_symlink() or dest.stat().st_nlink != 1 or sha(dest) != digest:
            raise ValueError("Backup is not an independently verified regular object")
        entries[path.relative_to(root).as_posix()] = {"sha256": digest, "size": after.st_size}
    name = f"{time.time_ns()}_{label}.json"
    manifest = root / "backups/manifests" / name
    write(manifest, {"files": entries, "scope": "task files only; large assets referenced, not copied",
                     "independent_readback": True}, immutable=True)
    receipt = {"manifest": manifest.relative_to(root).as_posix(), "sha256": sha(manifest),
               "objects": len(entries), "time_unix": time.time()}
    write(root / "backups/latest.json", receipt)
    return receipt
