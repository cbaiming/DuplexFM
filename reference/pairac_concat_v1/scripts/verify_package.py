"""Verify code-only package hashes; does not load checkpoints or contact servers."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT/'CODE_MANIFEST.json').read_text(encoding='utf-8'))
for name, expected in manifest['files_sha256'].items():
    p = ROOT/name
    if p.is_symlink() or not p.resolve().is_relative_to(ROOT):
        raise ValueError('Unsafe package path: '+name)
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError('Changed or incomplete code: '+name)
print('Verified', len(manifest['files_sha256']), 'code/documentation files.')
