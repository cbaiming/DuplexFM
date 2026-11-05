"""Read-only release inventory/SHA checks; Python standard library only."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRS = {'.git', '__pycache__', '.pytest_cache', '.venv', 'venv', 'review_runs', 'reproduced'}
FORBIDDEN_SUFFIXES = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin', '.npy', '.npz', '.pkl', '.pem', '.key', '.zip', '.gz'}
TEXT_SUFFIXES = {'.py', '.md', '.json', '.txt', '.csv', '.sh', '.yml', '.yaml'}
MOCK_PATH_FILES = {
    'reference/pairac_concat_v1/execution_reference/test_backup_compat.py': '0629369ec8ec76315ead30481bf97d64995ecb577f7ebf6d716833c087d3a220',
    'reference/pairac_concat_v1/execution_reference/test_backup_compat_v2.py': 'c64cc4a4e18259d617570b291f0c1731ae5d7a08328c56aa6fb070617f31cb1e',
}

def safe_path(root, name):
    logical = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or logical.is_absolute() or '..' in logical.parts:
        raise ValueError('Unsafe manifest path: ' + name)
    path = root.joinpath(*logical.parts)
    if any(parent.is_symlink() for parent in [path, *path.parents] if parent != root.parent):
        raise ValueError('Symlink in release path: ' + name)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Path escapes release: ' + name)
    return path

def included_files(root):
    for path in root.rglob('*'):
        rel = path.relative_to(root)
        if IGNORED_DIRS.intersection(rel.parts):
            continue
        if path.is_symlink():
            raise ValueError('Symlink not allowed: ' + rel.as_posix())
        if path.is_file() and path != root / 'RELEASE_MANIFEST.json':
            yield path

def suspicious_text(text):
    # These are detection patterns, not example credentials or host settings.
    patterns = {
        'private_key': r'-----BEGIN ' + r'(?:OPENSSH|RSA|EC|DSA|ENCRYPTED) PRIVATE KEY-----',
        'github_token': r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b',
        'aws_access_key': r'\bAKIA[A-Z0-9]{16}\b',
        'windows_home': r'[A-Za-z]:[/\\]+Users[/\\]+[A-Za-z0-9_.-]+[/\\]',
        'linux_home': r'/(?:home|root)/[A-Za-z0-9_.-]+/',
        'private_storage': r'/mnt/afs/[A-Za-z0-9_.-]+',
        'tunnel_host': r'https?://[A-Za-z0-9-]+\.\w+\.devtunnels\.ms',
    }
    return sorted(name for name, pattern in patterns.items() if re.search(pattern, text))

def scan_file(relative, data):
    text = data.decode('utf-8-sig')
    if relative in MOCK_PATH_FILES and hashlib.sha256(data).hexdigest() == MOCK_PATH_FILES[relative]:
        # Exact immutable synthetic worker-command fixture, never a real server path.
        fixture = '/' + 'root' + '/leases/x.json'
        text = text.replace(fixture, '<synthetic-path>')
    return suspicious_text(text)

def verify(root):
    root = root.resolve()
    manifest = json.loads((root / 'RELEASE_MANIFEST.json').read_text('utf-8'))
    expected = manifest['files']
    names = [entry['path'] for entry in expected]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate paths in manifest')
    actual = {p.relative_to(root).as_posix(): p for p in included_files(root)}
    if set(names) != set(actual):
        raise ValueError('Inventory mismatch: missing=%s extra=%s' % (
            sorted(set(names) - set(actual)), sorted(set(actual) - set(names))))
    data_manifest = root / 'DATA_MANIFEST.json'
    data_paths = {r['path'] for r in json.loads(data_manifest.read_text())['files']} if data_manifest.exists() else set()
    total = 0
    for entry in expected:
        path = safe_path(root, entry['path'])
        data = path.read_bytes()
        if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('SHA/size mismatch: ' + entry['path'])
        declared_data = entry['path'].startswith('data/') and entry['path'] in data_paths
        if ((path.suffix.lower() in FORBIDDEN_SUFFIXES and not declared_data) or path.name.startswith(('.env', 'id_rsa', 'id_ed25519'))):
            raise ValueError('Private/large-asset file type not allowed: ' + entry['path'])
        if len(data) > (95_000_000 if declared_data else 10 * 1024 * 1024):
            raise ValueError('File exceeds declared release size limit: ' + entry['path'])
        if path.suffix.lower() in TEXT_SUFFIXES:
            findings = scan_file(entry['path'], data)
            if findings:
                raise ValueError('Potential private metadata: %s: %s' % (entry['path'], findings))
        total += len(data)
    return {'status': 'passed', 'files': len(names), 'bytes': total,
            'integrity': 'SHA256 and file size; exact declared inventory',
            'privacy': 'bounded text-pattern and excluded-file-type scan; not a license clearance',
            'scientific_scope': 'No model execution or row-level metric recomputation.'}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))

if __name__ == '__main__':
    main()
