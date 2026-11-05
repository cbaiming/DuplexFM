"""Audited backup-only execution overlay for original locked PairAC checkpoints."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def verify_overlay(root, folder=HERE):
    root,folder=Path(root).resolve(),Path(folder).resolve()
    lock=json.loads((folder/'compat_lock.json').read_text())
    if lock['version']!='backup-compat-v1-20261019' or lock['scientific_change'] is not False:
        raise ValueError('Unknown repair contract')
    if str(root)!=lock['remote_root']:
        raise ValueError('Repair applies only to its original execution root')
    for name,digest in lock['original_files_sha256'].items():
        p=root/name
        if p.is_symlink() or not p.resolve().is_relative_to(root) or sha(p)!=digest:
            raise ValueError('Original lock changed: '+name)
    for name,digest in lock['repair_files_sha256'].items():
        p=folder/name
        if p.is_symlink() or not p.resolve().is_relative_to(folder) or sha(p)!=digest:
            raise ValueError('Repair code changed: '+name)
    release=json.loads((root/'release_manifest.json').read_text())
    for name,digest in release['files_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Original release code changed: '+name)
    ready=json.loads((folder/'repair_ready.json').read_text())
    if ready['status']!='passed' or ready['compat_lock_sha256']!=sha(folder/'compat_lock.json'):
        raise ValueError('Repair has not passed server CPU acceptance')
    for name,digest in ready['files_sha256'].items():
        if sha(folder/name)!=digest:raise ValueError('Repair acceptance receipt changed: '+name)
    return lock


def worker_command(root,assets,lease):
    return [sys.executable,'-u',str(HERE/'parallel_compat_run.py'),'worker',
            '--root',str(root),'--assets',str(assets),'--lease',str(lease)]


def main():
    if len(sys.argv)<2 or sys.argv[1] not in ('run','worker','recover','status'):
        raise ValueError('Only existing execution/resume actions are supported')
    if '--root' not in sys.argv:raise ValueError('Explicit original root required')
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    lock=verify_overlay(root)
    sys.path.insert(0,str(HERE));sys.path.insert(0,str(root/'code'))
    import backup_compat
    from duplexfm_mechanism import parallel
    from duplexfm_mechanism.durable_io import durable_json
    parallel.worker_command=worker_command
    receipt={'action':sys.argv[1],'pid':os.getpid(),'time_unix':time.time(),
             'compat_lock_sha256':sha(HERE/'compat_lock.json'),
             'original_release_sha256':sha(root/'release_manifest.json'),
             'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
             'scope':'backup I/O only; same original signatures, parameters, optimizer and cursor'}
    invocation=HERE/'invocations'/f'{time.time_ns()}_{os.getpid()}.json'
    durable_json(invocation,receipt,immutable=True)
    ready=json.loads((HERE/'repair_ready.json').read_text())
    extras={HERE/n for n in lock['repair_files_sha256']} | {HERE/'compat_lock.json',HERE/'repair_ready.json',invocation}
    extras |= {HERE/n for n in ready['files_sha256']}
    backup_compat.install(sorted(extras))
    spec=importlib.util.spec_from_file_location('pairac_original_parallel_entry',root/'code/parallel_run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.main()


if __name__=='__main__':
    raise SystemExit(main())
