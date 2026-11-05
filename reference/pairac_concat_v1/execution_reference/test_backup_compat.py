import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import threading

import pytest
import backup_compat as repair
import parallel_compat_run as entry
from duplexfm_mechanism import common, training_hourly, parallel, evaluation


def write_input(root,name='data.txt',payload=b'unchanged bytes'):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload);return p


def inspect(root,receipt):
    manifest=root/receipt['manifest']
    assert repair._sha(manifest)==receipt['sha256']
    rows=json.loads(manifest.read_text())
    for item in rows['files'].values():
        obj=root/'backups/objects'/item['sha256']
        assert repair._sha(obj)==item['sha256'] and obj.stat().st_size==item['size'] and obj.stat().st_nlink==1
    return rows


def test_old_error_reproduced_new_bounded_retry_succeeds(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('duplexfm_mechanism.old_common_fixture',common.__file__)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    p=write_input(tmp_path);original=old.sha
    def old_changing_sha(path):
        result=original(path)
        if Path(path)==p:
            s=p.stat();os.utime(p,ns=(s.st_atime_ns,s.st_mtime_ns+1000000000))
        return result
    monkeypatch.setattr(old,'sha',old_changing_sha)
    with pytest.raises(ValueError,match='Unstable backup input'):old.backup(tmp_path,[p],'original')
    actual=repair._copy_open_file;calls=[]
    def touch_once(source,target):
        value=actual(source,target);calls.append(source.name)
        if len(calls)==1:
            s=p.stat();os.utime(p,ns=(s.st_atime_ns,s.st_mtime_ns+1000000000))
        return value
    monkeypatch.setattr(repair,'_copy_open_file',touch_once)
    receipt=repair.backup(tmp_path,[p],'repaired')
    m=inspect(tmp_path,receipt)
    assert len(calls)==2 and receipt['snapshot_retry_count']==1
    assert m['snapshot_retries'][0]['path']=='data.txt'
    assert not list((tmp_path/'backups/objects').glob('.snapshot*'))


def test_persistent_instability_preserves_previous_manifest(tmp_path,monkeypatch):
    p=write_input(tmp_path);previous=repair.backup(tmp_path,[p],'good')
    before=(tmp_path/'backups/latest.json').read_bytes();actual=repair._copy_open_file;calls=[]
    def changing(source,target):
        value=actual(source,target);calls.append(1)
        s=p.stat();os.utime(p,ns=(s.st_atime_ns,s.st_mtime_ns+1000000000));return value
    monkeypatch.setattr(repair,'_copy_open_file',changing)
    with pytest.raises(repair.SnapshotChanged,match='data.txt'):repair.backup(tmp_path,[p],'bad')
    assert len(calls)==3 and (tmp_path/'backups/latest.json').read_bytes()==before
    assert len(list((tmp_path/'backups/failures').glob('*.json')))==1
    assert not list((tmp_path/'backups/objects').glob('.snapshot*'))
    inspect(tmp_path,previous)


def test_atomic_replace_cannot_mix_versions(tmp_path,monkeypatch):
    p=write_input(tmp_path,payload=b'version A');actual=repair._copy_open_file;calls=[]
    original_stat=Path.stat;armed=[False]
    def replace_once(source,target):
        value=actual(source,target);calls.append(1)
        if len(calls)==1:armed[0]=True
        return value
    def replace_after_close(path,*args,**kwargs):
        # Windows forbids replacing an open file. Mutate at the first pathname
        # stat after closing the original descriptor; Linux uses the same fixture.
        if path==p and armed[0]:
            armed[0]=False
            replacement=tmp_path/'replacement';replacement.write_bytes(b'version B');os.replace(replacement,p)
        return original_stat(path,*args,**kwargs)
    monkeypatch.setattr(repair,'_copy_open_file',replace_once)
    monkeypatch.setattr(Path,'stat',replace_after_close)
    r=repair.backup(tmp_path,[p],'atomic');m=inspect(tmp_path,r)
    assert r['snapshot_retry_count']==1 and m['files']['data.txt']['sha256']==hashlib.sha256(b'version B').hexdigest()


def test_same_size_byte_mutation_detected(tmp_path,monkeypatch):
    p=write_input(tmp_path,payload=b'AAAA');actual=repair._copy_open_file;calls=[]
    def change_preserve_mtime(source,target):
        before=p.stat();value=actual(source,target);calls.append(1)
        if len(calls)==1:
            p.write_bytes(b'BBBB');os.utime(p,ns=(before.st_atime_ns,before.st_mtime_ns))
        return value
    monkeypatch.setattr(repair,'_copy_open_file',change_preserve_mtime)
    r=repair.backup(tmp_path,[p],'bytes');m=inspect(tmp_path,r)
    assert r['snapshot_retry_count']==1 and m['files']['data.txt']['sha256']==repair._sha(p)


def test_corrupted_existing_object_is_not_silently_replaced(tmp_path):
    p=write_input(tmp_path);r=repair.backup(tmp_path,[p],'good')
    obj=tmp_path/'backups/objects'/repair._sha(p);obj.write_bytes(b'corrupt')
    before=(tmp_path/'backups/latest.json').read_bytes()
    with pytest.raises(ValueError,match='integrity mismatch'):repair.backup(tmp_path,[p],'reject')
    assert obj.read_bytes()==b'corrupt' and (tmp_path/'backups/latest.json').read_bytes()==before


def test_concurrent_backups_publish_valid_independent_objects(tmp_path):
    p=write_input(tmp_path,payload=b'shared'*1000);barrier=threading.Barrier(2)
    def task(label):barrier.wait(timeout=10);return repair.backup(tmp_path,[p],label)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a,b=[x.result(timeout=20) for x in [pool.submit(task,'worker_a'),pool.submit(task,'worker_b')]]
    inspect(tmp_path,a);inspect(tmp_path,b)
    assert a['manifest']!=b['manifest'] and len(list((tmp_path/'backups/objects').iterdir()))==1


def test_escape_duplicate_label_rejected(tmp_path):
    p=write_input(tmp_path);outside=tmp_path.parent/'outside.txt';outside.write_bytes(b'x')
    with pytest.raises(ValueError):repair.backup(tmp_path,[outside],'escape')
    with pytest.raises(ValueError,match='Duplicate'):repair.backup(tmp_path,[p,p],'duplicate')
    with pytest.raises(ValueError,match='Unsafe'):repair.backup(tmp_path,[p],'../escape')


def test_symlink_input_rejected(tmp_path):
    if os.name=='nt':
        # No symlink privileges are required for this safety regression.
        from unittest.mock import patch
        p=write_input(tmp_path)
        with patch.object(Path,'is_symlink',lambda x: x==p):
            with pytest.raises(ValueError,match='Symlink'):repair.backup(tmp_path,[p],'symlink')
    else:
        p=write_input(tmp_path);s=tmp_path/'symlink';s.symlink_to(p)
        with pytest.raises(ValueError,match='Symlink'):repair.backup(tmp_path,[s],'symlink')


@pytest.mark.parametrize('mismatch',['checkpoint','signature','missing'])
def test_checkpoint_commit_pair_rejects_inconsistent_snapshot(tmp_path,mismatch):
    cp=write_input(tmp_path,'runs/a/last.pt',b'checkpoint');sig=write_input(tmp_path,'runs/a/signature.json',b'{}')
    commit=write_input(tmp_path,'runs/a/last.commit.json',json.dumps({'sha256':repair._sha(cp),
       'signature_sha256':repair._sha(sig)}).encode())
    row=json.loads(commit.read_text())
    if mismatch!='missing':row['sha256' if mismatch=='checkpoint' else 'signature_sha256']='0'*64;commit.write_text(json.dumps(row))
    paths=[cp,sig] if mismatch=='missing' else [cp,sig,commit]
    with pytest.raises(ValueError,match='checkpoint|Checkpoint'):repair.backup(tmp_path,paths,'badpair')
    assert not (tmp_path/'backups/latest.json').exists()


def test_checkpoint_pair_valid_and_backup_does_not_consume_rng(tmp_path):
    cp=write_input(tmp_path,'runs/a/last.pt',b'checkpoint');sig=write_input(tmp_path,'runs/a/signature.json',b'{}')
    commit=write_input(tmp_path,'runs/a/last.commit.json',json.dumps({'sha256':repair._sha(cp),
       'signature_sha256':repair._sha(sig)}).encode())
    before=random.getstate();r=repair.backup(tmp_path,[cp,sig,commit],'pair');inspect(tmp_path,r)
    assert random.getstate()==before


def test_all_execution_aliases_and_worker_entry_use_repair():
    repair.install()
    assert all(m.backup is repair.backup for m in (common,training_hourly,parallel,evaluation))
    args=entry.worker_command('/root','/assets','/root/leases/x.json')
    assert Path(args[2]).name=='parallel_compat_run.py' and args[3]=='worker' and args[-1]=='/root/leases/x.json'


def test_execution_overlay_is_included_in_later_snapshots(tmp_path):
    p=write_input(tmp_path);extra=write_input(tmp_path,'execution/fix/patch.py',b'patch')
    repair.install([extra])
    try:
        receipt=repair.backup(tmp_path,[p],'includes_overlay')
        assert 'execution/fix/patch.py' in inspect(tmp_path,receipt)['files']
    finally:
        repair.install()


def test_overlay_tamper_and_unready_rejected(tmp_path):
    root=tmp_path/'root';folder=root/'execution/fix';folder.mkdir(parents=True)
    write_input(root,'code/original.py',b'original')
    write_input(root,'release_manifest.json',json.dumps({'files_sha256':{'code/original.py':repair._sha(root/'code/original.py')}}).encode())
    write_input(folder,'repair.py',b'patch');write_input(folder,'accept.json',b'{}')
    lock={'version':'backup-compat-v1-20261019','scientific_change':False,'remote_root':str(root.resolve()),
       'original_files_sha256':{'release_manifest.json':repair._sha(root/'release_manifest.json')},
       'repair_files_sha256':{'repair.py':repair._sha(folder/'repair.py')}}
    write_input(folder,'compat_lock.json',json.dumps(lock).encode())
    ready={'status':'passed','compat_lock_sha256':repair._sha(folder/'compat_lock.json'),
      'files_sha256':{'accept.json':repair._sha(folder/'accept.json')}}
    write_input(folder,'repair_ready.json',json.dumps(ready).encode())
    assert entry.verify_overlay(root,folder)==lock
    (folder/'repair.py').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Repair code changed'):entry.verify_overlay(root,folder)
    (folder/'repair.py').write_bytes(b'patch');ready['status']='pending';(folder/'repair_ready.json').write_text(json.dumps(ready))
    with pytest.raises(ValueError,match='not passed'):entry.verify_overlay(root,folder)
