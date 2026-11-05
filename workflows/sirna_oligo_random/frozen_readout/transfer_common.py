"""Experiment-specific helpers; unique module name prevents legacy path shadowing."""
import csv,hashlib,json,os,time,threading
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
_BACKUP_THREAD_LOCK=threading.RLock()
SETTINGS=('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix','wang2009','alas1_patent')
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def plain(x):
 if isinstance(x,np.ndarray):return plain(x.tolist())
 if isinstance(x,np.generic):return plain(x.item())
 if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [plain(v) for v in x]
 return x
def write(p,x,replace=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and not replace:raise FileExistsError(p)
 q=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with q.open('x',encoding='utf8') as f:json.dump(plain(x),f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def array(p,**arrays):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with q.open('xb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def table(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvsave(p,rr):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+f'.{os.getpid()}.tmp')
 with q.open('x',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(plain(rr));f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def checkfiles(root,files):
 for p,h in files.items():assert sha(Path(root)/p)==h,p
def load(name,p):
 import importlib.util
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def backup(folder,tag):
 with _BACKUP_THREAD_LOCK:
  with (ROOT/'backup.lock').open('a') as lease:
   if os.name!='nt':
    import fcntl
    fcntl.flock(lease,fcntl.LOCK_EX)
   try:return _backup_locked(folder,tag)
   finally:
    if os.name!='nt':fcntl.flock(lease,fcntl.LOCK_UN)
def _backup_locked(folder,tag):
 import shutil
 folder=Path(folder);objects=ROOT/'backup_objects';objects.mkdir(exist_ok=True);files={}
 for p in folder.rglob('*'):
  if not p.is_file() or p.suffix not in ('.json','.npz','.csv','.npy'):continue
  if p.name=='status.json':continue
  h=sha(p);q=objects/h
  if not q.exists():
   # Publish a complete object atomically; parallel workers may share a digest.
   temp=q.with_name(q.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
   with p.open('rb') as src,temp.open('xb') as dst:
    shutil.copyfileobj(src,dst);dst.flush();os.fsync(dst.fileno())
   assert sha(temp)==h
   try:os.replace(temp,q)
   except PermissionError:
    # Windows can deny replacing an identical object while another verifier
    # holds it open. Reuse only the already complete, independently checked file.
    assert q.is_file() and sha(q)==h
    temp.unlink()
  assert q.stat().st_nlink==1 and sha(q)==h
  files[p.relative_to(ROOT).as_posix()]={'sha256':h,'bytes':p.stat().st_size}
 write(ROOT/'backups'/f'{tag}.json',{'files':files,'second_readback_passed':True,'unix':time.time()})
def validate_inputs():
 rows=read(ROOT/'inputs/input_rows.json');plan=read(ROOT/'inputs/native_split_plan.json')
 assert sha(ROOT/'inputs/input_rows.json')=='a928975aa974a20bf3303ee46257336390a1e1a90b643a1d92960539c38c4980'
 assert sha(ROOT/'inputs/native_split_plan.json')=='d4e2ca306f7074ff643d5ec60dc0f6c72fe692e6ca60ec0211e701c7c0c4de5b'
 assert len(rows)==3688 and len({r['sample_id'] for r in rows})==3688 and len(plan['folds'])==26
 lut={r['sample_id']:i for i,r in enumerate(rows)}
 for s in SETTINGS:
  expected={r['sample_id'] for r in rows if r['cohort']==('Mixset' if s=='Inter_Hu_to_Mix' else s)}
  ids=[x for f in plan['folds'] if f['setting']==s for x in f['test_ids']]
  assert len(ids)==len(expected) and set(ids)==expected
 for f in plan['folds']:
  assert not set(f['fit_ids'])&set(f['test_ids']);cover=np.zeros(len(f['fit_ids']),int)
  for q in f['inner']:
   assert not set(q['train'])&set(q['valid']) and set(q['train']+q['valid'])==set(range(len(cover)))
   cover[q['valid']]+=1
  assert (cover==1).all()
 return rows,plan,lut
