import csv,gzip,hashlib,io,json,os,shutil,time
from pathlib import Path
import numpy as np
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def atom(p,b):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp')
 with q.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def write(p,v,exclusive=False):
 if exclusive and Path(p).exists():raise FileExistsError(p)
 atom(p,(json.dumps(v,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode())
def npz(p,**kw):
 b=io.BytesIO();np.savez_compressed(b,**kw);atom(p,b.getvalue())
def rows(root,name):
 with gzip.open(Path(root)/'prepared'/f'{name}.csv.gz','rt',encoding='utf-8',newline='') as f:r=list(csv.DictReader(f))
 for x in r:x['anchor']=int(x['anchor']);x['label']=float(x['label']);x['mismatches']=int(x['mismatches'])
 return r
def space(root):
 n=shutil.disk_usage(root).free
 if n<15*(1<<30):raise RuntimeError('Free disk below 15 GiB; nothing deleted')
 return n
def verify(root,files):
 for n,v in files.items():
  h=v['sha256'] if isinstance(v,dict) else v
  p=(Path(root)/n).resolve()
  if not p.is_relative_to(Path(root).resolve()) or sha(p)!=h:raise ValueError('SHA/path mismatch '+n)
def check(root):
 lock=read(Path(root)/'lock.json');verify(root,lock['files']);return sha(Path(root)/'lock.json')
class Backup:
 def __init__(self,root):self.root=Path(root);self.last=time.monotonic()
 def save(self,force=False):
  if not force and time.monotonic()-self.last<3600:return
  space(self.root);refs={}
  for folder in ('code','prepared','features','analysis'):
   base=self.root/folder
   for p in sorted(base.rglob('*')) if base.exists() else []:
    if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.tmp':continue
    h=sha(p);d=self.root/'backups/objects'/h[:2]/h;d.parent.mkdir(parents=True,exist_ok=True)
    if not d.exists():shutil.copyfile(p,d)
    assert sha(d)==h and d.stat().st_nlink==1
    refs[p.relative_to(self.root).as_posix()]=dict(sha256=h,bytes=p.stat().st_size)
  for p in sorted(self.root.glob('*.json')):
   h=sha(p);d=self.root/'backups/objects'/h[:2]/h;d.parent.mkdir(parents=True,exist_ok=True)
   if not d.exists():shutil.copyfile(p,d)
   assert sha(d)==h and d.stat().st_nlink==1
   refs[p.name]=dict(sha256=h,bytes=p.stat().st_size)
  name=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.json'
  write(self.root/'backups/manifests'/name,dict(files=refs,readback=True,same_disk=True,foundation_referenced=True))
  self.last=time.monotonic()
  write(self.root/'latest_backup.json',dict(manifest=name,sha256=sha(self.root/'backups/manifests'/name),files=len(refs)))
