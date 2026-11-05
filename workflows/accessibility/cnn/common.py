from pathlib import Path
import os,json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parent
DATA=Path(os.environ.get('ICSHAPE_DATA',str(ROOT/'inputs/data')))
ORIGINAL=DATA.parent
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def atomic_json(p,v):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp')
 with q.open('w') as f:json.dump(v,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(q,p)
def arrays():
 names=['sequences','lengths','median_accessibility_profiles','observed_mask','splits','transcript_components']
 return {k:np.load(DATA/(k+'.npy'),mmap_mode='r') for k in names}
def tokenize(seqs):
 b=np.asarray(seqs,dtype='S60').view('u1').reshape(-1,60)
 return np.stack([b==ord(c) for c in 'ACGU'],axis=1).astype('float32')
def require_disk():
 import shutil
 if shutil.disk_usage(ROOT).free<10*2**30:raise RuntimeError('Less than 10 GiB free: stop new work')
