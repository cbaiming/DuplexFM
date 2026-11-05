"""Label-free LinearFold-V binary state cache; verified completed-chunk resume."""
from pathlib import Path
import sys,os,time,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parent/'dependencies'))
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from common import ROOT,arrays,sha,atomic_json,require_disk
from linearfold_backend import fold_block,EXPECTED,prepare
def main():
 import fcntl
 cache=ROOT/'features_linearfold_v';cache.mkdir(exist_ok=True)
 with (cache/'lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run(cache)
def run(cache):
 started=time.time();require_disk();prepare();a=arrays();ids=np.flatnonzero(a['splits']<2)
 seqs,inverse=np.unique(a['sequences'][ids],return_inverse=True);chunk=2048
 tasks=[];reused=0
 for j in range(0,len(seqs),chunk):
  idx=j//chunk;p=cache/f'chunk_{idx:04d}.npy';receipt=p.with_suffix('.json');seqsha=hashlib.sha256(seqs[j:j+chunk].tobytes()).hexdigest()
  if receipt.exists():
   rr=json.loads(receipt.read_text());assert rr['sequence_sha256']==seqsha and sha(p)==rr['sha256'];reused+=1
  else:tasks.append((idx,seqs[j:j+chunk]))
 with ProcessPoolExecutor(max_workers=16) as pool:
  for idx,out in pool.map(fold_block,tasks,chunksize=1):
   require_disk();p=cache/f'chunk_{idx:04d}.npy';tmp=p.with_suffix('.tmp')
   with tmp.open('wb') as f:np.save(f,out);f.flush();os.fsync(f.fileno())
   os.replace(tmp,p);j=idx*chunk
   atomic_json(p.with_suffix('.json'),dict(sequence_sha256=hashlib.sha256(seqs[j:j+chunk].tobytes()).hexdigest(),sha256=sha(p),rows=len(out)))
   atomic_json(cache/'status.json',dict(completed_chunks=len(list(cache.glob('chunk_*.json'))),total_chunks=(len(seqs)+chunk-1)//chunk,seconds=time.time()-started))
 # Preserve original row order. No labels read by the folding worker.
 allpu=np.concatenate([np.load(cache/f'chunk_{i:04d}.npy') for i in range((len(seqs)+chunk-1)//chunk)])
 p=cache/'unpaired_state.npy';tmp=cache/'unpaired_state.tmp.npy';m=np.lib.format.open_memmap(tmp,mode='w+',dtype='uint8',shape=(len(a['splits']),60));m[:]=255
 for j in range(0,len(ids),8192):m[ids[j:j+8192]]=allpu[inverse[j:j+8192]]
 m.flush();del m;os.replace(tmp,p)
 atomic_json(cache/'complete.json',dict(status='complete',sha256=sha(p),rows=len(ids),unique_sequences=len(seqs),seconds=time.time()-started,reused_chunks=reused,workers=16,new_gpu_seconds=0,binary_sha256=EXPECTED,backend='LinearFold-V binary unpaired state',uncomputed_internal_test_sentinel=255))
 print((cache/'complete.json').read_text(),flush=True)
if __name__=='__main__':main()
