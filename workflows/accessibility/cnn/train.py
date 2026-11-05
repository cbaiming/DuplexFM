"""Fixed supervised comparison: no test read, atomic and exact batch-cursor resume."""
from pathlib import Path
import os,sys,time,json,argparse,random,shutil,hashlib
import numpy as np,torch
from common import ROOT,arrays,tokenize,sha,atomic_json,require_disk
from models import Baseline,loss
import accessibility_metrics as metrics

def save(path,payload):
 tmp=path.with_suffix('.tmp')
 with tmp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)
def backup(out):
 objects=out/'objects';objects.mkdir(exist_ok=True);refs={}
 for n in ['last.pt','best.pt','initial.pt','history.json','config.json']:
  p=out/n
  if not p.exists():continue
  h=sha(p);dest=objects/h
  if not dest.exists():
   tmp=objects/(h+'.tmp');shutil.copyfile(p,tmp);assert sha(tmp)==h;os.replace(tmp,dest)
  assert sha(dest)==h and dest.stat().st_nlink==1
  refs[n]=dict(sha256=h,bytes=p.stat().st_size)
 atomic_json(out/('backup_'+str(time.time_ns())+'.json'),dict(files=refs,readback=True))
def batch(a,ids,pu,device):
 x=tokenize(a['sequences'][ids]);l=np.asarray(a['lengths'][ids],dtype='int64')
 if pu is not None:
  f=np.asarray(pu[ids]);assert np.isin(f,[0,1]).all();x=np.concatenate([x,f[:,None]],axis=1)
 return (torch.from_numpy(x).to(device),torch.tensor(l,device=device),torch.tensor(np.asarray(a['median_accessibility_profiles'][ids]),device=device))
@torch.inference_mode()
def validate(m,a,ids,pu,device):
 m.eval();pred=[]
 for j in range(0,len(ids),1024):
  x,l,y=batch(a,ids[j:j+1024],pu,device);pred.append(m(x,l).cpu().numpy())
 return metrics.profile_metrics(np.concatenate(pred),a['median_accessibility_profiles'][ids],a['lengths'][ids])
def config(kind,device):
 return dict(kind=kind,seed=42,batch=512,lr=.0003,weight_decay=.0001,epochs=25,patience=5,precision='fp32',clip=1,
   device=device,science_lock_sha256=sha(ROOT/'science_lock.json'),input_manifest_sha256=sha(ROOT/'inputs/input_package_manifest.json'))
def main():
 p=argparse.ArgumentParser();p.add_argument('kind',choices=['sequence','structure']);p.add_argument('--device',default='cuda:0');p.add_argument('--resume',action='store_true');args=p.parse_args()
 out=ROOT/('run_'+args.kind);out.mkdir(exist_ok=True)
 import fcntl
 with (out/'lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  started=time.time();receipt=dict(start=started,status='running',device=args.device,visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),pid=os.getpid());phase=out/('phase_'+str(time.time_ns())+'.json');atomic_json(phase,receipt)
  try:run(args,out);receipt['status']='complete'
  except BaseException as e:receipt.update(status='failed_or_interrupted',error=repr(e));raise
  finally:
   receipt.update(end=time.time(),wall_seconds=time.time()-started);atomic_json(phase,receipt)
def run(args,out):
 require_disk();assert not (out/'complete.json').exists(),'Already complete'
 cfg=config(args.kind,args.device);device=torch.device(args.device)
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.manual_seed(42);np.random.seed(42);random.seed(42)
 a=arrays();ids=np.flatnonzero(a['splits']==0);vids=np.flatnonzero(a['splits']==1)
 assert len(ids)==1928795 and len(vids)==196020
 pu=None
 if args.kind=='structure':
  rr=json.loads((ROOT/'features_linearfold_v/complete.json').read_text());assert sha(ROOT/'features_linearfold_v/unpaired_state.npy')==rr['sha256'];pu=np.load(ROOT/'features_linearfold_v/unpaired_state.npy',mmap_mode='r')
 m=Baseline(args.kind).to(device);opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
 history=[];best=None;bestepoch=0;stale=0;epoch=1;cursor=0;step=0;running_loss=0.;rows_seen=0
 if args.resume:
  assert json.loads((out/'config.json').read_text())==cfg
  ck=torch.load(out/'last.pt',map_location=device,weights_only=False);assert ck['config']==cfg
  m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['torch_rng'].cpu())
  if device.type=='cuda':torch.cuda.set_rng_state(ck['cuda_rng'].cpu(),device)
  random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
  history,best,bestepoch,stale,epoch,cursor,step,running_loss,rows_seen=[ck[k] for k in ['history','best','bestepoch','stale','epoch','cursor','step','running_loss','rows_seen']]
 else:
  assert not (out/'last.pt').exists() and not (out/'config.json').exists(),'Use explicit resume'
  atomic_json(out/'config.json',cfg);save(out/'initial.pt',dict(model=m.state_dict(),config=cfg))
 lastbackup=time.monotonic()
 def checkpoint():
  save(out/'last.pt',dict(config=cfg,model=m.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
   history=history,best=best,bestepoch=bestepoch,stale=stale,epoch=epoch,cursor=cursor,step=step,running_loss=running_loss,rows_seen=rows_seen))
 checkpoint();backup(out)
 while epoch<=25 and stale<5:
  order=np.random.default_rng(np.random.SeedSequence([42,epoch,7301])).permutation(ids);t=time.monotonic();m.train()
  while cursor<len(order):
   ix=order[cursor:cursor+512];x,l,y=batch(a,ix,pu,device);opt.zero_grad(set_to_none=True);v=loss(m(x,l),y,l)
   if not torch.isfinite(v):raise FloatingPointError('nonfinite loss')
   v.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();cursor+=len(ix);step+=1;running_loss+=float(v.detach())*len(ix);rows_seen+=len(ix)
   if step%500==0:
    require_disk();checkpoint();atomic_json(out/'status.json',dict(epoch=epoch,cursor=cursor,total=len(order),step=step))
   if time.monotonic()-lastbackup>=3600:checkpoint();backup(out);lastbackup=time.monotonic()
  validation=validate(m,a,vids,pu,device);candidate=metrics.selection_value(validation)
  if metrics.is_better(candidate,best,1e-4):
   best=candidate;bestepoch=epoch;stale=0;save(out/'best.pt',dict(model=m.state_dict(),config=cfg,epoch=epoch,validation=validation))
  else:stale+=1
  history.append(dict(epoch=epoch,loss=running_loss/rows_seen,validation=validation,seconds=time.monotonic()-t))
  print(json.dumps(dict(epoch=epoch,kind=args.kind,loss=history[-1]['loss'],val_pcc=validation['micro_pearson'],bestepoch=bestepoch)),flush=True)
  epoch+=1;cursor=0;running_loss=0.;rows_seen=0;atomic_json(out/'history.json',history);checkpoint()
 backup(out)
 atomic_json(out/'complete.json',dict(status='complete',epochs=len(history),best_epoch=bestepoch,parameters=sum(p.numel() for p in m.parameters()),test_used=False,
   files={n:sha(out/n) for n in ['best.pt','initial.pt','history.json','config.json','last.pt']},science_lock_sha256=cfg['science_lock_sha256']))
if __name__=='__main__':main()
