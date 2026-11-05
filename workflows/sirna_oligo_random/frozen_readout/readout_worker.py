import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import sys,time,traceback
from transfer_common import *
M=ROOT/'math/sirna_increment_20261014'
sys.path[:0]=[str(M/'round4'),str(ROOT/'math/expert_first_20261014')]
head=load('_new_r55',M/'round55/head.py');scale=load('_new_scale',M/'round4/core.py')
grouping=load('_new_groups',M/'round49/group_nuisance.py');ga=load('_new_group_audit',M/'round49/verify.py');audit=load('_new_audit',M/'round55/audit_math.py')
sys.path.insert(0,str(ROOT/'common_reader'));E=load('_new_common_reader',ROOT/'common_reader/evaluate.py')
def fit_scaler(x):
 tm,ts=scale.scaler(x[:,:24]);sm,ss=scale.scaler(x[:,24:],True);return dict(td_mean=tm,td_scale=ts,src_mean=sm,src_scale=ss)
def get_features(name,mode,rows):
 if mode=='pca_td' or mode=='ridge_td':
  if name=='TD24_common':return np.array([[float(v) for v in r['td'].split(',')] for r in rows])
  p=ROOT/'common_features'/f'{name}.npz';a=np.load(p,allow_pickle=False)
  assert a['sample_ids'].tolist()==[r['sample_id'] for r in rows];return a['embedding'].astype(float)
 parts=[];ids=[]
 p=ROOT/'features'/f'{name}.npz'
 if p.exists():
  a=np.load(p,allow_pickle=False);parts.append(a['source']);ids+=a['sample_ids'].tolist()
 p=ROOT/'exports'/name/'source.npz'
 if p.exists():
  c=read(p.parent/'complete.json');assert c['passed'];checkfiles(p.parent,c['files']);a=np.load(p,allow_pickle=False);parts.append(a['source']);ids+=a['sample_ids'].tolist()
 assert ids==[r['sample_id'] for r in rows]
 src=np.concatenate(parts).astype(float);assert src.shape==(3688,192) and np.isfinite(src).all()
 td=np.array([[float(v) for v in r['td'].split(',')] for r in rows]) if mode=='TD24' else np.zeros((3688,24))
 return np.c_[td,src]
def one_r55(item,x,rows,lut,mode,dest,deadline):
 fi=np.array([lut[s] for s in item['fit_ids']]);ti=np.array([lut[s] for s in item['test_ids']]);yf=np.array([float(rows[i]['label']) for i in fi])
 oof=np.full((len(fi),5),np.nan);trans=[];certificates=[];bs=[];cs=[];groups=[];normalmax=0.;gapmax=0.
 for j in range(6):
  tr=np.arange(len(fi)) if j==5 else np.array(item['inner'][j]['train']);t=fit_scaler(x[fi[tr]]);z=head.transform(x[fi[tr]],t)
  targets=[rows[fi[i]]['target57'] for i in tr];g,gm=grouping.groups(targets);ind,_=ga.independent_groups(targets);np.testing.assert_array_equal(g,ind)
  profile=audit.profile(z,yf[tr],g);sub=[];betas=[];centers=[]
  if j==5:
   pc,ms=scale.score(yf,oof);chosen=head.choose(pc,ms);grid=[chosen]
  else:grid=range(5)
  for k in grid:
   b,c,cert=head.fit(z,yf[tr],g,head.ALPHAS[k],True,deadline=deadline);assert cert['certified']
   normal,gap=audit.check_source(z,yf[tr],g,b,c,cert,head.ALPHAS[k],cached=profile);normalmax=max(normalmax,normal);gapmax=max(gapmax,gap)
   if mode=='source':
    assert not np.count_nonzero(x[fi[tr],:24]) and not np.count_nonzero(b[:24]);np.testing.assert_array_equal(t['td_mean'],np.zeros(24));np.testing.assert_array_equal(t['td_scale'],np.ones(24))
   if j<5:
    va=np.array(item['inner'][j]['valid']);oof[va,k]=head.transform(x[fi[va]],t)@b+c
   else:
    pred=head.transform(x[ti],t)@b+c;db,dc=head.deployment(t,b,c);np.testing.assert_allclose(pred,x[ti]@db+dc,atol=1e-10,rtol=1e-10)
   sub.append(cert);betas.append(b);centers.append(c)
  trans.append(t);certificates.append(sub);bs.append(betas);cs.append(centers);groups.append(dict(labels=g,metadata=gm))
 write(dest/'transforms.json',trans);write(dest/'groups.json',groups);write(dest/'certificates.json',certificates)
 array(dest/'arrays.npz',inner_oof=oof,inner_coef=np.array(bs[:5]),inner_intercept=np.array(cs[:5]),final_coef=b,final_intercept=np.array(c),deployment_coef=db,deployment_intercept=np.array(dc),prediction=pred)
 write(dest/'selection.json',dict(alpha=head.ALPHAS[chosen],chosen=chosen,inner_PCC=pc,inner_MSE=ms,fit_ids=item['fit_ids'],test_ids=item['test_ids'],test_used_for_selection=False,normal_max=normalmax,gap_relative_max=gapmax))
 return pred,26
def one_common(item,x,td,rows,lut,mode,dest):
 fi=np.array([lut[s] for s in item['fit_ids']]);ti=np.array([lut[s] for s in item['test_ids']]);y=np.array([float(r['label']) for r in rows]);kind='direct' if mode=='ridge_td' else 'pca_td'
 oof=np.full((len(fi),5),np.nan)
 for j in range(6):
  tr=np.arange(len(fi)) if j==5 else np.array(item['inner'][j]['train']);ii=fi[tr];z,m=E.transform_fit(x[ii],kind,td[ii]);b,c=E.solve_all(z,y[ii])
  if j<5:
   va=np.array(item['inner'][j]['valid']);vv=fi[va];pr=E.transform(x[vv],kind,m,td[vv])@b+c;oof[va]=pr
   array(dest/f'inner{j}.npz',**m,coef_grid=b,intercept=np.array(c),train=ii,valid=vv,prediction_grid=pr)
  else:
   mse=((oof-y[fi,None])**2).mean(0);k=E.choose(mse);pred=E.transform(x[ti],kind,m,td[ti])@b[:,k]+c
   independent=np.linalg.solve(z.T@z+E.ALPHAS[k]*np.eye(z.shape[1]),z.T@(y[ii]-y[ii].mean()));np.testing.assert_allclose(independent,b[:,k],atol=1e-8,rtol=1e-7)
   array(dest/'final.npz',**m,coef_grid=b,intercept=np.array(c),train=ii,test=ti,prediction=pred)
   write(dest/'selection.json',dict(alpha=E.ALPHAS[k],chosen=k,inner_MSE=mse,fit_ids=item['fit_ids'],test_ids=item['test_ids'],test_used_for_selection=False))
 return pred,30
def run(name,mode):
 start=time.monotonic();rc=1;error=None;out=ROOT/'readouts'/f'{name}__{mode}';out.mkdir(parents=True,exist_ok=True)
 try:
  checkfiles(ROOT,read(ROOT/'package_lock.json')['files']);rows,plan,lut=validate_inputs();x=get_features(name,mode,rows);td=np.array([[float(v) for v in r['td'].split(',')] for r in rows]);predrows=[];solves=0;lastbackup=time.monotonic()
  feature_hash=hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
  if (out/'complete.json').exists():checkfiles(out,read(out/'complete.json')['files']);return
  for f in plan['folds']:
   dest=out/f"{f['setting']}_{f['fold']}";receipt=dest/'complete.json'
   if receipt.exists():
    rec=read(receipt);checkfiles(dest,rec['files']);assert rec['feature_hash']==feature_hash and rec['execution_lock_sha256']==sha(ROOT/'execution_lock.json');part=table(dest/'predictions.csv')
   else:
    assert not dest.exists() or not any(dest.iterdir()),'Partial fold retained; audit before retry';dest.mkdir(parents=True,exist_ok=True)
    if mode in ('source','TD24'):p,n=one_r55(f,x,rows,lut,mode,dest,start+7200)
    else:p,n=one_common(f,x,td,rows,lut,mode,dest)
    solves+=n;part=[dict(source=name,mode=mode,setting=f['setting'],fold=f['fold'],sample_id=s,label=float(rows[lut[s]]['label']),binary=int(rows[lut[s]]['y']),prediction=float(v)) for s,v in zip(f['test_ids'],p)]
    csvsave(dest/'predictions.csv',part);write(receipt,dict(passed=True,solves=n,feature_hash=feature_hash,execution_lock_sha256=sha(ROOT/'execution_lock.json'),files={p.name:sha(p) for p in dest.iterdir() if p.is_file()}))
   assert [p['sample_id'] for p in part]==f['test_ids'];predrows+=part
   print('FOLD_COMPLETE',name,mode,f['setting'],f['fold'],round(time.monotonic()-start,2),flush=True)
   if time.monotonic()-lastbackup>=3600:backup(out,f'{name}_{mode}_{time.time_ns()}');lastbackup=time.monotonic()
  assert len(predrows)==4157;csvsave(out/'predictions.csv',predrows);backup(out,f'{name}_{mode}_final')
  write(out/'complete.json',dict(passed=True,source=name,mode=mode,rows=4157,final_heads=26,new_solves=solves,wall_seconds=time.monotonic()-start,files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}));rc=0
 except BaseException:error=traceback.format_exc();write(out/'failure.json',dict(error=error));raise
 finally:write(out/'cost.json',dict(closed=True,rc=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_seconds=0))
if __name__=='__main__':run(sys.argv[1],sys.argv[2])
