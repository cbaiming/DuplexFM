"""Fixed nested, leakage-isolated lightweight regressions; no source fitting."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,csv,gzip,sys,time
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.stats import pearsonr,spearmanr
from sklearn.decomposition import PCA
from utils import read,write,sha,npz,rows,verify,check,Backup
ALPHAS=np.array([.1,1.,10.,100.,1000.])
SOURCES=('full42','initial42','full43','initial43','full44','initial44')
def scaler(x):
 m=x.mean(0);s=x.std(0);s=np.where(s>1e-12,s,1.)
 return m,s
def transform_fit(x,kind,td=None):
 a=np.asarray(x,float);meta={}
 if kind.startswith('pca'):
  m,s=scaler(a);p=PCA(n_components=16,svd_solver='randomized',random_state=42,iterated_power=7,n_oversamples=16)
  p.fit((a-m)/s);a=p.transform((a-m)/s)
  meta.update(source_mean=m,source_scale=s,pca_mean=p.mean_,components=p.components_)
  if kind=='pca_td':a=np.c_[td,a]
 m,s=scaler(a);meta.update(mean=m,scale=s);return (a-m)/s,meta
def transform(x,kind,m,td=None):
 a=np.asarray(x,float)
 if kind.startswith('pca'):
  a=((a-m['source_mean'])/m['source_scale']-m['pca_mean'])@m['components'].T
  if kind=='pca_td':a=np.c_[td,a]
 return (a-m['mean'])/m['scale']
def solve_all(z,y):
 gram=z.T@z;rhs=z.T@(y-y.mean());out=[]
 for a in ALPHAS:
  b=cho_solve(cho_factor(gram+a*np.eye(z.shape[1]),lower=True),rhs)
  err=np.max(abs((gram+a*np.eye(z.shape[1]))@b-rhs))
  assert err<1e-7*(1+np.max(abs(rhs)))
  out.append(b)
 return np.stack(out,axis=1),float(y.mean())
def choose(mse):
 best=min(mse);return int(np.flatnonzero(np.isclose(mse,best,rtol=1e-12,atol=1e-12))[-1])
def metric(y,p):
 assert np.isfinite(y).all() and np.isfinite(p).all()
 return dict(n=len(y),PCC=float(pearsonr(y,p).statistic) if np.std(p)>1e-12 and np.std(y)>1e-12 else None,
  Spearman=float(spearmanr(y,p).statistic) if np.std(p)>1e-12 and np.std(y)>1e-12 else None,
  MSE=float(np.mean((y-p)**2)),MAE=float(np.mean(abs(y-p))))
def group_metrics(y,p,g):
 vals=[]
 for v in np.unique(g):
  i=np.flatnonzero(g==v)
  if len(i)>=3 and np.std(y[i])>1e-12 and np.std(p[i])>1e-12:vals.append(float(pearsonr(y[i],p[i]).statistic))
 return dict(macro_PCC=float(np.mean(vals)) if vals else None,eligible_groups=len(vals))
def load_features(root,dataset,source):
 d=root/'features'/dataset/source;c=read(d/'complete.json');assert c['lock']==sha(root/'lock.json');verify(d,c['files'])
 a=np.concatenate([np.load(p,allow_pickle=False)['source'] for p in sorted(d.glob('*.npz'))])
 assert a.shape==(c['rows'],192) and np.isfinite(a).all();return a
def simple(rs):
 out=[]
 for r in rs:
  g=r['guide'];t=r['target'];a=r['anchor'];comp=str.maketrans('ACGU','UGCA')
  mm=[float(g[j].translate(comp)!=t[a-j]) for j in range(len(g))]
  counts=[s.count(c)/len(s) for s in (g,t) for c in 'ACGU']
  out.append(mm+counts)
 return np.asarray(out,float)
def head(root,dataset,name,x,kind,rs,plan,td=None):
 folder=root/'analysis'/dataset/name
 if (folder/'complete.json').exists():
  c=read(folder/'complete.json');verify(folder,c['files']);return np.load(folder/'predictions.npz',allow_pickle=False)['prediction']
 y=np.array([r['label'] for r in rs]);p=np.full(len(y),np.nan);coverage=np.zeros(len(y),int);selections=[]
 for f in plan['folds']:
  tr=np.array(f['train'],int);te=np.array(f['test'],int);assert set(tr).isdisjoint(te)
  if kind=='mean':pred=np.full(len(te),y[tr].mean());selected=None
  else:
   total=np.zeros(5);n=0;seen=np.zeros(len(y),int)
   for inner in f['inner']:
    ii=np.array(inner['train'],int);vv=np.array(inner['valid'],int)
    assert set(ii).isdisjoint(vv) and set(ii)|set(vv)<=set(tr)
    z,m=transform_fit(x[ii],kind,td[ii] if td is not None else None)
    zv=transform(x[vv],kind,m,td[vv] if td is not None else None)
    b,c=solve_all(z,y[ii]);preds=zv@b+c;total+=((preds-y[vv,None])**2).sum(0);n+=len(vv);seen[vv]+=1
   assert np.all(seen[tr]==1)
   k=choose(total/n);z,m=transform_fit(x[tr],kind,td[tr] if td is not None else None)
   zt=transform(x[te],kind,m,td[te] if td is not None else None)
   b,c=solve_all(z,y[tr]);pred=zt@b[:,k]+c;selected=float(ALPHAS[k])
   npz(folder/f"fold{f['fold']}_head.npz",**m,coef=b[:,k],intercept=np.array(c),inner_mse=total/n,alpha=np.array(selected),train=tr,test=te)
   # Independent direct normal-equation check of selected coefficient.
   beta=np.linalg.solve(z.T@z+selected*np.eye(z.shape[1]),z.T@(y[tr]-y[tr].mean()))
   np.testing.assert_allclose(beta,b[:,k],rtol=1e-7,atol=1e-8)
  p[te]=pred;coverage[te]+=1;selections.append(dict(fold=f['fold'],alpha=selected,train_n=len(tr),test_n=len(te)))
 assert np.all(coverage==1) and np.isfinite(p).all()
 npz(folder/'predictions.npz',prediction=p,label=y,sample_ids=np.array([r['sample_id'] for r in rs]))
 write(folder/'selection.json',selections)
 write(folder/'complete.json',dict(rows=len(y),readout=kind,input_lock=sha(root/'lock.json'),evaluation_code_sha256=sha(Path(__file__)),
  files={p.name:sha(p) for p in folder.iterdir() if p.is_file()}),exclusive=True)
 print('HEAD_COMPLETE',dataset,name,len(y),flush=True);return p
def boot(y,pa,pb,group,seed=20261027):
 _,ix=np.unique(group,return_inverse=True);K=ix.max()+1;summary=[]
 for p in (pa,pb):
  vals=[np.bincount(ix,weights=v,minlength=K) for v in (np.ones(len(y)),y,p,y*y,p*p,y*p,(y-p)**2)]
  summary.append(np.array(vals).T)
 weights=np.random.default_rng(seed).multinomial(K,np.full(K,1/K),size=1000)
 stats=[]
 for s in summary:
  a=weights@s;n,sy,sp,yy,pp,yp,se=a.T
  pc=(yp-sy*sp/n)/np.sqrt(np.maximum((yy-sy*sy/n)*(pp-sp*sp/n),1e-30))
  stats.append((pc,se/n))
 return dict(bootstrap_groups=int(K),replicates=1000,PCC_difference_ci95=np.quantile(stats[0][0]-stats[1][0],[.025,.975]).tolist(),
   MSE_relative_reduction_pct_ci95=np.quantile(100*(1-stats[0][1]/stats[1][1]),[.025,.975]).tolist())
def write_csv(p,rs):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def cas13(root):
 begin=time.monotonic();check(root);summ=[];comparisons=[]
 for dataset in ('ADAPT','TIGER'):
  rs=rows(root,dataset);y=np.array([r['label'] for r in rs]);g=np.array([r['parent'] if dataset=='ADAPT' else r['gene'] for r in rs]);parent=np.array([r['parent'] for r in rs])
  plan=read(root/'prepared'/f'{dataset}_splits.json');preds={}
  for name in ('mean','mismatch_composition')+SOURCES:
   if name=='mean':x=np.ones((len(rs),1));kind='mean'
   elif name=='mismatch_composition':x=simple(rs);kind='direct'
   else:x=load_features(root,dataset,name);kind='direct'
   p=head(root,dataset,name,x,kind,rs,plan);preds[name]=p
   m=metric(y,p);gm=group_metrics(y,p,g);center_y=y.copy();center_p=p.copy()
   _,ii=np.unique(parent,return_inverse=True);nn=np.bincount(ii)
   center_y-=np.bincount(ii,weights=y)[ii]/nn[ii];center_p-=np.bincount(ii,weights=p)[ii]/nn[ii]
   summ.append(dict(dataset=dataset,model=name,**m,**gm,within_parent_PCC=metric(center_y,center_p)['PCC']))
   for kind in sorted({r['kind'] for r in rs}):
    ix=np.array([r['kind']==kind for r in rs]);write(root/'analysis'/dataset/name/f'class_{kind}.json',metric(y[ix],p[ix]))
  for seed in (42,43,44):
   for ref in (f'initial{seed}','mismatch_composition'):
    a,b=preds[f'full{seed}'],preds[ref];ma,mb=metric(y,a),metric(y,b)
    comparisons.append(dict(dataset=dataset,model=f'full{seed}',reference=ref,PCC_difference=ma['PCC']-mb['PCC'],
     MSE_relative_reduction_pct=100*(1-ma['MSE']/mb['MSE']),**boot(y,a,b,g)))
  npz(root/'analysis'/dataset/'compact_predictions.npz',label=y,group=g,parent=parent,kind=np.array([r['kind'] for r in rs]),
    sample_ids=np.array([r['sample_id'] for r in rs]),**preds)
 write_csv(root/'analysis/cas13_summary.csv',summ);write(root/'analysis/cas13_comparisons.json',comparisons)
 write(root/'cas13_evaluation_complete.json',dict(wall_seconds=time.monotonic()-begin,source_training_updates=0,
  readout_conditions=8,datasets=2,rows=112354,evaluation_code_sha256=sha(Path(__file__))))
def sirna(root):
 begin=time.monotonic();check(root);rs0=rows(root,'siRNA50');idx={r['sample_id']:i for i,r in enumerate(rs0)}
 old=read(root/'prepared/siRNA_original_plan.json');verify(root/'cnn',read(root/'cnn/complete.json')['files']);cnn=np.load(root/'cnn/embedding.npz',allow_pickle=False)
 assert cnn['embedding'].shape==(3329,160);assert list(cnn['sample_ids'])==[r['sample_id'] for r in rs0]
 xx={'CNN':cnn['embedding']};xx.update({s:load_features(root,'siRNA50',s) for s in ('full42','full43','full44')})
 summary=[];pred_out=[]
 for setting in ('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix'):
  selected=[f for f in old['folds'] if f['setting']==setting]
  testids={v for f in selected for v in f['test_ids']};universe=sorted({v for f in selected for v in f['test_ids']+f['fit_ids']},key=idx.get)
  rr=[rs0[idx[v]] for v in universe];lut={s:i for i,s in enumerate(universe)};raw=np.array([idx[v] for v in universe]);td=np.array([[float(v) for v in r['td'].split(',')] for r in rr]);folds=[]
  for f in selected:
   tr=[lut[v] for v in f['fit_ids']];te=[lut[v] for v in f['test_ids']]
   inn=[dict(train=[tr[i] for i in z['train']],valid=[tr[i] for i in z['valid']]) for z in f['inner']]
   folds.append(dict(fold=f['fold'],train=tr,test=te,inner=inn))
  # Cross-domain task does not produce OOF for the Huesken training rows.
  for name in ('TD24',)+tuple(f'{source}_{kind}' for source in xx for kind in ('only','TD24')):
   if name=='TD24':x=td;kind='direct'
   else:
    source,tail=name.rsplit('_',1);x=xx[source][raw];kind='pca' if tail=='only' else 'pca_td'
   folder=root/'analysis'/setting/name
   # Shared head supports exactly-once test subset while preserving the original fit rows.
   p=head_subset(root,setting,name,x,kind,rr,dict(folds=folds),td)
   ix=np.array([r['sample_id'] in testids for r in rr]);y=np.array([r['label'] for r in rr])[ix]
   summary.append(dict(setting=setting,model=name,**metric(y,p[ix])))
   for i in np.flatnonzero(ix):pred_out.append(dict(setting=setting,model=name,sample_id=rr[i]['sample_id'],label=rr[i]['label'],prediction=p[i]))
 write_csv(root/'analysis/cnn_sirna_summary.csv',summary)
 with gzip.open(root/'analysis/cnn_sirna_predictions.csv.gz','wt',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(pred_out[0]));w.writeheader();w.writerows(pred_out)
 write(root/'cnn_sirna_evaluation_complete.json',dict(seconds=time.monotonic()-begin,conditions=9,settings=4,source_training_updates=0,evaluation_code_sha256=sha(Path(__file__))))
def head_subset(root,dataset,name,x,kind,rs,plan,td):
 # Same nested algebra as head; exact-once is enforced on the predefined test subset.
 folder=root/'analysis'/dataset/name;y=np.array([r['label'] for r in rs]);p=np.full(len(y),np.nan);coverage=np.zeros(len(y),int)
 if (folder/'complete.json').exists():
  c=read(folder/'complete.json');verify(folder,c['files']);return np.load(folder/'predictions.npz',allow_pickle=False)['prediction']
 for f in plan['folds']:
  tr=np.array(f['train']);te=np.array(f['test']);assert set(tr).isdisjoint(te);mse=np.zeros(5);n=0;cov=np.zeros(len(rs),int)
  for inner in f['inner']:
   ii=np.array(inner['train']);vv=np.array(inner['valid']);assert set(ii).isdisjoint(vv) and set(ii)|set(vv)<=set(tr)
   z,m=transform_fit(x[ii],kind,td[ii]);zv=transform(x[vv],kind,m,td[vv]);b,c=solve_all(z,y[ii]);mse+=((zv@b+c-y[vv,None])**2).sum(0);n+=len(vv);cov[vv]+=1
  assert np.all(cov[tr]==1);k=choose(mse/n);z,m=transform_fit(x[tr],kind,td[tr]);zt=transform(x[te],kind,m,td[te]);b,c=solve_all(z,y[tr]);p[te]=zt@b[:,k]+c;coverage[te]+=1
  npz(folder/f"fold{f['fold']}_head.npz",**m,coef=b[:,k],intercept=np.array(c),inner_mse=mse/n,alpha=ALPHAS[k],train=tr,test=te)
 test=np.array(sorted({i for f in plan['folds'] for i in f['test']}));assert np.all(coverage[test]==1) and np.isfinite(p[test]).all()
 npz(folder/'predictions.npz',prediction=p,label=y,test_indices=test,sample_ids=np.array([r['sample_id'] for r in rs]))
 write(folder/'complete.json',dict(rows=len(test),kind=kind,files={v.name:sha(v) for v in folder.iterdir() if v.is_file()}),exclusive=True)
 print('SIRNA_HEAD_COMPLETE',dataset,name,len(test),flush=True);return p
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('cas13','sirna'));ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
 (cas13 if a.mode=='cas13' else sirna)(a.root)
