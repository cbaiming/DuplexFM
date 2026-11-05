"""Feature-only overlap qualification on three predeclared outer-fit pools."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4'
spec=importlib.util.spec_from_file_location('_r4_diagnostic_io',P/'run.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
io=prior.io
S=('Huesken','Mixset','Takayuki');V=('core_real','full_real','initial_core')

def register():
    assert not (R/'diagnostic_lock.json').exists()
    prior.validate();old=io.read(P/'lock.json');deps=dict(old['dependencies'])
    for name in old['files']:deps[str((P/name).resolve())]=io.sha(P/name)
    for p in (P/'lock.json',P/'verification.json',R.parent/'round7/verification.json',R.parent/'round7/results/complete.json'):
        deps[str(p.resolve())]=io.sha(p)
    for setting in S:
        for view in V:
            p=P/'results/folds'/setting/'fold0'/view/'transforms.json';deps[str(p.resolve())]=io.sha(p)
    io.write(R/'diagnostic_lock.json',dict(files={n:io.sha(R/n) for n in ('diagnose.py','DIAGNOSTIC_PROTOCOL.md')},
        dependencies=deps,CPU_only=True,real_labels_used=False,source_seed=42,settings=S,outer_fold=0))
    print('DIAGNOSTIC_REGISTERED',io.sha(R/'diagnostic_lock.json'),flush=True)

def rank(x):
    s=np.linalg.svd(x,compute_uv=False)
    return int((s>s[0]*1e-10).sum()) if s[0]>0 else 0

def run():
    assert not (R/'diagnostic.json').exists()
    lock=io.read(R/'diagnostic_lock.json');io.check_files(R,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    rows,xx,identities=prior.inputs();index={r['sample_id']:i for i,r in enumerate(rows)}
    plan=io.read(P/'plan.json');results=[];pools={}
    for setting in S:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ii=np.array([index[x] for x in item['fit_ids']]);pools[setting]=item['fit_ids']
        for view in V:
            state=io.read(P/'results/folds'/setting/'fold0'/view/'transforms.json')['final']
            x=xx[view][ii];z=prior.transform(x,state);td,source=z[:,:24],z[:,24:]
            a=np.linalg.lstsq(td,source,rcond=1e-10)[0];residual=source-td@a
            var=float(np.sum(source**2));explained=float(1-np.sum(residual**2)/var)
            error=float(np.max(abs(td.T@residual))/len(td))
            raw=(x[:,24:]-np.asarray(state['src_mean']))/np.asarray(state['src_scale'])
            retained=float(np.sum(state['eigenvalues'])/np.var(raw,axis=0).sum())
            assert np.isfinite(a).all() and error<1e-10 and -.000001<=explained<=1.000001
            results.append(dict(setting=setting,source=view,n=len(td),TD_rank=rank(td),remaining_source_rank=rank(residual),
                source_variance_explained_by_TD=explained,max_TD_orthogonality_error=error,
                original_PCA16_variance_retained=retained,projection=a,
                source_identity=identities[view],original_transform_sha256=io.sha(P/'results/folds'/setting/'fold0'/view/'transforms.json')))
    core=[x for x in results if x['source']=='core_real']
    qualified=sum(x['source_variance_explained_by_TD']>=.1 for x in core)>=2 and all(x['remaining_source_rank']==16 and x['max_TD_orthogonality_error']<1e-10 for x in core)
    io.write(R/'diagnostic.json',dict(qualified=qualified,rows=results,fit_ids=pools,real_labels_used=False,
        new_prediction_metrics=False,GPU_hours=0,new_source_forward=0,lock_sha256=io.sha(R/'diagnostic_lock.json')))
    for row in results:print(row['setting'],row['source'],row['source_variance_explained_by_TD'],row['original_PCA16_variance_retained'],flush=True)
    print('QUALIFIED',qualified,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('register','run'));mode=ap.parse_args().mode
    if mode=='register':register()
    else:
        start=time.monotonic();rc=1;error=None
        try:run();rc=0
        except Exception:error=traceback.format_exc();raise
        finally:io.write(R/f'diagnostic_execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0))
