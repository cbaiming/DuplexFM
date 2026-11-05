"""Declared internal-OOF residual diagnostic, no model fitting."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4'
spec=importlib.util.spec_from_file_location('_r9_diag_io',P/'run.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
io=prior.io
SETTINGS=('Huesken','Mixset','Takayuki');VIEWS=('core_real','full_real','initial_core','TD24')

def stat(r):
    med=float(np.median(r));scale=max(float(1.4826*np.median(abs(r-med))),1e-6)
    count=int(np.ceil(.05*len(r)));sq=r*r;den=float(sq.sum())
    return dict(n=len(r),median_residual=med,MAD_scale=scale,top5_count=count,
        top5_SSE_fraction=float(np.sort(sq)[-count:].sum()/den) if den>0 else 0.,
        tail3MAD_fraction=float(np.mean(abs(r-med)>3*scale)),MSE=float(sq.mean()))

def register():
    assert not (R/'diagnostic_lock.json').exists();prior.validate()
    old=io.read(P/'lock.json');deps=dict(old['dependencies'])
    for name in old['files']:deps[str((P/name).resolve())]=io.sha(P/name)
    for p in (P/'lock.json',P/'results/complete.json',P/'verification.json',R.parent/'round8/verification.json'):
        deps[str(p.resolve())]=io.sha(p)
    for setting in SETTINGS:
        for view in VIEWS[:3]:
            p=P/'results/folds'/setting/'fold0'/view/'arrays.npz';deps[str(p.resolve())]=io.sha(p)
    io.write(R/'diagnostic_lock.json',dict(files={n:io.sha(R/n) for n in ('diagnose.py','DIAGNOSTIC_PROTOCOL.md')},
        dependencies=deps,CPU_only=True,new_fits=0,new_source_forward=0,settings=SETTINGS,outer_fold=0,
        source_candidate=18,TD_candidate=3,qualification='any Core pool: top5 SSE >=.35 OR tail3MAD >=.015'))
    print('DIAGNOSTIC_REGISTERED',io.sha(R/'diagnostic_lock.json'),flush=True)

def run():
    assert not (R/'diagnostic.json').exists();lock=io.read(R/'diagnostic_lock.json');io.check_files(R,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    # Only benchmark labels for predeclared fit IDs enter any calculation.
    plan=io.read(P/'plan.json');records=[];details=[];pools={}
    for setting in SETTINGS:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        lookup={r['sample_id']:r for r in io.table(prior.BENCH/'data'/f'{setting}.csv')}
        y=np.array([float(lookup[s]['label']) for s in item['fit_ids']]);pools[setting]=item['fit_ids']
        block=np.full(len(y),-1,dtype=int)
        for j,sp in enumerate(item['inner']):block[np.array(sp['valid'])]=j
        assert (block>=0).all()
        for view in VIEWS:
            path=P/'results/folds'/setting/'fold0'/('core_real' if view=='TD24' else view)/'arrays.npz'
            with np.load(path,allow_pickle=False) as a:p=a['inner_oof'][:,3 if view=='TD24' else 18].copy()
            r=y-p;row=dict(setting=setting,view=view,**stat(r),inner_blocks=[dict(block=j,**stat(r[block==j])) for j in range(5)])
            records.append(row)
            details.extend(dict(setting=setting,view=view,sample_id=s,inner_block=int(block[i]),label=float(y[i]),prediction=float(p[i]),residual=float(r[i])) for i,s in enumerate(item['fit_ids']))
            print(setting,view,row['top5_SSE_fraction'],row['tail3MAD_fraction'],flush=True)
    core=[r for r in records if r['view']=='core_real'];qualified=any(r['top5_SSE_fraction']>=.35 or r['tail3MAD_fraction']>=.015 for r in core)
    io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(qualified=qualified,statistics=records,fit_ids=pools,new_fits=0,
        new_source_forward=0,GPU_hours=0,outer_test_metrics_used=False,lock_sha256=io.sha(R/'diagnostic_lock.json'),
        diagnostic_rows_sha256=io.sha(R/'diagnostic_rows.csv')))
    print('QUALIFIED',qualified,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('register','run'));mode=ap.parse_args().mode
    if mode=='register':register()
    else:
        started=time.monotonic();rc=1;error=None
        try:run();rc=0
        except Exception:error=traceback.format_exc();raise
        finally:io.write(R/f'diagnostic_execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-started,GPU_hours=0,new_fits=0))
