"""One fixed training-internal additive-spline pilot; no outer evaluation."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4';sys.path.insert(0,str(R))
from additive import fit_transform,transform,fit_heads,base
spec=importlib.util.spec_from_file_location('_r13_old_io',P/'run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
io=old.io
SETTINGS=('Huesken','Mixset','Takayuki');VIEWS=('core_real','full_real','initial_core','TD24')
FILES=('diagnose.py','additive.py','test_additive.py','verify_diagnostic.py','DIAGNOSTIC_PROTOCOL.md','CODE_AUDIT.md','pytest.xml')

def register():
    assert not (R/'diagnostic_lock.json').exists();old.validate()
    tests=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(t.get('tests',0)) for t in tests)==8
    assert not any(int(t.get('errors',0))+int(t.get('failures',0))+int(t.get('skipped',0)) for t in tests)
    previous=io.read(P/'lock.json');deps=dict(previous['dependencies'])
    for name in previous['files']:deps[str((P/name).resolve())]=io.sha(P/name)
    for path in (P/'lock.json',P/'results/complete.json',P/'verification.json',R.parent/'round12/verification.json',R.parent/'round12/closure_receipt.json'):
        deps[str(path.resolve())]=io.sha(path)
    for setting in SETTINGS:
        for view in VIEWS[:3]:
            for name in ('arrays.npz','transforms.json'):
                path=P/'results/folds'/setting/'fold0'/view/name;deps[str(path.resolve())]=io.sha(path)
    io.write(R/'diagnostic_lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,
        versions=io.versions(),CPU_only=True,CPU_limit_seconds=300,new_source_forward=0,new_PCA_fits=0,
        fixed_alpha=100.,fixed_weight=1.,outer_fold=0,settings=SETTINGS,views=VIEWS,
        qualification='at least 2 Core fit pools dPCC>=.005 AND relative MSE reduction>=0; every Core dPCC>=-.005'))
    print('REGISTERED',io.sha(R/'diagnostic_lock.json'),flush=True)

def validate():
    lock=io.read(R/'diagnostic_lock.json');io.check_files(R,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    assert lock['versions']==io.versions();return lock

def stats(y,p):
    pc,ms=base.score(y,p[:,None]);return dict(PCC=float(pc[0]),MSE=float(ms[0]))

def run(start):
    lock=validate();root=R/'pilot';assert not root.exists() and not (R/'diagnostic.json').exists()
    rows,xx,_=old.inputs();lookup={r['sample_id']:i for i,r in enumerate(rows)};plan=io.read(P/'plan.json')
    records=[];pools={};details=[];count=0
    for setting in SETTINGS:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids'];pools[setting]=ids;fi=np.array([lookup[s] for s in ids])
        y=np.array([float(rows[i]['label']) for i in fi])
        for view in VIEWS:
            if time.monotonic()-start>lock['CPU_limit_seconds']:raise TimeoutError('CPU pilot limit; no new block')
            source='core_real' if view=='TD24' else view;x=xx[source][fi]
            reference=io.read(P/'results/folds'/setting/'fold0'/source/'transforms.json')
            with np.load(P/'results/folds'/setting/'fold0'/source/'arrays.npz',allow_pickle=False) as a:
                oldpred=a['inner_oof'][:,3 if view=='TD24' else 18].copy()
            predicted=np.full(len(y),np.nan);states=[];coefs=[];intercepts=[];blocks=[]
            for j,split in enumerate(item['inner']):
                tr=np.array(split['train']);va=np.array(split['valid'])
                z,t=fit_transform(x[tr],reference['inner'][j],view)
                b,c=fit_heads(z,y[tr],[dict(alpha=100.,weight=1.)])
                predicted[va]=transform(x[va],t)@b[0]+c;states.append(t);coefs.append(b[0]);intercepts.append(c);count+=1
                blocks.append(dict(inner=j,n=len(va),new=stats(y[va],predicted[va]),old=stats(y[va],oldpred[va])))
            assert np.isfinite(predicted).all()
            new=stats(y,predicted);previous=stats(y,oldpred)
            record=dict(setting=setting,view=view,n=len(y),feature_width=48 if view=='TD24' else 56,
                new=new,old=previous,delta_PCC=new['PCC']-previous['PCC'],
                MSE_relative_reduction_pct=100*(1-new['MSE']/previous['MSE']),inner_blocks=blocks)
            record['pool_qualified']=bool(record['delta_PCC']>=.005 and record['MSE_relative_reduction_pct']>=0)
            records.append(record);folder=root/setting/view
            io.write(folder/'states.json',states)
            old.npz_write(folder/'arrays.npz',inner_coef=np.array(coefs),inner_intercept=np.array(intercepts),
                inner_oof=predicted,reference_oof=oldpred)
            io.write(folder/'complete.json',dict(lock_sha256=io.sha(R/'diagnostic_lock.json'),new_fits=5,
                files_sha256={p.name:io.sha(p) for p in folder.iterdir() if p.is_file()}))
            details.extend(dict(setting=setting,view=view,sample_id=s,label=float(y[i]),new_prediction=float(predicted[i]),reference_prediction=float(oldpred[i])) for i,s in enumerate(ids))
            print(setting,view,record['delta_PCC'],record['MSE_relative_reduction_pct'],flush=True)
    assert count==60 and len(details)==4*sum(len(a) for a in pools.values())
    core=[r for r in records if r['view']=='core_real']
    qualified=bool(sum(r['pool_qualified'] for r in core)>=2 and all(r['delta_PCC']>=-.005 for r in core))
    io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(qualified=qualified,records=records,fit_ids=pools,new_efficiency_fits=count,
        new_PCA_fits=0,new_source_forward=0,GPU_hours=0,outer_test_metrics_used=False,
        lock_sha256=io.sha(R/'diagnostic_lock.json'),diagnostic_rows_sha256=io.sha(R/'diagnostic_rows.csv'),
        pilot_files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    validate();print('QUALIFIED',qualified,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('register','run'));mode=ap.parse_args().mode
    if mode=='register':register()
    else:
        mutex=R/'diagnostic.mutex'
        with mutex.open('x') as f:f.write(str(os.getpid()))
        start=time.monotonic();rc=1;error=None
        try:run(start);rc=0
        except Exception:error=traceback.format_exc();raise
        finally:
            io.write(R/f'diagnostic_execution_{time.time_ns()}.json',dict(returncode=rc,error=error,
                CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0))
            mutex.unlink()
