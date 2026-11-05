"""Fixed-input, label-free qualification of regional information retention."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;P=ROOT.parent/'round4'
spec=importlib.util.spec_from_file_location('_r10_r4_io',P/'run.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
io=prior.io
SETTINGS=('Huesken','Mixset','Takayuki');VIEWS=('core_real','full_real','initial_core')
MODEL=ROOT.parents[1]/'mechanism_transfer_20261010/results/h100_source_20261011/extracted/code'

def split(z):
    a,b,c=np.split(z,3,axis=1)
    return (a+b+c)/np.sqrt(3.),np.c_[(a-b)/np.sqrt(2.),(a+b-2*c)/np.sqrt(6.)]

def evidence(x,t):
    z=(x[:,24:]-t['src_mean'])/t['src_scale']-t['pca_mean']
    components=np.asarray(t['components'])
    reconstructed=z@components.T@components
    common,contrast=split(z);hc,hd=split(reconstructed)
    vc=float(np.sum(common*common));vd=float(np.sum(contrast*contrast));total=vc+vd
    assert total>0 and vc>0 and vd>0
    rc=1-float(np.sum((common-hc)**2))/vc
    rd=1-float(np.sum((contrast-hd)**2))/vd
    all_r=1-float(np.sum((z-reconstructed)**2))/total
    assert np.isclose(total,np.sum(z*z),atol=1e-8,rtol=1e-12)
    return dict(n=len(z),common_variance_fraction=vc/total,contrast_variance_fraction=vd/total,
                common_reconstruction_explained=rc,contrast_reconstruction_explained=rd,
                common_minus_contrast=rc-rd,global_PCA16_reconstruction_explained=all_r,
                qualifies=bool(vd/total>=.1 and rd<.5 and rc-rd>=.2))

def register():
    assert not (ROOT/'diagnostic_lock.json').exists()
    prior.validate();old=io.read(P/'lock.json');deps=dict(old['dependencies'])
    for n in old['files']:deps[str((P/n).resolve())]=io.sha(P/n)
    for path in (P/'lock.json',P/'verification.json',P/'results/complete.json',
                 ROOT.parent/'round9/verification.json',MODEL/'duplexfm_joint/model.py',MODEL/'duplexfm_mechanism/model.py'):
        deps[str(path.resolve())]=io.sha(path)
    for setting in SETTINGS:
        for view in VIEWS:
            path=P/'results/folds'/setting/'fold0'/view/'transforms.json'
            deps[str(path.resolve())]=io.sha(path)
    io.write(ROOT/'diagnostic_lock.json',dict(files={n:io.sha(ROOT/n) for n in ('diagnose.py','DIAGNOSTIC_PROTOCOL.md')},
        dependencies=deps,real_labels_used=False,new_source_forward=0,GPU_hours=0,
        qualification='at least 2 Core pools: contrast variance >=.1, retained <.5, common-minus-contrast >=.2'))
    print('REGISTERED',io.sha(ROOT/'diagnostic_lock.json'),flush=True)

def run():
    assert not (ROOT/'diagnostic.json').exists()
    lock=io.read(ROOT/'diagnostic_lock.json');io.check_files(ROOT,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    rows,xx,_=prior.inputs();ids={r['sample_id']:i for i,r in enumerate(rows)}
    plan=io.read(P/'plan.json');records=[];pools={}
    for setting in SETTINGS:
        item=next(p for p in plan['folds'] if p['setting']==setting and p['fold']==0)
        pools[setting]=item['fit_ids'];idx=np.array([ids[s] for s in item['fit_ids']])
        for view in VIEWS:
            t=io.read(P/'results/folds'/setting/'fold0'/view/'transforms.json')['final']
            record=dict(setting=setting,view=view,**evidence(xx[view][idx],t));records.append(record)
            print(record,flush=True)
    qualified=sum(r['qualifies'] for r in records if r['view']=='core_real')>=2
    io.write(ROOT/'diagnostic.json',dict(qualified=qualified,statistics=records,fit_ids=pools,
        real_labels_used=False,outer_test_metrics_used=False,new_fits=0,new_source_forward=0,GPU_hours=0,
        lock_sha256=io.sha(ROOT/'diagnostic_lock.json')))
    print('QUALIFIED',qualified,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('register','run'));mode=p.parse_args().mode
    if mode=='register':register()
    else:
        started=time.monotonic();rc=1;error=None
        try:run();rc=0
        except Exception:error=traceback.format_exc();raise
        finally:io.write(ROOT/f'diagnostic_execution_{time.time_ns()}.json',dict(returncode=rc,error=error,
            CPU_wall_seconds=time.monotonic()-started,GPU_hours=0,new_fits=0,new_source_forward=0))
