import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,importlib.util,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4'
spec=importlib.util.spec_from_file_location('_r12_old',P/'run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
io=old.io

def register():
    assert not (R/'diagnostic_lock.json').exists();old.validate()
    lock=io.read(P/'lock.json');deps=dict(lock['dependencies'])
    for n in lock['files']:deps[str((P/n).resolve())]=io.sha(P/n)
    for p in (P/'lock.json',P/'verification.json',P/'results/complete.json',R.parent/'round11/verification.json'):
        deps[str(p.resolve())]=io.sha(p)
    for s in old.SETTINGS[:3]:
        for source in old.SOURCES:
            p=P/'results/folds'/s/'fold0'/source/'transforms.json';deps[str(p.resolve())]=io.sha(p)
    io.write(R/'diagnostic_lock.json',dict(files={n:io.sha(R/n) for n in ('diagnose.py','DIAGNOSTIC_PROTOCOL.md')},
        dependencies=deps,real_labels_used=False,new_source_forward=0,GPU_hours=0,
        qualification='at least 2 Core pools with lambda1/lambda16>=3 and lambda16>1e-12'))
    print('REGISTERED',io.sha(R/'diagnostic_lock.json'),flush=True)

def run():
    assert not (R/'diagnostic.json').exists();lock=io.read(R/'diagnostic_lock.json');io.check_files(R,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    rows,xx,_=old.inputs();index={r['sample_id']:i for i,r in enumerate(rows)};plan=io.read(P/'plan.json')
    records=[];pools={}
    for setting in old.SETTINGS[:3]:
        f=next(t for t in plan['folds'] if t['setting']==setting and t['fold']==0);ids=f['fit_ids'];pools[setting]=ids
        order=sorted(range(len(ids)),key=lambda i:hashlib.sha256(('R12:42:'+ids[i]).encode()).hexdigest())
        halves=(order[:len(order)//2],order[len(order)//2:]);ii=[index[s] for s in ids]
        for source in old.SOURCES:
            t=io.read(P/'results/folds'/setting/'fold0'/source/'transforms.json')['final'];e=np.asarray(t['eigenvalues'])
            raw=(xx[source][ii,24:]-t['src_mean'])/t['src_scale'];components=[]
            for half in halves:
                z=raw[half]-raw[half].mean(0);_,v=np.linalg.eigh(z.T@z/len(z));components.append(v[:,-16:][:,::-1].T)
            leading=float(np.sum((components[0][:8]@components[1][:8].T)**2)/8)
            trailing=float(np.sum((components[0][8:]@components[1][8:].T)**2)/8)
            ratio=float(e[0]/max(e[-1],1e-12));qualified=bool(e[-1]>1e-12 and ratio>=3)
            record=dict(setting=setting,source=source,n=len(ids),eigenvalues=e,variance_ratio_first_to_16=ratio,
                half_top8_overlap=leading,half_next8_overlap=trailing,top_minus_next_overlap=leading-trailing,
                half_indices=halves,qualified=qualified);records.append(record)
            print(setting,source,'ratio',ratio,'half_overlap',leading,trailing,flush=True)
    qualified=sum(r['qualified'] for r in records if r['source']=='core_real')>=2
    io.write(R/'diagnostic.json',dict(qualified=qualified,rows=records,fit_ids=pools,real_labels_used=False,
        new_prediction_metrics=False,unsupervised_half_decompositions=18,new_efficiency_fits=0,
        GPU_hours=0,new_source_forward=0,lock_sha256=io.sha(R/'diagnostic_lock.json')))
    print('QUALIFIED',qualified,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('register','run'));mode=ap.parse_args().mode
    if mode=='register':register()
    else:
        start=time.monotonic();rc=1;error=None
        try:run();rc=0
        except Exception:error=traceback.format_exc();raise
        finally:io.write(R/f'diagnostic_execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0,new_efficiency_fits=0))
