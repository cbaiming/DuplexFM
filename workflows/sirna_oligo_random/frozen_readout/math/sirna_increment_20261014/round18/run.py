"""One preregistered inner-training-only two-block evidence pilot."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4';sys.path.insert(0,str(R))
from evidence import fit,predict
spec=importlib.util.spec_from_file_location('_r18_prior4',P/'run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
io=old.io;SETTINGS=('Huesken','Mixset','Takayuki');VIEWS=('core_real','full_real','initial_core','TD24')
FILES=('run.py','evidence.py','verify.py','test_evidence.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')


def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==9
    assert not any(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites)
    prior=io.read(P/'lock.json');deps=dict(prior['dependencies'])
    for name in prior['files']:deps[str((P/name).resolve())]=io.sha(P/name)
    for path in (P/'lock.json',P/'results/complete.json',P/'verification.json',R.parent/'round17/closure_receipt.json'):
        deps[str(path.resolve())]=io.sha(path)
    for setting in SETTINGS:
        for view in VIEWS[:3]:
            for name in ('transforms.json','arrays.npz'):
                p=P/'results/folds'/setting/'fold0'/view/name;deps[str(p.resolve())]=io.sha(p)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=900,new_PCA_fits=0,new_source_forward=0,models=60,
        primary='core_real',settings=SETTINGS,views=VIEWS,lambda_initial=100.,lambda_bounds=[1e-6,1e8],
        max_iterations=1000,projected_gradient_tolerance=1e-6))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    assert lock['versions']==io.versions();return lock


def stats(y,p):
    pc,ms=old.score(y,p[:,None]);return dict(PCC=float(pc[0]),MSE=float(ms[0]))


def run(start):
    validate();root=R/'pilot';assert not root.exists() and not (R/'diagnostic.json').exists()
    threads=threadpool_info();assert all(t['num_threads']==1 for t in threads)
    rows,xx,identities=old.inputs();lookup={r['sample_id']:i for i,r in enumerate(rows)};plan=io.read(P/'plan.json')
    records=[];details=[];pools={};count=0;allgood=True;locksha=io.sha(R/'lock.json')
    for setting in SETTINGS:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids'];pools[setting]=ids;assert set(ids).isdisjoint(item['test_ids'])
        fi=np.array([lookup[s] for s in ids]);y=np.array([float(rows[i]['label']) for i in fi])
        for view in VIEWS:
            source='core_real' if view=='TD24' else view;raw=xx[source][fi];widths=(24,) if view=='TD24' else (24,16)
            pf=P/'results/folds'/setting/'fold0'/source/'transforms.json';projections=io.read(pf)
            with np.load(P/'results/folds'/setting/'fold0'/source/'arrays.npz',allow_pickle=False) as a:
                ref=a['inner_oof'][:,3 if view=='TD24' else 18].copy()
            prediction=np.full(len(y),np.nan);blocks=[]
            for j,sp in enumerate(item['inner']):
                if time.monotonic()-start>900:raise TimeoutError('No new fit: stage cap')
                tr=np.array(sp['train']);va=np.array(sp['valid'])
                z=old.transform(raw[tr],projections['inner'][j])[:,:sum(widths)]
                zv=old.transform(raw[va],projections['inner'][j])[:,:sum(widths)]
                folder=root/setting/view/f'inner{j}'
                def initial(theta,beta,mean,intercept):old.npz_write(folder/'initial.npz',theta=theta,beta=beta,mean=mean,intercept=np.array(intercept))
                def checkpoint(k,theta):old.npz_write(folder/f'parameters_iter_{k:06d}.npz',theta=theta,iteration=np.array(k))
                before=time.monotonic();state=fit(z,y[tr],widths,initial,checkpoint,start+900);elapsed=time.monotonic()-before
                pp=predict(zv,state);prediction[va]=pp;count+=1;allgood &= state['numerically_qualified']
                old.npz_write(folder/'model.npz',beta=state['beta'],mean=state['mean'],theta=state['theta'],prediction=pp)
                metadata={k:v for k,v in state.items() if k not in ('beta','mean')}
                io.write(folder/'state.json',dict(setting=setting,view=view,inner=j,model=metadata,
                    train_indices=tr,valid_indices=va,projection_sha256=io.sha(pf),CPU_fit_wall_seconds=elapsed))
                io.write(folder/'complete.json',dict(lock_sha256=locksha,files_sha256={p.name:io.sha(p) for p in folder.iterdir() if p.is_file()}))
                blocks.append(dict(inner=j,n=len(va),metrics=stats(y[va],pp),reference=stats(y[va],ref[va]),
                    lambdas=state['lambdas'],iterations=state['iterations'],numerically_qualified=state['numerically_qualified']))
            assert np.isfinite(prediction).all();metric=stats(y,prediction);reference=stats(y,ref)
            delta=metric['PCC']-reference['PCC'];reduction=100*(1-metric['MSE']/reference['MSE'])
            records.append(dict(setting=setting,view=view,n=len(y),metrics=metric,reference=reference,
                delta_PCC_vs_R4=delta,MSE_reduction_vs_R4_pct=reduction,
                pool_qualified=bool(delta>=.005 and reduction>=0),inner_blocks=blocks))
            details.extend(dict(setting=setting,view=view,sample_id=s,label=float(y[k]),prediction=float(prediction[k]),
                reference_prediction=float(ref[k])) for k,s in enumerate(ids))
            print(setting,view,'deltaPCC',delta,'MSE_reduction_pct',reduction,flush=True)
    core=[r for r in records if r['view']=='core_real']
    performance=sum(r['pool_qualified'] for r in core)>=2 and all(r['delta_PCC_vs_R4']>=-.005 for r in core)
    qualified=bool(performance and allgood);assert count==60 and len(details)==4*sum(len(v) for v in pools.values())==9196
    io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(qualified=qualified,performance_gate=bool(performance),all_numerically_qualified=bool(allgood),
        records=records,fit_ids=pools,source_identities=identities,new_efficiency_fits=count,new_source_forward=0,new_PCA_fits=0,
        GPU_hours=0,outer_test_metrics_used=False,threadpool=threads,lock_sha256=locksha,
        diagnostic_rows_sha256=io.sha(R/'diagnostic_rows.csv'),
        pilot_files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    validate();print('QUALIFIED',qualified,'PERFORMANCE',performance,'NUMERICAL',allgood,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('register','run'));mode=ap.parse_args().mode
    if mode=='register':register()
    else:
        mutex=R/'execution.mutex'
        with mutex.open('x') as f:f.write(str(os.getpid()))
        start=time.monotonic();rc=1;error=None
        try:run(start);rc=0
        except Exception:error=traceback.format_exc();raise
        finally:
            io.write(R/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0))
            mutex.unlink()
