"""Registered 60-fit CPU-only training-internal neural readout pilot."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent; P=R.parent/'round4'; sys.path.insert(0,str(R))
from mlp import fit,predict
spec=importlib.util.spec_from_file_location('_r16_old_io',P/'run.py'); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
io=old.io; SETTINGS=('Huesken','Mixset','Takayuki'); VIEWS=('core_real','full_real','initial_core','TD24')
FILES=('run.py','mlp.py','verify.py','test_mlp.py','test_audit.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml','pytest_math_first.xml')


def register():
    assert not (R/'lock.json').exists(); old.validate()
    tests=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in tests)==8
    assert not any(int(s.get('errors',0))+int(s.get('failures',0))+int(s.get('skipped',0)) for s in tests)
    previous=io.read(P/'lock.json'); deps=dict(previous['dependencies'])
    for n in previous['files']: deps[str((P/n).resolve())]=io.sha(P/n)
    for path in (P/'lock.json',P/'verification.json',P/'results/complete.json',R.parent/'round15/closure_receipt.json'):
        deps[str(path.resolve())]=io.sha(path)
    for setting in SETTINGS:
        for source in VIEWS[:3]:
            for name in ('arrays.npz','transforms.json'):
                path=P/'results/folds'/setting/'fold0'/source/name; deps[str(path.resolve())]=io.sha(path)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=900,settings=SETTINGS,views=VIEWS,outer_fold=0,
        alpha=10.,hidden=16,head_seed=42,new_source_forward=0,new_PCA_fits=0,
        qualification='Core: at least two dPCC>=.005 and MSE nonworse; all dPCC>=-.005; all 60 optimization numerically qualified'))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock=io.read(R/'lock.json'); io.check_files(R,lock['files'])
    for p,h in lock['dependencies'].items(): assert io.sha(Path(p))==h,p
    assert lock['versions']==io.versions(); return lock


def stats(y,p):
    pc,ms=old.score(y,p[:,None]); return dict(PCC=float(pc[0]),MSE=float(ms[0]))


def run(start):
    lock=validate(); root=R/'pilot'; assert not root.exists() and not (R/'diagnostic.json').exists()
    threads=threadpool_info(); assert all(r['num_threads']==1 for r in threads)
    rows,xx,identities=old.inputs(); lookup={r['sample_id']:i for i,r in enumerate(rows)}; plan=io.read(P/'plan.json')
    records=[]; details=[]; pools={}; count=0; all_converged=True; locksha=io.sha(R/'lock.json')
    for setting in SETTINGS:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids']; pools[setting]=ids; fi=np.array([lookup[s] for s in ids]); y=np.array([float(rows[i]['label']) for i in fi])
        assert set(ids).isdisjoint(item['test_ids'])
        for view in VIEWS:
            source='core_real' if view=='TD24' else view; x=xx[source][fi]; folder=root/setting/view
            projection_file=P/'results/folds'/setting/'fold0'/source/'transforms.json'; reference=io.read(projection_file)
            with np.load(P/'results/folds'/setting/'fold0'/source/'arrays.npz',allow_pickle=False) as a:
                previous=a['inner_oof'][:,3 if view=='TD24' else 18].copy()
            predicted=np.full(len(y),np.nan); blocks=[]; view_converged=True
            for j,split in enumerate(item['inner']):
                if time.monotonic()-start>lock['CPU_limit_seconds']: raise TimeoutError('No new MLP fit after deadline')
                tr=np.array(split['train']); va=np.array(split['valid']); out=folder/f'inner{j}'; t=reference['inner'][j]
                z=old.transform(x[tr],t); zv=old.transform(x[va],t)
                if view=='TD24': z=z[:,:24]; zv=zv[:,:24]
                before=time.monotonic()
                def initial(theta,mean,scale):
                    old.npz_write(out/'initial.npz',theta=theta,y_mean=np.array(mean),y_scale=np.array(scale))
                def checkpoint(k,theta): old.npz_write(out/f'parameters_iter_{k:06d}.npz',theta=theta,iteration=np.array(k))
                state=fit(z,y[tr],save_initial=initial,save_checkpoint=checkpoint,deadline=start+lock['CPU_limit_seconds'])
                elapsed=time.monotonic()-before; predicted[va]=predict(zv,state)
                old.npz_write(out/'model.npz',theta=state['theta'],trace=state['trace'],prediction=predicted[va])
                metadata={k:v for k,v in state.items() if k not in ('theta','initial','trace')}
                metadata.update(setting=setting,view=view,inner=j,source=source,head_seed=42,
                    n_train=len(tr),n_valid=len(va),train_indices=tr,valid_indices=va,
                    projection_sha256=io.sha(projection_file),CPU_fit_wall_seconds=elapsed)
                io.write(out/'state.json',metadata)
                io.write(out/'complete.json',dict(lock_sha256=locksha,files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
                count+=1; view_converged &= state['qualified_numerically']; all_converged &= state['qualified_numerically']
                blocks.append(dict(inner=j,n=len(va),new=stats(y[va],predicted[va]),old=stats(y[va],previous[va]),
                    n_iter=state['n_iter'],max_abs_gradient=state['max_abs_gradient'],qualified_numerically=state['qualified_numerically']))
            assert np.isfinite(predicted).all()
            m,pm=stats(y,predicted),stats(y,previous); delta=m['PCC']-pm['PCC']; reduction=100*(1-m['MSE']/pm['MSE'])
            record=dict(setting=setting,view=view,n=len(y),input_width=24 if view=='TD24' else 40,
                parameters=417 if view=='TD24' else 673,new=m,old=pm,delta_PCC=delta,MSE_relative_reduction_pct=reduction,
                pool_qualified=bool(delta>=.005 and reduction>=0),all_fits_numerically_qualified=bool(view_converged),inner_blocks=blocks)
            records.append(record)
            details.extend(dict(setting=setting,view=view,sample_id=s,label=float(y[i]),new_prediction=float(predicted[i]),reference_prediction=float(previous[i])) for i,s in enumerate(ids))
            print(setting,view,'dPCC',delta,'MSE_reduction_pct',reduction,'converged',view_converged,flush=True)
    assert count==60 and len(details)==4*sum(len(v) for v in pools.values())
    core=[r for r in records if r['view']=='core_real']; metric_gate=sum(r['pool_qualified'] for r in core)>=2 and all(r['delta_PCC']>=-.005 for r in core)
    io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(qualified=bool(metric_gate and all_converged),metric_gate=bool(metric_gate),
        all_numerically_qualified=bool(all_converged),records=records,fit_ids=pools,identities=identities,
        new_efficiency_fits=count,new_PCA_fits=0,new_source_forward=0,GPU_hours=0,outer_test_metrics_used=False,
        threadpool=threads,lock_sha256=locksha,diagnostic_rows_sha256=io.sha(R/'diagnostic_rows.csv'),
        pilot_files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    validate(); print('QUALIFIED',bool(metric_gate and all_converged),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=('register','run')); mode=ap.parse_args().mode
    if mode=='register': register()
    else:
        mutex=R/'execution.mutex'
        with mutex.open('x') as f: f.write(str(os.getpid()))
        start=time.monotonic(); rc=1; error=None
        try: run(start); rc=0
        except Exception: error=traceback.format_exc(); raise
        finally:
            io.write(R/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,
                GPU_hours=0,new_source_forward=0))
            mutex.unlink()
