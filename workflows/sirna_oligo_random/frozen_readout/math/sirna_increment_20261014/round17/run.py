"""One fixed paired convex-interaction pilot in existing training pools."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent; P=R.parent/'round4'; sys.path.insert(0,str(R))
from nuclear import fit_map,transform,fit_pair
spec=importlib.util.spec_from_file_location('_r17_old_io',P/'run.py'); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
io=old.io; SETTINGS=('Huesken','Mixset','Takayuki'); VIEWS=('core_real','full_real','initial_core','TD24')
FILES=('run.py','nuclear.py','verify.py','test_nuclear.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')

def register():
    assert not (R/'lock.json').exists(); old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite')); assert sum(int(s.get('tests',0)) for s in suites)==7
    assert not any(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites)
    prior=io.read(P/'lock.json'); deps=dict(prior['dependencies'])
    for name in prior['files']: deps[str((P/name).resolve())]=io.sha(P/name)
    for p in (P/'lock.json',P/'results/complete.json',P/'verification.json',R.parent/'round16/closure_receipt.json'):
        deps[str(p.resolve())]=io.sha(p)
    for s in SETTINGS:
        for view in VIEWS[:3]:
            for name in ('transforms.json','arrays.npz'):
                p=P/'results/folds'/s/'fold0'/view/name; deps[str(p.resolve())]=io.sha(p)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_limit_seconds=900,CPU_only=True,alpha=100.,lambda_fraction=.25,new_PCA_fits=0,new_source_forward=0,
        models=120,primary='core_real_Nuclear',settings=SETTINGS,views=VIEWS))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)

def validate():
    lock=io.read(R/'lock.json'); io.check_files(R,lock['files'])
    for p,h in lock['dependencies'].items(): assert io.sha(Path(p))==h,p
    assert lock['versions']==io.versions(); return lock

def stats(y,p):
    pc,ms=old.score(y,p[:,None]); return dict(PCC=float(pc[0]),MSE=float(ms[0]))

def run(start):
    lock=validate(); root=R/'pilot'; assert not root.exists() and not (R/'diagnostic.json').exists()
    threads=threadpool_info(); assert all(x['num_threads']==1 for x in threads)
    rows,xx,identities=old.inputs(); lookup={r['sample_id']:i for i,r in enumerate(rows)}; plan=io.read(P/'plan.json')
    records=[]; details=[]; pools={}; total=0; certified=True; locksha=io.sha(R/'lock.json')
    for setting in SETTINGS:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0); ids=item['fit_ids']; pools[setting]=ids
        fi=np.array([lookup[i] for i in ids]); y=np.array([float(rows[i]['label']) for i in fi]); assert set(ids).isdisjoint(item['test_ids'])
        for view in VIEWS:
            source='core_real' if view=='TD24' else view; rawx=xx[source][fi]; kind='TD24' if view=='TD24' else 'source'
            pf=P/'results/folds'/setting/'fold0'/source/'transforms.json'; projections=io.read(pf)
            with np.load(P/'results/folds'/setting/'fold0'/source/'arrays.npz',allow_pickle=False) as a:
                ref=a['inner_oof'][:,3 if view=='TD24' else 18].copy()
            pred={m:np.full(len(y),np.nan) for m in ('Ridge','Nuclear')}; blocks=[]
            for j,sp in enumerate(item['inner']):
                if time.monotonic()-start>900: raise TimeoutError('No new paired fit')
                tr=np.array(sp['train']); va=np.array(sp['valid']); z=old.transform(rawx[tr],projections['inner'][j]); zv=old.transform(rawx[va],projections['inner'][j])
                if kind=='TD24': z=z[:,:24]; zv=zv[:,:24]
                (x,jx),state=fit_map(z,kind); xv,jv=transform(zv,state); folder=root/setting/view/f'inner{j}'
                def initial(b,c,mu): old.npz_write(folder/'initial.npz',b=b,c=c,intercept=np.array(mu))
                def checkpoint(k,c): old.npz_write(folder/f'parameters_iter_{k:06d}.npz',c=c,iteration=np.array(k))
                before=time.monotonic(); fit=fit_pair(x,jx,y[tr],state['matrix_shape'],initial,checkpoint,start+900); elapsed=time.monotonic()-before
                metadata={}; arrays={}; block=dict(inner=j,n=len(va),reference=stats(y[va],ref[va]))
                for mode,result in fit.items():
                    prediction=xv@result['b']+jv@result['c']+result['intercept']; pred[mode][va]=prediction
                    for key in ('b','c','dual'): arrays[mode+'_'+key]=result[key]
                    arrays[mode+'_prediction']=prediction
                    metadata[mode]={k:v for k,v in result.items() if k not in ('b','c','dual')}
                    block[mode]=dict(metrics=stats(y[va],prediction),rank=result['rank'],certified=result['certified'],iterations=result['iterations'])
                    certified &= result['certified']; total+=1
                io.write(folder/'state.json',dict(setting=setting,view=view,inner=j,map=state,models=metadata,
                    projection_sha256=io.sha(pf),train_indices=tr,valid_indices=va,CPU_fit_wall_seconds=elapsed))
                old.npz_write(folder/'models.npz',**arrays)
                io.write(folder/'complete.json',dict(lock_sha256=locksha,files_sha256={p.name:io.sha(p) for p in folder.iterdir() if p.is_file()}))
                blocks.append(block)
            reference=stats(y,ref); metrics={m:stats(y,pred[m]) for m in pred}
            nuc,paired=metrics['Nuclear'],metrics['Ridge']; delta=nuc['PCC']-reference['PCC']; reduction=100*(1-nuc['MSE']/reference['MSE'])
            record=dict(setting=setting,view=view,n=len(y),metrics=metrics,reference=reference,
                delta_PCC_vs_R4=delta,MSE_reduction_vs_R4_pct=reduction,
                delta_PCC_vs_paired=nuc['PCC']-paired['PCC'],MSE_reduction_vs_paired_pct=100*(1-nuc['MSE']/paired['MSE']),
                pool_qualified=bool(delta>=.005 and reduction>=0),inner_blocks=blocks)
            records.append(record)
            details.extend(dict(setting=setting,view=view,sample_id=s,label=float(y[k]),ridge_prediction=float(pred['Ridge'][k]),
                nuclear_prediction=float(pred['Nuclear'][k]),reference_prediction=float(ref[k])) for k,s in enumerate(ids))
            print(setting,view,'dPCC_R4',delta,'MSE_R4',reduction,'MSE_paired',record['MSE_reduction_vs_paired_pct'],flush=True)
    core=[r for r in records if r['view']=='core_real']; first=sum(r['pool_qualified'] for r in core)>=2 and all(r['delta_PCC_vs_R4']>=-.005 for r in core)
    second=sum(r['MSE_reduction_vs_paired_pct']>=0 for r in core)>=2 and np.mean([r['MSE_reduction_vs_paired_pct'] for r in core])>=0
    qualified=bool(first and second and certified); assert total==120 and len(details)==4*sum(len(v) for v in pools.values())
    io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(qualified=qualified,R4_gate=bool(first),paired_gate=bool(second),all_certified=bool(certified),
        records=records,fit_ids=pools,source_identities=identities,models=120,new_efficiency_fits=120,new_source_forward=0,
        new_PCA_fits=0,GPU_hours=0,outer_test_metrics_used=False,threadpool=threads,lock_sha256=locksha,
        diagnostic_rows_sha256=io.sha(R/'diagnostic_rows.csv'),
        pilot_files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    validate(); print('QUALIFIED',qualified,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=('register','run')); mode=ap.parse_args().mode
    if mode=='register': register()
    else:
        mutex=R/'execution.mutex'
        with mutex.open('x') as f:f.write(str(os.getpid()))
        start=time.monotonic(); rc=1; error=None
        try:run(start);rc=0
        except Exception:error=traceback.format_exc();raise
        finally:
            io.write(R/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0))
            mutex.unlink()
