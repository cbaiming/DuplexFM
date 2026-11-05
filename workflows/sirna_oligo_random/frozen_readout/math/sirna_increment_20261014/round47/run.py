"""One fixed CPU pilot of conditional spectral amplitude refitting."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;B=R.parent;P=B/'round4';D=B/'round21';N=B/'round25'
sys.path.insert(0,str(R))
from spectral_refit import fit,transform,deployment,qualify
spec=importlib.util.spec_from_file_location('_r47_prior4',P/'run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
io=old.io
FILES=('run.py','spectral_refit.py','verify.py','test_spectral_refit.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')


def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==10
    assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
    previous=io.read(B/'round46/closure_receipt.json');assert previous['status']=='closed' and previous['readback_passed']
    for name,h in previous['files_sha256'].items():assert io.sha(B/name)==h,name
    prior=io.read(N/'lock.json');io.check_files(N,prior['files']);deps=dict(prior['dependencies'])
    for name in prior['files']:deps[str((N/name).resolve())]=io.sha(N/name)
    for path in (N/'lock.json',N/'closure_receipt.json',N/'diagnostic.json',N/'verification.json',Path(io.__file__),
                 B/'round46/closure_receipt.json',B/'round36/verify.py',P/'verify.py'):
        deps[str(path.resolve())]=io.sha(path)
    assert io.read(N/'verification.json')['passed']
    for name,h in io.read(N/'diagnostic.json')['pilot_files_sha256'].items():
        path=N/'pilot'/name;assert io.sha(path)==h;deps[str(path.resolve())]=h
    for setting in old.SETTINGS[:3]:
        path=P/'results/folds'/setting/'fold0/core_real/arrays.npz';deps[str(path.resolve())]=io.sha(path)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=900,source_seed=42,settings=old.SETTINGS[:3],sources=old.SOURCES,primary='core_real',
        alpha=100.,previous_lambda_ratio=.1,rank_threshold=1e-8,nonnegative_amplitudes=True,
        new_heads=45,new_supervised_direction_decompositions=45,new_internal_OOF_rows=6897,reused_native_heads=45,reused_unique_TD_heads=15,
        new_source_forward=0,new_PCA=0,new_original_scalers=0))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for path,h in lock['dependencies'].items():assert io.sha(Path(path))==h,path
    assert io.versions()==lock['versions'];return lock


def stats(y,p):
    pc,ms=old.score(y,p[:,None]);return dict(PCC=float(pc[0]),MSE=float(ms[0]))


def run(start):
    validate();root=R/'pilot';assert not root.exists() and not (R/'diagnostic.json').exists()
    threads=threadpool_info();assert all(t['num_threads']==1 for t in threads)
    rows,xx,identities=old.inputs();lookup={r['sample_id']:i for i,r in enumerate(rows)};plan=io.read(P/'plan.json')
    records=[];details=[];pools={};count=0;solves=0;numeric=True
    for setting in old.SETTINGS[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids'];pools[setting]=ids
        assert set(ids).isdisjoint(item['test_ids']);fi=np.array([lookup[s] for s in ids]);y=np.array([float(rows[i]['label']) for i in fi])
        tdpath=P/'results/folds'/setting/'fold0/core_real/arrays.npz'
        with np.load(tdpath,allow_pickle=False) as a:tdpred=a['inner_oof'][:,3].copy()
        for source in old.SOURCES:
            raw=xx[source][fi];pred=np.full(len(ids),np.nan);ref=pred.copy();blocks=[];coverage=[]
            for j,split in enumerate(item['inner']):
                if time.monotonic()-start>900:raise TimeoutError('CPU stage cap')
                tr,va=np.array(split['train']),np.array(split['valid']);coverage+=va.tolist();assert set(tr).isdisjoint(va)
                previous=N/'pilot'/setting/source/f'inner{j}';control=D/'pilot'/setting/source/'dense'/f'inner{j}'
                nc=io.read(previous/'complete.json');io.check_files(previous,nc['files_sha256'])
                assert nc['lock_sha256']==io.sha(N/'lock.json')
                ns=io.read(previous/'state.json');st=io.read(control/'transform.json')
                assert ns['train_indices']==tr.tolist() and ns['valid_indices']==va.tolist() and ns['transform_sha256']==io.sha(control/'transform.json')
                assert ns['numerically_qualified']
                with np.load(previous/'model.npz',allow_pickle=False) as a:ref[va]=a['prediction'];oldbeta=a['beta'].copy()
                z=transform(raw,st);out=root/setting/source/f'inner{j}';before=time.monotonic()
                model,cert=fit(z[tr],y[tr],oldbeta);assert cert['selected_rank']==ns['certificate']['rank']
                pp=z[va]@model['beta']+float(model['intercept']);coef,offset=deployment(st,model['beta'],float(model['intercept']))
                pred[va]=pp;count+=1;solves+=cert['active_system_solves'];valid=bool(cert['certified']);numeric &= valid
                old.npz_write(out/'model.npz',**model,prediction=pp,deployment_coef=coef,deployment_intercept=np.array(offset))
                io.write(out/'state.json',dict(setting=setting,source=source,inner=j,train_indices=tr,valid_indices=va,
                    certificate=cert,numerically_qualified=valid,transform_sha256=io.sha(control/'transform.json'),
                    native_reference_sha256=io.sha(previous/'model.npz'),TD_reference_sha256=io.sha(tdpath),CPU_fit_wall_seconds=time.monotonic()-before))
                io.write(out/'complete.json',dict(lock_sha256=io.sha(R/'lock.json'),files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
                blocks.append(dict(inner=j,n=len(va),metrics=stats(y[va],pp),reference=stats(y[va],ref[va]),TD_metrics=stats(y[va],tdpred[va]),
                    selected_rank=cert['selected_rank'],active_amplitudes=cert['active_amplitudes'],
                    amplitude_change_norm=float(np.linalg.norm(model['amplitudes']-model['old_singular_values'])),
                    objective_reduction=cert['reference_objective_without_trace']-cert['objective_without_trace'],numerically_qualified=valid))
            assert sorted(coverage)==list(range(len(ids))) and np.isfinite(pred).all() and np.isfinite(ref).all()
            m,rr,td=stats(y,pred),stats(y,ref),stats(y,tdpred)
            rec=dict(setting=setting,source=source,n=len(ids),metrics=m,reference=rr,TD_metrics=td,
                delta_PCC_vs_R25=m['PCC']-rr['PCC'],MSE_reduction_vs_R25_pct=100*(1-m['MSE']/rr['MSE']),inner_blocks=blocks)
            records.append(rec)
            details.extend(dict(setting=setting,source=source,sample_id=s,label=float(y[k]),prediction=float(pred[k]),
                native_reference_prediction=float(ref[k]),TD_prediction=float(tdpred[k])) for k,s in enumerate(ids))
            print(setting,source,'deltaPCC',rec['delta_PCC_vs_R25'],'MSEred',rec['MSE_reduction_vs_R25_pct'],flush=True)
    assert count==45 and len(details)==6897;gates=qualify(records,numeric)
    io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(**gates,records=records,fit_ids=pools,source_identities=identities,
        new_heads=45,new_supervised_direction_decompositions=45,new_active_system_solves=solves,new_TD_profile_factors=45,
        new_internal_OOF_rows=6897,reused_native_heads=45,reused_unique_TD_heads=15,reused_unique_TD_rows=2299,
        new_source_forward=0,new_PCA=0,new_original_scalers=0,GPU_hours=0,outer_test_metrics_used=False,threadpool=threads,
        lock_sha256=io.sha(R/'lock.json'),diagnostic_rows_sha256=io.sha(R/'diagnostic_rows.csv'),
        pilot_files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    validate();print('QUALIFIED',gates['qualified'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('register','run'))
    if ap.parse_args().mode=='register':register()
    else:
        mutex=R/'execution.mutex'
        with mutex.open('x') as f:f.write(str(os.getpid()))
        start=time.monotonic();rc=1;error=None
        try:run(start);rc=0
        except Exception:error=traceback.format_exc();raise
        finally:
            io.write(R/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0))
            mutex.unlink()
