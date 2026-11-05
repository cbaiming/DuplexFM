"""One finite source-response-surface pilot; all reference heads are reused."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4';Q=R.parent/'round5'
sys.path.insert(0,str(R))
from quadratic import fit_map,transform,fit,direct_predict,qualify
sp=importlib.util.spec_from_file_location('_r28_prior4',P/'run.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
io=old.io;SETTINGS=old.SETTINGS[:3];SOURCES=old.SOURCES
FILES=('run.py','quadratic.py','verify.py','test_quadratic.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')


def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==8
    assert not any(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites)
    previous=io.read(R.parent/'round27/closure_receipt.json')
    assert previous['status']=='closed' and previous['readback_passed']
    for name,h in previous['files_sha256'].items():assert io.sha(R.parent/name)==h,name
    plock=io.read(P/'lock.json');qlock=io.read(Q/'lock.json');io.check_files(Q,qlock['files'])
    assert io.read(Q/'plan.json')['folds']==io.read(P/'plan.json')['folds']
    qv=io.read(Q/'verification.json');receipt=io.read(Q/'artifact_metric_execution_receipt.json')
    assert qv['passed'] and receipt['returncode']==0 and receipt['verification_sha256']==io.sha(Q/'verification.json')
    assert qv['complete_sha256']==receipt['results_complete_sha256']==io.sha(Q/'results/complete.json')
    assert qv['lock_sha256']==receipt['original_lock_sha256']==io.sha(Q/'lock.json')
    assert qv['compat_execution_lock_sha256']==receipt['execution_lock_sha256']==io.sha(Q/'artifact_metric_execution_lock.json')
    executionlock=io.read(Q/'artifact_metric_execution_lock.json');io.check_files(Q,executionlock['files'])
    deps=dict(plock['dependencies'])
    for folder,source in ((P,plock),(Q,qlock)):
        for name in source['files']:deps[str((folder/name).resolve())]=io.sha(folder/name)
    for name in executionlock['files']:deps[str((Q/name).resolve())]=io.sha(Q/name)
    for p in (P/'lock.json',P/'plan.json',P/'results/complete.json',P/'verification.json',Path(io.__file__),
              Q/'lock.json',Q/'results/complete.json',Q/'verification.json',Q/'artifact_metric_execution_lock.json',
              Q/'artifact_metric_execution_receipt.json',R.parent/'round27/closure_receipt.json'):
        deps[str(p.resolve())]=io.sha(p)
    for setting in SETTINGS:
        for source in SOURCES:
            for name in ('transforms.json','arrays.npz'):
                p=P/'results/folds'/setting/'fold0'/source/name;deps[str(p.resolve())]=io.sha(p)
        folder=Q/'results/folds'/setting/'fold0'/'TDQuadratic'
        c=io.read(folder/'complete.json');io.check_files(folder,c['files_sha256'])
        assert c['lock_sha256']==io.sha(Q/'lock.json')
        for p in [folder/'complete.json']+[folder/name for name in c['files_sha256']]:deps[str(p.resolve())]=io.sha(p)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=900,source_seed=42,settings=SETTINGS,sources=SOURCES,primary='core_real',
        alpha=100.,new_quadratic_terms=136,new_heads=45,new_maps=45,new_internal_OOF_rows=6897,
        reused_linear_heads=45,reused_TD_quadratic_heads=15,reused_unique_TD_OOF_rows=2299,
        new_source_forward=0,new_PCA=0))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for name,h in lock['dependencies'].items():assert io.sha(Path(name))==h,name
    assert lock['versions']==io.versions();return lock


def stats(y,p):
    pc,ms=old.score(y,p[:,None]);return dict(PCC=float(pc[0]),MSE=float(ms[0]))


def run(start):
    validate();root=R/'pilot';assert not root.exists() and not (R/'diagnostic.json').exists()
    threads=threadpool_info();assert all(t['num_threads']==1 for t in threads)
    rows,xx,identities=old.inputs();lookup={r['sample_id']:i for i,r in enumerate(rows)};plan=io.read(P/'plan.json')
    records=[];details=[];pools={};count=0;numeric=True;locksha=io.sha(R/'lock.json')
    for setting in SETTINGS:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids'];pools[setting]=ids;assert set(ids).isdisjoint(item['test_ids'])
        fi=np.array([lookup[s] for s in ids]);y=np.array([float(rows[i]['label']) for i in fi])
        tdpath=Q/'results/folds'/setting/'fold0'/'TDQuadratic'/'arrays.npz'
        with np.load(tdpath,allow_pickle=False) as a:tdref=a['inner_oof'][:,18].copy()
        for source in SOURCES:
            raw=xx[source][fi];pf=P/'results/folds'/setting/'fold0'/source/'transforms.json';states=io.read(pf)
            with np.load(pf.parent/'arrays.npz',allow_pickle=False) as a:ref=a['inner_oof'][:,18].copy()
            prediction=np.full(len(y),np.nan);blocks=[]
            for j,split in enumerate(item['inner']):
                if time.monotonic()-start>900:raise TimeoutError('No new head after CPU cap')
                tr,va=np.array(split['train']),np.array(split['valid']);z=old.transform(raw,states['inner'][j]);out=root/setting/source/f'inner{j}'
                before=time.monotonic();x,t=fit_map(z[tr]);beta,center,cert=fit(x,y[tr]);pp=transform(z[va],t)@beta+center
                np.testing.assert_allclose(pp,direct_predict(z[va],t,beta,center),atol=1e-10,rtol=1e-10)
                prediction[va]=pp;count+=1;numeric &= cert['numerically_qualified']
                old.npz_write(out/'model.npz',beta=beta,intercept=np.array(center),prediction=pp)
                io.write(out/'map.json',t)
                io.write(out/'state.json',dict(setting=setting,source=source,inner=j,train_indices=tr,valid_indices=va,
                    certificate=cert,reference_transform_sha256=io.sha(pf),TD_reference_sha256=io.sha(tdpath),
                    CPU_fit_wall_seconds=time.monotonic()-before))
                io.write(out/'complete.json',dict(lock_sha256=locksha,files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
                blocks.append(dict(inner=j,n=len(va),metrics=stats(y[va],pp),reference=stats(y[va],ref[va]),
                    TD_quadratic_metrics=stats(y[va],tdref[va]),numerically_qualified=cert['numerically_qualified']))
            assert np.isfinite(prediction).all();m,rr,tm=stats(y,prediction),stats(y,ref),stats(y,tdref)
            rec=dict(setting=setting,source=source,n=len(y),metrics=m,reference=rr,TD_quadratic_metrics=tm,
                delta_PCC_vs_R4=m['PCC']-rr['PCC'],MSE_reduction_vs_R4_pct=100*(1-m['MSE']/rr['MSE']),inner_blocks=blocks)
            records.append(rec)
            details.extend(dict(setting=setting,source=source,sample_id=s,label=float(y[k]),prediction=float(prediction[k]),
                reference_prediction=float(ref[k]),TD_quadratic_prediction=float(tdref[k])) for k,s in enumerate(ids))
            print(setting,source,'deltaPCC',rec['delta_PCC_vs_R4'],'MSEred',rec['MSE_reduction_vs_R4_pct'],flush=True)
    assert count==45 and len(details)==6897
    gates=qualify(records,numeric);io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(**gates,records=records,fit_ids=pools,source_identities=identities,
        new_heads=count,new_maps=count,new_internal_OOF_rows=6897,reused_linear_heads=45,reused_TD_quadratic_heads=15,
        reused_unique_TD_OOF_rows=2299,new_source_forward=0,new_PCA=0,GPU_hours=0,outer_test_metrics_used=False,
        threadpool=threads,lock_sha256=locksha,diagnostic_rows_sha256=io.sha(R/'diagnostic_rows.csv'),
        pilot_files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    validate();print('QUALIFIED',gates['qualified'],flush=True)


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
