"""One native matrix-penalized head; exact dense controls reused by hash."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4';D=R.parent/'round21'
sys.path.insert(0,str(R))
from native_trace import fit,transform,deployment,qualify
sp=importlib.util.spec_from_file_location('_r25_old4',P/'run.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
io=old.io;SETTINGS=old.SETTINGS[:3];SOURCES=old.SOURCES
FILES=('run.py','native_trace.py','verify.py','test_native_trace.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')


def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==8
    assert not any(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites)
    previous=io.read(R.parent/'round24/closure_receipt.json')
    for name,h in previous['files_sha256'].items():assert io.sha(R.parent/name)==h,name
    dlock=io.read(D/'lock.json');io.check_files(D,dlock['files'])
    deps=dict(dlock['dependencies'])
    for name in dlock['files']:deps[str((D/name).resolve())]=io.sha(D/name)
    for p in (P/'plan.json',D/'lock.json',D/'closure_receipt.json',D/'diagnostic.json',D/'verification.json',
              R.parent/'round17/nuclear.py',R.parent/'round17/lock.json',R.parent/'round24/closure_receipt.json'):
        deps[str(p.resolve())]=io.sha(p)
    nlock=io.read(R.parent/'round17/lock.json');assert io.sha(R.parent/'round17/nuclear.py')==nlock['files']['nuclear.py']
    for setting in SETTINGS:
        for source in SOURCES:
            for j in range(5):
                for p in (D/'pilot'/setting/source/'dense'/f'inner{j}').iterdir():
                    if p.is_file():deps[str(p.resolve())]=io.sha(p)
    srcroot=old.PACKAGE/'code'
    for name in ('duplexfm_joint/model.py','duplexfm_mechanism/model.py'):
        p=srcroot/name;deps[str(p.resolve())]=io.sha(p)
    assert io.read(D/'verification.json')['passed']
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=900,source_seed=42,primary='core_real_native_trace',settings=SETTINGS,sources=SOURCES,
        alpha=100.,lambda_ratio=.1,matrix_shape=[6,32],new_heads=45,reused_dense_heads=45,new_internal_OOF_rows=6897,
        new_source_forward=0,new_PCA_fits=0,source_scale=float(np.sqrt(12.)),
        pool_order=['positions2_8_mean','positions2_8_max','positions9_12_mean','positions9_12_max','positions1_13_19_mean','positions1_13_19_max']))
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
    records=[];details=[];pools={};count=0;all_numeric=True;locksha=io.sha(R/'lock.json')
    for setting in SETTINGS:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids'];pools[setting]=ids
        assert set(ids).isdisjoint(item['test_ids']);fi=np.array([lookup[s] for s in ids]);y=np.array([float(rows[i]['label']) for i in fi])
        for source in SOURCES:
            raw=xx[source][fi];prediction=np.full(len(y),np.nan);dense=np.full(len(y),np.nan);blocks=[]
            with np.load(P/'results/folds'/setting/'fold0'/source/'arrays.npz',allow_pickle=False) as a:ref=a['inner_oof'][:,18].copy()
            for j,split in enumerate(item['inner']):
                if time.monotonic()-start>900:raise TimeoutError('No new head: CPU cap')
                tr,va=np.array(split['train']),np.array(split['valid']);control=D/'pilot'/setting/source/'dense'/f'inner{j}'
                oldcomplete=io.read(control/'complete.json');io.check_files(control,oldcomplete['files_sha256'])
                st=io.read(control/'transform.json');cs=io.read(control/'state.json')
                assert cs['train_indices']==tr.tolist() and cs['valid_indices']==va.tolist() and cs['source']==source
                with np.load(control/'model.npz',allow_pickle=False) as a:dense[va]=a['prediction']
                z=transform(raw,st);out=root/setting/source/f'inner{j}';before=time.monotonic()
                beta,center,cert=fit(z[tr],y[tr],deadline=start+900);pp=z[va]@beta+center;coef,offset=deployment(st,beta,center)
                prediction[va]=pp;count+=1;valid=bool(cert['certified'] and cert['gap']<=1e-9*cert['gap_scale']);all_numeric &= valid
                old.npz_write(out/'model.npz',beta=beta,intercept=np.array(center),prediction=pp,deployment_coef=coef,deployment_intercept=np.array(offset))
                io.write(out/'state.json',dict(setting=setting,source=source,inner=j,train_indices=tr,valid_indices=va,certificate=cert,
                    numerically_qualified=valid,transform_sha256=io.sha(control/'transform.json'),dense_model_sha256=io.sha(control/'model.npz'),
                    dense_complete_sha256=io.sha(control/'complete.json'),CPU_fit_wall_seconds=time.monotonic()-before))
                io.write(out/'complete.json',dict(lock_sha256=locksha,files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
                blocks.append(dict(inner=j,n=len(va),metrics=stats(y[va],pp),dense_metrics=stats(y[va],dense[va]),
                    reference=stats(y[va],ref[va]),rank=cert['rank'],iterations=cert['iterations'],numerically_qualified=valid))
            assert np.isfinite(prediction).all() and np.isfinite(dense).all();m,dm,rr=stats(y,prediction),stats(y,dense),stats(y,ref)
            rec=dict(setting=setting,source=source,n=len(y),metrics=m,dense_metrics=dm,reference=rr,
                delta_PCC_vs_R4=m['PCC']-rr['PCC'],MSE_reduction_vs_R4_pct=100*(1-m['MSE']/rr['MSE']),inner_blocks=blocks)
            records.append(rec)
            details.extend(dict(setting=setting,source=source,sample_id=s,label=float(y[i]),prediction=float(prediction[i]),
                dense_prediction=float(dense[i]),reference_prediction=float(ref[i])) for i,s in enumerate(ids))
            print(setting,source,'deltaPCC',rec['delta_PCC_vs_R4'],'MSEred',rec['MSE_reduction_vs_R4_pct'],flush=True)
    assert count==45 and len(details)==6897
    gates=qualify(records,all_numeric);io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(**gates,records=records,fit_ids=pools,source_identities=identities,
        new_efficiency_heads=45,reused_dense_heads=45,new_internal_OOF_rows=6897,reused_dense_internal_OOF_rows=6897,
        new_source_forward=0,new_PCA_fits=0,GPU_hours=0,outer_test_metrics_used=False,threadpool=threads,
        lock_sha256=locksha,diagnostic_rows_sha256=io.sha(R/'diagnostic_rows.csv'),
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
