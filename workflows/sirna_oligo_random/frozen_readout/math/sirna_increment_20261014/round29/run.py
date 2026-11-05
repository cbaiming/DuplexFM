"""Bounded TD-coefficient trust experiment on unchanged native source inputs."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4';D=R.parent/'round21';N=R.parent/'round25'
sys.path.insert(0,str(R))
from anchored import fit,transform,deployment,qualify
sp=importlib.util.spec_from_file_location('_r29_prior4',P/'run.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
io=old.io;SETTINGS=old.SETTINGS[:3];SOURCES=old.SOURCES
FILES=('run.py','anchored.py','verify.py','test_anchored.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')


def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==8
    assert not any(int(s.get('errors',0))+int(s.get('failures',0))+int(s.get('skipped',0)) for s in suites)
    previous=io.read(R.parent/'round28/closure_receipt.json');assert previous['status']=='closed' and previous['readback_passed']
    for name,h in previous['files_sha256'].items():assert io.sha(R.parent/name)==h,name
    nlock=io.read(N/'lock.json');io.check_files(N,nlock['files']);nv=io.read(N/'verification.json')
    assert nv['passed'] and nv['lock_sha256']==io.sha(N/'lock.json') and nv['diagnostic_sha256']==io.sha(N/'diagnostic.json')
    io.check_files(N/'pilot',io.read(N/'diagnostic.json')['pilot_files_sha256'])
    deps=dict(nlock['dependencies'])
    for name in nlock['files']:deps[str((N/name).resolve())]=io.sha(N/name)
    for p in (N/'lock.json',N/'closure_receipt.json',N/'diagnostic.json',N/'verification.json',Path(io.__file__),
              R.parent/'round28/closure_receipt.json'):
        deps[str(p.resolve())]=io.sha(p)
    for name,h in io.read(N/'diagnostic.json')['pilot_files_sha256'].items():deps[str((N/'pilot'/name).resolve())]=h
    for setting in SETTINGS:
        p=P/'results/folds'/setting/'fold0'/'core_real'/'arrays.npz';deps[str(p.resolve())]=io.sha(p)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=900,source_seed=42,settings=SETTINGS,sources=SOURCES,primary='core_real',
        alpha=100.,kappa=100.,lambda_ratio=.1,matrix_shape=[6,32],new_heads=45,new_internal_OOF_rows=6897,
        reused_native_heads=45,reused_unique_TD_heads=15,reused_unique_TD_rows=2299,new_source_forward=0,new_PCA=0))
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
    count=0;numeric=True;details=[];records=[];pools={};locksha=io.sha(R/'lock.json')
    for setting in SETTINGS:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids'];pools[setting]=ids;assert set(ids).isdisjoint(item['test_ids'])
        fi=np.array([lookup[s] for s in ids]);y=np.array([float(rows[i]['label']) for i in fi])
        tdpath=P/'results/folds'/setting/'fold0'/'core_real'/'arrays.npz'
        with np.load(tdpath,allow_pickle=False) as a:
            b0=a['inner_coef'][:,3,:24].copy();tdpred=a['inner_oof'][:,3].copy()
        for source in SOURCES:
            raw=xx[source][fi];prediction=np.full(len(y),np.nan);reference=np.full(len(y),np.nan);blocks=[]
            for j,split in enumerate(item['inner']):
                if time.monotonic()-start>900:raise TimeoutError('No new head after stage cap')
                tr,va=np.array(split['train']),np.array(split['valid'])
                native=N/'pilot'/setting/source/f'inner{j}';control=D/'pilot'/setting/source/'dense'/f'inner{j}'
                ns=io.read(native/'state.json');st=io.read(control/'transform.json')
                assert ns['transform_sha256']==io.sha(control/'transform.json') and ns['train_indices']==tr.tolist() and ns['valid_indices']==va.tolist()
                with np.load(native/'model.npz',allow_pickle=False) as a:
                    reference[va]=a['prediction'];oldbeta=a['beta'].copy()
                z=transform(raw,st);out=root/setting/source/f'inner{j}';before=time.monotonic()
                beta,center,cert=fit(z[tr],y[tr],b0[j],deadline=start+900)
                pp=z[va]@beta+center;coef,offset=deployment(st,beta,center);prediction[va]=pp
                oldshift=float(np.linalg.norm(oldbeta[:24]-b0[j]));count+=1;valid=bool(cert['certified']);numeric &= valid
                old.npz_write(out/'model.npz',beta=beta,intercept=np.array(center),prediction=pp,TD_reference=b0[j],
                    deployment_coef=coef,deployment_intercept=np.array(offset))
                io.write(out/'state.json',dict(setting=setting,source=source,inner=j,train_indices=tr,valid_indices=va,
                    certificate=cert,numerically_qualified=valid,transform_sha256=io.sha(control/'transform.json'),
                    native_reference_sha256=io.sha(native/'model.npz'),TD_reference_sha256=io.sha(tdpath),
                    reference_TD_shift_norm=oldshift,CPU_fit_wall_seconds=time.monotonic()-before))
                io.write(out/'complete.json',dict(lock_sha256=locksha,files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
                blocks.append(dict(inner=j,n=len(va),metrics=stats(y[va],pp),reference=stats(y[va],reference[va]),
                    TD_metrics=stats(y[va],tdpred[va]),TD_shift_norm=cert['TD_shift_norm'],reference_TD_shift_norm=oldshift,
                    rank=cert['rank'],iterations=cert['iterations'],numerically_qualified=valid))
            assert np.isfinite(prediction).all() and np.isfinite(reference).all()
            m,rr,tm=stats(y,prediction),stats(y,reference),stats(y,tdpred)
            rec=dict(setting=setting,source=source,n=len(y),metrics=m,reference=rr,TD_metrics=tm,
                delta_PCC_vs_R25=m['PCC']-rr['PCC'],MSE_reduction_vs_R25_pct=100*(1-m['MSE']/rr['MSE']),inner_blocks=blocks)
            records.append(rec)
            details.extend(dict(setting=setting,source=source,sample_id=s,label=float(y[k]),prediction=float(prediction[k]),
                native_reference_prediction=float(reference[k]),TD_prediction=float(tdpred[k])) for k,s in enumerate(ids))
            print(setting,source,'deltaPCC',rec['delta_PCC_vs_R25'],'MSEred',rec['MSE_reduction_vs_R25_pct'],flush=True)
    assert count==45 and len(details)==6897
    gates=qualify(records,numeric);io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(**gates,records=records,fit_ids=pools,source_identities=identities,
        new_heads=45,new_internal_OOF_rows=6897,reused_native_heads=45,reused_unique_TD_heads=15,reused_unique_TD_rows=2299,
        new_source_forward=0,new_PCA=0,GPU_hours=0,outer_test_metrics_used=False,threadpool=threads,
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
