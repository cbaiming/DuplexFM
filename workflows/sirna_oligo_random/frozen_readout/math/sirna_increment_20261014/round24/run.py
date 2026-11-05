"""Bounded training-internal CPU experiment; one shared head per condition."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4';Q=R.parent/'round23'
sys.path.insert(0,str(R))
from robust_head import fit,qualify
sp=importlib.util.spec_from_file_location('_r24_prior4',P/'run.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
io=old.io;SETTINGS=old.SETTINGS[:3];SOURCES=(*old.SOURCES,'TD_only');MODES=('balanced','robust')
FILES=('run.py','robust_head.py','verify.py','test_robust_head.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')


def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==10
    assert not any(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites)
    prior=io.read(P/'lock.json');deps=dict(prior['dependencies'])
    for name in prior['files']:deps[str((P/name).resolve())]=io.sha(P/name)
    for p in (P/'lock.json',P/'plan.json',P/'results/complete.json',P/'verification.json',
              Q/'lock.json',Q/'diagnostic.json',Q/'verification.json',Q/'closure_receipt.json'):
        deps[str(p.resolve())]=io.sha(p)
    for p in (Q/'gates').rglob('*'):
        if p.is_file():deps[str(p.resolve())]=io.sha(p)
    for setting in SETTINGS:
        for source in old.SOURCES:
            for name in ('transforms.json','arrays.npz'):
                p=P/'results/folds'/setting/'fold0'/source/name;deps[str(p.resolve())]=io.sha(p)
    assert io.read(Q/'verification.json')['passed'] and not io.read(Q/'diagnostic.json')['qualified']
    previous=io.read(Q/'closure_receipt.json')
    for name,h in previous['files_sha256'].items():assert io.sha(R.parent/name)==h,name
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=900,primary='core_real_robust',source_seed=42,
        settings=SETTINGS,sources=SOURCES,modes=MODES,alpha=100.,heads=120,internal_OOF_rows=18392,
        new_source_forward=0,new_PCA_fits=0,new_KMeans_fits=0,
        qualification=dict(minimum_pools=2,minimum_delta_PCC=.005,minimum_MSE_reduction=0.,
                           all_pool_delta_floor=-.005,paired_nondegrading_pools=2,paired_mean_MSE_floor=0.)))
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
    records=[];details=[];pools={};count=0;solves=0;all_numeric=True;locksha=io.sha(R/'lock.json')
    for setting in SETTINGS:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids'];pools[setting]=ids;assert set(ids).isdisjoint(item['test_ids'])
        fi=np.array([lookup[s] for s in ids]);y=np.array([float(rows[i]['label']) for i in fi])
        for source in SOURCES:
            src='core_real' if source=='TD_only' else source
            raw=xx[src][fi];pf=P/'results/folds'/setting/'fold0'/src/'transforms.json';states=io.read(pf)
            with np.load(P/'results/folds'/setting/'fold0'/src/'arrays.npz',allow_pickle=False) as a:
                ref=a['inner_oof'][:,3 if source=='TD_only' else 18].copy()
            for mode in MODES:
                pred=np.full(len(y),np.nan);blocks=[]
                for j,split in enumerate(item['inner']):
                    if time.monotonic()-start>900:raise TimeoutError('Stage cap; no new head')
                    tr,va=np.asarray(split['train']),np.asarray(split['valid']);out=root/setting/source/mode/f'inner{j}'
                    st=states['inner'][j];z=old.transform(raw,st)
                    if source=='TD_only':z=z[:,:24]
                    group_path=Q/'gates'/setting/f'inner{j}'/'assignments.npz'
                    with np.load(group_path,allow_pickle=False) as a:assignment=a['all_fit_assignment'].copy()
                    before=time.monotonic();beta,intercept,cert=fit(z[tr],y[tr],assignment[tr],mode)
                    prediction=z[va]@beta+intercept;pred[va]=prediction
                    count+=1;solves+=cert['weighted_ridge_solves'];numeric=cert['numerically_qualified'];all_numeric &= numeric
                    old.npz_write(out/'model.npz',beta=beta,intercept=np.array(intercept),prediction=prediction)
                    io.write(out/'state.json',dict(setting=setting,source=source,mode=mode,inner=j,
                        train_indices=tr,valid_indices=va,reference_transform_sha256=io.sha(pf),
                        group_assignments_sha256=io.sha(group_path),certificate=cert,CPU_fit_wall_seconds=time.monotonic()-before))
                    io.write(out/'complete.json',dict(lock_sha256=locksha,files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
                    blocks.append(dict(inner=j,n=len(va),metrics=stats(y[va],prediction),reference=stats(y[va],ref[va]),
                                       group0_weight=cert['group0_weight'],numerically_qualified=numeric))
                assert np.isfinite(pred).all();m,rr=stats(y,pred),stats(y,ref)
                rec=dict(setting=setting,source=source,mode=mode,n=len(y),metrics=m,reference=rr,
                    delta_PCC_vs_R4=m['PCC']-rr['PCC'],MSE_reduction_vs_R4_pct=100*(1-m['MSE']/rr['MSE']),inner_blocks=blocks)
                records.append(rec)
                details.extend(dict(setting=setting,source=source,mode=mode,sample_id=s,label=float(y[k]),
                                    prediction=float(pred[k]),reference_prediction=float(ref[k])) for k,s in enumerate(ids))
                print(setting,source,mode,'deltaPCC',rec['delta_PCC_vs_R4'],'MSEred',rec['MSE_reduction_vs_R4_pct'],flush=True)
    assert count==120 and len(details)==18392
    gates=qualify(records,all_numeric);io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(**gates,records=records,fit_ids=pools,source_identities=identities,
        new_efficiency_heads=count,weighted_ridge_solves=solves,new_source_forward=0,new_PCA_fits=0,new_KMeans_fits=0,GPU_hours=0,
        outer_test_metrics_used=False,threadpool=threads,lock_sha256=locksha,diagnostic_rows_sha256=io.sha(R/'diagnostic_rows.csv'),
        pilot_files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    validate();print('QUALIFIED',gates['qualified'],'weighted_solves',solves,flush=True)


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
