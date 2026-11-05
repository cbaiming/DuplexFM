"""Fixed input-only geometry gate, one bounded CPU development pilot."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;B=R.parent;P=B/'round4';D=B/'round21';N=B/'round25'
sys.path.insert(0,str(R))
from leverage_gate import fit_gate,apply_design,fit_head,transform,qualify
spec=importlib.util.spec_from_file_location('_r48_prior4',P/'run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
io=old.io
FILES=('run.py','leverage_gate.py','verify.py','test_leverage_gate.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')


def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==11
    assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
    previous=io.read(B/'round47/closure_receipt.json');assert previous['status']=='closed' and previous['readback_passed']
    for name,h in previous['files_sha256'].items():assert io.sha(B/name)==h,name
    prior=io.read(N/'lock.json');io.check_files(N,prior['files']);deps=dict(prior['dependencies'])
    for name in prior['files']:deps[str((N/name).resolve())]=io.sha(N/name)
    for path in (N/'lock.json',N/'closure_receipt.json',N/'diagnostic.json',N/'verification.json',Path(io.__file__),
                 B/'round47/closure_receipt.json',B/'round36/verify.py',P/'verify.py'):
        deps[str(path.resolve())]=io.sha(path)
    assert io.read(N/'verification.json')['passed']
    for name,h in io.read(N/'diagnostic.json')['pilot_files_sha256'].items():
        path=N/'pilot'/name;assert io.sha(path)==h;deps[str(path.resolve())]=h
    for setting in old.SETTINGS[:3]:
        path=P/'results/folds'/setting/'fold0/core_real/arrays.npz';deps[str(path.resolve())]=io.sha(path)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=900,source_seed=42,settings=old.SETTINGS[:3],sources=old.SOURCES,
        primary='core_real',auxiliary_TD_control=True,alpha=100.,lambda_ratio=.1,
        metric_ridge=100.,quantile=.9,quantile_method='higher',new_heads=60,new_source_heads=45,new_TD_gated_heads=15,
        new_input_geometries=60,new_gate_centering_means=60,new_internal_OOF_rows=9196,
        reused_native_heads=45,reused_unique_TD_heads=15,new_source_forward=0,new_PCA=0,new_original_scalers=0))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for path,h in lock['dependencies'].items():assert io.sha(Path(path))==h,path
    assert io.versions()==lock['versions'];return lock


def stats(y,p):
    assert np.isfinite(p).all() and np.std(p)>1e-14 and np.std(y)>1e-14
    pc,ms=old.score(y,p[:,None]);return dict(PCC=float(pc[0]),MSE=float(ms[0]))


def geometry_stats(h,w):
    return dict(n=len(h),leverage_min=float(h.min()),leverage_max=float(h.max()),leverage_mean=float(h.mean()),
        weight_min=float(w.min()),weight_mean=float(w.mean()),clipped_count=int(np.count_nonzero(w<1)))


def run(start):
    validate();root=R/'pilot';assert not root.exists() and not (R/'diagnostic.json').exists()
    threads=threadpool_info();assert all(t['num_threads']==1 for t in threads)
    rows,xx,identities=old.inputs();lookup={r['sample_id']:i for i,r in enumerate(rows)};plan=io.read(P/'plan.json')
    records=[];details=[];pools={};count=0;numeric=True
    for setting in old.SETTINGS[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids'];pools[setting]=ids
        assert set(ids).isdisjoint(item['test_ids']);fi=np.array([lookup[s] for s in ids]);y=np.array([float(rows[i]['label']) for i in fi])
        tdpath=P/'results/folds'/setting/'fold0/core_real/arrays.npz'
        with np.load(tdpath,allow_pickle=False) as a:tdpred=a['inner_oof'][:,3].copy()
        for source in ('TD_only',)+tuple(old.SOURCES):
            is_source=source!='TD_only';origin=source if is_source else 'core_real';raw=xx[origin][fi]
            pred=np.full(len(ids),np.nan);ref=pred.copy();hh=pred.copy();ww=pred.copy();blocks=[];coverage=[]
            for j,split in enumerate(item['inner']):
                if time.monotonic()-start>900:raise TimeoutError('CPU stage cap')
                tr,va=np.array(split['train']),np.array(split['valid']);coverage+=va.tolist();assert set(tr).isdisjoint(va)
                control=D/'pilot'/setting/origin/'dense'/f'inner{j}';st=io.read(control/'transform.json')
                previous=N/'pilot'/setting/origin/f'inner{j}';reference_sha=None
                if is_source:
                    nc=io.read(previous/'complete.json');io.check_files(previous,nc['files_sha256'])
                    assert nc['lock_sha256']==io.sha(N/'lock.json')
                    ns=io.read(previous/'state.json')
                    assert ns['train_indices']==tr.tolist() and ns['valid_indices']==va.tolist() and ns['transform_sha256']==io.sha(control/'transform.json')
                    assert ns['numerically_qualified']
                    with np.load(previous/'model.npz',allow_pickle=False) as a:ref[va]=a['prediction']
                    reference_sha=io.sha(previous/'model.npz')
                else:ref[va]=tdpred[va]
                z=transform(raw,st);out=root/setting/source/f'inner{j}';before=time.monotonic()
                gate=fit_gate(z[tr,24:] if is_source else z[tr,:24])
                beta,center,cert=fit_head(z[tr],y[tr],gate,is_source,deadline=start+900)
                train_design,th,tw=apply_design(z[tr],gate,is_source);valid_design,vh,vw=apply_design(z[va],gate,is_source)
                pp=valid_design@beta+center;pred[va]=pp;hh[va]=vh;ww[va]=vw;count+=1
                valid=bool(cert['certified']);numeric &= valid
                old.npz_write(out/'model.npz',beta=beta,intercept=np.array(center),prediction=pp,
                    gate_precision=gate['precision'],gate_mean=gate['mean'],train_leverage=th,train_weight=tw,valid_leverage=vh,valid_weight=vw)
                io.write(out/'state.json',dict(setting=setting,source=source,inner=j,train_indices=tr,valid_indices=va,
                    certificate=cert,numerically_qualified=valid,gate={k:v for k,v in gate.items() if k not in ('precision','mean')},
                    transform_sha256=io.sha(control/'transform.json'),native_reference_sha256=reference_sha,
                    TD_reference_sha256=io.sha(tdpath),CPU_fit_wall_seconds=time.monotonic()-before))
                io.write(out/'complete.json',dict(lock_sha256=io.sha(R/'lock.json'),files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
                blocks.append(dict(inner=j,n=len(va),metrics=stats(y[va],pp),reference=stats(y[va],ref[va]),TD_metrics=stats(y[va],tdpred[va]),
                    threshold=gate['threshold'],training_geometry=geometry_stats(th,tw),validation_geometry=geometry_stats(vh,vw),numerically_qualified=valid))
            assert sorted(coverage)==list(range(len(ids))) and np.isfinite(pred).all() and np.isfinite(ref).all()
            m,rr,td=stats(y,pred),stats(y,ref),stats(y,tdpred)
            rec=dict(setting=setting,source=source,n=len(ids),metrics=m,reference=rr,TD_metrics=td,
                delta_PCC_vs_reference=m['PCC']-rr['PCC'],MSE_reduction_vs_reference_pct=100*(1-m['MSE']/rr['MSE']),inner_blocks=blocks)
            records.append(rec)
            details.extend(dict(setting=setting,source=source,sample_id=s,label=float(y[k]),prediction=float(pred[k]),
                reference_prediction=float(ref[k]),TD_prediction=float(tdpred[k]),leverage=float(hh[k]),gate_weight=float(ww[k])) for k,s in enumerate(ids))
            print(setting,source,'deltaPCC',rec['delta_PCC_vs_reference'],'MSEred',rec['MSE_reduction_vs_reference_pct'],flush=True)
    assert count==60 and len(details)==9196;gates=qualify(records,numeric)
    io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(**gates,records=records,fit_ids=pools,source_identities=identities,
        new_heads=60,new_source_heads=45,new_TD_gated_heads=15,new_input_geometries=60,new_gate_centering_means=60,
        new_internal_OOF_rows=9196,reused_native_heads=45,reused_unique_TD_heads=15,reused_unique_TD_rows=2299,
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
