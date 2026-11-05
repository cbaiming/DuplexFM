"""Native representation versus fixed PCA16 kernel, cache-only CPU pilot."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4';K=R.parent/'round33';N=R.parent/'round25';D=R.parent/'round21'
sys.path.insert(0,str(R))
from native_kernel import transform,fit,predict,qualify
spec=importlib.util.spec_from_file_location('_r40_r4',P/'run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
io=old.io;SOURCES=old.SOURCES
FILES=('run.py','native_kernel.py','verify.py','test_native_kernel.py','PROTOCOL.md','CODE_AUDIT.md','PCC_OBJECTIVE_AUDIT.md','pytest.xml')
ARRAY_KEYS=('train_u','train_td','dual','td_coef','raw_row_mean')
SCALAR_KEYS=('intercept','raw_grand_mean','aux_scale')


def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==8
    assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
    c=io.read(R.parent/'round39/closure_receipt.json');assert c['status']=='closed' and c['readback_passed']
    for n,h in c['files_sha256'].items():assert io.sha(R.parent/n)==h,n
    deps={}
    for folder in (K,N):
        lock=io.read(folder/'lock.json');io.check_files(folder,lock['files'])
        assert io.read(folder/'verification.json')['passed']
        for p,h in lock['dependencies'].items():
            assert io.sha(Path(p))==h,p
            if p in deps:assert deps[p]==h
            deps[p]=h
        for n,h in lock['files'].items():deps[str((folder/n).resolve())]=h
        for p in (folder/'lock.json',folder/'diagnostic.json',folder/'diagnostic_rows.csv',folder/'verification.json',folder/'closure_receipt.json'):
            deps[str(p.resolve())]=io.sha(p)
        diagnostic=io.read(folder/'diagnostic.json')
        for n,h in diagnostic['pilot_files_sha256'].items():
            p=folder/'pilot'/n;assert io.sha(p)==h;deps[str(p.resolve())]=h
    deps[str((R.parent/'round39/closure_receipt.json').resolve())]=io.sha(R.parent/'round39/closure_receipt.json')
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=900,source_seed=42,settings=old.SETTINGS[:3],sources=SOURCES,primary='core_real',
        alpha=100.,aux_average_trace=16.,kernel_degree=1,kernel_depth=1,no_kernel_bias=True,
        source_coordinates=192,source_divisor=float(np.sqrt(192)),TD_divisor=float(np.sqrt(24)),
        source_preprocessing='existing_R4_train_coordinate_scalers_without_PCA_or_PC_whitening',
        new_heads=45,new_internal_OOF_rows=6897,reused_TD_heads=15,reused_TD_OOF_rows=2299,
        reused_R33_source_heads=45,reused_R4_source_heads=45,reused_R25_source_heads=45,
        combined_internal_OOF_rows=9196,new_source_forward=0,new_PCA=0,new_scaler_fits=0))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    assert io.versions()==lock['versions'];return lock


def score(y,p):
    pc,ms=old.score(y,p[:,None]);return dict(PCC=float(pc[0]),MSE=float(ms[0]))


def run(start):
    validate();root=R/'pilot';assert not root.exists() and not (R/'diagnostic.json').exists()
    threads=threadpool_info();assert all(t['num_threads']==1 for t in threads)
    rows,xx,identities=old.inputs();lookup={r['sample_id']:i for i,r in enumerate(rows)}
    plan=io.read(P/'plan.json');old_kernel=io.read(K/'diagnostic.json')
    td_records=[];records=[];details=[];pools={};heads=training_rows=0;numeric=True
    for setting in old.SETTINGS[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids'];pools[setting]=ids;assert set(ids).isdisjoint(item['test_ids'])
        fi=np.array([lookup[s] for s in ids]);y=np.array([float(rows[i]['label']) for i in fi])
        td_prediction=np.full(len(ids),np.nan)
        for j,sp in enumerate(item['inner']):
            path=K/'pilot'/setting/'TD_only'/f'inner{j}'/'model.npz'
            with np.load(path,allow_pickle=False) as a:td_prediction[sp['valid']]=a['prediction']
        assert np.isfinite(td_prediction).all();tm=score(y,td_prediction)
        historical=next(r for r in old_kernel['records'] if r['setting']==setting and r['source']=='TD_only')
        assert all(abs(tm[k]-historical['metrics'][k])<1e-12 for k in tm)
        td_records.append(dict(setting=setting,source='TD_only',metrics=tm,n=len(ids),new_fit=False))
        details.extend(dict(setting=setting,source='TD_only',sample_id=s,label=float(y[k]),prediction=float(td_prediction[k]),
            new_fit=False,R4_reference_prediction='',R33_reference_prediction=float(td_prediction[k]),R25_reference_prediction='') for k,s in enumerate(ids))
        for source in SOURCES:
            raw=xx[source][fi];pf=P/'results/folds'/setting/'fold0'/source/'transforms.json';af=pf.parent/'arrays.npz';states=io.read(pf)
            with np.load(af,allow_pickle=False) as a:r4=a['inner_oof'][:,18].copy()
            pred=np.full(len(ids),np.nan);r33=np.full(len(ids),np.nan);r25=np.full(len(ids),np.nan);blocks=[]
            for j,sp in enumerate(item['inner']):
                if time.monotonic()-start>900:raise TimeoutError('CPU stage cap')
                tr,va=np.array(sp['train']),np.array(sp['valid']);z=transform(raw,states['inner'][j])
                kp=K/'pilot'/setting/source/f'inner{j}'/'model.npz';npth=N/'pilot'/setting/source/f'inner{j}'/'model.npz'
                with np.load(kp,allow_pickle=False) as a:r33[va]=a['prediction']
                with np.load(npth,allow_pickle=False) as a:r25[va]=a['prediction']
                before=time.monotonic();m=fit(z[tr],y[tr],deadline=start+900);pp=predict(z[va],m);assert np.isfinite(pp).all();pred[va]=pp
                out=root/setting/source/f'inner{j}';arrays={k:np.asarray(m[k]) for k in ARRAY_KEYS+SCALAR_KEYS};arrays['prediction']=pp
                old.npz_write(out/'model.npz',**arrays)
                cert={k:v for k,v in m.items() if k not in ARRAY_KEYS+SCALAR_KEYS}
                io.write(out/'state.json',dict(setting=setting,source=source,inner=j,train_indices=tr,valid_indices=va,
                    certificate=cert,transform_sha256=io.sha(pf),R4_reference_sha256=io.sha(af),R33_reference_sha256=io.sha(kp),
                    R25_reference_sha256=io.sha(npth),CPU_fit_wall_seconds=time.monotonic()-before,saved_float_values=sum(a.size for a in arrays.values())))
                io.write(out/'complete.json',dict(lock_sha256=io.sha(R/'lock.json'),files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
                heads+=1;training_rows+=len(tr);numeric &= m['numerically_qualified']
                blocks.append(dict(inner=j,n=len(va),training_rows=len(tr),metrics=score(y[va],pp),R4_reference=score(y[va],r4[va]),
                    R33_reference=score(y[va],r33[va]),R25_reference=score(y[va],r25[va]),normal_error=m['normal_error'],numerically_qualified=m['numerically_qualified']))
            assert np.isfinite(pred).all() and np.isfinite(r33).all() and np.isfinite(r25).all()
            m,refs=score(y,pred),dict(R4=score(y,r4),R33=score(y,r33),R25=score(y,r25))
            rec=dict(setting=setting,source=source,n=len(ids),metrics=m,inner_blocks=blocks)
            for name,ref in refs.items():rec.update({name+'_reference':ref,'delta_PCC_vs_'+name:m['PCC']-ref['PCC'],'MSE_reduction_vs_'+name+'_pct':100*(1-m['MSE']/ref['MSE'])})
            records.append(rec)
            details.extend(dict(setting=setting,source=source,sample_id=s,label=float(y[k]),prediction=float(pred[k]),new_fit=True,
                R4_reference_prediction=float(r4[k]),R33_reference_prediction=float(r33[k]),R25_reference_prediction=float(r25[k])) for k,s in enumerate(ids))
            print(setting,source,'delta_vs_PCA_kernel',rec['delta_PCC_vs_R33'],'delta_vs_native',rec['delta_PCC_vs_R25'],flush=True)
    assert heads==45 and len(details)==9196;gates=qualify(records,numeric,td_records)
    io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(**gates,records=records,TD_records=td_records,fit_ids=pools,source_identities=identities,
        new_heads=heads,new_linear_solves=heads,new_internal_OOF_rows=6897,stored_training_row_occurrences=training_rows,
        reused_TD_heads=15,reused_TD_OOF_rows=2299,reused_R33_source_heads=45,reused_R4_source_heads=45,reused_R25_source_heads=45,
        combined_internal_OOF_rows=9196,new_source_forward=0,new_PCA=0,new_scaler_fits=0,GPU_hours=0,outer_test_metrics_used=False,
        threadpool=threads,lock_sha256=io.sha(R/'lock.json'),diagnostic_rows_sha256=io.sha(R/'diagnostic_rows.csv'),
        pilot_files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    validate();print('QUALIFIED',gates['qualified'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('register','run'))
    if parser.parse_args().mode=='register':register()
    else:
        mutex=R/'execution.mutex'
        with mutex.open('x') as f:f.write(str(os.getpid()))
        started=time.monotonic();rc=1;error=None
        try:run(started);rc=0
        except Exception:error=traceback.format_exc();raise
        finally:
            io.write(R/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-started,GPU_hours=0,new_source_forward=0))
            mutex.unlink()
