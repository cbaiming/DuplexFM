"""One fixed semiparametric pilot; all parameters fit in a single objective."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4';N=R.parent/'round25';K=R.parent/'round40';D=R.parent/'round21'
sys.path.insert(0,str(R))
from joint_kernel import fit,predict,transform,qualify
sp=importlib.util.spec_from_file_location('_r41_old4',P/'run.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
io=old.io;SOURCES=(*old.SOURCES,'TD_only')
FILES=('run.py','joint_kernel.py','verify.py','test_joint_kernel.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')
ARRAYS=('train_u','raw_row_mean','beta','dual');SCALARS=('raw_grand_mean','aux_scale','aux_variance','aux_trace','intercept')


def register():
    assert not (R/'lock.json').exists();old.validate()
    tests=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'));assert sum(int(t.get('tests',0)) for t in tests)==9
    assert not any(int(t.get(k,0)) for t in tests for k in ('errors','failures','skipped'))
    closure=io.read(K/'closure_receipt.json');assert closure['status']=='closed' and closure['readback_passed']
    for n,h in closure['files_sha256'].items():assert io.sha(R.parent/n)==h,n
    deps=dict(io.read(K/'lock.json')['dependencies'])
    for folder in (N,K):
        lock=io.read(folder/'lock.json');io.check_files(folder,lock['files']);assert io.read(folder/'verification.json')['passed']
        for path,h in lock['dependencies'].items():
            assert io.sha(Path(path))==h,path
            if path in deps:assert deps[path]==h
            deps[path]=h
        for n,h in lock['files'].items():deps[str((folder/n).resolve())]=h
        for p in (folder/'lock.json',folder/'diagnostic.json',folder/'diagnostic_rows.csv',folder/'verification.json',folder/'closure_receipt.json'):
            deps[str(p.resolve())]=io.sha(p)
        for n,h in io.read(folder/'diagnostic.json')['pilot_files_sha256'].items():
            p=folder/'pilot'/n;assert io.sha(p)==h;deps[str(p.resolve())]=h
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=900,source_seed=42,settings=old.SETTINGS[:3],sources=SOURCES,primary='core_real',
        alpha=100.,aux_average_trace=16.,lambda_ratio=.1,lambda_definition='exact_original_R25_without_kernel',
        kernel='arc1_minus_half_linear',linear_source_divisor=float(np.sqrt(12)),auxiliary_source_divisor_from_linear=4.,
        joint_objective=True,reference_predictions_used_for_training=False,new_heads=60,new_internal_OOF_rows=9196,
        new_nuclear_heads=45,new_TD_heads=15,reused_R25_source_heads=45,reused_R40_source_heads=45,reused_R4_TD_heads=15,
        maximum_iterations=5000,gap_tolerance=1e-9,new_source_forward=0,new_PCA=0,new_scaler_fits=0))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    assert io.versions()==lock['versions'];return lock


def metric(y,p):
    pc,ms=old.score(y,p[:,None]);return dict(PCC=float(pc[0]),MSE=float(ms[0]))


def run(start):
    validate();root=R/'pilot';assert not root.exists() and not (R/'diagnostic.json').exists()
    threads=threadpool_info();assert all(t['num_threads']==1 for t in threads)
    rows,xx,identities=old.inputs();lookup={r['sample_id']:i for i,r in enumerate(rows)};plan=io.read(P/'plan.json')
    records=[];details=[];pools={};count=updates=training_rows=0;numeric=True
    for setting in old.SETTINGS[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids'];pools[setting]=ids
        assert set(ids).isdisjoint(item['test_ids']);fi=np.array([lookup[s] for s in ids]);y=np.array([float(rows[i]['label']) for i in fi])
        for source in SOURCES:
            src='core_real' if source=='TD_only' else source;raw=xx[src][fi];pf=P/'results/folds'/setting/'fold0'/src/'transforms.json';af=pf.parent/'arrays.npz'
            with np.load(af,allow_pickle=False) as a:r4=a['inner_oof'][:,3 if source=='TD_only' else 18].copy()
            pred=np.full(len(ids),np.nan);r25=pred.copy();r40=pred.copy();blocks=[]
            for j,s in enumerate(item['inner']):
                if time.monotonic()-start>900:raise TimeoutError('CPU cap')
                tr,va=np.array(s['train']),np.array(s['valid']);sp=D/'pilot'/setting/src/'dense'/f'inner{j}'/'transform.json';state=io.read(sp)
                z=transform(raw,state)
                if source=='TD_only':z=z[:,:24]
                # Old prediction arrays are read only for reports; never passed to fit.
                references=dict(R4=io.sha(af))
                if source!='TD_only':
                    for folder,ref in ((N,r25),(K,r40)):
                        p=folder/'pilot'/setting/source/f'inner{j}'/'model.npz'
                        with np.load(p,allow_pickle=False) as a:ref[va]=a['prediction']
                        references['R25' if folder==N else 'R40']=io.sha(p)
                before=time.monotonic();m=fit(z[tr],y[tr],deadline=start+900);pp=predict(z[va],m);assert np.isfinite(pp).all();pred[va]=pp
                out=root/setting/source/f'inner{j}';arrays={k:np.asarray(m[k]) for k in ARRAYS+SCALARS};arrays['prediction']=pp
                old.npz_write(out/'model.npz',**arrays)
                io.write(out/'state.json',dict(setting=setting,source=source,inner=j,train_indices=tr,valid_indices=va,
                    transform_sha256=io.sha(sp),reference_sha256=references,certificate=m['certificate'],
                    CPU_fit_wall_seconds=time.monotonic()-before,saved_float_values=sum(a.size for a in arrays.values())))
                io.write(out/'complete.json',dict(lock_sha256=io.sha(R/'lock.json'),files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
                cert=m['certificate'];count+=1;updates+=cert['profile']['iterations'];training_rows+=len(tr);numeric &= cert['numerically_qualified']
                block=dict(inner=j,n=len(va),metrics=metric(y[va],pp),R4_reference=metric(y[va],r4[va]),
                    relative_gap=max(0.,cert['full_gap'])/cert['gap_scale'],iterations=cert['profile']['iterations'],numerically_qualified=cert['numerically_qualified'])
                if source!='TD_only':block.update(R25_reference=metric(y[va],r25[va]),R40_reference=metric(y[va],r40[va]))
                blocks.append(block)
            assert np.isfinite(pred).all();m=metric(y,pred);ref=dict(R4=metric(y,r4))
            if source!='TD_only':
                assert np.isfinite(r25).all() and np.isfinite(r40).all();ref.update(R25=metric(y,r25),R40=metric(y,r40))
            rec=dict(setting=setting,source=source,n=len(ids),metrics=m,inner_blocks=blocks)
            for name,rr in ref.items():rec.update({name+'_reference':rr,'delta_PCC_vs_'+name:m['PCC']-rr['PCC'],'MSE_reduction_vs_'+name+'_pct':100*(1-m['MSE']/rr['MSE'])})
            records.append(rec)
            details.extend(dict(setting=setting,source=source,sample_id=s,label=float(y[i]),prediction=float(pred[i]),R4_reference_prediction=float(r4[i]),
                R25_reference_prediction='' if source=='TD_only' else float(r25[i]),R40_reference_prediction='' if source=='TD_only' else float(r40[i])) for i,s in enumerate(ids))
            print(setting,source,'delta',rec.get('delta_PCC_vs_R25',rec['delta_PCC_vs_R4']),'MSEred',rec.get('MSE_reduction_vs_R25_pct',rec['MSE_reduction_vs_R4_pct']),flush=True)
    assert count==60 and len(details)==9196;gates=qualify(records,numeric);io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(**gates,records=records,fit_ids=pools,source_identities=identities,new_heads=count,new_nuclear_heads=45,new_TD_heads=15,
        new_proximal_updates=updates,new_internal_OOF_rows=9196,stored_training_row_occurrences=training_rows,
        reused_R25_source_heads=45,reused_R40_source_heads=45,reused_R4_TD_heads=15,new_source_forward=0,new_PCA=0,new_scaler_fits=0,GPU_hours=0,
        outer_test_metrics_used=False,threadpool=threads,lock_sha256=io.sha(R/'lock.json'),diagnostic_rows_sha256=io.sha(R/'diagnostic_rows.csv'),
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
