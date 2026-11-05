"""Fixed loss-only pilot on the previously closed R33 inputs and kernel."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4';K=R.parent/'round33'
sys.path.insert(0,str(R));sys.path.insert(1,str(K))
from epsilon_kernel import solve,qualify,ALPHA,EPSILON_FRACTION
from relu_kernel import arc,predict
spec=importlib.util.spec_from_file_location('_r39_r33',K/'run.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
old,io=prior.old,prior.io;SOURCES=prior.SOURCES
FILES=('run.py','epsilon_kernel.py','verify.py','test_epsilon_kernel.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')


def register():
    assert not (R/'lock.json').exists();prior.validate()
    tests=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(x.get('tests',0)) for x in tests)==8
    assert not any(int(x.get(k,0)) for x in tests for k in ('errors','failures','skipped'))
    closure=io.read(R.parent/'round38/closure_receipt.json');assert closure['status']=='closed' and closure['readback_passed']
    for n,h in closure['files_sha256'].items():assert io.sha(R.parent/n)==h,n
    lock=io.read(K/'lock.json');deps=dict(lock['dependencies'])
    for n in lock['files']:deps[str((K/n).resolve())]=io.sha(K/n)
    for p in (K/'lock.json',K/'diagnostic.json',K/'verification.json',K/'diagnostic_rows.csv',K/'closure_receipt.json',R.parent/'round38/closure_receipt.json'):
        deps[str(p.resolve())]=io.sha(p)
    d=io.read(K/'diagnostic.json')
    for n,h in d['pilot_files_sha256'].items():
        p=K/'pilot'/n;assert io.sha(p)==h;deps[str(p.resolve())]=h
    for p in sorted((R/'prelock_test_v1').iterdir()):
        if p.is_file():deps[str(p.resolve())]=io.sha(p)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=900,source_seed=42,alpha=ALPHA,epsilon_fraction=EPSILON_FRACTION,
        loss='half_squared_epsilon_insensitive',free_intercept=True,kernel_unchanged_from='round33',
        maximum_iterations=5000,gap_tolerance=1e-9,settings=old.SETTINGS[:3],sources=SOURCES,primary='core_real',
        new_heads=60,new_OOF_rows=9196,reused_R33_heads=60,reused_R4_heads=60,new_PCA=0,new_source_forward=0))
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
    plan=io.read(P/'plan.json');records=[];details=[];pools={};heads=updates=training_rows=0;numeric=True
    for setting in old.SETTINGS[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids'];pools[setting]=ids;assert set(ids).isdisjoint(item['test_ids'])
        fi=np.array([lookup[s] for s in ids]);y=np.array([float(rows[i]['label']) for i in fi])
        for source in SOURCES:
            src='core_real' if source=='TD_only' else source;raw=xx[src][fi]
            pf=P/'results/folds'/setting/'fold0'/src/'transforms.json';af=pf.parent/'arrays.npz';states=io.read(pf)
            with np.load(af,allow_pickle=False) as a:r4=a['inner_oof'][:,3 if source=='TD_only' else 18].copy()
            pred=np.full(len(ids),np.nan);r33=np.full(len(ids),np.nan);blocks=[]
            for j,split in enumerate(item['inner']):
                if time.monotonic()-start>900:raise TimeoutError('CPU stage cap')
                tr,va=np.array(split['train']),np.array(split['valid']);z=old.transform(raw,states['inner'][j])
                if source=='TD_only':z=z[:,:24]
                refpath=K/'pilot'/setting/source/f'inner{j}'/'model.npz'
                with np.load(refpath,allow_pickle=False) as a:reference={k:a[k].copy() for k in a.files}
                r33[va]=reference['prediction'];u0=reference['train_u'];td=reference['train_td']
                np.testing.assert_allclose(td,z[tr,:24],atol=0,rtol=0)
                rawk=arc(u0,u0);rawk=(rawk+rawk.T)/2;np.fill_diagonal(rawk,np.sum(u0*u0,axis=1))
                kc=rawk-reference['raw_row_mean'][:,None]-reference['raw_row_mean'][None,:]+reference['raw_grand_mean']
                gram=td@td.T+reference['aux_scale']*kc;gram=(gram+gram.T)/2
                before=time.monotonic();sd=float(np.std(y[tr]));eps=EPSILON_FRACTION*sd
                u,cert=solve(gram,y[tr],eps,deadline=start+900)
                model={k:reference[k] for k in ('train_u','train_td','raw_row_mean','raw_grand_mean','aux_scale')}
                model.update(dual=u/ALPHA,td_coef=td.T@u/ALPHA,intercept=cert['intercept'])
                pp=predict(z[va],model);assert np.isfinite(pp).all();pred[va]=pp
                out=root/setting/source/f'inner{j}';arrays={k:np.asarray(v) for k,v in model.items()}
                arrays.update(epsilon_dual=u,prediction=pp)
                old.npz_write(out/'model.npz',**arrays)
                io.write(out/'state.json',dict(setting=setting,source=source,inner=j,train_indices=tr,valid_indices=va,
                    certificate=cert,training_label_population_SD=sd,training_rows=len(tr),transform_sha256=io.sha(pf),
                    R4_reference_sha256=io.sha(af),R33_reference_sha256=io.sha(refpath),CPU_fit_wall_seconds=time.monotonic()-before,
                    saved_float_values=sum(a.size for a in arrays.values())))
                io.write(out/'complete.json',dict(lock_sha256=io.sha(R/'lock.json'),files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
                heads+=1;updates+=cert['iterations'];training_rows+=len(tr);numeric &= cert['numerically_qualified']
                blocks.append(dict(inner=j,n=len(va),metrics=score(y[va],pp),R4_reference=score(y[va],r4[va]),
                    R33_reference=score(y[va],r33[va]),epsilon=eps,inactive_fraction=cert['inactive_fraction'],
                    relative_gap=cert['relative_gap'],iterations=cert['iterations'],numerically_qualified=cert['numerically_qualified']))
            assert np.isfinite(pred).all() and np.isfinite(r33).all()
            m,ref4,ref33=score(y,pred),score(y,r4),score(y,r33)
            rec=dict(setting=setting,source=source,n=len(ids),metrics=m,R4_reference=ref4,R33_reference=ref33,
                delta_PCC_vs_R4=m['PCC']-ref4['PCC'],MSE_reduction_vs_R4_pct=100*(1-m['MSE']/ref4['MSE']),
                delta_PCC_vs_R33=m['PCC']-ref33['PCC'],MSE_reduction_vs_R33_pct=100*(1-m['MSE']/ref33['MSE']),inner_blocks=blocks)
            records.append(rec)
            details.extend(dict(setting=setting,source=source,sample_id=s,label=float(y[k]),prediction=float(pred[k]),
                R4_reference_prediction=float(r4[k]),R33_reference_prediction=float(r33[k])) for k,s in enumerate(ids))
            print(setting,source,'delta_vs_R33',rec['delta_PCC_vs_R33'],'MSEred',rec['MSE_reduction_vs_R33_pct'],flush=True)
    assert heads==60 and len(details)==9196;gates=qualify(records,numeric)
    io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(**gates,records=records,fit_ids=pools,source_identities=identities,new_heads=heads,
        new_proximal_updates=updates,new_internal_OOF_rows=9196,stored_training_row_occurrences=training_rows,
        reused_R33_heads=60,reused_R4_heads=60,new_source_forward=0,new_PCA=0,GPU_hours=0,outer_test_metrics_used=False,
        threadpool=threads,lock_sha256=io.sha(R/'lock.json'),diagnostic_rows_sha256=io.sha(R/'diagnostic_rows.csv'),
        pilot_files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    validate();print('QUALIFIED',gates['qualified'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('register','run'))
    if p.parse_args().mode=='register':register()
    else:
        mutex=R/'execution.mutex'
        with mutex.open('x') as f:f.write(str(os.getpid()))
        start=time.monotonic();rc=1;error=None
        try:run(start);rc=0
        except Exception:error=traceback.format_exc();raise
        finally:
            io.write(R/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,
                GPU_hours=0,new_source_forward=0))
            mutex.unlink()
