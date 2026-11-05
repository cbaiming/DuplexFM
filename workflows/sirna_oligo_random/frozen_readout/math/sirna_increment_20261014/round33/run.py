"""Fixed CPU kernel pilot with original inner splits and cache-only inputs."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4'
sys.path.insert(0,str(R))
from relu_kernel import fit,predict,qualify
spec=importlib.util.spec_from_file_location('_r33_r4',P/'run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
io=old.io;SOURCES=(*old.SOURCES,'TD_only')
FILES=('run.py','relu_kernel.py','verify.py','test_relu_kernel.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')
ARRAY_KEYS=('train_u','train_td','dual','td_coef','raw_row_mean')
SCALAR_KEYS=('intercept','raw_grand_mean','aux_scale')


def register():
    assert not (R/'lock.json').exists();old.validate()
    tests=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(x.get('tests',0)) for x in tests)==9
    assert not any(int(x.get(k,0)) for x in tests for k in ('errors','failures','skipped'))
    c=io.read(R.parent/'round32/closure_receipt.json');assert c['status']=='closed' and c['readback_passed']
    for name,h in c['files_sha256'].items():assert io.sha(R.parent/name)==h,name
    prior=io.read(P/'lock.json');deps=dict(prior['dependencies'])
    for name in prior['files']:deps[str((P/name).resolve())]=io.sha(P/name)
    for path in (P/'lock.json',P/'plan.json',P/'results/complete.json',P/'verification.json',Path(io.__file__),R.parent/'round32/closure_receipt.json'):
        deps[str(path.resolve())]=io.sha(path)
    for setting in old.SETTINGS[:3]:
        for source in old.SOURCES:
            for name in ('transforms.json','arrays.npz'):
                p=P/'results/folds'/setting/'fold0'/source/name;deps[str(p.resolve())]=io.sha(p)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=900,source_seed=42,alpha=100.,aux_average_trace=16.,kernel_degree=1,kernel_depth=1,
        no_kernel_bias=True,exact_training_Gram=True,maximum_training_rows=2000,
        settings=old.SETTINGS[:3],sources=SOURCES,primary='core_real',new_heads=60,new_OOF_rows=9196,
        reused_reference_heads=60,new_PCA=0,new_source_forward=0))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for path,h in lock['dependencies'].items():assert io.sha(Path(path))==h,path
    assert io.versions()==lock['versions'];return lock


def score(y,p):
    pc,ms=old.score(y,p[:,None]);return dict(PCC=float(pc[0]),MSE=float(ms[0]))


def run(start):
    validate();root=R/'pilot';assert not root.exists() and not (R/'diagnostic.json').exists()
    threads=threadpool_info();assert all(t['num_threads']==1 for t in threads)
    rows,xx,identities=old.inputs();lookup={r['sample_id']:i for i,r in enumerate(rows)}
    plan=io.read(P/'plan.json');records=[];details=[];pools={};heads=training_rows=0;all_numeric=True
    for setting in old.SETTINGS[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids'];pools[setting]=ids;assert set(ids).isdisjoint(item['test_ids'])
        fi=np.array([lookup[s] for s in ids]);y=np.array([float(rows[i]['label']) for i in fi])
        for source in SOURCES:
            src='core_real' if source=='TD_only' else source;raw=xx[src][fi]
            pf=P/'results/folds'/setting/'fold0'/src/'transforms.json';af=pf.parent/'arrays.npz';states=io.read(pf)
            with np.load(af,allow_pickle=False) as a:reference=a['inner_oof'][:,3 if source=='TD_only' else 18].copy()
            pred=np.full(len(ids),np.nan);blocks=[]
            for j,split in enumerate(item['inner']):
                if time.monotonic()-start>900:raise TimeoutError('CPU stage cap')
                tr,va=np.array(split['train']),np.array(split['valid']);z=old.transform(raw,states['inner'][j])
                if source=='TD_only':z=z[:,:24]
                out=root/setting/source/f'inner{j}';before=time.monotonic();model=fit(z[tr],y[tr],deadline=start+900)
                pp=predict(z[va],model);assert np.isfinite(pp).all();pred[va]=pp
                heads+=1;training_rows+=len(tr);all_numeric &= model['numerically_qualified']
                arrays={k:np.asarray(model[k]) for k in ARRAY_KEYS+SCALAR_KEYS};arrays['prediction']=pp
                old.npz_write(out/'model.npz',**arrays)
                certificate={k:v for k,v in model.items() if k not in ARRAY_KEYS+SCALAR_KEYS}
                io.write(out/'state.json',dict(setting=setting,source=source,inner=j,train_indices=tr,valid_indices=va,certificate=certificate,
                    transform_sha256=io.sha(pf),reference_arrays_sha256=io.sha(af),CPU_fit_wall_seconds=time.monotonic()-before,
                    saved_float_values=sum(a.size for a in arrays.values())))
                io.write(out/'complete.json',dict(lock_sha256=io.sha(R/'lock.json'),files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
                blocks.append(dict(inner=j,n=len(va),training_rows=len(tr),metrics=score(y[va],pp),reference=score(y[va],reference[va]),
                    normal_error=model['normal_error'],numerically_qualified=model['numerically_qualified']))
            assert np.isfinite(pred).all();m,ref=score(y,pred),score(y,reference)
            rec=dict(setting=setting,source=source,n=len(ids),metrics=m,reference=ref,
                delta_PCC_vs_R4=m['PCC']-ref['PCC'],MSE_reduction_vs_R4_pct=100*(1-m['MSE']/ref['MSE']),inner_blocks=blocks)
            records.append(rec)
            details.extend(dict(setting=setting,source=source,sample_id=s,label=float(y[k]),prediction=float(pred[k]),reference_prediction=float(reference[k])) for k,s in enumerate(ids))
            print(setting,source,'deltaPCC',rec['delta_PCC_vs_R4'],'MSEred',rec['MSE_reduction_vs_R4_pct'],flush=True)
    assert heads==60 and len(details)==9196;gates=qualify(records,all_numeric)
    io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(**gates,records=records,fit_ids=pools,source_identities=identities,new_heads=heads,
        new_linear_solves=heads,stored_training_row_occurrences=training_rows,new_internal_OOF_rows=9196,reused_reference_heads=60,
        new_source_forward=0,new_PCA=0,GPU_hours=0,outer_test_metrics_used=False,threadpool=threads,lock_sha256=io.sha(R/'lock.json'),
        diagnostic_rows_sha256=io.sha(R/'diagnostic_rows.csv'),pilot_files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    validate();print('QUALIFIED',gates['qualified'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('register','run'))
    if parser.parse_args().mode=='register':register()
    else:
        mutex=R/'execution.mutex'
        with mutex.open('x') as f:f.write(str(os.getpid()))
        start=time.monotonic();rc=1;error=None
        try:run(start);rc=0
        except Exception:error=traceback.format_exc();raise
        finally:
            io.write(R/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0))
            mutex.unlink()
