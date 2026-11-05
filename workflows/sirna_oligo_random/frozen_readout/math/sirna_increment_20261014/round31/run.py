"""Bounded smooth conditional readout; all selection is internal to training."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4'
sys.path.insert(0,str(R))
from soft_route import fit,predict,branch_predict,qualify
sp=importlib.util.spec_from_file_location('_r31_old4',P/'run.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
io=old.io;SETTINGS=old.SETTINGS[:3];SOURCES=(*old.SOURCES,'TD_only')
FILES=('run.py','soft_route.py','verify.py','test_soft_route.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')


def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==8
    assert not any(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites)
    previous=io.read(R.parent/'round30/closure_receipt.json')
    assert previous['status']=='closed' and previous['readback_passed']
    for name,h in previous['files_sha256'].items():assert io.sha(R.parent/name)==h,name
    prior=io.read(P/'lock.json');deps=dict(prior['dependencies'])
    for name in prior['files']:deps[str((P/name).resolve())]=io.sha(P/name)
    for p in (P/'lock.json',P/'plan.json',P/'results/complete.json',P/'verification.json',Path(io.__file__),R.parent/'round30/closure_receipt.json'):
        deps[str(p.resolve())]=io.sha(p)
    for setting in SETTINGS:
        for source in old.SOURCES:
            for name in ('transforms.json','arrays.npz'):
                p=P/'results/folds'/setting/'fold0'/source/name;deps[str(p.resolve())]=io.sha(p)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),CPU_only=True,
        CPU_limit_seconds=900,source_seed=42,head_seed=42,settings=SETTINGS,sources=SOURCES,primary='core_real',
        shared_alpha=100.,local_multiplier=4.,gate_fraction=.01,maxiter=500,maxfun=10000,maxls=50,ftol=1e-12,gtol=1e-8,
        audit_gradient_limit=1e-6,new_heads=60,new_internal_OOF_rows=9196,reused_reference_heads=60,
        new_source_forward=0,new_PCA=0))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for name,h in lock['dependencies'].items():assert io.sha(Path(name))==h,name
    assert lock['versions']==io.versions();return lock


def stats(y,p):
    pc,ms=old.score(y,p[:,None]);return dict(PCC=float(pc[0]),MSE=float(ms[0]))


def gate_stats(g):return dict(mean=float(g.mean()),std=float(g.std()),minimum=float(g.min()),maximum=float(g.max()))


def run(start):
    validate();root=R/'pilot';assert not root.exists() and not (R/'diagnostic.json').exists()
    threads=threadpool_info();assert all(t['num_threads']==1 for t in threads)
    rows,xx,identities=old.inputs();lookup={r['sample_id']:i for i,r in enumerate(rows)}
    plan=io.read(P/'plan.json');locksha=io.sha(R/'lock.json');records=[];details=[];pools={};count=solves=0;numeric=True
    for setting in SETTINGS:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids'];pools[setting]=ids;assert set(ids).isdisjoint(item['test_ids'])
        fi=np.array([lookup[s] for s in ids]);y=np.array([float(rows[i]['label']) for i in fi])
        for source in SOURCES:
            src='core_real' if source=='TD_only' else source;raw=xx[src][fi]
            pf=P/'results/folds'/setting/'fold0'/src/'transforms.json';af=pf.parent/'arrays.npz';states=io.read(pf)
            with np.load(af,allow_pickle=False) as a:reference=a['inner_oof'][:,3 if source=='TD_only' else 18].copy()
            prediction=np.full(len(y),np.nan);gate_oof=np.full(len(y),np.nan);blocks=[]
            for j,split in enumerate(item['inner']):
                if time.monotonic()-start>900:raise TimeoutError('No new head after stage cap')
                tr,va=np.array(split['train']),np.array(split['valid']);z=old.transform(raw,states['inner'][j])
                if source=='TD_only':z=z[:,:24]
                out=root/setting/source/f'inner{j}';before=time.monotonic()
                def save_initial(gate):old.npz_write(out/'initial.npz',gate=gate)
                model=fit(z[tr],y[tr],save_initial=save_initial,deadline=start+900)
                pp,gg=predict(z[va],model);bp,bg=branch_predict(z[va],model)
                np.testing.assert_allclose(pp,bp,atol=1e-11,rtol=1e-11);np.testing.assert_array_equal(gg,bg)
                _,train_gate=predict(z[tr],model);prediction[va]=pp;gate_oof[va]=gg
                count+=1;solves+=model['new_linear_solves'];numeric &= model['numerically_qualified']
                old.npz_write(out/'model.npz',gate=model['gate'],beta=model['beta'],intercept=np.array(model['intercept']),
                    h_mean=np.array(model['h_mean']),j_mean=model['j_mean'],prediction=pp,gate_probability=gg,accepted_gates=model['accepted_gates'])
                keys=('initial_objective','objective','gradient','gradient_max','gamma','loss_scale','normal_error','normal_limit','train_MSE',
                    'numerically_qualified','success','message','iterations','function_evaluations','new_linear_solves',
                    'shared_alpha','local_multiplier','gate_fraction')
                info={k:model[k] for k in keys}
                io.write(out/'state.json',dict(setting=setting,source=source,inner=j,train_indices=tr,valid_indices=va,
                    certificate=info,reference_transform_sha256=io.sha(pf),reference_arrays_sha256=io.sha(af),
                    train_gate=gate_stats(train_gate),valid_gate=gate_stats(gg),CPU_fit_wall_seconds=time.monotonic()-before))
                io.write(out/'complete.json',dict(lock_sha256=locksha,files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
                blocks.append(dict(inner=j,n=len(va),metrics=stats(y[va],pp),reference=stats(y[va],reference[va]),
                    train_gate=gate_stats(train_gate),valid_gate=gate_stats(gg),iterations=model['iterations'],new_linear_solves=model['new_linear_solves'],
                    gradient_max=model['gradient_max'],numerically_qualified=model['numerically_qualified']))
            assert np.isfinite(prediction).all() and np.isfinite(gate_oof).all()
            m,rr=stats(y,prediction),stats(y,reference)
            rec=dict(setting=setting,source=source,n=len(y),metrics=m,reference=rr,
                delta_PCC_vs_R4=m['PCC']-rr['PCC'],MSE_reduction_vs_R4_pct=100*(1-m['MSE']/rr['MSE']),inner_blocks=blocks)
            records.append(rec)
            details.extend(dict(setting=setting,source=source,sample_id=s,label=float(y[k]),prediction=float(prediction[k]),
                gate_probability=float(gate_oof[k]),reference_prediction=float(reference[k])) for k,s in enumerate(ids))
            print(setting,source,'deltaPCC',rec['delta_PCC_vs_R4'],'MSEred',rec['MSE_reduction_vs_R4_pct'],flush=True)
    assert count==60 and len(details)==9196
    gates=qualify(records,numeric);io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(**gates,records=records,fit_ids=pools,source_identities=identities,new_heads=count,new_linear_solves=solves,
        new_internal_OOF_rows=9196,reused_reference_heads=60,new_source_forward=0,new_PCA=0,GPU_hours=0,outer_test_metrics_used=False,
        threadpool=threads,lock_sha256=locksha,diagnostic_rows_sha256=io.sha(R/'diagnostic_rows.csv'),
        pilot_files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    validate();print('QUALIFIED',gates['qualified'],'NEW_SOLVES',solves,flush=True)


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
