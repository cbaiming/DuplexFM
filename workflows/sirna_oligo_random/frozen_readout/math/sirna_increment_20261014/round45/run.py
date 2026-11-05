"""Single bounded optimistic ordering diagnostic. Never publish these projections as OOF."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4';N=R.parent/'round25'
sys.path.insert(0,str(R))
from monotone_bound import project,qualify
spec=importlib.util.spec_from_file_location('_r45_prior4',P/'run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
io=old.io;CONDITIONS=(*old.SOURCES,'TD_only')
FILES=('run.py','monotone_bound.py','verify.py','test_monotone_bound.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')


def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==9
    assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
    previous=io.read(R.parent/'round44/closure_receipt.json')
    assert previous['status']=='closed' and previous['readback_passed']
    for name,h in previous['files_sha256'].items():assert io.sha(R.parent/name)==h,name
    prior=io.read(N/'lock.json');io.check_files(N,prior['files']);deps=dict(prior['dependencies'])
    for name in prior['files']:deps[str((N/name).resolve())]=io.sha(N/name)
    for path in (N/'lock.json',N/'closure_receipt.json',N/'diagnostic.json',N/'verification.json',Path(io.__file__),
                 R.parent/'round44/closure_receipt.json',P/'verify.py'):
        deps[str(path.resolve())]=io.sha(path)
    assert io.read(N/'verification.json')['passed']
    for name,h in io.read(N/'diagnostic.json')['pilot_files_sha256'].items():
        path=N/'pilot'/name;assert io.sha(path)==h;deps[str(path.resolve())]=h
    for setting in old.SETTINGS[:3]:
        path=P/'results/folds'/setting/'fold0/core_real/arrays.npz';deps[str(path.resolve())]=io.sha(path)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=900,source_seed=42,settings=old.SETTINGS[:3],conditions=CONDITIONS,primary='core_real',
        new_label_using_isotonic_projections=72,pool_projections=12,inner_valid_projections=60,reused_native_heads=45,reused_unique_TD_heads=15,
        source_score_rows=9196,new_efficiency_heads=0,new_OOF_rows=0,new_source_forward=0,GPU_hours=0,
        main_PCC_room_min=.01,main_MSE_beyond_affine_min_pct=1.,paired_PCC_margin=.005,paired_noninferiority=-.005,
        optimistic_diagnostic_only=True,not_deployable=True))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for path,h in lock['dependencies'].items():assert io.sha(Path(path))==h,path
    assert io.versions()==lock['versions'];return lock


def run(start):
    validate();root=R/'projection_audit';assert not root.exists() and not (R/'diagnostic.json').exists()
    threads=threadpool_info();assert all(t['num_threads']==1 for t in threads)
    plan=io.read(P/'plan.json');native_diag=io.read(N/'diagnostic.json')
    records=[];details=[];pools={};count=0;numeric=True
    for setting in old.SETTINGS[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids'];pools[setting]=ids
        assert set(ids).isdisjoint(item['test_ids'])
        rowmap={r['sample_id']:r for r in io.table(old.BENCH/'data'/f'{setting}.csv')}
        y=np.array([float(rowmap[s]['label']) for s in ids]);foldid=np.full(len(ids),-1,dtype=int)
        for j,sp in enumerate(item['inner']):
            assert set(sp['train']).isdisjoint(sp['valid']);foldid[np.array(sp['valid'])]=j
        assert np.all(foldid>=0)
        for condition in CONDITIONS:
            prediction=np.full(len(ids),np.nan);references=[]
            if condition=='TD_only':
                path=P/'results/folds'/setting/'fold0/core_real/arrays.npz'
                with np.load(path,allow_pickle=False) as a:prediction=a['inner_oof'][:,3].copy()
                references.append(dict(path=str(path.resolve()),sha256=io.sha(path),key='inner_oof[:,3]'))
            else:
                for j,sp in enumerate(item['inner']):
                    folder=N/'pilot'/setting/condition/f'inner{j}';receipt=io.read(folder/'complete.json');io.check_files(folder,receipt['files_sha256'])
                    state=io.read(folder/'state.json');assert state['numerically_qualified']
                    assert state['train_indices']==sp['train'] and state['valid_indices']==sp['valid']
                    path=folder/'model.npz'
                    with np.load(path,allow_pickle=False) as a:prediction[np.array(sp['valid'])]=a['prediction']
                    references.append(dict(path=str(path.resolve()),sha256=io.sha(path),key='prediction',inner=j))
            assert np.isfinite(prediction).all();all_projected=None;inner_projected=np.full(len(ids),np.nan);blocks=[]
            for j in range(-1,5):
                if time.monotonic()-start>900:raise TimeoutError('CPU diagnostic cap')
                indices=np.arange(len(ids)) if j==-1 else np.array(item['inner'][j]['valid'])
                scope='pool' if j==-1 else f'inner_valid{j}';out=root/setting/condition/scope
                p,state=project(prediction[indices],y[indices]);count+=1;numeric &= state['certificate']['certified']
                old.npz_write(out/'optimistic_projection.npz',fixed_OOF_score=prediction[indices],optimistic_label_using_projection=p)
                io.write(out/'state.json',dict(setting=setting,condition=condition,inner=j,indices=indices,sample_ids=[ids[k] for k in indices],
                    original_references=references,**state))
                io.write(out/'complete.json',dict(lock_sha256=io.sha(R/'lock.json'),uses_evaluation_labels=True,
                    deployable=False,files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
                if j==-1:all_projected=p;mainmetrics=state['metrics']
                else:
                    inner_projected[indices]=p
                    blocks.append(dict(inner=j,n=len(indices),metrics=state['metrics'],numerically_qualified=state['certificate']['certified']))
            assert np.isfinite(inner_projected).all()
            records.append(dict(setting=setting,condition=condition,n=len(ids),metrics=mainmetrics,inner_blocks=blocks))
            details.extend(dict(setting=setting,condition=condition,sample_id=s,inner=int(foldid[k]),label=float(y[k]),
                original_OOF_prediction=float(prediction[k]),optimistic_pool_projection=float(all_projected[k]),
                optimistic_inner_projection=float(inner_projected[k])) for k,s in enumerate(ids))
            print(setting,condition,'rawPCC',mainmetrics['raw_PCC'],'optimistic_upper',mainmetrics['upper_PCC'],flush=True)
    assert count==72 and len(details)==9196 and len(records)==12;gates=qualify(records,numeric)
    io.write_csv(R/'optimistic_projection_rows.csv',details)
    io.write(R/'diagnostic.json',dict(**gates,records=records,fit_ids=pools,source_identities=native_diag['source_identities'],
        new_label_using_isotonic_projections=72,pool_projections=12,inner_valid_projections=60,reused_native_heads=45,reused_unique_TD_heads=15,
        source_score_rows=9196,new_efficiency_heads=0,new_OOF_rows=0,new_source_forward=0,GPU_hours=0,
        outer_test_metrics_used=False,optimistic_diagnostic_only=True,deployable=False,threadpool=threads,
        lock_sha256=io.sha(R/'lock.json'),projection_rows_sha256=io.sha(R/'optimistic_projection_rows.csv'),
        projection_files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    validate();print('QUALIFICATION_ONLY',gates['qualified'],flush=True)


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
