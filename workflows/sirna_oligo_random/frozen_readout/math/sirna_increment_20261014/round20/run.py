"""Preregistered fixed-map CPU pilot. No test labels or source forward."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4';sys.path.insert(0,str(R))
from random_subspace import MAP_SEEDS,make_projection,fit_transform,transform,fit_head,deployment
spec=importlib.util.spec_from_file_location('_r20_prior4',P/'run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
io=old.io;SETTINGS=('Huesken','Mixset','Takayuki');SOURCES=('core_real','full_real','initial_core')
FILES=('run.py','random_subspace.py','verify.py','test_random_subspace.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml','projections.npz')


def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==8
    assert not any(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites)
    old.npz_write(R/'projections.npz',**{f'seed{s}':make_projection(s) for s in MAP_SEEDS})
    prior=io.read(P/'lock.json');deps=dict(prior['dependencies'])
    for name in prior['files']:deps[str((P/name).resolve())]=io.sha(P/name)
    for p in (P/'lock.json',P/'results/complete.json',P/'verification.json',R.parent/'round19/closure_receipt.json'):deps[str(p.resolve())]=io.sha(p)
    for setting in SETTINGS:
        for source in SOURCES:
            for name in ('transforms.json','arrays.npz'):
                p=P/'results/folds'/setting/'fold0'/source/name;deps[str(p.resolve())]=io.sha(p)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=900,new_PCA_fits=0,new_covariance_whitenings=135,new_source_forward=0,models=135,
        primary='core_real_map42',settings=SETTINGS,sources=SOURCES,map_seeds=MAP_SEEDS,source_seed=42,
        source_dimensions=16,alpha=100.,weight=1.,prediction_averaging=False))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    assert lock['versions']==io.versions();return lock


def stats(y,p):
    pc,ms=old.score(y,p[:,None]);return dict(PCC=float(pc[0]),MSE=float(ms[0]))


def run(start):
    validate();root=R/'pilot';assert not root.exists() and not (R/'diagnostic.json').exists()
    threads=threadpool_info();assert all(t['num_threads']==1 for t in threads)
    with np.load(R/'projections.npz',allow_pickle=False) as a:maps={s:a[f'seed{s}'].copy() for s in MAP_SEEDS}
    for seed,p in maps.items():np.testing.assert_array_equal(p,make_projection(seed))
    rows,xx,identities=old.inputs();lookup={r['sample_id']:i for i,r in enumerate(rows)};plan=io.read(P/'plan.json')
    records=[];details=[];pools={};count=0;allvalid=True;locksha=io.sha(R/'lock.json')
    for setting in SETTINGS:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids'];pools[setting]=ids;assert set(ids).isdisjoint(item['test_ids'])
        fi=np.array([lookup[s] for s in ids]);y=np.array([float(rows[i]['label']) for i in fi])
        for source in SOURCES:
            raw=xx[source][fi];pf=P/'results/folds'/setting/'fold0'/source/'transforms.json';projections=io.read(pf)
            with np.load(P/'results/folds'/setting/'fold0'/source/'arrays.npz',allow_pickle=False) as a:ref=a['inner_oof'][:,18].copy()
            for map_seed in MAP_SEEDS:
                prediction=np.full(len(y),np.nan);blocks=[]
                for j,sp in enumerate(item['inner']):
                    if time.monotonic()-start>900:raise TimeoutError('No new block: stage cap')
                    tr=np.array(sp['train']);va=np.array(sp['valid']);folder=root/setting/source/f'map{map_seed}'/f'inner{j}'
                    before=time.monotonic();z,state=fit_transform(raw[tr],projections['inner'][j],maps[map_seed])
                    io.write(folder/'transform.json',state)
                    beta,center=fit_head(z,y[tr]);zv=transform(raw[va],state);pp=zv@beta+center
                    coef,offset=deployment(state,beta,center);prediction[va]=pp;count+=1;allvalid &= state['rank16_valid']
                    old.npz_write(folder/'model.npz',beta=beta,intercept=np.array(center),prediction=pp,
                        deployment_coef=coef,deployment_intercept=np.array(offset))
                    io.write(folder/'state.json',dict(setting=setting,source=source,map_seed=map_seed,inner=j,
                        train_indices=tr,valid_indices=va,projection_sha256=io.sha(pf),
                        fixed_maps_sha256=io.sha(R/'projections.npz'),CPU_fit_wall_seconds=time.monotonic()-before))
                    io.write(folder/'complete.json',dict(lock_sha256=locksha,files_sha256={p.name:io.sha(p) for p in folder.iterdir() if p.is_file()}))
                    blocks.append(dict(inner=j,n=len(va),metrics=stats(y[va],pp),reference=stats(y[va],ref[va]),rank16_valid=state['rank16_valid']))
                assert np.isfinite(prediction).all();metric,reference=stats(y,prediction),stats(y,ref)
                delta=metric['PCC']-reference['PCC'];reduction=100*(1-metric['MSE']/reference['MSE'])
                records.append(dict(setting=setting,source=source,map_seed=map_seed,n=len(y),metrics=metric,reference=reference,
                    delta_PCC_vs_R4=delta,MSE_reduction_vs_R4_pct=reduction,
                    pool_qualified=bool(delta>=.005 and reduction>=0),inner_blocks=blocks))
                details.extend(dict(setting=setting,source=source,map_seed=map_seed,sample_id=s,label=float(y[k]),
                    prediction=float(prediction[k]),reference_prediction=float(ref[k])) for k,s in enumerate(ids))
                print(setting,source,map_seed,'deltaPCC',delta,'MSE_reduction_pct',reduction,flush=True)
    gates={}
    for seed in MAP_SEEDS:
        core=[r for r in records if r['source']=='core_real' and r['map_seed']==seed]
        gates[str(seed)]=bool(sum(r['pool_qualified'] for r in core)>=2 and all(r['delta_PCC_vs_R4']>=-.005 for r in core))
    qualified=bool(all(gates.values()) and allvalid);assert count==135 and len(details)==9*sum(len(v) for v in pools.values())==20691
    io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(qualified=qualified,core_map_gates=gates,all_rank16_valid=bool(allvalid),
        records=records,fit_ids=pools,source_identities=identities,new_efficiency_fits=count,new_source_forward=0,
        new_PCA_fits=0,new_covariance_whitenings=135,GPU_hours=0,outer_test_metrics_used=False,
        prediction_averaging=False,threadpool=threads,lock_sha256=locksha,
        diagnostic_rows_sha256=io.sha(R/'diagnostic_rows.csv'),
        pilot_files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    validate();print('QUALIFIED',qualified,'GATES',gates,'RANK',allvalid,flush=True)


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
