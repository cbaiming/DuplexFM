"""Registered paired-initialization compression pilot; frozen CPU caches only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse, importlib.util, sys, time, traceback, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R = Path(__file__).resolve().parent
P = R.parent/'round4'
sys.path.insert(0,str(R))
from contrast import subspace, fitted_transform, transform, fit, deployment, qualify
spec = importlib.util.spec_from_file_location('_r34_r4',P/'run.py')
old = importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
io = old.io
CONDITIONS = ('core_contrast','core_pooled','initial_contrast')
FILES = ('run.py','contrast.py','verify.py','test_contrast.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')


def register():
    assert not (R/'lock.json').exists()
    old.validate()
    suites = list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites) == 8
    assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
    prior = io.read(P/'lock.json');deps = dict(prior['dependencies'])
    for n in prior['files']:
        deps[str((P/n).resolve())] = io.sha(P/n)
    closure = io.read(R.parent/'round33/closure_receipt.json')
    assert closure['status'] == 'closed' and closure['readback_passed']
    for n,h in closure['files_sha256'].items():
        assert io.sha(R.parent/n) == h,n
    for path in (P/'lock.json',P/'plan.json',P/'results/complete.json',P/'verification.json',Path(io.__file__),R.parent/'round33/closure_receipt.json'):
        deps[str(path.resolve())] = io.sha(path)
    for setting in old.SETTINGS[:3]:
        for source in ('core_real','initial_core'):
            for name in ('transforms.json','arrays.npz'):
                path = P/'results/folds'/setting/'fold0'/source/name
                deps[str(path.resolve())] = io.sha(path)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=900,source_seed=42,alpha=100.,source_weight=1.,dimensions=16,
        contrast_strength='trace_foreground/trace_background',primary='core_contrast',conditions=CONDITIONS,
        settings=old.SETTINGS[:3],new_heads=45,new_spectral_decompositions=30,new_OOF_rows=6897,
        reused_reference_heads=45,new_source_forward=0,prediction_averaging=False))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock = io.read(R/'lock.json');io.check_files(R,lock['files'])
    for path,h in lock['dependencies'].items():
        assert io.sha(Path(path)) == h,path
    assert io.versions() == lock['versions']
    return lock


def metrics(y,p):
    pc,ms = old.score(y,p[:,None]);return dict(PCC=float(pc[0]),MSE=float(ms[0]))


def run(started):
    validate();root = R/'pilot'
    assert not root.exists() and not (R/'diagnostic.json').exists()
    threads = threadpool_info();assert all(t['num_threads']==1 for t in threads)
    rows,xx,identities = old.inputs()
    fg,bg = identities['core_real'],identities['initial_core']
    assert fg['initial_sha256'] == bg['checkpoint_sha256'] == bg['initial_sha256']
    assert fg['variant'] == bg['variant'] == 'core_real'
    assert fg['source_receipt_sha256'] == bg['source_receipt_sha256']
    assert fg['initial_type'] == bg['initial_type'] == 'exact_pre_update_task_state_with_pretrained_public_foundations'
    lookup = {r['sample_id']:i for i,r in enumerate(rows)}
    plan = io.read(P/'plan.json');records=[];details=[];pools={};heads=maps=0;numeric=True
    for setting in old.SETTINGS[:3]:
        item = next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids = item['fit_ids'];pools[setting]=ids
        assert set(ids).isdisjoint(item['test_ids'])
        fi = np.array([lookup[s] for s in ids]);y = np.array([float(rows[i]['label']) for i in fi])
        raw = {source:xx[source][fi] for source in ('core_real','initial_core')}
        states={};reference={};refsha={}
        for source in raw:
            pf=P/'results/folds'/setting/'fold0'/source/'transforms.json';af=pf.parent/'arrays.npz'
            states[source]=io.read(pf);refsha[source]=dict(transforms=io.sha(pf),arrays=io.sha(af))
            with np.load(af,allow_pickle=False) as a:
                reference[source]=a['inner_oof'][:,18].copy()
                if source=='core_real':tdref=a['inner_oof'][:,3].copy()
        predictions={c:np.full(len(ids),np.nan) for c in CONDITIONS};blocks={c:[] for c in CONDITIONS}
        for j,split in enumerate(item['inner']):
            if time.monotonic()-started>900:raise TimeoutError('CPU stage cap')
            tr,va=np.array(split['train']),np.array(split['valid']);base=states['core_real']['inner'][j]
            projections={m:subspace(raw['core_real'][tr],raw['initial_core'][tr],base,m) for m in ('contrast','pooled')};maps+=2
            for condition in CONDITIONS:
                source='initial_core' if condition.startswith('initial') else 'core_real'
                mode='pooled' if condition=='core_pooled' else 'contrast';mapping=projections[mode]
                out=root/setting/condition/f'inner{j}';before=time.monotonic()
                z,projection=fitted_transform(raw[source][tr],base,mapping)
                # Persist unsupervised state before the efficiency solve.
                io.write(out/'transform.json',dict(projection=projection,mapping=mapping))
                model=fit(z,y[tr]);zv=transform(raw[source][va],projection);pp=zv@model['beta']+model['intercept']
                coef,intercept=deployment(projection,model['beta'],model['intercept'])
                np.testing.assert_allclose(raw[source][va]@coef+intercept,pp,rtol=1e-10,atol=1e-10)
                assert np.isfinite(pp).all();predictions[condition][va]=pp;heads+=1
                valid=bool(model['numerically_qualified'] and mapping['spectral_qualified']);numeric &= valid
                old.npz_write(out/'model.npz',beta=model['beta'],intercept=np.array(model['intercept']),
                              raw_coef=coef,raw_intercept=np.array(intercept),prediction=pp)
                certificate={k:v for k,v in model.items() if k not in ('beta','intercept')}
                io.write(out/'state.json',dict(setting=setting,condition=condition,source=source,inner=j,
                    train_indices=tr,valid_indices=va,certificate=certificate,numerically_qualified=valid,
                    reference_sha256=refsha,CPU_fit_wall_seconds=time.monotonic()-before))
                io.write(out/'complete.json',dict(lock_sha256=io.sha(R/'lock.json'),files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
                blocks[condition].append(dict(inner=j,n=len(va),metrics=metrics(y[va],pp),reference=metrics(y[va],reference[source][va]),
                    gamma=mapping['gamma'],positive_eigenvalues=mapping['positive_eigenvalues'],
                    smallest_selected_eigenvalue=float(mapping['eigenvalues'][-1]),normal_error=model['normal_error'],numerically_qualified=valid))
        for condition in CONDITIONS:
            source='initial_core' if condition.startswith('initial') else 'core_real'
            pred=predictions[condition];assert np.isfinite(pred).all()
            m,ref=metrics(y,pred),metrics(y,reference[source])
            record=dict(setting=setting,condition=condition,source=source,n=len(ids),metrics=m,reference=ref,TD_reference=metrics(y,tdref),
                delta_PCC_vs_R4=m['PCC']-ref['PCC'],MSE_reduction_vs_R4_pct=100*(1-m['MSE']/ref['MSE']),inner_blocks=blocks[condition])
            records.append(record)
            details.extend(dict(setting=setting,condition=condition,sample_id=s,label=float(y[k]),prediction=float(pred[k]),reference_prediction=float(reference[source][k]),TD_prediction=float(tdref[k])) for k,s in enumerate(ids))
            print(setting,condition,'deltaPCC',record['delta_PCC_vs_R4'],'MSEred',record['MSE_reduction_vs_R4_pct'],flush=True)
    assert heads==45 and maps==30 and len(details)==6897
    io.write_csv(R/'diagnostic_rows.csv',details)
    gates=qualify(records,numeric)
    io.write(R/'diagnostic.json',dict(**gates,records=records,fit_ids=pools,source_identities={k:identities[k] for k in ('core_real','initial_core')},
        new_heads=45,new_spectral_decompositions=30,new_transform_standardizations=45,new_internal_OOF_rows=6897,
        reused_reference_heads=45,new_source_forward=0,GPU_hours=0,outer_test_metrics_used=False,
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
