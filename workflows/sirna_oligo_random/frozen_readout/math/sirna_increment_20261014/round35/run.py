"""One fixed nested-subset learning-curve diagnostic, no model selection."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4';Q=R.parent/'round25'
sys.path.insert(0,str(R))
from curve import subset,fit_transform,fit,head,metrics,perturbation_variance,FRACTIONS,DRAWS
spec=importlib.util.spec_from_file_location('_r35_r4',P/'run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
io=old.io;CONDITIONS=(*old.SOURCES,'TD_only')
FILES=('run.py','curve.py','verify.py','test_curve.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml','subsets.json')


def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==7
    assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
    closure=io.read(R.parent/'round34/closure_receipt.json');assert closure['status']=='closed' and closure['readback_passed']
    for n,h in closure['files_sha256'].items():assert io.sha(R.parent/n)==h,n
    prior=io.read(P/'lock.json');deps=dict(prior['dependencies'])
    for n in prior['files']:deps[str((P/n).resolve())]=io.sha(P/n)
    blocks=[];plan=io.read(P/'plan.json')
    for setting in old.SETTINGS[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids']
        for j,sp in enumerate(item['inner']):
            eligible=[ids[i] for i in sp['train']]
            for draw in DRAWS:
                for fraction in FRACTIONS:
                    chosen=[sp['train'][i] for i in subset(eligible,fraction,draw)]
                    blocks.append(dict(setting=setting,inner=j,draw=draw,fraction=fraction,train_indices=chosen,
                        valid_indices=sp['valid'],eligible_indices=sp['train'],train_ids=[ids[i] for i in chosen],full_n=len(eligible)))
        for source in old.SOURCES:
            for n in ('transforms.json','arrays.npz'):
                path=P/'results/folds'/setting/'fold0'/source/n;deps[str(path.resolve())]=io.sha(path)
            for j in range(5):
                for path in (Q/'pilot'/setting/source/f'inner{j}').iterdir():
                    if path.is_file():deps[str(path.resolve())]=io.sha(path)
    io.write(R/'subsets.json',dict(blocks=blocks,draws=DRAWS,fractions=FRACTIONS,source_seed=42,algorithm='sha256(draw NUL sample_id), take ceil(f*n), restore input order'))
    for path in (P/'lock.json',P/'plan.json',P/'results/complete.json',P/'verification.json',Q/'lock.json',Q/'diagnostic.json',Q/'verification.json',
                 R.parent/'round34/closure_receipt.json',R.parent/'round26/head.py',R.parent/'round17/nuclear.py',Path(io.__file__)):
        deps[str(path.resolve())]=io.sha(path)
    expected=io.read(R.parent/'round26/lock.json');assert io.sha(R.parent/'round26/head.py')==expected['files']['head.py']
    assert io.read(Q/'verification.json')['passed']
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=900,source_seed=42,draws=DRAWS,fractions=FRACTIONS,conditions=CONDITIONS,
        new_heads=360,new_native_heads=270,new_TD_heads=90,new_scalers=270,new_PCA=0,new_source_forward=0,
        reused_reference_heads=60,new_OOF_rows=55176,reused_OOF_rows=9196,model_selection=False,outer_test_evaluation=False,
        baseline_sum_alpha=100.,lambda_ratio=.1,row_weight='full_inner_train_count/subset_count'))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for path,h in lock['dependencies'].items():assert io.sha(Path(path))==h,path
    assert io.versions()==lock['versions'];return lock


def record(setting,condition,fraction,draw,y,p,td):
    m,base=metrics(y,p),metrics(y,td)
    return dict(setting=setting,condition=condition,fraction=fraction,draw=draw,n=len(y),**m,
        TD_PCC=base['PCC'],TD_MSE=base['MSE'],PCC_delta=m['PCC']-base['PCC'],MSE_reduction_vs_TD_pct=100*(1-m['MSE']/base['MSE']))


def run(start):
    validate();root=R/'pilot';assert not root.exists() and not (R/'diagnostic.json').exists()
    threads=threadpool_info();assert all(t['num_threads']==1 for t in threads)
    rows,xx,identities=old.inputs();lookup={r['sample_id']:i for i,r in enumerate(rows)};plan=io.read(P/'plan.json');selection=io.read(R/'subsets.json')
    details=[];refdetails=[];records=[];aggregates=[];pools={};count=native=tdcount=0;allvalid=True
    for setting in old.SETTINGS[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids'];pools[setting]=ids
        assert set(ids).isdisjoint(item['test_ids']);fi=np.array([lookup[s] for s in ids]);y=np.array([float(rows[i]['label']) for i in fi])
        raw={s:xx[s][fi] for s in old.SOURCES};reference={s:np.full(len(ids),np.nan) for s in old.SOURCES}
        for source in old.SOURCES:
            for j,split in enumerate(item['inner']):
                out=Q/'pilot'/setting/source/f'inner{j}';receipt=io.read(out/'complete.json');io.check_files(out,receipt['files_sha256'])
                state=io.read(out/'state.json');assert state['train_indices']==split['train'] and state['valid_indices']==split['valid']
                with np.load(out/'model.npz',allow_pickle=False) as a:reference[source][np.array(split['valid'])]=a['prediction']
        with np.load(P/'results/folds'/setting/'fold0/core_real/arrays.npz',allow_pickle=False) as a:reference['TD_only']=a['inner_oof'][:,3].copy()
        for condition in CONDITIONS:
            assert np.isfinite(reference[condition]).all()
            records.append(record(setting,condition,1.,None,y,reference[condition],reference['TD_only']))
            refdetails.extend(dict(setting=setting,condition=condition,sample_id=s,label=float(y[i]),prediction=float(reference[condition][i])) for i,s in enumerate(ids))
        predictions={(fraction,draw,c):np.full(len(ids),np.nan) for fraction in FRACTIONS for draw in DRAWS for c in CONDITIONS}
        for block in (b for b in selection['blocks'] if b['setting']==setting):
            if time.monotonic()-start>900:raise TimeoutError('CPU stage cap')
            tr,va=np.array(block['train_indices']),np.array(block['valid_indices']);j=block['inner'];draw=block['draw'];fraction=block['fraction']
            maps={s:fit_transform(raw[s][tr]) for s in old.SOURCES}
            for condition in CONDITIONS:
                source='core_real' if condition=='TD_only' else condition;z,t=maps[source]
                out=root/setting/f'p{int(100*fraction)}'/f'draw{draw}'/condition/f'inner{j}';before=time.monotonic()
                io.write(out/'transform.json',t)
                beta,center,cert=fit(z,y[tr],block['full_n'],'TD' if condition=='TD_only' else 'native',deadline=start+900)
                zv=head.transform(raw[source][va],t);pp=zv@beta+center;coef,offset=head.deployment(t,beta,center)
                np.testing.assert_allclose(raw[source][va]@coef+offset,pp,rtol=1e-9,atol=1e-9)
                assert np.isfinite(pp).all();predictions[fraction,draw,condition][va]=pp
                valid=bool(cert['certified']);allvalid &= valid;count+=1
                if condition=='TD_only':tdcount+=1
                else:native+=1
                old.npz_write(out/'model.npz',beta=beta,intercept=np.array(center),raw_coef=coef,raw_intercept=np.array(offset),prediction=pp)
                io.write(out/'state.json',dict(**block,condition=condition,source=source,certificate=cert,numerically_qualified=valid,CPU_fit_wall_seconds=time.monotonic()-before))
                io.write(out/'complete.json',dict(lock_sha256=io.sha(R/'lock.json'),files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
        for fraction in FRACTIONS:
            for draw in DRAWS:
                for condition in CONDITIONS:
                    p=predictions[fraction,draw,condition];assert np.isfinite(p).all()
                    records.append(record(setting,condition,fraction,draw,y,p,predictions[fraction,draw,'TD_only']))
                    details.extend(dict(setting=setting,condition=condition,fraction=fraction,draw=draw,sample_id=s,label=float(y[i]),prediction=float(p[i])) for i,s in enumerate(ids))
        for fraction in (*FRACTIONS,1.):
            for condition in CONDITIONS:
                rr=[r for r in records if r['setting']==setting and r['fraction']==fraction and r['condition']==condition]
                row=dict(setting=setting,condition=condition,fraction=fraction,n_draws=len(rr),n_validation=len(ids))
                for key in ('PCC','MSE','PCC_delta','MSE_reduction_vs_TD_pct'):
                    values=[r[key] for r in rr];row[key+'_mean']=float(np.mean(values));row[key+'_SD']=float(np.std(values,ddof=1)) if len(rr)>1 else None
                row['subset_perturbation_variance']=perturbation_variance(np.stack([predictions[fraction,s,condition] for s in DRAWS])) if fraction<1 else None
                aggregates.append(row)
                if condition=='core_real':print(setting,fraction,'Core deltaPCC mean',row['PCC_delta_mean'],'MSEgain',row['MSE_reduction_vs_TD_pct_mean'],flush=True)
    assert count==360 and native==270 and tdcount==90 and len(details)==55176 and len(refdetails)==9196 and len(records)==84 and len(aggregates)==36
    io.write_csv(R/'diagnostic_rows.csv',details);io.write_csv(R/'reference_rows.csv',refdetails);io.write_csv(R/'aggregate.csv',aggregates)
    io.write(R/'diagnostic.json',dict(records=records,aggregate=aggregates,fit_ids=pools,source_identities=identities,
        new_heads=count,new_native_heads=native,new_TD_heads=tdcount,new_scalers=270,new_internal_OOF_rows=len(details),
        reused_reference_heads=60,reused_internal_OOF_rows=len(refdetails),new_PCA=0,new_source_forward=0,GPU_hours=0,
        outer_test_metrics_used=False,model_selection=False,all_numerically_qualified=bool(allvalid),threadpool=threads,
        lock_sha256=io.sha(R/'lock.json'),output_sha256={n:io.sha(R/n) for n in ('diagnostic_rows.csv','reference_rows.csv','aggregate.csv')},
        pilot_files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    validate();print('CURVE_COMPLETE',allvalid,flush=True)


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
