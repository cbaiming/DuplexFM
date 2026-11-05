"""One bounded qualification using cached inputs and old within-block TD OOF residuals."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;B=R.parent;P=B/'round4'
sys.path.insert(0,str(R))
from sample_graph import graph,energies,pool,qualify,K


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


old=load('_r46_prior4',P/'run.py');io=old.io
native=load('_r46_native_transform',B/'round21/coordinate_sparse.py')
benchmark=load('_r46_sequence_adjacency',old.RESEARCH/'sirna_benchmark_20261007/benchmark_core.py')
CONDITIONS=(*old.SOURCES,'TD_only')
FILES=('run.py','sample_graph.py','verify.py','test_sample_graph.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')


def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==10
    assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
    previous=io.read(B/'round45/closure_receipt.json');assert previous['status']=='closed' and previous['readback_passed']
    for name,h in previous['files_sha256'].items():assert io.sha(B/name)==h,name
    p=io.read(P/'lock.json');deps=dict(p['dependencies'])
    for name in p['files']:deps[str((P/name).resolve())]=io.sha(P/name)
    deps[str((P/'lock.json').resolve())]=io.sha(P/'lock.json')
    for path in (B/'round45/closure_receipt.json',Path(io.__file__),Path(native.__file__),Path(benchmark.__file__),P/'verify.py'):
        deps[str(path.resolve())]=io.sha(path)
    for setting in old.SETTINGS[:3]:
        for source in old.SOURCES:
            for j in range(5):
                path=B/'round21/pilot'/setting/source/'dense'/f'inner{j}'/'transform.json'
                deps[str(path.resolve())]=io.sha(path)
            folder=P/'results/folds'/setting/'fold0'/source
            receipt=io.read(folder/'complete.json')
            assert receipt['lock_sha256']==io.sha(P/'lock.json')
            io.check_files(folder,receipt['files_sha256'])
            for name in ('transforms.json','complete.json'):
                path=folder/name;deps[str(path.resolve())]=io.sha(path)
        path=P/'results/folds'/setting/'fold0/core_real/arrays.npz';deps[str(path.resolve())]=io.sha(path)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=900,source_seed=42,settings=old.SETTINGS[:3],conditions=CONDITIONS,
        primary='core_real',K=10,new_graphs=60,eligibility_masks=15,condition_rows=9196,reused_unique_TD_heads=15,
        new_efficiency_heads=0,new_OOF_rows=0,new_source_forward=0,new_PCA_fits=0,GPU_hours=0,
        main_max_ratio=.90,margin_vs_TD=.05,minimum_blocks_below_null=4,minimum_qualified_pools=2,minimum_row_coverage=.9))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for path,h in lock['dependencies'].items():assert io.sha(Path(path))==h,path
    assert io.versions()==lock['versions'];return lock


def run(start):
    validate();root=R/'graph_audit';assert not root.exists() and not (R/'diagnostic.json').exists()
    threads=threadpool_info();assert all(t['num_threads']==1 for t in threads)
    rows,raw,identities=old.inputs();rowindex={r['sample_id']:i for i,r in enumerate(rows)}
    plan=io.read(P/'plan.json');records=[];details=[];pools={};graphs=0;masks=0
    for setting,expected_n in zip(old.SETTINGS[:3],(1564,319,416)):
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids'];assert len(ids)==expected_n and set(ids).isdisjoint(item['test_ids'])
        pools[setting]=ids;fi=np.array([rowindex[s] for s in ids]);fr=[rows[i] for i in fi]
        y=np.array([float(r['label']) for r in fr]);tdpath=P/'results/folds'/setting/'fold0/core_real/arrays.npz'
        with np.load(tdpath,allow_pickle=False) as a:tdpred=a['inner_oof'][:,3].copy()
        assert tdpred.shape==y.shape and np.isfinite(tdpred).all()
        blocks={name:[] for name in CONDITIONS};coverage=np.zeros(len(ids),int)
        for j,sp in enumerate(item['inner']):
            if time.monotonic()-start>900:raise TimeoutError('CPU diagnostic cap')
            tr,va=np.array(sp['train']),np.array(sp['valid']);coverage[va]+=1
            assert set(tr).isdisjoint(va)
            valid_ids=[ids[i] for i in va]
            frame=pd.DataFrame([dict(guide=fr[i]['guide'],target57=fr[i]['target57']) for i in va])
            eligible=~benchmark.build_adjacency(frame);np.fill_diagonal(eligible,False)
            maskpath=root/setting/'eligibility'/f'inner{j}.npz';old.npz_write(maskpath,eligible=eligible);masks+=1
            yy=y[va];pp=tdpred[va];residual=yy-pp
            for condition in CONDITIONS:
                source='core_real' if condition=='TD_only' else condition
                tp=B/'round21/pilot'/setting/source/'dense'/f'inner{j}'/'transform.json';state=io.read(tp)
                previous=io.read(P/'results/folds'/setting/'fold0'/source/'transforms.json')['inner'][j]
                for key in ('td_mean','td_scale','src_mean','src_scale'):
                    np.testing.assert_array_equal(state[key],previous[key])
                allz=native.transform(raw[source][fi[va]],state)
                z=allz[:,:24] if condition=='TD_only' else allz[:,24:]
                selected=graph(z,eligible,valid_ids)
                residual_stat,rkn,rnull=energies(residual,eligible,selected)
                response_stat,ykn,ynull=energies(yy,eligible,selected)
                out=root/setting/condition/f'inner{j}'
                old.npz_write(out/'graph.npz',neighbors=selected,eligible_count=eligible.sum(1),
                              residual_knn=rkn,residual_null=rnull,response_knn=ykn,response_null=ynull)
                block=dict(inner=j,n=len(va),residual=residual_stat,response=response_stat)
                io.write(out/'state.json',dict(setting=setting,condition=condition,**block,
                    train_indices=sp['train'],valid_indices=sp['valid'],sample_ids=valid_ids,
                    transform_path=str(tp.resolve()),transform_sha256=io.sha(tp),
                    eligibility_path=str(maskpath.resolve()),eligibility_sha256=io.sha(maskpath),
                    TD_OOF_path=str(tdpath.resolve()),TD_OOF_sha256=io.sha(tdpath),TD_OOF_key='inner_oof[:,3]',
                    input_only_graph=True,uses_known_fit_responses_for_diagnostic=True,deployable=False,new_predictions=False))
                io.write(out/'complete.json',dict(lock_sha256=io.sha(R/'lock.json'),
                    files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
                graphs+=1;blocks[condition].append(block)
                details.extend(dict(setting=setting,condition=condition,inner=j,sample_id=s,
                    eligible_count=int(eligible.sum(1)[i]),neighbor_count=int((selected[i]>=0).sum()),
                    neighbors=';'.join(valid_ids[k] for k in selected[i] if k>=0),label=float(yy[i]),TD_OOF=float(pp[i]),
                    residual=float(residual[i]),residual_knn=None if np.isnan(rkn[i]) else float(rkn[i]),
                    residual_null=None if np.isnan(rnull[i]) else float(rnull[i]),response_knn=None if np.isnan(ykn[i]) else float(ykn[i]),
                    response_null=None if np.isnan(ynull[i]) else float(ynull[i])) for i,s in enumerate(valid_ids))
        assert (coverage==1).all()
        for condition in CONDITIONS:
            agg=pool(blocks[condition]);assert agg['n']==expected_n
            records.append(dict(setting=setting,condition=condition,aggregate=agg,inner_blocks=blocks[condition]))
            print(setting,condition,'TD_residual_neighbor_ratio',agg['residual']['ratio'],flush=True)
    assert graphs==60 and masks==15 and len(records)==12 and len(details)==9196
    result=qualify(records);io.write_csv(R/'diagnostic_rows.csv',details)
    io.write(R/'diagnostic.json',dict(**result,records=records,fit_ids=pools,source_identities=identities,
        new_graphs=graphs,eligibility_masks=masks,condition_rows=len(details),reused_unique_TD_heads=15,
        new_efficiency_heads=0,new_OOF_rows=0,new_source_forward=0,new_PCA_fits=0,GPU_hours=0,
        outer_test_metrics_used=False,deployable=False,threadpool=threads,
        lock_sha256=io.sha(R/'lock.json'),rows_sha256=io.sha(R/'diagnostic_rows.csv'),
        graph_files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    validate();print('QUALIFICATION_ONLY',result['qualified'],flush=True)


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
            io.write(R/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,
                GPU_hours=0,new_source_forward=0))
            mutex.unlink()
