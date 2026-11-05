"""Training-internal routing qualification; no new efficiency regression."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,itertools,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4';sys.path.insert(0,str(R))
from route import fit,predict,summary,qualify
sp=importlib.util.spec_from_file_location('_r23_prior4',P/'run.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
io=old.io;SETTINGS=old.SETTINGS[:3];SOURCES=old.SOURCES
FILES=('diagnose.py','route.py','verify.py','test_route.py','DIAGNOSTIC_PROTOCOL.md','CODE_AUDIT.md','pytest.xml','pytest_first6.xml')

def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==7
    assert not any(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites)
    previous=io.read(R.parent/'round22/closure_receipt.json')
    for n,h in previous['files_sha256'].items():assert io.sha(R.parent/n)==h,n
    original=io.read(P/'lock.json');deps=dict(original['dependencies'])
    for n in original['files']:deps[str((P/n).resolve())]=io.sha(P/n)
    for p in (P/'lock.json',P/'results/complete.json',P/'verification.json',R.parent/'round22/closure_receipt.json'):deps[str(p.resolve())]=io.sha(p)
    for setting in SETTINGS:
        for source in SOURCES:
            for n in ('arrays.npz','transforms.json'):
                p=P/'results/folds'/setting/'fold0'/source/n;deps[str(p.resolve())]=io.sha(p)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),CPU_only=True,
        new_unsupervised_gates=15,new_efficiency_fits=0,source_forward=0,source_seed=42,CPU_stage_cap_seconds=300,
        primary='core_real',settings=SETTINGS,sources=SOURCES,K=2,n_init=10,random_state=42,max_iter=300,tol=1e-4,
        minimum_group_fraction=.15,minimum_median_ARI=.5,minimum_abs_gain_gap=.05,minimum_consistent_inner_blocks=4,minimum_pools=2))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)

def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for n,h in lock['dependencies'].items():assert io.sha(Path(n))==h,n
    assert lock['versions']==io.versions();return lock

def run(start):
    validate();assert not (R/'diagnostic.json').exists() and not (R/'gates').exists();threads=threadpool_info();assert all(t['num_threads']==1 for t in threads)
    plan=io.read(P/'plan.json');records=[];gate_records=[];rowsout=[];route_rows=[];pools={};gate_count=0
    for setting in SETTINGS:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids'];pools[setting]=ids
        assert set(ids).isdisjoint(item['test_ids'])
        lookup={r['sample_id']:r for r in io.table(old.BENCH/'data'/f'{setting}.csv')}
        td=np.array([[float(v) for v in lookup[s]['td'].split(',')] for s in ids]);y=np.array([float(lookup[s]['label']) for s in ids])
        refpath=P/'results/folds'/setting/'fold0/core_real/transforms.json';states=io.read(refpath)
        gate_oof=np.full(len(ids),-1,int);inner=np.full(len(ids),-1,int);allassign=[];balanced=True;converged=True;gblocks=[]
        for j,split in enumerate(item['inner']):
            if time.monotonic()-start>300:raise TimeoutError('No new gate: bounded stage')
            tr,va=np.array(split['train']),np.array(split['valid']);st=states['inner'][j]
            z=(td-st['td_mean'])/st['td_scale'];before=time.monotonic();g=fit(z[tr]);gate_count+=1
            assignment=predict(z,g['centers']);allassign.append(assignment);gate_oof[va]=assignment[va];inner[va]=j
            vc=np.bincount(assignment[va],minlength=2);bc=bool(min(g['training_counts'])/len(tr)>=.15 and min(vc)/len(va)>=.15)
            conv=bool(g['iterations']<300 and g['distinct']);balanced &= bc;converged &= conv
            out=R/'gates'/setting/f'inner{j}';io.write(out/'gate.json',dict(**g,inner=j,train_indices=tr,valid_indices=va,
                valid_counts=vc,balanced=bc,converged=conv,TD_mean=st['td_mean'],TD_scale=st['td_scale'],
                reference_transform_sha256=io.sha(refpath),CPU_gate_seconds=time.monotonic()-before))
            old.npz_write(out/'assignments.npz',all_fit_assignment=assignment,OOF_assignment=assignment[va])
            io.write(out/'complete.json',dict(lock_sha256=io.sha(R/'lock.json'),files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
            gblocks.append(dict(inner=j,training_counts=g['training_counts'],valid_counts=vc,balanced=bc,converged=conv))
        assert (gate_oof>=0).all() and (inner>=0).all()
        aris=[dict(first=a,second=b,ARI=float(adjusted_rand_score(allassign[a],allassign[b]))) for a,b in itertools.combinations(range(5),2)]
        gr=dict(setting=setting,balanced=bool(balanced),all_converged=bool(converged),median_ARI=float(np.median([r['ARI'] for r in aris])),ARI_pairs=aris,inner_blocks=gblocks)
        gate_records.append(gr)
        route_rows.extend(dict(setting=setting,sample_id=s,inner_block=int(inner[i]),OOF_route=int(gate_oof[i])) for i,s in enumerate(ids))
        for source in SOURCES:
            with np.load(P/'results/folds'/setting/'fold0'/source/'arrays.npz',allow_pickle=False) as a:pt=a['inner_oof'][:,3].copy();ps=a['inner_oof'][:,18].copy()
            rec=dict(setting=setting,source=source,summary=summary(y,pt,ps,gate_oof),inner_blocks=[dict(inner=j,summary=summary(y[inner==j],pt[inner==j],ps[inner==j],gate_oof[inner==j])) for j in range(5)])
            rec['qualified']=qualify(rec,gr);records.append(rec)
            rowsout.extend(dict(setting=setting,source=source,sample_id=s,inner_block=int(inner[i]),OOF_route=int(gate_oof[i]),
                label=float(y[i]),TD_prediction=float(pt[i]),source_prediction=float(ps[i]),gain=float((y[i]-pt[i])**2-(y[i]-ps[i])**2)) for i,s in enumerate(ids))
            print(setting,source,'gap',rec['summary']['normalized_gain_gap'],'ARI',gr['median_ARI'],'qualified',rec['qualified'],flush=True)
    qualified=bool(sum(r['qualified'] for r in records if r['source']=='core_real')>=2)
    assert gate_count==15 and len(rowsout)==6897 and len(route_rows)==2299
    io.write_csv(R/'route_rows.csv',route_rows);io.write_csv(R/'diagnostic_rows.csv',rowsout)
    io.write(R/'diagnostic.json',dict(qualified=qualified,records=records,gate_records=gate_records,fit_ids=pools,
        new_unsupervised_gates=gate_count,new_efficiency_fits=0,new_source_forward=0,GPU_hours=0,outer_test_metrics_used=False,
        threadpool=threads,lock_sha256=io.sha(R/'lock.json'),diagnostic_rows_sha256=io.sha(R/'diagnostic_rows.csv'),route_rows_sha256=io.sha(R/'route_rows.csv'),
        gates_files_sha256={p.relative_to(R/'gates').as_posix():io.sha(p) for p in sorted((R/'gates').rglob('*')) if p.is_file()}))
    validate();print('QUALIFIED',qualified,flush=True)

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
            io.write(R/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,new_efficiency_fits=0,GPU_hours=0))
            mutex.unlink()
