"""Label-free paired-background qualification on the original training pools."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4';D=R.parent/'round21';N=R.parent/'round25'
sys.path.insert(0,str(R))
from background import fit_map,permutation,statistics,aggregate,qualify
s=importlib.util.spec_from_file_location('_r38_old4',P/'run.py');old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
io=old.io;FILES=('run.py','background.py','verify.py','test_background.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')
CONDITIONS=('paired_initial','shuffled_initial','TD24')


def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'));assert sum(int(x.get('tests',0)) for x in suites)==9
    assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
    previous=io.read(R.parent/'round37/closure_receipt.json');assert previous['status']=='closed' and previous['readback_passed']
    for name,h in previous['files_sha256'].items():assert io.sha(R.parent/name)==h,name
    prior=io.read(N/'lock.json');io.check_files(N,prior['files']);deps=dict(prior['dependencies'])
    for name in prior['files']:deps[str((N/name).resolve())]=io.sha(N/name)
    for p in (N/'lock.json',N/'verification.json',Path(io.__file__),R.parent/'round37/closure_receipt.json'):
        deps[str(p.resolve())]=io.sha(p)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),CPU_only=True,
        CPU_limit_seconds=900,source_seed=42,settings=old.SETTINGS[:3],conditions=CONDITIONS,sources=['core_real','initial_core'],alpha=100.,
        paired_R2_min=.1,delta_vs_TD_min=.05,delta_vs_shuffle_min=.05,required_pools=2,new_maps=45,vector_OOF_rows=6897,vector_width=192,
        unique_fit_rows=2299,new_efficiency_heads=0,new_source_forward=0,new_PCA=0,numeric_labels_read=False))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    assert io.versions()==lock['versions'];return lock


def run(start):
    validate();root=R/'maps';assert not root.exists() and not (R/'diagnostic.json').exists()
    threads=threadpool_info();assert all(t['num_threads']==1 for t in threads)
    rows,xx,identities=old.inputs();identity={n:identities[n] for n in ('core_real','initial_core')}
    assert identity['core_real']['initial_sha256']==identity['initial_core']['checkpoint_sha256']
    assert identity['core_real']['variant']==identity['initial_core']['variant']=='core_real'
    assert identity['core_real']['source_receipt_sha256']==identity['initial_core']['source_receipt_sha256']
    lookup={r['sample_id']:i for i,r in enumerate(rows)};plan=io.read(P/'plan.json');records=[];details=[];pools={};count=0
    for setting in old.SETTINGS[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids'];pools[setting]=ids
        assert set(ids).isdisjoint(item['test_ids']);fi=np.array([lookup[s] for s in ids]);parts={k:[] for k in CONDITIONS};blocks=[]
        for j,split in enumerate(item['inner']):
            if time.monotonic()-start>900:raise TimeoutError('CPU stage cap')
            tr,va=np.array(split['train']),np.array(split['valid']);z={};stpaths={}
            for source in ('core_real','initial_core'):
                path=D/'pilot'/setting/source/'dense'/f'inner{j}'/'transform.json';t=io.read(path);stpaths[source]=str(path.resolve())
                z[source]=(xx[source][fi,24:]-t['src_mean'])/t['src_scale']/np.sqrt(12.)
                if source=='core_real':td=(xx[source][fi,:24]-t['td_mean'])/t['td_scale']
            f=z['core_real'];b=z['initial_core'];pi=permutation([ids[i] for i in tr],setting,j);bparts={}
            for condition in CONDITIONS:
                x=td if condition=='TD24' else b;train=x[tr][pi] if condition=='shuffled_initial' else x[tr]
                before=time.monotonic();a,g,cert=fit_map(train,f[tr]);pred=x[va]@a;st=statistics(f[va],pred)
                out=root/setting/condition/f'inner{j}'
                old.npz_write(out/'model.npz',coefficient=a,training_prediction_gram=g,prediction=pred,training_permutation=pi if condition=='shuffled_initial' else np.arange(len(tr)))
                io.write(out/'state.json',dict(setting=setting,condition=condition,inner=j,train_indices=tr,valid_indices=va,statistics=st,certificate=cert,
                    transform_sha256={k:io.sha(Path(v)) for k,v in stpaths.items()},shuffled_fixed_points=int(sum(pi==np.arange(len(tr)))) if condition=='shuffled_initial' else None,
                    CPU_map_wall_seconds=time.monotonic()-before))
                io.write(out/'complete.json',dict(lock_sha256=io.sha(R/'lock.json'),files_sha256={p.name:io.sha(p) for p in out.iterdir() if p.is_file()}))
                parts[condition].append(st);bparts[condition]=st;count+=1
                details.extend(dict(setting=setting,condition=condition,inner=j,sample_id=ids[i],SSE=float(np.sum((f[i]-pred[k])**2)),null_SSE=float(np.sum(f[i]**2))) for k,i in enumerate(va))
            blocks.append(dict(inner=j,n=len(va),conditions=bparts))
        result={k:aggregate(parts[k]) for k in CONDITIONS};records.append(dict(setting=setting,n=len(ids),conditions=result,inner_blocks=blocks))
        print(setting,{k:result[k]['R2'] for k in CONDITIONS},flush=True)
    assert count==45 and len(details)==6897 and sum(len(v) for v in pools.values())==2299
    io.write_csv(R/'diagnostic_rows.csv',details);gates=qualify(records)
    io.write(R/'diagnostic.json',dict(**gates,records=records,fit_ids=pools,source_identities=identity,new_maps=count,vector_OOF_rows=len(details),vector_width=192,
        unique_fit_rows=2299,new_efficiency_heads=0,new_source_forward=0,new_PCA=0,GPU_hours=0,numeric_labels_read=False,outer_test_metrics_used=False,threadpool=threads,
        lock_sha256=io.sha(R/'lock.json'),diagnostic_rows_sha256=io.sha(R/'diagnostic_rows.csv'),map_files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
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
            io.write(R/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0,new_efficiency_heads=0));mutex.unlink()
