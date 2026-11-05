"""Bounded label-free input audit, without supervised fits or source forwards."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4'
sys.path.insert(0,str(R))
from tails import inspect,qualify
spec=importlib.util.spec_from_file_location('_r32_old4',P/'run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
io=old.io
FILES=('run.py','tails.py','verify.py','test_tails.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')


def register():
    assert not (R/'lock.json').exists();old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(x.get('tests',0)) for x in suites)==7
    assert not any(int(x.get(k,0)) for x in suites for k in ('errors','failures','skipped'))
    c=io.read(R.parent/'round31/closure_receipt.json');assert c['status']=='closed' and c['readback_passed']
    for name,h in c['files_sha256'].items():assert io.sha(R.parent/name)==h,name
    prior=io.read(P/'lock.json');deps=dict(prior['dependencies'])
    for name in prior['files']:deps[str((P/name).resolve())]=io.sha(P/name)
    for p in (P/'lock.json',P/'plan.json',P/'verification.json',Path(io.__file__),R.parent/'round31/closure_receipt.json'):
        deps[str(p.resolve())]=io.sha(p)
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=300,source_seed=42,settings=old.SETTINGS[:3],sources=old.SOURCES,
        top_fraction=.01,share_threshold=.2,max_z_threshold=6.,minimum_columns=16,minimum_fraction=.1,required_core_pools=2,
        fit_pool_rows=2299,source_blocks=9,TD_blocks=3,source_rows=6897,new_efficiency_heads=0,new_source_forward=0,new_PCA=0))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)


def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files'])
    for path,h in lock['dependencies'].items():assert io.sha(Path(path))==h,path
    assert io.versions()==lock['versions'];return lock


def run(start):
    validate();assert not (R/'diagnostic.json').exists()
    threads=threadpool_info();assert all(t['num_threads']==1 for t in threads)
    rows,xx,identity=old.inputs();lookup={r['sample_id']:i for i,r in enumerate(rows)}
    plan=io.read(P/'plan.json');records=[];controls=[];pools={}
    for setting in old.SETTINGS[:3]:
        item=next(x for x in plan['folds'] if x['setting']==setting and x['fold']==0)
        ids=item['fit_ids'];assert set(ids).isdisjoint(item['test_ids']);pools[setting]=ids
        fi=np.array([lookup[s] for s in ids]);td=xx['core_real'][fi,:24]
        controls.append(dict(setting=setting,source='TD24',statistics=inspect(td)))
        for source in old.SOURCES:
            if time.monotonic()-start>300:raise TimeoutError('Input audit stage cap')
            result=inspect(xx[source][fi,24:]);records.append(dict(setting=setting,source=source,statistics=result))
            print(setting,source,'flagged',result['flagged_columns'],'eligible',result['eligible_columns'],
                  'required',result['required_columns'],'qualified',result['pool_qualified'],flush=True)
    assert sum(len(x) for x in pools.values())==2299 and len(records)==9 and len(controls)==3
    validate()
    io.write(R/'diagnostic.json',dict(qualified=qualify(records),records=records,TD_controls=controls,fit_ids=pools,source_identities=identity,
        source_blocks=9,TD_blocks=3,source_row_occurrences=6897,unique_input_rows=2299,
        new_efficiency_heads=0,new_source_forward=0,new_PCA=0,GPU_hours=0,numeric_labels_read=False,outer_test_metrics_used=False,
        threadpool=threads,lock_sha256=io.sha(R/'lock.json')))
    print('QUALIFIED',qualify(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('register','run'))
    if p.parse_args().mode=='register':register()
    else:
        mutex=R/'execution.mutex'
        with mutex.open('x') as f:f.write(str(os.getpid()))
        started=time.monotonic();rc=1;error=None
        try:run(started);rc=0
        except Exception:error=traceback.format_exc();raise
        finally:
            io.write(R/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-started,
                GPU_hours=0,new_efficiency_heads=0,new_source_forward=0))
            mutex.unlink()
