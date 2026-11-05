"""Locked input-only radial geometry audit; fixed outer fold0 fit pools."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,importlib.util,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent; P=R.parent/'round4'; sys.path.insert(0,str(R))
from geometry import source_norm,statistics,qualification
spec=importlib.util.spec_from_file_location('_r15_old_io',P/'run.py'); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
io=old.io
FILES=('diagnose.py','geometry.py','test_geometry.py','verify_geometry.py','DIAGNOSTIC_PROTOCOL.md','CODE_AUDIT.md','pytest.xml')


def register():
    assert not (R/'geometry_lock.json').exists(); old.validate()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==6
    assert not any(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites)
    previous=io.read(P/'lock.json'); deps=dict(previous['dependencies'])
    for n in previous['files']: deps[str((P/n).resolve())]=io.sha(P/n)
    for path in (P/'lock.json',P/'verification.json',P/'results/complete.json',R.parent/'round14/closure_receipt.json'):
        deps[str(path.resolve())]=io.sha(path)
    for setting in old.SETTINGS[:3]:
        for source in old.SOURCES:
            path=P/'results/folds'/setting/'fold0'/source/'transforms.json'; deps[str(path.resolve())]=io.sha(path)
    io.write(R/'geometry_lock.json',dict(files={n:io.sha(R/n) for n in FILES},dependencies=deps,versions=io.versions(),
        CPU_only=True,CPU_limit_seconds=300,labels_numerically_used=False,outer_test_metrics_used=False,
        new_source_forward=0,GPU_hours=0,qualification='Core at least 2 of 3 CV>=.10 and r2>=.20; all Core norms>1e-12'))
    print('REGISTERED',io.sha(R/'geometry_lock.json'),flush=True)


def validate():
    lock=io.read(R/'geometry_lock.json'); io.check_files(R,lock['files'])
    for path,h in lock['dependencies'].items(): assert io.sha(Path(path))==h,path
    assert lock['versions']==io.versions(); return lock


def run(start):
    lock=validate(); assert not (R/'geometry_result.json').exists()
    rows,xx,identities=old.inputs(); index={r['sample_id']:i for i,r in enumerate(rows)}
    plan=io.read(P/'plan.json'); records=[]; pools={}
    for setting in old.SETTINGS[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids']; pools[setting]=ids; ii=np.array([index[s] for s in ids])
        assert set(ids).isdisjoint(item['test_ids'])
        for source in old.SOURCES:
            if time.monotonic()-start>lock['CPU_limit_seconds']: raise TimeoutError('No new geometry block')
            x=xx[source][ii]; t=io.read(P/'results/folds'/setting/'fold0'/source/'transforms.json')['final']
            pc1=old.transform(x,t)[:,24]; record=dict(setting=setting,source=source,n=len(ii),**statistics(source_norm(x),pc1))
            records.append(record)
            print(setting,source,'CV',record['norm_CV'],'r2',record['norm_PC1_r2'],'qualified',record['pool_qualified'],flush=True)
    assert len(records)==9
    io.write(R/'geometry_result.json',dict(qualified=qualification(records),records=records,fit_ids=pools,
        identities=identities,lock_sha256=io.sha(R/'geometry_lock.json'),labels_numerically_used=False,
        new_efficiency_fits=0,new_PCA_fits=0,new_source_forward=0,outer_test_metrics_used=False,GPU_hours=0))
    validate(); print('QUALIFIED',qualification(records),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=('register','run')); mode=ap.parse_args().mode
    if mode=='register': register()
    else:
        mutex=R/'geometry.mutex'
        with mutex.open('x') as f: f.write(str(os.getpid()))
        start=time.monotonic(); rc=1; error=None
        try: run(start); rc=0
        except Exception: error=traceback.format_exc(); raise
        finally:
            io.write(R/f'geometry_execution_{time.time_ns()}.json',dict(returncode=rc,error=error,
                CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0,new_efficiency_fits=0))
            mutex.unlink()
