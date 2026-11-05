"""Explicit metadata-only repair and fresh output namespace; original locks intact."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,importlib.util,inspect,json,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def rewrite_run(source):
    old="certs.append(dict(inner=j,mode=mode,candidate=k,**cert))";new="certs.append(dict(inner=j,candidate=k,**cert))"
    assert source.count(old)==1 and source.count("root=R/'results'")==1
    return source.replace(old,new).replace("root=R/'results'","root=R/'results_v2'")
def rewrite_verify(source):
    assert source.count("R.glob('execution_*.json')")==1
    return source.replace("R/'results/","R/'results_v2/").replace("R/'results'","R/'results_v2'").replace("R.glob('execution_*.json')","R.glob('compat_execution_*.json')")
def validate():
    lock=read(R/'execution_lock.json')
    for n,h in lock['files_sha256'].items():assert sha(R/n)==h,n
    original=read(R/'lock.json')
    for n,h in original['files'].items():assert sha(R/n)==h,n
    return lock
def register():
    assert not (R/'execution_lock.json').exists() and not (R/'execution.mutex').exists()
    suites=list(ET.parse(R/'compat_pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==3 and not any(int(s.get('errors',0))+int(s.get('failures',0))+int(s.get('skipped',0)) for s in suites)
    failed=list(R.glob('execution_*.json'));assert len(failed)==1 and read(failed[0])['returncode']==1
    files=list((R/'results').rglob('*'));files=[p for p in files if p.is_file()]
    assert len(files)==1 and files[0].name=='inner0_dense_alpha0_certificate.json'
    files +=[R/n for n in ('execution_compat.py','test_execution_compat.py','EXECUTION_COMPAT.md','compat_pytest.xml','lock.json','run.py','verify.py','core.py')]+failed
    write(R/'execution_lock.json',dict(scientific_lock_sha256=sha(R/'lock.json'),preserve_partial=True,new_results='results_v2',
        files_sha256={p.relative_to(R).as_posix():sha(p) for p in files},scientific_change=False,discarded_failed_candidate_count=1))
    validate();print('COMPAT_REGISTERED',sha(R/'execution_lock.json'))
def execute(mode):
    validate();assert not (R/'execution.mutex').exists();start=time.monotonic();rc=1;error=None
    mutex=R/'execution.mutex' if mode=='run' else R/'verify_compat.mutex'
    with mutex.open('x') as f:f.write(str(os.getpid()))
    try:
        if mode=='run':
            spec=importlib.util.spec_from_file_location('_r22_original_runtime',R/'run.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
            exec(compile(rewrite_run(inspect.getsource(mod.run)),str(R/'run.py'),'exec'),mod.__dict__)
            mod.run(start)
        else:
            ns={'__name__':'_r22_verifier_compat','__file__':str(R/'verify.py')}
            exec(compile(rewrite_verify((R/'verify.py').read_text(encoding='utf-8')),str(R/'verify.py'),'exec'),ns)
            ns['main']()
        validate();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        prefix='compat_execution' if mode=='run' else 'compat_verification_execution'
        write(R/f'{prefix}_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,
            execution_lock_sha256=sha(R/'execution_lock.json'),scientific_lock_sha256=sha(R/'lock.json')))
        mutex.unlink()
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('register','run','verify'));mode=ap.parse_args().mode
    if mode=='register':register()
    else:execute(mode)
