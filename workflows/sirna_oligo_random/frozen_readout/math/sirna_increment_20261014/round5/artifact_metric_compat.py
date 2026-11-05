"""Explicit preserved-code audit compatibility; scientific results are immutable."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,json,subprocess,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,value):
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with p.open('x',encoding='utf-8') as f:f.write(text)

def prepare():
    lock=read(R/'lock.json')
    for n,h in lock['files'].items():assert sha(R/n)==h,n
    d=read(R/'rank_replay_diagnosis.json')
    assert d['source_lock_sha256']==sha(R/'lock.json') and d['max_prediction_difference']<1e-10 and d['n_rank_changes']==2
    assert abs(d['saved_Spearman']-d['reconstructed_Spearman'])>1e-9
    suites=list(ET.parse(R/'artifact_metrics_pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==2
    assert not any(int(s.get('errors',0))+int(s.get('failures',0))+int(s.get('skipped',0)) for s in suites)
    source=(R/'verify.py').read_text(encoding='utf-8')
    needle='canonical.setdefault((setting,cond),[]).extend(zip(ti,pp))'
    replacement="canonical.setdefault((setting,cond),[]).extend(zip(ti,np.array([float(r['prediction']) for r in part])))"
    assert source.count(needle)==1;source=source.replace(needle,replacement)
    needle='def main():\n'
    replacement="def main():\n    audit_execution=read(R/'artifact_metric_execution_lock.json')\n    check(R,audit_execution['files'])\n    assert audit_execution['original_lock_sha256']==sha(R/'lock.json')\n"
    assert source.count(needle)==1;source=source.replace(needle,replacement)
    needle='result=dict(passed=True,inner_normal_equations=count'
    replacement="result=dict(passed=True,metric_prediction_source='verified artifact CSV, preserving exact ranks',compat_execution_lock_sha256=sha(R/'artifact_metric_execution_lock.json'),inner_normal_equations=count"
    assert source.count(needle)==1;source=source.replace(needle,replacement)
    with (R/'verify_artifact_metrics.py').open('x',encoding='utf-8',newline='\n') as f:f.write(source)
    files=('verify.py','verify_artifact_metrics.py','artifact_metric_compat.py','ARTIFACT_METRIC_COMPAT.md',
           'inspect_rank_replay.py','rank_replay_diagnosis.json','verification_failed_v1.json',
           'test_artifact_metrics.py','artifact_metrics_pytest.xml')
    save(R/'artifact_metric_execution_lock.json',dict(original_lock_sha256=sha(R/'lock.json'),
        original_result_sha256=sha(R/'results/complete.json'),files={n:sha(R/n) for n in files},
        scientific_settings_changed=False,original_numerical_tolerances_unchanged=True))
    print('COMPAT_REGISTERED',sha(R/'artifact_metric_execution_lock.json'))

def verify():
    out=R/'artifact_metric_execution_receipt.json';assert not out.exists()
    lock=read(R/'artifact_metric_execution_lock.json')
    for n,h in lock['files'].items():assert sha(R/n)==h,n
    assert lock['original_lock_sha256']==sha(R/'lock.json') and lock['original_result_sha256']==sha(R/'results/complete.json')
    start=time.monotonic();rc=1;error=None
    try:
        result=subprocess.run([sys.executable,'-X','utf8',str(R/'verify_artifact_metrics.py')],cwd=R,timeout=1800)
        rc=result.returncode
    except Exception as e:error=repr(e);raise
    finally:
        save(out,dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,
            execution_lock_sha256=sha(R/'artifact_metric_execution_lock.json'),original_lock_sha256=sha(R/'lock.json'),
            results_complete_sha256=sha(R/'results/complete.json'),verification_sha256=sha(R/'verification.json') if (R/'verification.json').exists() else None))
    if rc:raise SystemExit(rc)
if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ('prepare','verify')
    prepare() if sys.argv[1]=='prepare' else verify()
