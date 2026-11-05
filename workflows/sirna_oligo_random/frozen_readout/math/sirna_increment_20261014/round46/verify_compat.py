"""Explicit one-dimensional legacy intercept compatibility, no scientific changes."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent
OLD="tdintercept=a['inner_intercept'][:,3].copy()"
NEW="tdintercept=a['inner_intercept'].copy();assert tdintercept.shape==(5,)"
FILES=('verify_compat.py','test_verify_compat.py','VERIFY_COMPAT.md','pytest_compat.xml')


def patched(text):
    if text.count(OLD)!=1:raise ValueError('Exact single legacy-read site required')
    return text.replace(OLD,NEW)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def main(mode):
    original=R/'verify.py';science=read(R/'lock.json')
    assert sha(original)==science['files']['verify.py']
    effective=patched(original.read_text(encoding='utf-8'));effective_sha=hashlib.sha256(effective.encode()).hexdigest()
    if mode=='register':
        assert not (R/'verification.json').exists() and not (R/'verification_execution_lock.json').exists()
        failures=list(R.glob('verification_execution_*.json'));assert len(failures)==1
        failed=read(failures[0]);assert failed['returncode']==1 and 'IndexError: too many indices' in failed['error']
        suites=list(ET.parse(R/'pytest_compat.xml').getroot().iter('testsuite'))
        assert sum(int(s.get('tests',0)) for s in suites)==3
        assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
        value=dict(files_sha256={p:sha(R/p) for p in FILES},original_science_lock_sha256=sha(R/'lock.json'),
            original_verifier_sha256=sha(original),effective_verifier_sha256=effective_sha,
            diagnostic_sha256=sha(R/'diagnostic.json'),failed_execution_sha256=sha(failures[0]),
            original_read=OLD,compatible_read=NEW,only_intercept_storage_shape_changed=True,scientific_changes=False)
        with (R/'verification_execution_lock.json').open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
        print('REGISTERED_COMPAT',sha(R/'verification_execution_lock.json'),flush=True)
    else:
        assert not (R/'verification.json').exists() and not (R/'execution.mutex').exists()
        lock=read(R/'verification_execution_lock.json')
        for p,h in lock['files_sha256'].items():assert sha(R/p)==h,p
        assert sha(R/'lock.json')==lock['original_science_lock_sha256'] and sha(original)==lock['original_verifier_sha256']
        assert effective_sha==lock['effective_verifier_sha256'] and sha(R/'diagnostic.json')==lock['diagnostic_sha256']
        exec(compile(effective,str(original),'exec'),{'__name__':'__main__','__file__':str(original)})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('register','run'));main(parser.parse_args().mode)
