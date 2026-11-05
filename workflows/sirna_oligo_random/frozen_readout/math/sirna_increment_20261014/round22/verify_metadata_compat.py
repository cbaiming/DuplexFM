"""Fix only final verification-report variable spelling; no prediction changes."""
import importlib.util,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('_r22_original_execution',R/'execution_compat.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
def rewrite(source):
    out=old.rewrite_verify(source);assert out.count('support_records=support,')==1
    return out.replace('support_records=support,','support_records=supports,')
def register():
    old.validate();assert not (R/'verification_metadata_lock.json').exists() and not (R/'verification.json').exists()
    failed=list(R.glob('compat_verification_execution_*.json'));assert len(failed)==1 and old.read(failed[0])['returncode']==1
    assert "NameError: name 'support'" in old.read(failed[0])['error']
    suites=list(ET.parse(R/'metadata_pytest.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==2
    assert not any(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites)
    files=[R/n for n in ('verify_metadata_compat.py','test_verify_metadata_compat.py','VERIFICATION_METADATA_REPAIR.md','metadata_pytest.xml','execution_lock.json','results_v2/complete.json')]+failed
    old.write(R/'verification_metadata_lock.json',dict(scientific_change=False,results_unchanged=True,prior_execution_lock_sha256=old.sha(R/'execution_lock.json'),files_sha256={p.relative_to(R).as_posix():old.sha(p) for p in files}))
    print('METADATA_REGISTERED',old.sha(R/'verification_metadata_lock.json'))
def verify():
    lock=old.read(R/'verification_metadata_lock.json')
    for n,h in lock['files_sha256'].items():assert old.sha(R/n)==h,n
    initial=old.rewrite_verify;start=time.monotonic()
    def fixed(source):
        out=initial(source);assert out.count('support_records=support,')==1
        return out.replace('support_records=support,','support_records=supports,')
    try:old.rewrite_verify=fixed;old.execute('verify')
    finally:old.rewrite_verify=initial
    old.write(R/'verification_metadata_receipt.json',dict(passed=True,metadata_lock_sha256=old.sha(R/'verification_metadata_lock.json'),verification_sha256=old.sha(R/'verification.json'),complete_sha256=old.sha(R/'results_v2/complete.json'),CPU_wall_seconds=time.monotonic()-start))
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('register','verify'));mode=ap.parse_args().mode
    if mode=='register':register()
    else:verify()
