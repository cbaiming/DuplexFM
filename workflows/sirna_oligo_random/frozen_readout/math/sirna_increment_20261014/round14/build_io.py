"""Derive a one-source controller from the already audited R4 implementation."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;P=R.parent/'round4'
def replace(s,a,b):
    assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
original=(P/'run.py').read_text(encoding='utf-8')
prefix='''"""Fixed same-readout test of one pre-existing Core+Access source."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;PRIOR=HERE.parent/'round4';sys.path.insert(0,str(HERE))
from assets import inputs as asset_inputs,old
io=old.io;OLD=old.OLD;SETTINGS=old.SETTINGS
configs,fit_transform,transform,fit_heads,score,choose=(old.configs,old.fit_transform,old.transform,old.fit_heads,old.score,old.choose)
npz_write=old.npz_write
SOURCES=('core_access_real',);MODES=('Uniform40','Adaptive40')
CODE=('run.py','verify.py','assets.py','test_assets.py','build_io.py','build_io_v1_failed.py','BUILD_REPAIR.md','ASSET_PROTOCOL.md','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')

def inputs():
    rows,xx,identity,_=asset_inputs();return rows,xx,{'core_access_real':identity}

def register():
    assert not (HERE/'lock.json').exists()
    tests=list(ET.parse(HERE/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(t.get('tests',0)) for t in tests)==4
    assert not any(int(t.get('errors',0))+int(t.get('failures',0))+int(t.get('skipped',0)) for t in tests)
    asset=io.read(HERE/'asset_verification.json');assert asset['passed'] and asset['rows']==3329
    assert asset['code_sha256']==io.sha(HERE/'assets.py') and asset['protocol_sha256']==io.sha(HERE/'ASSET_PROTOCOL.md')
    for p,h in asset['files_sha256'].items():assert io.sha(Path(p))==h,p
    rows,xx,identity,newdeps=asset_inputs();assert identity==asset['source_identity']
    old.validate();ref=io.read(PRIOR/'lock.json');deps=dict(ref['dependencies']);deps.update(newdeps)
    assert io.read(PRIOR/'verification.json')['passed']
    assert io.read(PRIOR/'verification.json')['complete_sha256']==io.sha(PRIOR/'results/complete.json')
    for n in ref['files']:deps[str((PRIOR/n).resolve())]=io.sha(PRIOR/n)
    for p in (PRIOR/'lock.json',PRIOR/'verification.json',PRIOR/'results/complete.json',HERE/'asset_verification.json',HERE.parent/'round13/closure_receipt.json'):
        deps[str(p.resolve())]=io.sha(p)
    for n,h in io.read(PRIOR/'results/complete.json')['files_sha256'].items():
        p=PRIOR/'results'/n;assert io.sha(p)==h;deps[str(p.resolve())]=h
    plan=io.read(PRIOR/'plan.json')
    io.write(HERE/'plan.json',dict(folds=plan['folds'],configs=configs(),sources=SOURCES,modes=MODES,
        source_identity={'core_access_real':identity},primary_condition='core_access_real_Adaptive40',
        excluded=['Wang','ALAS1'],CPU_phase_limit_seconds=1800,reference_R4_complete_sha256=io.sha(PRIOR/'results/complete.json')))
    io.write(HERE/'lock.json',dict(files={n:io.sha(HERE/n) for n in CODE+('plan.json',)},
        dependencies=deps,versions=io.versions(),CPU_only=True,GPU_hours=0,source_updates=0,new_source_forward=0,
        mean_relative_PCC_goal_pct=5,CPU_phase_limit_seconds=1800))
    print('REGISTERED',io.sha(HERE/'lock.json'),flush=True)

def validate():
    lock=io.read(HERE/'lock.json');io.check_files(HERE,lock['files'])
    for p,h in lock['dependencies'].items():assert io.sha(Path(p))==h,p
    assert lock['versions']==io.versions()

'''
run=prefix+original[original.index('def run():'):]
run=replace(run,'def run():\n    validate();',"def run():\n    phase_start=time.monotonic()\n    validate();")
run=replace(run,"        for source in SOURCES:\n            x=xx[source];out=", "        for source in SOURCES:\n            if time.monotonic()-phase_start>1800:raise TimeoutError('CPU phase cap; no new block')\n            x=xx[source];out=")
run=replace(run,'assert len(predictions)==22788 and len(selections)==96','assert len(predictions)==7596 and len(selections)==32')
run=replace(run,'outer_folds=16,sources=3,inner_projection_blocks=240,','outer_folds=16,sources=1,inner_projection_blocks=80,final_projection_blocks=16,')
run=replace(run,'inner_fits=6000,final_heads=96,rows=22788','inner_fits=2000,final_heads=32,rows=7596')
run=replace(run,'ROUND4_COMPLETE','ROUND14_COMPLETE')
verify=(P/'verify.py').read_text(encoding='utf-8')
verify=replace(verify,"EXPORT=B/'core_six_settings_20261011/results/exports'","EXPORT=B/'core_access_transfer_20261012/results/extracted'")
verify=replace(verify,"SOURCES=('core_real','full_real','initial_core')","SOURCES=('core_access_real',)")
verify=replace(verify,"table(EXPORT/name/'row_identity.csv')","table(EXPORT/'row_identity.csv')")
verify=replace(verify,"np.load(EXPORT/name/'source.npy'","np.load(EXPORT/'source.npy'")
verify=replace(verify,'assert len(pred)==22788 and len(summary)==24','assert len(pred)==7596 and len(summary)==8')
verify=replace(verify,'assert count==6000 and final==96','assert count==2000 and final==32')
verify=replace(verify,"r['condition']=='core_real_Adaptive40'","r['condition']=='core_access_real_Adaptive40'")
verify=replace(verify,'OOF_rows=22788,source_models=3','OOF_rows=7596,source_models=1')
needle="    y=np.array([float(r['label']) for r in rows]);binary=np.array([int(r['y']) for r in rows])"
verify=replace(verify,needle,needle+"\n    asset=read(R/'asset_verification.json');assert asset['passed'] and asset['rows']==3329\n    assert plan['sources']==list(SOURCES) and plan['primary_condition']=='core_access_real_Adaptive40'\n    assert plan['source_identity']['core_access_real']==asset['source_identity']\n    assert plan['reference_R4_complete_sha256']==sha(R.parent/'round4/results/complete.json')\n    for p,h in asset['files_sha256'].items():assert sha(Path(p))==h,p\n    selections=table(R/'results/selection.csv');assert len(selections)==32")
needle="                canonical.setdefault((s,condition),[]).extend(zip(ti,canonical_p))"
verify=replace(verify,needle,needle+"\n                sr=[r for r in selections if r['setting']==s and int(r['fold'])==fold and r['condition']==condition];assert len(sr)==1\n                assert int(sr[0]['candidate'])==chosen[mode]\n                for name in ('alpha','weight'):close(float(sr[0][name]),cfg[chosen[mode]][name],0)")
verify=replace(verify,"    result=dict(passed=True,", "    assert complete['inner_fits']==2000 and complete['final_heads']==32 and complete['rows']==7596\n    assert complete['inner_projection_blocks']==80 and complete['final_projection_blocks']==16\n    result=dict(passed=True,")
verify=replace(verify,"if __name__=='__main__':main()",'''if __name__=='__main__':
    import traceback
    started=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-started,GPU_hours=0),f,indent=2)
''')
for name,code in [('run.py',run),('verify.py',verify)]:
    compile(code,str(R/name),'exec')
    with (R/name).open('x',encoding='utf-8',newline='\n') as f:f.write(code)
receipt=dict(source_files_sha256={str((P/n).resolve()):hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ('run.py','verify.py')},
    generated_files_sha256={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ('run.py','verify.py')})
with (R/'build_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
