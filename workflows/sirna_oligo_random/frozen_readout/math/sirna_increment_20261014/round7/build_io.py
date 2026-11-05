"""Generate an R7 controller from the audited R5 lifecycle, before registration."""
from pathlib import Path
R=Path(__file__).resolve().parent
assert not (R/'lock.json').exists()

def replace(text,old,new,n=None):
    assert old in text,old
    if n is not None:assert text.count(old)==n,(old,text.count(old),n)
    return text.replace(old,new)

s=(R.parent/'round5/run.py').read_text(encoding='utf-8')
s=replace(s,'from conditional import','from kernel import')
s=replace(s,'TDQuadratic','TDKernel')
s=replace(s,"CODE=('run.py','conditional.py','verify.py','test_conditional.py','PROTOCOL.md','CODE_AUDIT.md')",
    "CODE=('run.py','kernel.py','verify.py','audit_kernel.py','test_kernel.py','test_execution.py','build_io.py','calibrate_cpu.py','PROTOCOL.md','CODE_AUDIT.md','CPU_QUALIFICATION.json','execution_pytest.xml')")
s=replace(s,"return view+'_'+mode+('Quadratic' if view=='TDKernel' else 'Bilinear')","return view+'_'+mode+'Kernel'")
s=replace(s,"return 'td_quadratic' if view=='TDKernel' else 'bilinear'","return 'td' if view=='TDKernel' else 'joint'")
s=replace(s,'core_real_AdaptiveBilinear','core_real_AdaptiveKernel')
s=replace(s,'ROUND5_COMPLETE','ROUND7_COMPLETE')
anchor="    previous.validate();v=io.read(PRIOR/'verification.json')"
s=replace(s,anchor,"""    extra=list(ET.parse(HERE/'execution_pytest.xml').getroot().iter('testsuite'))
    assert sum(int(x.get('tests',0)) for x in extra)==3
    assert not any(int(x.get('errors',0))+int(x.get('failures',0))+int(x.get('skipped',0)) for x in extra)
    qualification=io.read(HERE/'CPU_QUALIFICATION.json')
    assert qualification['synthetic_only'] and qualification['GPU_hours']==0 and qualification['landmarks']==64
    assert qualification['kernel_code_sha256']==io.sha(HERE/'kernel.py')
    previous.validate();v=io.read(PRIOR/'verification.json')""")
s=replace(s,"HERE.parent/'r4_training_diagnostic.json',HERE.parent/'r4_training_diagnostic.py'",
    "HERE.parent/'round6/lock.json',HERE.parent/'round6/verification.json',HERE.parent/'round6/results/complete.json'")
with (R/'run.py').open('x',encoding='utf-8',newline='\n') as f:f.write(s)

s=(R.parent/'round5/verify.py').read_text(encoding='utf-8')
s=replace(s,'Independent R5 audit','Independent R7 audit')
s=replace(s,'import importlib.util,json,time','import importlib.util,json,time,traceback,sys')
s=replace(s,'TDQuadratic','TDKernel')
s=replace(s,"return view+'_'+mode+('Quadratic' if view=='TDKernel' else 'Bilinear')","return view+'_'+mode+'Kernel'")
a=s.index('def expand(');b=s.index('\ndef equations(',a)
s=s[:a]+"sys.path.insert(0,str(R))\nfrom audit_kernel import expand\n\n"+s[b:]
s=replace(s,'core_real_AdaptiveBilinear','core_real_AdaptiveKernel')
s=replace(s,"width=324 if view=='TDKernel' else 424","width=88")
s=replace(s,"                canonical.setdefault((setting,cond),[]).extend(zip(ti,pp))",
    "                canonical.setdefault((setting,cond),[]).extend(zip(ti,[float(r['prediction']) for r in part]))")
s=replace(s,'all_PCA_interaction_selection_predictions_metrics_checked','all_PCA_kernel_selection_predictions_metrics_checked')
anchor="            oof=np.full((len(fi),25),np.nan);coverage=[]"
s=replace(s,anchor,"""            if view!='TDKernel':
                prior=read(P/'results/folds'/setting/f'fold{fold}'/view/'transforms.json')
                for nt,ot in zip(t['inner']+[t['final']],prior['inner']+[prior['final']]):
                    for key in ot:close(nt['projection'][key],ot[key],1e-12)
            oof=np.full((len(fi),25),np.nan);coverage=[]""")
s=replace(s,"    assert count==8000 and final==128","""    assert count==8000 and final==128
    assert complete['inner_fits']==8000 and complete['final_heads']==128 and complete['rows']==30384
    assert complete['GPU_hours']==0 and complete['source_updates']==0 and complete['new_source_forward']==0""")
s=replace(s,"if __name__=='__main__':main()","""if __name__=='__main__':
    begin=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-begin,GPU_hours=0),f,indent=2)
""")
with (R/'verify.py').open('x',encoding='utf-8',newline='\n') as f:f.write(s)
print('Created R7 lifecycle and independent verifier; no registration or data run')
