"""Explicit, non-overwriting R8-to-R12 controller derivation before locking."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;OLD=R.parent/'round8'
def replace(s,a,b):
    assert s.count(a)==1,(a,s.count(a))
    return s.replace(a,b)
run=(OLD/'run.py').read_text(encoding='utf-8')
run=replace(run,'from orthogonal import','from spectral import')
run=replace(run,"'orthogonal.py','verify.py','audit_projection.py','test_orthogonal.py'","'spectral.py','verify.py','audit_spectral.py','test_spectral.py'")
run=run.replace('Orthogonal','Spectral')
run=replace(run,"for s in suites)==5","for s in suites)==6")
needle="    previous.validate();v=io.read(PRIOR/'verification.json')"
run=replace(run,needle,"    dv=io.read(HERE/'diagnostic_verification.json')\n    assert dv['passed'] and dv['qualified'] and dv['independent_half_SVDs']==18\n    assert dv['diagnostic_sha256']==io.sha(HERE/'diagnostic.json')\n    assert dv['lock_sha256']==io.sha(HERE/'diagnostic_lock.json')\n"+needle)
run=replace(run,"HERE.parent/'round7/verification.json'","HERE/'diagnostic_verification.json',HERE.parent/'round11/verification.json'")
run=replace(run,'z,t=fit_transform(x[fi[tr]])',"z,t=fit_transform(x[fi[tr]],ref['inner'][j])")
run=replace(run,'z,t=fit_transform(x[fi])',"z,t=fit_transform(x[fi],ref['final'])")
run=replace(run,'inner_projection_blocks=240,','inner_scale_blocks=240,final_scale_blocks=48,new_PCA_fits=0,')
run=replace(run,'ROUND8_COMPLETE','ROUND12_COMPLETE')
verify=(OLD/'verify.py').read_text(encoding='utf-8')
verify=replace(verify,'Independent R8 audit','Independent R12 audit')
verify=verify.replace('Orthogonal','Spectral')
verify=replace(verify,'from audit_projection import','from audit_spectral import')
verify=replace(verify,"                for k in range(25):close(a['inner_coef'][j,k,:24],a['inner_coef'][j,k%5,:24],1e-10)\n",'')
verify=replace(verify,"source_coef=coef[:,24:]/np.asarray(pstate['pc_scale'])","source_coef=coef[:,24:]/t['final']['common_pc_scale']")
verify=replace(verify,"td_coef=(coef[:,:24]-coef[:,24:]@np.asarray(t['final']['td_to_source']).T)/np.asarray(pstate['td_scale'])","td_coef=coef[:,:24]/np.asarray(pstate['td_scale'])")
verify=replace(verify,"offset=float(a['final_intercept'])-coef[:,24:]@np.asarray(t['final']['residual_mean'])","offset=np.full(2,float(a['final_intercept']))")
start=verify.index('    diagnostics_checked=0\n')
end=verify.index("    assert complete['inner_fits']==6000",start)
verify=verify[:start]+"""    dv=read(R/'diagnostic_verification.json')
    assert dv['passed'] and dv['qualified'] and dv['independent_half_SVDs']==18
    assert not dv['real_labels_used'] and dv['new_efficiency_fits']==0
    assert dv['diagnostic_sha256']==sha(R/'diagnostic.json') and dv['lock_sha256']==sha(R/'diagnostic_lock.json')
    assert dv['code_sha256']==sha(R/'audit_spectral.py')
    diagnostics_checked=9
    assert complete['new_PCA_fits']==0 and complete['inner_scale_blocks']==240 and complete['final_scale_blocks']==48
"""+verify[end:]
verify=replace(verify,'all_PCA_orthogonalization_deployment_selection_metrics_checked','all_PCA_common_scale_deployment_selection_metrics_checked')
for name,code in [('run.py',run),('verify.py',verify)]:
    compile(code,str(R/name),'exec')
    with (R/name).open('x',encoding='utf-8',newline='\n') as f:f.write(code)
receipt=dict(source_files_sha256={str((OLD/n).resolve()):hashlib.sha256((OLD/n).read_bytes()).hexdigest() for n in ('run.py','verify.py')},
    generated_files_sha256={n:hashlib.sha256((R/n).read_bytes()).hexdigest() for n in ('run.py','verify.py')},
    source_results_untouched=True,CPU_only=True)
with (R/'build_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
