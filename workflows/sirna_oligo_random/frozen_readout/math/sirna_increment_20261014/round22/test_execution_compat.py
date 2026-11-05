from pathlib import Path
import ast
from execution_compat import rewrite_run,rewrite_verify
R=Path(__file__).resolve().parent
def test_metadata_no_duplicate_and_same_values():
    s="def f():\n root=R/'results'\n certs.append(dict(inner=j,mode=mode,candidate=k,**cert))\n"
    fixed=rewrite_run(s);assert 'mode=mode' not in fixed and "root=R/'results_v2'" in fixed
    ns=dict(R=R,certs=[],j=0,k=1,cert={'mode':'dense','alpha':1.});exec(fixed,ns);ns['f']()
    assert ns['certs']==[dict(inner=0,candidate=1,mode='dense',alpha=1.)]
def test_real_function_edits_only_two_literals():
    src=(R/'run.py').read_text();node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    body=ast.get_source_segment(src,node);out=rewrite_run(body)
    assert out.replace("root=R/'results_v2'","root=R/'results'").replace('dict(inner=j,candidate=k,**cert)','dict(inner=j,mode=mode,candidate=k,**cert)')==body
    compile(out,'<synthetic>','exec')
def test_verifier_only_paths_and_completed_execution_change():
    src=(R/'verify.py').read_text();out=rewrite_verify(src)
    assert out.replace('results_v2','results').replace('compat_execution_','execution_')==src
    compile(out,'<synthetic>','exec')
