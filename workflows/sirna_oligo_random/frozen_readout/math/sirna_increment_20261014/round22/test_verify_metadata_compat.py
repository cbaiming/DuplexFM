from pathlib import Path
from verify_metadata_compat import rewrite,old
def test_only_final_metadata_symbol_changes():
    p=Path(__file__).resolve().parent/'verify.py';src=p.read_text();base=old.rewrite_verify(src);fixed=rewrite(src)
    assert fixed.replace('support_records=supports,','support_records=support,')==base
    compile(fixed,'<metadata-only>','exec')
def test_output_has_expected_support_value():
    src="result=dict(support_records=support,other=123)\npath=R/'results/complete.json'\nxs=R.glob('execution_*.json')"
    ns={'supports':[1,2,3],'R':Path('.')};exec(rewrite(src),ns)
    assert ns['result']=={'support_records':[1,2,3],'other':123}
