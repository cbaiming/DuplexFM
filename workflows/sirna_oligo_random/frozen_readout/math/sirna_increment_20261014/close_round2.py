"""Read-only numeric-repair audit and derived documentation, no fitting."""
from pathlib import Path
import csv, hashlib, json
ROOT=Path(__file__).resolve().parent
R=ROOT/'round2'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
lock=read(R/'numeric_execution_lock.json')
assert sha(R/'lock.json')==lock['original_lock_sha256']
for section in ('files_sha256','prior_failure','reused_folds'):
    for name,digest in lock[section].items():assert sha(R/name)==digest,name
assert len(lock['reused_folds'])==15
receipt=read(next(R.glob('numeric_compat_execution_*.json')))
assert receipt['numeric_execution_lock_sha256']==sha(R/'numeric_execution_lock.json')
assert receipt['complete_sha256']==sha(R/'results/complete.json') and receipt['returncode']==0
audit=read(R/'verification.json');assert audit['passed']
assert audit['complete_sha256']==sha(R/'results/complete.json')
costs={p.name:read(p) for p in R.glob('execution_*.json')}
assert sorted(r['returncode'] for r in costs.values())==[0,1]
wall=sum(r['CPU_wall_seconds'] for r in costs.values())
usage=list(csv.DictReader((R/'results/usage.csv').open(encoding='utf-8')))
routes={}
for row in usage:
    key=row['condition']+'/'+row['setting']
    entry=routes.setdefault(key,{'total':0,'TD_only':0})
    entry['total']+=int(row['count'])
    if row['mask']=='0':entry['TD_only']+=int(row['count'])
save(R/'numeric_compat_readback.json',dict(passed=True,checks='original lock, repair files, failed receipt, 15 reused fold receipts, completed results and independent numeric audit',
    original_lock_sha256=sha(R/'lock.json'),numeric_lock_sha256=sha(R/'numeric_execution_lock.json'),
    complete_sha256=sha(R/'results/complete.json'),verification_sha256=sha(R/'verification.json'),
    execution_receipts={p.name:sha(p) for p in R.glob('execution_*.json')},
    CPU_pipeline_wall_seconds_including_failure=wall,independent_verification_wall_seconds=audit['wall_seconds'],
    overlapping_wrapper_cost_not_added=True,GPU_hours=0,source_updates=0,routes=routes))
td=[.609,.479,.599,.556,.284,.367]
duplex=[.626,.556,.637,.562,.362,.464]
gain=[100*(b/a-1) for a,b in zip(td,duplex)]
save(ROOT/'paper_gain_arithmetic.json',dict(source='manuscript_revision_20260923/Table4_RNAiSpline_OligoGraph_completed.md; PDF-rounded non-ensemble values, not fresh prediction reproduction',
    table_sha256=sha(ROOT.parent/'manuscript_revision_20260923/Table4_RNAiSpline_OligoGraph_completed.md'),
    settings=['Huesken','Mixset','Takayuki','Inter_Hu_to_Mix','Wang','ALAS1'],TD24=td,DuplexFM=duplex,
    relative_PCC_gain_pct=gain,four_setting_mean=sum(gain[:4])/4,six_setting_mean=sum(gain)/6))
print(json.dumps(dict(passed=True,CPU_wall_seconds=wall,routes=routes,paper_four_mean=sum(gain[:4])/4,paper_six_mean=sum(gain)/6),ensure_ascii=False))
