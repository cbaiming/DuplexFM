"""Derived locked-result report; no training or model selection."""
from pathlib import Path
import csv,hashlib,json
R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audit=read(R/'verification.json');assert audit['passed'] and audit['complete_sha256']==sha(R/'results/complete.json')
with (R/'results/summary.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
paired=[]
for s in ('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix'):
    a=next(r for r in rows if r['setting']==s and r['condition']=='P12_native_head')
    b=next(r for r in rows if r['setting']==s and r['condition']=='P13_native_head')
    paired.append(dict(setting=s,P12_PCC=float(a['PCC']),P13_PCC=float(b['PCC']),
        P13_vs_P12_relative_PCC_gain_pct=100*(float(b['PCC'])/float(a['PCC'])-1),
        P13_vs_TD24_Ridge_relative_PCC_gain_pct=float(b['PCC_relative_gain_pct'])))
costs=[read(p) for p in R.glob('execution_*.json')]
result=dict(passed=True,paired=paired,mean_P13_vs_P12_relative_PCC_gain_pct=sum(p['P13_vs_P12_relative_PCC_gain_pct'] for p in paired)/4,
    mean_P13_vs_fixed_TD24_relative_PCC_gain_pct=sum(p['P13_vs_TD24_Ridge_relative_PCC_gain_pct'] for p in paired)/4,
    CPU_pipeline_seconds=sum(r['CPU_wall_seconds'] for r in costs),CPU_verification_seconds=audit['CPU_wall_seconds'],
    source_updates=0,GPU_hours=0,verification_sha256=sha(R/'verification.json'),complete_sha256=sha(R/'results/complete.json'))
with (R/'derived_comparison.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
