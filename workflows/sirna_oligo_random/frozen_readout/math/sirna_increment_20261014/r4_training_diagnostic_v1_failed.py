"""Post-result training-OOF description only. No model fitting/test metrics."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
P=R/'round4'
B=R.parent/'core_six_settings_20261011/benchmark/data'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def corr(a,b):
    a=a-a.mean();b=b-b.mean();d=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/d) if d>1e-14 else None
def describe(y,base,aug):
    delta=aug-base;error=y-base
    return dict(n=len(y),residual_correction_correlation=corr(error,delta),
        correction_std=float(delta.std()),residual_std=float(error.std()),
        PCC_delta=corr(y,aug)-corr(y,base),
        MSE_relative_reduction_pct=100*(1-np.mean((y-aug)**2)/np.mean((y-base)**2)))
start=time.monotonic()
out=R/'r4_training_diagnostic.json'
assert not out.exists()
v=read(P/'verification.json');assert v['passed'] and v['complete_sha256']==sha(P/'results/complete.json')
plan=read(P/'plan.json');data={r['sample_id']:float(r['label']) for name in ('Huesken','Mixset','Takayuki') for r in table(B/f'{name}.csv')}
rows=[]
for fold in plan['folds']:
    y=np.array([data[i] for i in fold['fit_ids']])
    for source in plan['sources']:
        path=P/'results/folds'/fold['setting']/f"fold{fold['fold']}"/source
        receipt=read(path/'complete.json')
        assert sha(path/'arrays.npz')==receipt['files_sha256']['arrays.npz']
        with np.load(path/'arrays.npz',allow_pickle=False) as z:
            p=z['inner_oof']
        # Fixed alpha=100, w=1 vs same-alpha TD24. Not chosen from test or OOF results.
        assert plan['configs'][18]=={'alpha':100.,'weight':1.}
        assert plan['configs'][3]=={'alpha':100.,'weight':0.}
        pooled=describe(y,p[:,3],p[:,18]);parts=[]
        for inner in fold['inner']:
            ix=np.array(inner['valid']);parts.append(describe(y[ix],p[ix,3],p[ix,18]))
        rows.append(dict(setting=fold['setting'],fold=fold['fold'],source=source,
            pooled=pooled,inner=parts,positive_MSE_inner_folds=sum(q['MSE_relative_reduction_pct']>0 for q in parts)))
summary=[]
for setting in ('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix'):
    for source in plan['sources']:
        selected=[r for r in rows if r['setting']==setting and r['source']==source]
        summary.append(dict(setting=setting,source=source,
            mean_fit_OOF_PCC_delta=float(np.mean([q['pooled']['PCC_delta'] for q in selected])),
            mean_fit_OOF_MSE_reduction_pct=float(np.mean([q['pooled']['MSE_relative_reduction_pct'] for q in selected])),
            positive_inner_MSE=sum(q['positive_MSE_inner_folds'] for q in selected),total_inner=len(selected)*5))
receipt=dict(kind='fixed_configuration_training_OOF_diagnostic',round4_complete_sha256=v['complete_sha256'],
    source_script_sha256=sha(Path(__file__)),new_fits=0,GPU_hours=0,new_source_forward=0,
    limitations='Repeated data across outer folds; this is a descriptive development diagnostic, not independent confirmation. No outer-test predictions/metrics opened by this script. No model or hyperparameter selection.',
    config_indices=[3,18],summary=summary,rows=rows,CPU_wall_seconds=time.monotonic()-start)
with out.open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(json.dumps(summary,ensure_ascii=False,indent=2))
