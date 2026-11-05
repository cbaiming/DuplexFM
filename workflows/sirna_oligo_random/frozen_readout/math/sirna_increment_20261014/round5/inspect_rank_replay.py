"""Bounded diagnosis of the failed Mixset/Core uniform Spearman comparison."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time
from pathlib import Path
import numpy as np
from scipy.stats import rankdata,spearmanr
R=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_r5_failed_audit',R/'verify.py');v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
started=time.monotonic();output=R/'rank_replay_diagnosis.json';assert not output.exists()
plan=v.read(R/'plan.json');rows=v.table(v.BENCH/'Mixset.csv');lookup={r['sample_id']:i for i,r in enumerate(rows)}
meta=v.table(v.EXPORT/'core_real/row_identity.csv');mi={r['sample_id']:i for i,r in enumerate(meta)}
source=np.load(v.EXPORT/'core_real/source.npy',allow_pickle=False)[[mi[r['sample_id']] for r in rows]].astype(float)
td=np.array([[float(z) for z in r['td'].split(',')] for r in rows]);x=np.c_[td,source]
ids=[];labels=[];saved=[];replayed=[]
for item in plan['folds']:
    if item['setting']!='Mixset':continue
    fi=np.array([lookup[i] for i in item['fit_ids']]);ti=np.array([lookup[i] for i in item['test_ids']])
    out=R/'results/folds/Mixset'/f"fold{item['fold']}"/'core_real';t=v.read(out/'transforms.json')['final']
    with np.load(out/'arrays.npz',allow_pickle=False) as a:p=v.expand(x[fi],x[ti],t,'core_real')@a['final_coef'][0]+a['final_intercept']
    r=[r for r in v.table(out/'predictions.csv') if r['condition']=='core_real_UniformBilinear']
    assert [r['sample_id'] for r in r]==item['test_ids']
    ids+=item['test_ids'];labels +=[float(row['label']) for row in r];saved +=[float(row['prediction']) for row in r];replayed+=p.tolist()
a=np.array(saved);b=np.array(replayed);ra=rankdata(a);rb=rankdata(b);changed=np.flatnonzero(ra!=rb)
maxerr=float(np.max(abs(a-b)));assert maxerr<1e-10
stored=v.table(R/'results/summary.csv');row=next(r for r in stored if r['setting']=='Mixset' and r['condition']=='core_real_UniformBilinear')
actual=float(spearmanr(labels,a).statistic);rebuilt=float(spearmanr(labels,b).statistic)
assert abs(actual-float(row['Spearman']))<1e-15 and abs(actual-rebuilt)>1e-9
result=dict(cause='rank discontinuity under independent floating-point reconstruction',source_lock_sha256=v.sha(R/'lock.json'),
    max_prediction_difference=maxerr,saved_Spearman=actual,reconstructed_Spearman=rebuilt,
    n_rank_changes=len(changed),changed=[dict(sample_id=ids[i],saved=float(a[i]),reconstructed=float(b[i]),saved_rank=float(ra[i]),reconstructed_rank=float(rb[i])) for i in changed],
    new_fits=0,GPU_hours=0,CPU_wall_seconds=time.monotonic()-started)
with output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
