"""Describe candidate sensitivity using only saved inner predictions and fit labels."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
from stability import select
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
start=time.monotonic();out=R/'diagnostic.json';assert not out.exists()
bench=R.parents[1]/'core_six_settings_20261011/benchmark/data'
dependency=read(R.parent/'round5/lock.json')['dependencies']
for name in ('Huesken','Mixset','Takayuki'):
    path=(bench/f'{name}.csv').resolve();assert sha(path)==dependency[str(path)]
labels={r['sample_id']:float(r['label']) for name in ('Huesken','Mixset','Takayuki') for r in table(bench/f'{name}.csv')}
results=[];fingerprints={};summaries=[]
for number in (4,5):
    old=R.parent/f'round{number}';plan=read(old/'plan.json');v=read(old/'verification.json')
    assert v['passed'] and v['complete_sha256']==sha(old/'results/complete.json')
    fingerprints[str(old/'results/complete.json')]=sha(old/'results/complete.json');manifest=read(old/'results/complete.json')['files_sha256']
    views=plan['sources'] if number==4 else plan['views']
    for fold in plan['folds']:
        y=np.array([labels[i] for i in fold['fit_ids']]);blocks=[d['valid'] for d in fold['inner']]
        for view in views:
            folder=old/'results/folds'/fold['setting']/f"fold{fold['fold']}"/view
            assert sha(folder/'complete.json')==manifest[(folder/'complete.json').relative_to(old/'results').as_posix()]
            c=read(folder/'complete.json');assert sha(folder/'arrays.npz')==c['files_sha256']['arrays.npz']
            with np.load(folder/'arrays.npz',allow_pickle=False) as a:p=a['inner_oof']
            for mode,allowed in (('Uniform',range(15,20)),('Adaptive',range(25)),('TD_only',range(5))):
                answer=select(y,p,plan['configs'],blocks,allowed)
                results.append(dict(round=number,setting=fold['setting'],fold=fold['fold'],view=view,mode=mode,**answer))
    for view in views:
        for mode in ('Uniform','Adaptive','TD_only'):
            part=[x for x in results if x['round']==number and x['view']==view and x['mode']==mode]
            summaries.append(dict(round=number,view=view,mode=mode,outer_folds=len(part),
                changed=sum(x['original']!=x['selected'] for x in part),
                original_supported_at_most_two=sum(x['original_support']<=2 for x in part),
                mean_original_support=float(np.mean([x['original_support'] for x in part]))))
receipt=dict(kind='training_OOF_selection_sensitivity',no_new_fits=True,no_outer_metrics=True,GPU_hours=0,
    code_sha256={p.name:sha(p) for p in (Path(__file__),R/'stability.py',R/'DIAGNOSTIC_PROTOCOL.md')},
    parent_complete_sha256=fingerprints,summary=summaries,rows=results,CPU_wall_seconds=time.monotonic()-start)
text=json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
with out.open('x',encoding='utf-8') as f:f.write(text)
print(json.dumps(summaries,ensure_ascii=False,indent=2))
