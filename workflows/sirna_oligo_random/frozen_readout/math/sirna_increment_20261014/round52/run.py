"""Train-pool variance qualification only; immutable prior OOF is reused."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,hashlib,json,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;B=R.parent;Q=B/'round49';F=B/'round50';DATA=B.parent/'core_six_settings_20261011/benchmark/data'
sys.path.insert(0,str(R));import variance
SETTINGS=('Huesken','Mixset','Takayuki');SOURCES=('core_real','full_real','initial_core','TD_only')
CODE=('run.py','variance.py','verify.py','close.py','test_variance.py','PROTOCOL.md','CODE_AUDIT.md','pytest.xml')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def table(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def write(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(clean(obj),f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def write_csv(p,rows):
    with Path(p).open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(clean(rows))

def register():
    assert not (R/'lock.json').exists();suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==8
    assert not any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped'))
    deps={}
    def pin(p,h=None):
        p=Path(p).resolve();actual=sha(p)
        if h is not None:assert actual==h,str(p)
        deps[str(p)]=actual
    for parent in (Q,F,B/'round51'):
        cl=read(parent/'closure_receipt.json');assert cl['status']=='closed' and cl['readback_passed'] and not (parent/'execution.mutex').exists()
        for name,h in cl['files_sha256'].items():assert sha(B/name)==h,name
        for name in ('closure_receipt.json','verification.json','lock.json','final_readback.json'):pin(parent/name)
    assert read(B/'round51/final_readback.json')['status']=='passed'
    d=read(Q/'diagnostic.json');pin(Q/'diagnostic.json');pin(Q/'diagnostic_rows.csv',d['diagnostic_rows_sha256'])
    for name,h in d['pilot_files_sha256'].items():pin(Q/'pilot'/name,h)
    pin(F/'plan.json');pin(F/'results/complete.json');c=read(F/'results/complete.json')
    olddeps={str(Path(p).resolve()):h for p,h in read(F/'lock.json')['dependencies'].items()}
    for s in SETTINGS:
        p=(DATA/f'{s}.csv').resolve();pin(p,olddeps[str(p)])
        rel=f'groups/{s}/fold0/final.json';pin(F/'results'/rel,c['files_sha256'][rel])
    write(R/'lock.json',dict(files={n:sha(R/n) for n in CODE},dependencies=deps,numpy_version=np.__version__,
        settings=SETTINGS,sources=SOURCES,layouts=[0,1],minimum_group_n=8,minimum_groups=8,minimum_coverage=.5,
        minimum_rho=.3,minimum_score_improvement=.01,prior_df=5.,required_pools=2,
        fit_pool_n=[1564,319,416],OOF_rows=9196,source_seed=42,CPU_only=True,maximum_seconds=180,
        new_efficiency_fits=0,new_source_forward=0,new_PCA=0,GPU_hours=0,outer_test_read=False))
    print('REGISTERED',sha(R/'lock.json'),flush=True)

def validate():
    lock=read(R/'lock.json')
    for name,h in lock['files'].items():assert sha(R/name)==h,name
    for name,h in lock['dependencies'].items():assert sha(name)==h,name
    assert np.__version__==lock['numpy_version'];return lock

def prepare():
    plan=read(F/'plan.json');prior=read(Q/'diagnostic.json');out={}
    for s,n in zip(SETTINGS,(1564,319,416)):
        item=next(r for r in plan['folds'] if r['setting']==s and r['fold']==0);ids=item['fit_ids']
        assert ids==prior['fit_ids'][s] and len(ids)==n and set(ids).isdisjoint(item['test_ids']) and set(ids).isdisjoint(item['validation_ids'])
        lookup={r['sample_id']:r for r in table(DATA/f'{s}.csv')};rows=[lookup[id] for id in ids]
        gs=read(F/'results/groups'/s/'fold0/final.json');assert gs['train_ids']==ids and gs['train_indices']==list(range(n))
        assert gs['target_strings_sha256']==hashlib.sha256('\n'.join(r['target57'] for r in rows).encode()).hexdigest()
        g=np.array(gs['labels'],int);layouts=[variance.halves(ids,g,l) for l in (0,1)]
        out[s]=dict(item=item,ids=ids,rows=rows,groups=g,layouts=layouts)
    return out

def run(start):
    validate();assert not (R/'diagnostic.json').exists();assert all(x['num_threads']==1 for x in threadpool_info())
    pools=prepare();write(R/'halves_fixed.json',{s:dict(ids=p['ids'],labels=p['groups'],layouts=p['layouts']) for s,p in pools.items()})
    prior=table(Q/'diagnostic_rows.csv');assert len(prior)==9196
    rec=[];detail=[];oldblocks=0;rowcount=0
    for s,pool in pools.items():
        ids=pool['ids'];y=np.array([float(r['label']) for r in pool['rows']])
        for source in SOURCES:
            part=[r for r in prior if r['setting']==s and r['source']==source];assert [r['sample_id'] for r in part]==ids
            np.testing.assert_allclose([float(r['label']) for r in part],y,rtol=0,atol=1e-14)
            pred=np.array([float(r['prediction']) for r in part]);cover=[]
            for j,split in enumerate(pool['item']['inner']):
                path=Q/'pilot'/s/source/f'inner{j}';st=read(path/'state.json');va=split['valid'];cover+=va
                assert st['train_indices']==split['train'] and st['valid_indices']==va and set(st['train_indices']).isdisjoint(va)
                with np.load(path/'model.npz',allow_pickle=False) as a:np.testing.assert_allclose(a['prediction'],pred[va],rtol=0,atol=1e-14)
                oldblocks+=1
            assert sorted(cover)==list(range(len(ids)));residual=y-pred
            for l in (0,1):
                d=variance.diagnose(residual,pool['layouts'][l],len(ids));r=dict(setting=s,source=source,layout=l,**d);rec.append(r)
                detail.append({k:v for k,v in r.items() if k not in ('A_to_B','B_to_A','groups')})
            rowcount+=len(ids)
            if time.monotonic()-start>180:raise TimeoutError('CPU cap')
    assert len(rec)==24 and oldblocks==60 and rowcount==9196;decision=variance.gate(rec)
    write_csv(R/'summary.csv',detail);write(R/'diagnostic.json',dict(**decision,records=rec,unique_fit_samples=2299,OOF_rows=9196,
        old_OOF_blocks_verified=60,condition_layout_blocks=24,source_seed=42,no_outer_test_evaluation=True,
        new_efficiency_fits=0,new_source_forward=0,new_PCA=0,GPU_hours=0,halves_sha256=sha(R/'halves_fixed.json'),
        lock_sha256=sha(R/'lock.json'),summary_sha256=sha(R/'summary.csv')))
    validate();print('QUALIFIED',decision,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('register','run'));mode=ap.parse_args().mode
    if mode=='register':register()
    else:
        mutex=R/'execution.mutex'
        with mutex.open('x') as f:f.write(str(os.getpid()))
        start=time.monotonic();rc=1;error=None
        try:run(start);rc=0
        except Exception:error=traceback.format_exc();raise
        finally:write(R/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_efficiency_fits=0));mutex.unlink()
