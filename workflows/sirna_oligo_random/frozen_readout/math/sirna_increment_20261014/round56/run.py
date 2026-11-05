"""Once-only fixed-model stability audit. No fitting or feature forward."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,hashlib,json,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;B=R.parent;RESEARCH=B.parent
sys.path.insert(0,str(R))
import stability as st
from reuse import load_fixed_sampling,stitch
SETTINGS=('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix')
DATA=RESEARCH/'core_six_settings_20261011/benchmark/data'
BASE=RESEARCH/'expert_banded_20261014/results'
CODE=('PROTOCOL.md','CODE_AUDIT.md','stability.py','run.py','verify.py','close.py','test_reuse.py','pytest.xml','reuse.py','build.py','TEST_ENVIRONMENT_REPAIR.md','pytest_attempt1_failed.xml')
SEED=20261015;DRAWS=2000

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
    with p.open('x',encoding='utf-8') as f:json.dump(clean(obj),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def write_csv(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(clean(rows))
def save_npz(p,**arrays):
    with Path(p).open('xb') as f:np.savez_compressed(f,**arrays)

def register():
    assert not (R/'lock.json').exists()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==4
    assert not any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped'))
    deps={}
    def pin(p,expected=None):
        p=Path(p).resolve();h=sha(p)
        if expected is not None:assert h==expected,str(p)
        deps[str(p)]=h
    for number in (50,55):
        parent=B/f'round{number}';cl=read(parent/'closure_receipt.json')
        assert cl['status']=='closed' and cl['readback_passed'] and not (parent/'execution.mutex').exists()
        for name,h in cl['files_sha256'].items():assert sha(B/name)==h,name
        for name in ('closure_receipt.json','lock.json','verification.json','results/complete.json'):pin(parent/name)
        comp=read(parent/'results/complete.json')
        for name in ('oof_predictions.csv','summary.csv'):pin(parent/'results'/name,comp['files_sha256'][name])
    pin(B/'round55/final_readback.json');assert read(B/'round55/final_readback.json')['passed']
    olddeps={str(Path(p).resolve()):h for p,h in read(B/'round55/lock.json')['dependencies'].items()}
    for s in SETTINGS[:3]:
        p=(DATA/f'{s}.csv').resolve();pin(p,olddeps[str(p)])
    complete=read(BASE/'complete.json');pin(BASE/'complete.json')
    for name in ('oof_predictions.csv','summary.csv'):pin(BASE/name,complete['files_sha256'][name])
    pin(B/'round55/plan.json');pin(RESEARCH/'sirna_benchmark_20261007/benchmark_core.py')
    prior=B/'round51';receipt=read(prior/'closure_receipt.json')
    assert receipt['status']=='closed' and receipt['readback_passed']
    for name,h in receipt['files_sha256'].items():assert sha(B/name)==h,name
    assert read(prior/'verification.json')['independent_original_graph']
    for name in ('lock.json','closure_receipt.json','verification.json','final_readback.json','stability.py','results/complete.json'):
        pin(prior/name)
    for name in ('groups_fixed.json','statistics.npz'):
        pin(prior/'results'/name,read(prior/'results/complete.json')['files_sha256'][name])
    write(R/'lock.json',dict(files={n:sha(R/n) for n in CODE},dependencies=deps,numpy_version=np.__version__,
        settings=SETTINGS,models=st.MODELS,comparisons=st.COMPARISONS,seed=SEED,draws=DRAWS,CPU_only=True,
        maximum_seconds=180,source_seed=42,new_fits=0,new_forward=0,GPU_hours=0,goal_complete=False))
    print('REGISTERED',sha(R/'lock.json'),flush=True)

def validate():
    lock=read(R/'lock.json')
    for name,h in lock['files'].items():assert sha(R/name)==h,name
    for name,h in lock['dependencies'].items():assert sha(name)==h,name
    assert np.__version__==lock['numpy_version'];return lock

def input_rows():
    rows=[]
    for s,n in zip(SETTINGS[:3],(2196,469,664)):
        part=table(DATA/f'{s}.csv');assert len(part)==n and all(r['cohort']==s for r in part);rows+=part
    assert len({r['sample_id'] for r in rows})==3329
    return rows

def bundle(rows):
    indexes={s:np.array([i for i,r in enumerate(rows) if r['cohort']==('Mixset' if s==SETTINGS[3] else s)]) for s in SETTINGS}
    paths=(BASE/'oof_predictions.csv',B/'round55/results/oof_predictions.csv',B/'round50/results/oof_predictions.csv')
    maps=[{} for _ in st.MODELS]
    for j,p in enumerate(paths):
        for r in table(p):
            if r['setting'] not in SETTINGS:continue
            if j==0:
                if r['condition']!='TD24_Ridge_CV':continue
                model=0
            elif j==1:model={'core_real':1,'full_real':2,'initial_core':3,'TD_only':4}[r['source']]
            else:
                if r['source']!='core_real':continue
                model=5
            key=(r['setting'],r['sample_id']);assert key not in maps[model];maps[model][key]=r
    out=[]
    for s in SETTINGS:
        ix=indexes[s];y=np.array([float(rows[i]['label']) for i in ix]);p=np.empty((len(ix),6));folds=None
        for model,m in enumerate(maps):
            part=[m[s,rows[i]['sample_id']] for i in ix]
            np.testing.assert_allclose([float(r['label']) for r in part],y,rtol=0,atol=2e-14)
            ff=np.array([int(r['fold']) for r in part]);folds=ff if folds is None else folds;assert np.array_equal(ff,folds)
            p[:,model]=[float(r['prediction']) for r in part]
            assert len([k for k in m if k[0]==s])==len(ix)
        out.append((ix,y,p,folds))
    assert np.array_equal(out[1][0],out[3][0]) and np.array_equal(out[1][1],out[3][1])
    assert all(len(m)==3798 for m in maps)
    return out

def run(start):
    validate();root=R/'results';assert not root.exists();root.mkdir()
    assert all(t['num_threads']==1 for t in threadpool_info())
    rows=input_rows();prior=B/'round51/results'
    labels,counts,refs=load_fixed_sampling(prior,[r['sample_id'] for r in rows],read(prior/'complete.json')['files_sha256']);K=int(labels.max()+1)
    write(root/'groups_fixed.json',dict(ids=[r['sample_id'] for r in rows],labels=labels,global_components=K,
        input_only=True,before_numeric_labels=True,reused_groups_sha256=sha(prior/'groups_fixed.json'),reused_statistics_sha256=sha(prior/'statistics.npz'),lock_sha256=sha(R/'lock.json')))
    data=bundle(rows);quals=[st.qualification(labels[ix]) for ix,_,_,_ in data]
    pc=np.empty((4,6));ms=pc.copy();observed=np.empty((4,7,3));foldrows=[]
    for j,(ix,y,p,folds) in enumerate(data):
        npcc,nmse=st.statistics(y,p[:,1:5]);pc[j]=stitch(npcc,refs['observed_PCC'][j]);ms[j]=stitch(nmse,refs['observed_MSE'][j]);observed[j]=st.paired(pc[j],ms[j])
        for f in sorted(set(folds.tolist())):
            keep=folds==f;fp,fm=st.statistics(y[keep],p[keep]);pp=st.paired(fp,fm)
            for c,(a,b) in enumerate(st.COMPARISONS):foldrows.append(dict(setting=SETTINGS[j],fold=f,n=int(keep.sum()),
                method=st.MODELS[a],reference=st.MODELS[b],PCC_method=fp[a],PCC_reference=fp[b],delta_PCC=pp[c,0],relative_PCC_pct=pp[c,1],MSE_reduction_pct=pp[c,2]))
    loo=np.full((K,4,7,3),np.nan);removed=np.zeros((K,4),int);loopc=np.full((K,4,6),np.nan);looms=loopc.copy()
    for g in range(K):
        for j,(ix,y,p,_) in enumerate(data):
            w=(labels[ix]!=g).astype(float);removed[g,j]=int(len(w)-w.sum());npcc,nmse=st.statistics(y,p[:,1:5],w)
            loopc[g,j]=stitch(npcc,refs['leave_PCC'][g,j]);looms[g,j]=stitch(nmse,refs['leave_MSE'][g,j]);loo[g,j]=st.paired(loopc[g,j],looms[g,j])
        if time.monotonic()-start>180:raise TimeoutError('CPU cap')
    boot=np.full((DRAWS,4,7,3),np.nan);bootpc=np.full((DRAWS,4,6),np.nan);bootms=bootpc.copy()
    for draw,weights in enumerate(counts):
        for j,(ix,y,p,_) in enumerate(data):
            if quals[j]['eligible']:
                npcc,nmse=st.statistics(y,p[:,1:5],weights[labels[ix]])
                bootpc[draw,j]=stitch(npcc,refs['bootstrap_PCC'][draw,j]);bootms[draw,j]=stitch(nmse,refs['bootstrap_MSE'][draw,j])
                boot[draw,j]=st.paired(bootpc[draw,j],bootms[draw,j])
        if time.monotonic()-start>180:raise TimeoutError('CPU cap')
    summaries=[]
    for j,s in enumerate(SETTINGS):
        for c,(a,b) in enumerate(st.COMPARISONS):
            vals=loo[removed[:,j]>0,j,c,1];finite=vals[np.isfinite(vals)]
            summaries.append(dict(setting=s,method=st.MODELS[a],reference=st.MODELS[b],PCC_method=pc[j,a],PCC_reference=pc[j,b],
                delta_PCC=observed[j,c,0],relative_PCC_pct=observed[j,c,1],MSE_reduction_pct=observed[j,c,2],
                relative_PCC_interval=st.interval(boot[:,j,c,1],quals[j]['eligible']),
                delta_PCC_interval=st.interval(boot[:,j,c,0],quals[j]['eligible']),
                MSE_reduction_interval=st.interval(boot[:,j,c,2],quals[j]['eligible']),
                leave_component_valid=len(finite),leave_component_undefined=len(vals)-len(finite),
                leave_component_min=float(min(finite)) if len(finite) else None,leave_component_max=float(max(finite)) if len(finite) else None))
    mac=st.macro(observed[:,:,1]);loomac=np.mean(loo[:,:,:,1],axis=1);bootmac=np.mean(boot[:,:,:,1],axis=1);eligible=all(q['eligible'] for q in quals)
    macrows=[]
    for c,(a,b) in enumerate(st.COMPARISONS):
        vals=loomac[:,c];finite=vals[np.isfinite(vals)]
        macrows.append(dict(method=st.MODELS[a],reference=st.MODELS[b],mean_relative_PCC_pct=mac[c],
            interval=st.interval(bootmac[:,c],eligible),leave_component_valid=len(finite),leave_component_undefined=K-len(finite),
            leave_component_min=float(min(finite)) if len(finite) else None,leave_component_max=float(max(finite)) if len(finite) else None))
    save_npz(root/'statistics.npz',labels=labels,counts=counts,observed_PCC=pc,observed_MSE=ms,observed_paired=observed,
        leave_PCC=loopc,leave_MSE=looms,leave_paired=loo,removed=removed,bootstrap_PCC=bootpc,bootstrap_MSE=bootms,
        bootstrap_paired=boot,macro=mac,leave_macro=loomac,bootstrap_macro=bootmac)
    write_csv(root/'fold_comparisons.csv',foldrows)
    write(root/'diagnostic.json',dict(settings=SETTINGS,models=st.MODELS,comparisons=st.COMPARISONS,
        unique_inputs=3329,predictions_per_condition=3798,conditions=6,global_components=K,
        qualification=dict(zip(SETTINGS,quals)),summaries=summaries,macro=macrows,joint_interval_eligible=eligible,
        bootstrap_draws=DRAWS,bootstrap_seed=SEED,bootstrap_draw_settings_computed=DRAWS*sum(q['eligible'] for q in quals),
        paired_Mix_weights_shared=True,reused_R51_groups=True,reused_R51_draws=True,reused_reference_model_statistics=2,new_condition_statistics=4,no_new_model=True,conditional_fixed_predictions_only=True,goal_complete=False))
    validate();write(root/'complete.json',dict(lock_sha256=sha(R/'lock.json'),files_sha256={p.name:sha(p) for p in sorted(root.iterdir()) if p.is_file()},new_fits=0,source_forward=0,GPU_hours=0))
    print(json.dumps(clean(dict(status='complete',global_components=K,qualification=quals,macro=macrows[0])),ensure_ascii=False),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('register','run'));mode=ap.parse_args().mode
    if mode=='register':register()
    else:
        mutex=R/'execution.mutex'
        with mutex.open('x') as f:f.write(str(os.getpid()))
        start=time.monotonic();rc=1;error=None
        try:run(start);rc=0
        except Exception:error=traceback.format_exc();raise
        finally:write(R/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_fits=0));mutex.unlink()
