"""Independent sample-variance, split and graph audit for R52."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,math,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import run as io
MAX_ERROR=0.

def close(a,b):
    global MAX_ERROR
    if b is None:assert a is None;return
    if isinstance(b,dict):
        assert set(a)==set(b)
        for key in b:close(a[key],b[key])
    elif isinstance(b,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):close(x,y)
    elif isinstance(b,(bool,str,int)):assert a==b
    else:
        assert math.isfinite(float(a)) and math.isfinite(float(b));MAX_ERROR=max(MAX_ERROR,abs(float(a)-float(b)))
        assert abs(float(a)-float(b))<=1e-10*(1+abs(float(b))),str((a,b))

def independent_groups(rows):
    keys={};ii=[];jj=[];table=str.maketrans('ACGT','TGCA')
    for i,row in enumerate(rows):
        s=row['target57'].upper().replace('U','T');assert len(s)==57 and not set(s)-set('ACGT')
        for k in {min(s[j:j+25],s[j:j+25].translate(table)[::-1]) for j in range(33)}:
            if k not in keys:keys[k]=len(keys)
            ii.append(i);jj.append(keys[k])
    a=csr_matrix((np.ones(len(ii)),(ii,jj)),shape=(len(rows),len(keys)))
    _,g=connected_components(a@a.T,directed=False);order=sorted(set(g.tolist()),key=lambda c:int(np.flatnonzero(g==c)[0]));mp={v:i for i,v in enumerate(order)}
    return np.array([mp[x] for x in g])

def part_stats(r,ix):
    x=r[ix];n=len(x);assert n>=4
    sse=float(np.sum((x[:,None]-x[None,:])**2)/(2*n))
    return dict(n=n,df=n-1,SSE=sse,variance=sse/(n-1))

def score(cal,val):
    if not cal:return dict(defined=False,reason='no_eligible_groups',predicted_variances=[],global_variance=None,score_improvement=None)
    v0=math.fsum(x['SSE'] for x in cal)/sum(x['df'] for x in cal)
    pred=[(x['SSE']+5*v0)/(x['df']+5) for x in cal]
    if v0<=0 or any(x['SSE']<=0 for x in cal) or any(x['variance']<=0 for x in val):return dict(defined=False,reason='zero_or_nonfinite_variance',predicted_variances=[],global_variance=v0,score_improvement=None)
    base=[.5*(math.log(v0)+x['variance']/v0) for x in val]
    alt=[.5*(math.log(p)+x['variance']/p) for p,x in zip(pred,val)]
    weights=[x['df']/sum(v['df'] for v in val) for x in val]
    return dict(defined=True,predicted_variances=pred,global_variance=v0,baseline_score=math.fsum(w*x for w,x in zip(weights,base)),
        group_score=math.fsum(w*x for w,x in zip(weights,alt)),score_improvement=math.fsum(w*(a-b) for w,a,b in zip(weights,base,alt)),
        group_score_improvements=[a-b for a,b in zip(base,alt)],validation_degrees_of_freedom=[float(x['df']) for x in val])

def main(start):
    io.validate();d=io.read(R/'diagnostic.json');fixed=io.read(R/'halves_fixed.json');prior=io.table(io.Q/'diagnostic_rows.csv')
    assert not (R/'execution.mutex').exists();runs=list(R.glob('execution_*.json'));assert len(runs)==1 and io.read(runs[0])['returncode']==0
    assert d['halves_sha256']==io.sha(R/'halves_fixed.json') and d['summary_sha256']==io.sha(R/'summary.csv') and d['lock_sha256']==io.sha(R/'lock.json')
    plan=io.read(io.F/'plan.json');summary=io.table(R/'summary.csv');assert len(summary)==24
    expected_records=[];nblocks=nrows=halfgroups=0;groupcounts={}
    for setting,n in zip(io.SETTINGS,(1564,319,416)):
        item=next(p for p in plan['folds'] if p['setting']==setting and p['fold']==0);ids=item['fit_ids'];assert len(ids)==n
        raw={r['sample_id']:r for r in io.table(io.DATA/f'{setting}.csv')};rows=[raw[id] for id in ids]
        assert set(ids).isdisjoint(item['test_ids']) and set(ids).isdisjoint(item['validation_ids'])
        g=independent_groups(rows);assert fixed[setting]['ids']==ids and fixed[setting]['labels']==g.tolist();groupcounts[setting]=int(g.max()+1)
        layouts=[]
        for layout in (0,1):
            blocks=[]
            for group in sorted(set(g.tolist())):
                members=[i for i in range(n) if g[i]==group]
                if len(members)<8:continue
                byhash=sorted((hashlib.sha256(('R52|20261015|layout'+str(layout)+'|'+ids[i]).encode()).hexdigest(),ids[i],i) for i in members)
                order=[v[2] for v in byhash];blocks.append(dict(group=group,A=order[::2],B=order[1::2]))
            assert blocks==fixed[setting]['layouts'][layout];layouts.append(blocks);halfgroups+=len(blocks)
        y=np.array([float(r['label']) for r in rows])
        for source in io.SOURCES:
            part=[r for r in prior if r['source']==source and r['setting']==setting];assert [r['sample_id'] for r in part]==ids
            pred=np.array([float(r['prediction']) for r in part]);np.testing.assert_allclose([float(r['label']) for r in part],y,rtol=0,atol=1e-14)
            coverage=[]
            for j,sp in enumerate(item['inner']):
                p=io.Q/'pilot'/setting/source/f'inner{j}';state=io.read(p/'state.json')
                assert state['valid_indices']==sp['valid'] and state['train_indices']==sp['train'] and set(sp['valid']).isdisjoint(sp['train']);coverage+=sp['valid']
                with np.load(p/'model.npz',allow_pickle=False) as a:np.testing.assert_allclose(a['prediction'],pred[sp['valid']],rtol=0,atol=1e-14)
                nblocks+=1
            assert sorted(coverage)==list(range(n));res=y-pred;nrows+=n
            for layout,blocks in enumerate(layouts):
                groups=[dict(group=b['group'],A=part_stats(res,b['A']),B=part_stats(res,b['B'])) for b in blocks]
                aa=np.array([x['A']['variance'] for x in groups]);bb=np.array([x['B']['variance'] for x in groups]);rho=None
                if len(aa)>=2 and np.all(aa>0) and np.all(bb>0) and np.ptp(aa)>0 and np.ptp(bb)>0:rho=float(np.corrcoef(rankdata(aa),rankdata(bb))[0,1])
                ab=score([x['A'] for x in groups],[x['B'] for x in groups]);ba=score([x['B'] for x in groups],[x['A'] for x in groups])
                mean=(ab['score_improvement']+ba['score_improvement'])/2 if ab['defined'] and ba['defined'] else None
                covered=sum(x['A']['n']+x['B']['n'] for x in groups);eligible=len(groups)>=8 and covered/n>=.5
                eligible_signal=eligible and rho is not None and rho>=.3 and mean is not None and mean>=.01
                expected=dict(setting=setting,source=source,layout=layout,eligible_groups=len(groups),covered_samples=covered,coverage=covered/n,
                    structurally_eligible=bool(eligible),variance_spearman=rho,A_to_B=ab,B_to_A=ba,mean_score_improvement=mean,qualified=bool(eligible_signal),groups=groups)
                actual=next(r for r in d['records'] if r['setting']==setting and r['source']==source and r['layout']==layout);close(actual,expected);expected_records.append(expected)
                ss=next(r for r in summary if r['setting']==setting and r['source']==source and int(r['layout'])==layout)
                for k in ('eligible_groups','covered_samples'):assert int(ss[k])==expected[k]
                for k in ('coverage','variance_spearman','mean_score_improvement'):close(None if ss[k]=='' else float(ss[k]),expected[k])
                assert (ss['qualified']=='True')==expected['qualified'] and (ss['structurally_eligible']=='True')==expected['structurally_eligible']
            if time.monotonic()-start>180:raise TimeoutError('Independent audit cap')
    pools={s:all(r['qualified'] for r in expected_records if r['source']=='core_real' and r['setting']==s) for s in io.SETTINGS}
    assert d['primary_pool_qualifications']==pools and d['qualified']==(sum(pools.values())>=2)
    assert d['required_pools']==2 and nblocks==60 and nrows==9196 and len(expected_records)==24
    assert d['no_outer_test_evaluation'] and d['new_efficiency_fits']==d['new_source_forward']==d['new_PCA']==d['GPU_hours']==0
    assert (R/'halves_fixed.json').stat().st_mtime_ns<=(R/'diagnostic.json').stat().st_mtime_ns
    io.validate();result=dict(passed=True,qualified=d['qualified'],primary_pool_qualifications=pools,independent_training_graphs=3,
        target_component_counts=groupcounts,independent_layout_group_splits=halfgroups,independent_condition_layout_blocks=24,
        old_OOF_blocks_verified=nblocks,OOF_rows=nrows,maximum_absolute_difference=MAX_ERROR,new_efficiency_fits=0,
        new_source_forward=0,GPU_hours=0,no_outer_test_evaluation=True,lock_sha256=io.sha(R/'lock.json'),diagnostic_sha256=io.sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-start)
    io.write(R/'verification.json',result);print(result,flush=True)

if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main(start);rc=0
    except Exception:error=traceback.format_exc();raise
    finally:io.write(R/f'verification_execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_efficiency_fits=0))
