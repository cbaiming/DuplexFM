"""Independent graph, centered-moment and complete paired-output verification."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
import run as io
PAIRS=((1,0),(1,4),(1,3),(1,5),(2,0),(3,0),(4,0))
MAX_ERROR=0.

def close(a,b):
    global MAX_ERROR
    a=np.asarray(a,float);b=np.asarray(b,float);assert a.shape==b.shape
    assert np.array_equal(np.isnan(a),np.isnan(b))
    good=np.isfinite(a)&np.isfinite(b)
    if good.any():MAX_ERROR=max(MAX_ERROR,float(np.max(abs(a[good]-b[good]))))
    np.testing.assert_allclose(a,b,rtol=1e-10,atol=2e-10,equal_nan=True)

def direct(y,p,w):
    w=np.asarray(w,float);take=w>0
    if not take.any():return np.full(6,np.nan),np.full(6,np.nan)
    y=y[take];p=p[take];w=w[take];w=w/w.sum()
    a=y-float(np.sum(w*y));c=p-np.sum(w[:,None]*p,axis=0)
    vy=np.sum(w*a*a);vp=np.sum(w[:,None]*c*c,axis=0);den=np.sqrt(vy*vp)
    valid=(vy>1e-14)&(vp>1e-14);pc=np.full(6,np.nan);pc[valid]=np.sum(w[:,None]*a[:,None]*c,axis=0)[valid]/den[valid]
    return pc,np.sum(w[:,None]*(p-y[:,None])**2,axis=0)

def paired(pc,ms):
    out=np.full((7,3),np.nan)
    for j,(a,b) in enumerate(PAIRS):
        out[j,0]=pc[a]-pc[b]
        if np.isfinite(pc[a]) and np.isfinite(pc[b]) and pc[b]>0:out[j,1]=(pc[a]-pc[b])/pc[b]*100
        if np.isfinite(ms[a]) and np.isfinite(ms[b]) and ms[b]>0:out[j,2]=(ms[b]-ms[a])/ms[b]*100
    return out

def verify_interval(saved,x,eligible):
    finite=x[np.isfinite(x)];available=bool(eligible and len(finite)>=.95*len(x))
    assert saved['eligible']==eligible and saved['valid_draws']==len(finite) and saved['invalid_draws']==len(x)-len(finite)
    assert saved['interval_available']==available
    if available:close(saved['percentile95'],np.percentile(finite,[2.5,97.5]))
    else:assert saved['percentile95'] is None

def main(start):
    io.validate();root=R/'results';complete=io.read(root/'complete.json')
    assert complete['lock_sha256']==io.sha(R/'lock.json') and complete['new_fits']==0 and complete['source_forward']==0 and complete['GPU_hours']==0
    assert not (R/'execution.mutex').exists()
    runs=list(R.glob('execution_*.json'));assert len(runs)==1 and io.read(runs[0])['returncode']==0
    for name,h in complete['files_sha256'].items():assert io.sha(root/name)==h,name
    rows=[]
    for s in io.SETTINGS[:3]:rows+=io.table(io.DATA/f'{s}.csv')
    assert len(rows)==3329 and len({r['sample_id'] for r in rows})==3329
    prior=io.B/'round51';pv=io.read(prior/'verification.json');pg=io.read(prior/'results/groups_fixed.json')
    assert pv['passed'] and pv['independent_original_graph'] and pg['ids']==[r['sample_id'] for r in rows]
    g=np.array(pg['labels'],int);K=pg['global_components'];assert K==pv['global_components']
    saved=io.read(root/'groups_fixed.json');assert saved['ids']==pg['ids'] and saved['global_components']==K and saved['input_only'] and saved['before_numeric_labels']
    assert saved['reused_groups_sha256']==io.sha(prior/'results/groups_fixed.json') and saved['reused_statistics_sha256']==io.sha(prior/'results/statistics.npz')
    close(saved['labels'],g)
    with np.load(root/'statistics.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    close(a['labels'],g);d=io.read(root/'diagnostic.json');assert d['global_components']==K
    with np.load(prior/'results/statistics.npz',allow_pickle=False) as old:
        for key in ('observed_PCC','observed_MSE','leave_PCC','leave_MSE','bootstrap_PCC','bootstrap_MSE'):
            close(a[key][...,[0,5]],old[key][...,[0,1]])
        close(a['counts'],old['counts'])
    assert d['reused_R51_groups'] and d['reused_R51_draws'] and d['reused_reference_model_statistics']==2 and d['new_condition_statistics']==4

    # Independent identity-based merge, numeric labels from benchmark only.
    tables=[io.table(io.BASE/'oof_predictions.csv'),io.table(io.B/'round55/results/oof_predictions.csv'),io.table(io.B/'round50/results/oof_predictions.csv')]
    data=[];all_eligible=True;eligible_list=[];foldrows=io.table(root/'fold_comparisons.csv');assert len(foldrows)==112
    indices_by_cohort={s:np.flatnonzero(np.array([r['cohort']==s for r in rows])) for s in io.SETTINGS[:3]}
    for j,s in enumerate(io.SETTINGS):
        cohort='Mixset' if j==3 else s;ix=indices_by_cohort[cohort];ids=[rows[i]['sample_id'] for i in ix]
        y=np.array([float(rows[i]['label']) for i in ix]);p=np.zeros((len(ix),6));fs=[]
        conditions=((0,'condition','TD24_Ridge_CV'),(1,'source','core_real'),(1,'source','full_real'),(1,'source','initial_core'),(1,'source','TD_only'),(2,'source','core_real'))
        for m,(t,key,val) in enumerate(conditions):
            part=[r for r in tables[t] if r['setting']==s and r[key]==val];lookup={r['sample_id']:r for r in part}
            assert len(part)==len(lookup)==len(ids) and set(lookup)==set(ids)
            part=[lookup[id] for id in ids];close([float(r['label']) for r in part],y);fs.append([int(r['fold']) for r in part]);p[:,m]=[float(r['prediction']) for r in part]
        assert all(v==fs[0] for v in fs);folds=np.array(fs[0]);data.append((ix,y,p,folds))
        counts=np.bincount(g[ix],minlength=K);positive=counts[counts>0];eff=len(ix)**2/float(np.sum(positive**2));largest=max(positive)/len(ix)
        eligible=bool(len(positive)>=10 and eff>=10 and largest<=.5);eligible_list.append(eligible);all_eligible=all_eligible and eligible
        q=d['qualification'][s];assert q['n']==len(ix) and q['components']==len(positive) and q['eligible']==eligible
        close(q['effective_components'],eff);close(q['largest_fraction'],largest)
        pc,ms=direct(y,p,np.ones(len(y)));close(pc,a['observed_PCC'][j]);close(ms,a['observed_MSE'][j]);close(paired(pc,ms),a['observed_paired'][j])
        for f in sorted(set(folds.tolist())):
            keep=folds==f;fp,fm=direct(y[keep],p[keep],np.ones(keep.sum()));out=paired(fp,fm)
            for c,(aa,bb) in enumerate(PAIRS):
                rr=next(r for r in foldrows if r['setting']==s and int(r['fold'])==f and r['method']==d['models'][aa] and r['reference']==d['models'][bb])
                assert int(rr['n'])==int(keep.sum())
                close([float(rr[k]) if rr[k] else np.nan for k in ('PCC_method','PCC_reference','delta_PCC','relative_PCC_pct','MSE_reduction_pct')],[fp[aa],fp[bb],*out[c]])
    assert np.array_equal(data[1][0],data[3][0]) and np.array_equal(data[1][1],data[3][1])
    assert d['joint_interval_eligible']==all_eligible and d['paired_Mix_weights_shared']
    counts=np.random.default_rng(20261015).multinomial(K,np.full(K,1/K),size=2000);close(a['counts'],counts)
    assert (counts>=0).all() and (counts.sum(1)==K).all()
    # Each draw is recalculated directly in sample space after explicit centering.
    for draw,w in enumerate(counts):
        for j,(ix,y,p,_) in enumerate(data):
            if eligible_list[j]:
                pc,ms=direct(y,p,w[g[ix]]);close(pc,a['bootstrap_PCC'][draw,j]);close(ms,a['bootstrap_MSE'][draw,j]);close(paired(pc,ms),a['bootstrap_paired'][draw,j])
            else:assert np.isnan(a['bootstrap_PCC'][draw,j]).all() and np.isnan(a['bootstrap_MSE'][draw,j]).all() and np.isnan(a['bootstrap_paired'][draw,j]).all()
        if time.monotonic()-start>180:raise TimeoutError('Independent audit cap')
    for group in range(K):
        for j,(ix,y,p,_) in enumerate(data):
            keep=g[ix]!=group;assert a['removed'][group,j]==int((~keep).sum())
            pc,ms=direct(y[keep],p[keep],np.ones(keep.sum()));close(pc,a['leave_PCC'][group,j]);close(ms,a['leave_MSE'][group,j]);close(paired(pc,ms),a['leave_paired'][group,j])
    for j,s in enumerate(io.SETTINGS):
        for c,(aa,bb) in enumerate(PAIRS):
            r=next(v for v in d['summaries'] if v['setting']==s and v['method']==d['models'][aa] and v['reference']==d['models'][bb])
            close([r['delta_PCC'],r['relative_PCC_pct'],r['MSE_reduction_pct']],a['observed_paired'][j,c])
            close([r['PCC_method'],r['PCC_reference']],a['observed_PCC'][j,[aa,bb]])
            for key,k in (('delta_PCC_interval',0),('relative_PCC_interval',1),('MSE_reduction_interval',2)):verify_interval(r[key],a['bootstrap_paired'][:,j,c,k],eligible_list[j])
            vals=a['leave_paired'][a['removed'][:,j]>0,j,c,1];v=vals[np.isfinite(vals)];assert r['leave_component_valid']==len(v) and r['leave_component_undefined']==len(vals)-len(v)
            if len(v):close([r['leave_component_min'],r['leave_component_max']],[v.min(),v.max()])
            else:assert r['leave_component_min'] is None and r['leave_component_max'] is None
    close(a['macro'],a['observed_paired'][:,:,1].mean(0));close(a['leave_macro'],a['leave_paired'][:,:,:,1].mean(1));close(a['bootstrap_macro'],a['bootstrap_paired'][:,:,:,1].mean(1))
    for c,r in enumerate(d['macro']):
        close(r['mean_relative_PCC_pct'],a['macro'][c]);verify_interval(r['interval'],a['bootstrap_macro'][:,c],all_eligible)
        vals=a['leave_macro'][:,c];v=vals[np.isfinite(vals)];assert r['leave_component_valid']==len(v) and r['leave_component_undefined']==K-len(v)
        if len(v):close([r['leave_component_min'],r['leave_component_max']],[v.min(),v.max()])
    previous=io.read(io.B/'round55/verification.json')['goal_summary'];primary=next(r for r in previous if r['primary'])
    close(a['macro'][0],primary['mean_relative_PCC_gain_pct']);assert a['macro'][0]>=5
    assert (root/'groups_fixed.json').stat().st_mtime_ns<=(root/'statistics.npz').stat().st_mtime_ns
    io.validate();result=dict(passed=True,reused_independently_verified_graph=True,new_graph_computations=0,global_components=K,direct_edges=pv['direct_edges'],
        verified_prediction_appearances=6*3798,unique_inputs=3329,bootstrap_global_draws=2000,
        bootstrap_eligible_settings=sum(eligible_list),bootstrap_setting_metric_blocks=2000*sum(eligible_list),
        leave_component_setting_blocks=K*4,fold_comparison_rows=112,maximum_absolute_statistic_difference=MAX_ERROR,
        joint_interval_available=all_eligible,source_forward=0,new_fits=0,GPU_hours=0,goal_complete=False,
        lock_sha256=io.sha(R/'lock.json'),complete_sha256=io.sha(root/'complete.json'),CPU_wall_seconds=time.monotonic()-start)
    io.write(R/'verification.json',result);print(result,flush=True)

if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main(start);rc=0
    except Exception:error=traceback.format_exc();raise
    finally:io.write(R/f'verification_execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_fits=0))
