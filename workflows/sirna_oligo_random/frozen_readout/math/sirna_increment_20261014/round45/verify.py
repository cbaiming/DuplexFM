"""Independent SciPy projection, scalar KKT certificate and ceiling arithmetic."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
from scipy.optimize import isotonic_regression
R=Path(__file__).resolve().parent;P=R.parent/'round4';N=R.parent/'round25'
spec=importlib.util.spec_from_file_location('_r45_independent4',P/'verify.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,sha,table,check=old.read,old.sha,old.table,old.check


def close(a,b,tol=1e-9):np.testing.assert_allclose(a,b,rtol=tol,atol=tol)


def independent_projection(score,y,state,projection):
    buckets={}
    for s,v in zip(score,y):buckets.setdefault(float(s),[]).append(float(v))
    levels=sorted(buckets);weights=np.array([len(buckets[s]) for s in levels],float)
    means=np.array([sum(buckets[s])/len(buckets[s]) for s in levels])
    fit=isotonic_regression(means,weights=weights,increasing=True).x
    mapping=dict(zip(levels,fit));pp=np.array([mapping[float(s)] for s in score])
    close(state['score_levels'],levels,0);close(state['counts'],weights,0);close(state['group_mean_response'],means,1e-12)
    close(state['fitted_levels'],fit,1e-11);close(projection,pp,1e-11)
    multipliers=[];residual=0.
    for w,m,p in zip(weights,means,fit):residual+=w*(m-p);multipliers.append(max(0.,residual))
    duals=np.array(multipliers[:-1]);close(state['dual_multipliers'],duals,1e-10)
    transpose=np.zeros(len(levels))
    for i,l in enumerate(duals):transpose[i]+=l;transpose[i+1]-=l
    within=.5*sum(sum((v-means[j])**2 for v in buckets[s]) for j,s in enumerate(levels))
    primal=.5*sum((float(v)-float(p))**2 for v,p in zip(y,pp))
    dual=sum(float(d*m-d*d/(2*w)) for d,m,w in zip(transpose,means,weights))+within
    cert=state['certificate'];gap=primal-dual;yc=y-y.mean();variance=float(sum(v*v for v in yc))
    threshold=1e-9*max(1.,variance,float(max(weights)))
    stationarity=float(max(abs(weights*(fit-means)+transpose)))
    complement=float(max(abs(duals*np.diff(fit)))) if len(fit)>1 else 0.
    violation=max(0.,float(-min(np.diff(fit)))) if len(fit)>1 else 0.
    for key,value in (('primal',primal),('dual',dual),('gap',gap),('threshold',threshold),('stationarity_max',stationarity),
                      ('complementarity_max',complement),('monotonicity_violation',violation),
                      ('centered_projection_orthogonality',float((yc-(pp-pp.mean()))@(pp-pp.mean()))),('mean_difference',float(pp.mean()-y.mean()))):
        close(cert[key],value,1e-9)
    qualified=bool(gap>=-threshold and gap<=threshold and stationarity<=threshold and complement<=threshold and violation<=threshold)
    assert cert['certified']==qualified
    assert abs(pp.mean()-y.mean())<1e-10 and abs((yc-(pp-pp.mean()))@(pp-pp.mean()))<1e-8
    sc=score-score.mean();sv=float(sc@sc);pc=pp-pp.mean();pv=float(pc@pc)
    raw=float(np.corrcoef(score,y)[0,1]) if sv>1e-14 else None
    slope=max(0.,float(sc@yc/sv)) if sv>1e-14 else 0.
    affine=y.mean()+slope*sc;affine_mse=float(np.mean((y-affine)**2))
    if raw is not None:close(affine_mse,np.var(y)*(1-max(raw,0)**2),1e-11)
    upper=float(np.sqrt(max(0.,min(1.,pv/variance))));projection_pcc=float(np.corrcoef(pp,y)[0,1]) if pv>1e-14 else None
    if projection_pcc is not None:close(projection_pcc,upper,1e-10)
    if raw is not None:assert upper>=raw-1e-10
    metrics=dict(raw_PCC=raw,raw_MSE=float(np.mean((score-y)**2)),upper_PCC=upper,upper_attained=bool(pv>1e-14),
        optimistic_projection_PCC=projection_pcc,optimistic_projection_MSE=float(np.mean((y-pp)**2)),positive_affine_MSE=affine_mse,
        positive_affine_slope=slope,absolute_PCC_room=None if raw is None else upper-raw,
        optimistic_MSE_reduction_vs_affine_pct=100*(1-np.mean((y-pp)**2)/affine_mse) if affine_mse>1e-14 else 0.,
        unique_score_count=len(levels),unique_fitted_levels=len(np.unique(fit)))
    for k,value in metrics.items():
        if value is None:assert state['metrics'][k] is None
        else:close(state['metrics'][k],value,1e-10)
    assert state['uses_evaluation_labels'] and not state['deployable'] and not state['new_OOF_predictions']
    ranges=state['block_ranges'];assert ranges[0][0]==0 and ranges[-1][1]==len(levels)
    assert all(a[1]==b[0] for a,b in zip(ranges[:-1],ranges[1:]))
    for lo,hi in ranges:
        assert lo<hi;close(fit[lo:hi],np.average(means[lo:hi],weights=weights[lo:hi]),1e-10)
    return metrics,qualified,abs(gap),float(np.max(np.abs(pp-projection))),stationarity


def main():
    start=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    assert lock['optimistic_diagnostic_only'] and lock['not_deployable'] and lock['new_efficiency_heads']==lock['new_OOF_rows']==0
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json')
    assert not d['outer_test_metrics_used'] and d['optimistic_diagnostic_only'] and not d['deployable']
    assert d['new_source_forward']==d['GPU_hours']==d['new_efficiency_heads']==d['new_OOF_rows']==0
    assert all(t['num_threads']==1 for t in d['threadpool'])
    check(R/'projection_audit',d['projection_files_sha256']);assert sha(R/'optimistic_projection_rows.csv')==d['projection_rows_sha256']
    runs=[read(p) for p in R.glob('execution_*.json')];assert len(runs)==1 and runs[0]['returncode']==0
    assert d['source_identities']==read(N/'diagnostic.json')['source_identities']
    plan=read(P/'plan.json');details=table(R/'optimistic_projection_rows.csv');records=[]
    counts=dict(projections=0,pool=0,inner=0,native_reused=0,TD_reused=0,rows=0)
    numeric=True;maxgap=maxdifference=maxstationarity=0.
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids']
        assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        rowmap={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')};y=np.array([float(rowmap[s]['label']) for s in ids])
        for condition in (*old.SOURCES,'TD_only'):
            original=np.full(len(ids),np.nan);refpaths=[];foldid=np.full(len(ids),-1,dtype=int);coverage=[]
            if condition=='TD_only':
                path=P/'results/folds'/setting/'fold0/core_real/arrays.npz'
                with np.load(path,allow_pickle=False) as a:original=a['inner_oof'][:,3].copy()
                refpaths.append(dict(path=str(path.resolve()),sha256=sha(path),key='inner_oof[:,3]'));counts['TD_reused']+=5
            for j,sp in enumerate(item['inner']):
                assert set(sp['train']).isdisjoint(sp['valid']);va=np.array(sp['valid']);coverage+=sp['valid'];foldid[va]=j
                if condition!='TD_only':
                    folder=N/'pilot'/setting/condition/f'inner{j}';c=read(folder/'complete.json');check(folder,c['files_sha256'])
                    assert c['lock_sha256']==sha(N/'lock.json');s=read(folder/'state.json')
                    assert s['train_indices']==sp['train'] and s['valid_indices']==sp['valid'] and s['numerically_qualified']
                    path=folder/'model.npz'
                    with np.load(path,allow_pickle=False) as a:original[va]=a['prediction']
                    refpaths.append(dict(path=str(path.resolve()),sha256=sha(path),key='prediction',inner=j));counts['native_reused']+=1
            assert sorted(coverage)==list(range(len(ids))) and np.isfinite(original).all()
            pool=None;inner=np.full(len(ids),np.nan);blocks=[]
            for j in range(-1,5):
                if time.monotonic()-start>900:raise TimeoutError('CPU audit cap')
                indices=np.arange(len(ids)) if j==-1 else np.array(item['inner'][j]['valid'])
                scope='pool' if j==-1 else f'inner_valid{j}';folder=R/'projection_audit'/setting/condition/scope
                c=read(folder/'complete.json');check(folder,c['files_sha256'])
                assert c['lock_sha256']==sha(R/'lock.json') and c['uses_evaluation_labels'] and not c['deployable']
                state=read(folder/'state.json')
                assert (state['setting'],state['condition'],state['inner'])==(setting,condition,j)
                assert state['indices']==indices.tolist() and state['sample_ids']==[ids[i] for i in indices] and state['original_references']==refpaths
                with np.load(folder/'optimistic_projection.npz',allow_pickle=False) as a:
                    close(a['fixed_OOF_score'],original[indices],0);projection=a['optimistic_label_using_projection'].copy()
                m,valid,gap,difference,stationarity=independent_projection(original[indices],y[indices],state,projection)
                numeric &= valid;maxgap=max(maxgap,gap);maxdifference=max(maxdifference,difference);maxstationarity=max(maxstationarity,stationarity)
                counts['projections']+=1
                if j==-1:pool=projection;poolmetrics=m;counts['pool']+=1
                else:inner[indices]=projection;blocks.append(dict(inner=j,n=len(indices),metrics=m,numerically_qualified=valid));counts['inner']+=1
            part=[r for r in details if r['setting']==setting and r['condition']==condition]
            assert [r['sample_id'] for r in part]==ids and [int(r['inner']) for r in part]==foldid.tolist()
            for field,value in (('label',y),('original_OOF_prediction',original),('optimistic_pool_projection',pool),('optimistic_inner_projection',inner)):
                close([float(r[field]) for r in part],value,0)
            counts['rows']+=len(part)
            expected=next(r for r in d['records'] if r['setting']==setting and r['condition']==condition)
            assert expected['n']==len(ids) and len(expected['inner_blocks'])==5
            for k,value in poolmetrics.items():
                if value is None:assert expected['metrics'][k] is None
                else:close(expected['metrics'][k],value,1e-10)
            for actual,block in zip(expected['inner_blocks'],blocks):
                assert actual['inner']==block['inner'] and actual['n']==block['n'] and actual['numerically_qualified']==block['numerically_qualified']
                for k,value in block['metrics'].items():
                    if value is None:assert actual['metrics'][k] is None
                    else:close(actual['metrics'][k],value,1e-10)
            records.append(dict(setting=setting,condition=condition,metrics=poolmetrics))
    assert counts==dict(projections=72,pool=12,inner=60,native_reused=45,TD_reused=15,rows=9196) and len(details)==9196
    assert d['new_label_using_isotonic_projections']==72 and d['pool_projections']==12 and d['inner_valid_projections']==60
    assert d['source_score_rows']==9196 and d['reused_native_heads']==45 and d['reused_unique_TD_heads']==15
    core=[r for r in records if r['condition']=='core_real'];rooms=[];pairs=[]
    for r in core:
        m=r['metrics'];tm=next(x['metrics'] for x in records if x['setting']==r['setting'] and x['condition']=='TD_only')
        rooms.append(m['upper_attained'] and m['absolute_PCC_room'] is not None and m['absolute_PCC_room']>=.01 and m['optimistic_MSE_reduction_vs_affine_pct']>=1.)
        pairs.append(dict(setting=r['setting'],Core_upper_PCC=m['upper_PCC'],TD_upper_PCC=tm['upper_PCC'],delta_upper_PCC=m['upper_PCC']-tm['upper_PCC']))
    for p in pairs:
        expected=next(x for x in d['paired_ceilings'] if x['setting']==p['setting'])
        for k in ('Core_upper_PCC','TD_upper_PCC','delta_upper_PCC'):close(expected[k],p[k],1e-10)
    room=sum(rooms)>=2;ordering=sum(p['delta_upper_PCC']>=.005 for p in pairs)>=2 and all(p['delta_upper_PCC']>=-.005 for p in pairs)
    qualified=bool(room and ordering and numeric)
    assert d['qualified']==qualified and d['calibration_room_gate']==room and d['paired_ordering_gate']==ordering and d['all_numerically_qualified']==numeric
    assert d['qualification_only'] and not d['goal_achieved']
    result=dict(passed=True,qualified=qualified,counts=counts,independent_SciPy_projections=72,all_numerically_qualified=bool(numeric),
        max_absolute_duality_gap=maxgap,max_projection_difference=maxdifference,max_stationarity_error=maxstationarity,
        new_source_forward=0,GPU_hours=0,new_OOF_rows=0,new_efficiency_heads=0,optimistic_diagnostic_only=True,deployable=False,
        outer_test_metrics_used=False,lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-start)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0),f,indent=2)
