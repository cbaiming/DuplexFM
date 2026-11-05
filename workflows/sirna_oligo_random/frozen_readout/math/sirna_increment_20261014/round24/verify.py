"""Independent augmented least-squares and convex primal/dual certificates."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4';Q=R.parent/'round23'
sp=importlib.util.spec_from_file_location('_r24_audit4',P/'verify.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close


def metric(y,p):
    return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))


def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for name,h in lock['dependencies'].items():assert sha(Path(name))==h,name
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json')
    assert d['new_source_forward']==d['new_PCA_fits']==d['new_KMeans_fits']==d['GPU_hours']==0
    assert not d['outer_test_metrics_used'] and all(t['num_threads']==1 for t in d['threadpool'])
    check(R/'pilot',d['pilot_files_sha256']);assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    ex=[read(p) for p in R.glob('execution_*.json')];assert len(ex)==1 and ex[0]['returncode']==0
    plan=read(P/'plan.json');details=table(R/'diagnostic_rows.csv');count=coverage=solves=0
    maxcoef=maxpred=maxnorm=maxgap=0.;all_numeric=True;records=[];dualweights=[]
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids']
        assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        rows={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')}
        td=np.array([[float(v) for v in rows[s]['td'].split(',')] for s in ids]);y=np.array([float(rows[s]['label']) for s in ids])
        for source in (*old.SOURCES,'TD_only'):
            src='core_real' if source=='TD_only' else source;folder=old.EXPORT/src
            identity=read(folder/'complete.json');assert d['source_identities'][src]==identity['source_identity']
            meta=table(folder/'row_identity.csv');index={r['sample_id']:i for i,r in enumerate(meta)}
            for s in ids:
                assert all(rows[s][k]==meta[index[s]][k] for k in ('sample_id','cohort','guide','target57'))
            rawsource=np.load(folder/'source.npy',allow_pickle=False)[[index[s] for s in ids]].astype(float)
            raw=np.c_[td,rawsource];pf=P/'results/folds'/setting/'fold0'/src/'transforms.json';states=read(pf)
            with np.load(P/'results/folds'/setting/'fold0'/src/'arrays.npz',allow_pickle=False) as a:ref=a['inner_oof'][:,3 if source=='TD_only' else 18].copy()
            projections=[];assignments=[]
            for j,split in enumerate(item['inner']):
                tr,va=np.array(split['train']),np.array(split['valid']);st=states['inner'][j]
                if source=='TD_only':
                    mu,sd=old.scale(td[tr],False);close(st['td_mean'],mu,1e-11);close(st['td_scale'],sd,1e-11);z=(td-mu)/sd
                else:z=old.project(raw[tr],raw,st)
                gp=Q/'gates'/setting/f'inner{j}'/'assignments.npz';gg=read(gp.parent/'gate.json')
                assert gg['train_indices']==tr.tolist() and gg['valid_indices']==va.tolist()
                zz=(td-gg['TD_mean'])/gg['TD_scale'];centers=np.array(gg['centers'])
                a=np.argmin(np.sum((zz[:,None,:]-centers[None,:,:])**2,axis=2),axis=1)
                with np.load(gp,allow_pickle=False) as saved:np.testing.assert_array_equal(a,saved['all_fit_assignment'])
                projections.append(z);assignments.append(a)
            for mode in ('balanced','robust'):
                prediction=np.full(len(ids),np.nan);cover=[];blockweights=[];blocknumeric=[]
                for j,split in enumerate(item['inner']):
                    tr,va=np.array(split['train']),np.array(split['valid']);z=projections[j];group=assignments[j][tr]
                    out=R/'pilot'/setting/source/mode/f'inner{j}';c=read(out/'complete.json');check(out,c['files_sha256']);assert c['lock_sha256']==sha(R/'lock.json')
                    s=read(out/'state.json');cert=s['certificate'];assert s['train_indices']==tr.tolist() and s['valid_indices']==va.tolist()
                    assert s['reference_transform_sha256']==sha(pf) and s['group_assignments_sha256']==sha(Q/'gates'/setting/f'inner{j}'/'assignments.npz')
                    assert (s['setting'],s['source'],s['mode'],s['inner'])==(setting,source,mode,j)
                    with np.load(out/'model.npz',allow_pickle=False) as model:
                        b=np.r_[float(model['intercept']),model['beta']];savedprediction=model['prediction'].copy()
                    n=len(tr);p=z.shape[1];x=np.c_[np.ones(n),z[tr]];q=cert['group0_weight'];assert 0<=q<=1
                    counts=np.bincount(group,minlength=2);assert counts.min()>0
                    np.testing.assert_array_equal(cert['training_counts'],counts);assert cert['alpha']==100 and cert['n']==n and cert['features']==p and cert['mode']==mode
                    if mode=='balanced':assert q==.5 and cert['location']=='fixed_equal_group_weight' and cert['weighted_ridge_solves']==1
                    w=np.where(group==0,n*q/counts[0],n*(1-q)/counts[1]);close(w.sum(),n,1e-11)
                    penalty=np.r_[0.,np.full(p,100.)];augment=np.diag(np.sqrt(penalty))
                    replay=np.linalg.lstsq(np.r_[np.sqrt(w)[:,None]*x,augment],np.r_[np.sqrt(w)*y[tr],np.zeros(p+1)],rcond=None)[0]
                    close(b,replay,1e-9);maxcoef=max(maxcoef,float(abs(b-replay).max()))
                    h=x.T@(w[:,None]*x)+np.diag(penalty);v=x.T@(w*y[tr]);normal=float(abs(h@b-v).max());limit=1e-9*(1+float(abs(v).max()))
                    residual=x@b-y[tr];loss=np.array([n*np.mean(residual[group==k]**2) for k in (0,1)])
                    reg=float(np.dot(penalty,b*b));weighted=float(q*loss[0]+(1-q)*loss[1]+reg);worst=float(max(loss)+reg)
                    gap=worst-weighted if mode=='robust' else 0.;gaplimit=1e-9*(1+abs(worst));numeric=bool(normal<=limit and -gaplimit<=gap<=gaplimit)
                    for key,value in [('group_scaled_losses',loss),('regularizer',reg),('weighted_objective',weighted),('worst_objective',worst),
                                      ('duality_gap',gap),('gap_limit',gaplimit),('normal_equation_error',normal),('normal_equation_limit',limit)]:close(cert[key],value,1e-10)
                    if mode=='robust':
                        assert cert['weighted_ridge_solves']>=2
                        if q==0:assert loss[0]-loss[1]<=gaplimit and cert['location']=='lower_boundary'
                        elif q==1:assert loss[0]-loss[1]>=-gaplimit and cert['location']=='upper_boundary'
                        else:assert abs(loss[0]-loss[1])<=1e-7*(1+abs(worst)) and cert['location']=='interior'
                    assert cert['numerically_qualified']==numeric;all_numeric &= numeric
                    pred=np.c_[np.ones(len(va)),z[va]]@b;close(pred,savedprediction,1e-10);maxpred=max(maxpred,float(abs(pred-savedprediction).max()))
                    prediction[va]=savedprediction;cover+=va.tolist();count+=1;solves+=cert['weighted_ridge_solves']
                    blockweights.append(q);blocknumeric.append(numeric);maxnorm=max(maxnorm,normal);maxgap=max(maxgap,abs(gap)/(1+abs(worst)))
                    dualweights.append(dict(setting=setting,source=source,mode=mode,inner=j,group0_weight=q,location=cert['location']))
                assert sorted(cover)==list(range(len(ids))) and np.isfinite(prediction).all()
                part=[r for r in details if r['setting']==setting and r['source']==source and r['mode']==mode];assert [r['sample_id'] for r in part]==ids
                close([float(r['label']) for r in part],y,0);close([float(r['prediction']) for r in part],prediction,0);close([float(r['reference_prediction']) for r in part],ref,0)
                coverage+=len(part);m,rr=metric(y,prediction),metric(y,ref);delta=m['PCC']-rr['PCC'];red=100*(1-m['MSE']/rr['MSE'])
                rec=next(r for r in d['records'] if r['setting']==setting and r['source']==source and r['mode']==mode)
                assert rec['n']==len(ids)
                for k in m:close(rec['metrics'][k],m[k],1e-11);close(rec['reference'][k],rr[k],1e-11)
                close(rec['delta_PCC_vs_R4'],delta,1e-11);close(rec['MSE_reduction_vs_R4_pct'],red,1e-10)
                for j,split in enumerate(item['inner']):
                    va=np.array(split['valid']);bl=rec['inner_blocks'][j];assert bl['inner']==j and bl['n']==len(va) and bl['numerically_qualified']==blocknumeric[j]
                    close(bl['group0_weight'],blockweights[j],0)
                    for key,expected in [('metrics',metric(y[va],prediction[va])),('reference',metric(y[va],ref[va]))]:
                        for k in expected:close(bl[key][k],expected[k],1e-11)
                records.append(dict(setting=setting,source=source,mode=mode,delta=delta,reduction=red,PCC=m['PCC'],MSE=m['MSE']))
    assert count==120==d['new_efficiency_heads'] and coverage==18392==len(details) and solves==d['weighted_ridge_solves']
    core=[r for r in records if r['source']=='core_real' and r['mode']=='robust']
    perf=sum(r['delta']>=.005 and r['reduction']>=0 for r in core)>=2 and all(r['delta']>=-.005 for r in core)
    paired=[]
    for a in core:
        b=next(r for r in records if r['source']=='core_real' and r['mode']=='balanced' and r['setting']==a['setting'])
        red=100*(1-a['MSE']/b['MSE']);dp=a['PCC']-b['PCC'];paired.append(red)
        item=next(r for r in d['paired_core_records'] if r['setting']==a['setting']);close(item['delta_PCC'],dp,1e-11);close(item['MSE_reduction_pct'],red,1e-10)
    pg=sum(v>=0 for v in paired)>=2 and np.mean(paired)>=0;qualified=bool(perf and pg and all_numeric)
    assert d['performance_gate']==perf and d['paired_gate']==pg and d['qualified']==qualified and d['all_numerically_qualified']==all_numeric
    result=dict(passed=True,qualified=qualified,verified_heads=count,independent_augmented_linear_audits=count,internal_OOF_rows=coverage,
        science_weighted_ridge_solves=solves,max_independent_coefficient_difference=maxcoef,max_prediction_difference=maxpred,
        max_normal_equation_error=maxnorm,max_relative_duality_gap=maxgap,all_numerically_qualified=bool(all_numeric),dual_weights=dualweights,
        new_source_forward=0,new_PCA_fits=0,new_KMeans_fits=0,GPU_hours=0,outer_test_metrics_used=False,
        lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-started)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='dual_weights'}),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0),f,indent=2)
