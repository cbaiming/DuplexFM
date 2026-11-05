"""Independent finite split, normal-equation and one-hot deployment audit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4'
sp=importlib.util.spec_from_file_location('_r27_audit4',P/'verify.py')
old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close


def metric(y,p):
    yc=y-y.mean();pc=p-p.mean();den=np.sqrt((yc@yc)*(pc@pc))
    return dict(PCC=float(yc@pc/den) if den>1e-14 else -2.,MSE=float(np.mean((y-p)**2)))


def main():
    started=time.monotonic()
    assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for name,h in lock['dependencies'].items():assert sha(Path(name))==h,name
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json')
    assert d['new_source_forward']==d['new_PCA']==d['GPU_hours']==0 and not d['outer_test_metrics_used']
    assert all(t['num_threads']==1 for t in d['threadpool'])
    check(R/'pilot',d['pilot_files_sha256']);assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    ex=[read(p) for p in R.glob('execution_*.json')];assert len(ex)==1 and ex[0]['returncode']==0
    plan=read(P/'plan.json');details=table(R/'diagnostic_rows.csv')
    count=coverage=solves=reused=certificates=0;maxcoef=maxpred=maxnormal=0.
    records=[];gates=[];all_numeric=True
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids']
        assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        rows={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')}
        td=np.array([[float(v) for v in rows[s]['td'].split(',')] for s in ids])
        y=np.array([float(rows[s]['label']) for s in ids])
        for source in (*old.SOURCES,'TD_only'):
            src='core_real' if source=='TD_only' else source;folder=old.EXPORT/src
            assert d['source_identities'][src]==read(folder/'complete.json')['source_identity']
            meta=table(folder/'row_identity.csv');idx={r['sample_id']:i for i,r in enumerate(meta)}
            for s in ids:assert all(rows[s][k]==meta[idx[s]][k] for k in ('sample_id','cohort','guide','target57'))
            source_array=np.load(folder/'source.npy',allow_pickle=False)[[idx[s] for s in ids]].astype(float)
            raw=np.c_[td,source_array];pf=P/'results/folds'/setting/'fold0'/src/'transforms.json'
            states=read(pf);af=pf.parent/'arrays.npz'
            with np.load(af,allow_pickle=False) as a:
                index=3 if source=='TD_only' else 18
                ref=a['inner_oof'][:,index].copy();reference_beta=a['inner_coef'][:,index].copy()
            prediction=np.full(len(ids),np.nan);covered=[];block_states=[]
            for j,split in enumerate(item['inner']):
                tr,va=np.array(split['train']),np.array(split['valid']);st=states['inner'][j]
                if source=='TD_only':
                    mu,sd=old.scale(td[tr],False);close(mu,st['td_mean'],1e-11);close(sd,st['td_scale'],1e-11);z=(td-mu)/sd
                else:z=old.project(raw[tr],raw,st)
                n,p=len(tr),z.shape[1];q=24 if p==24 else 16;w=z if p==24 else z[:,24:]
                folder2=R/'pilot'/setting/source/f'inner{j}';c=read(folder2/'complete.json')
                assert c['lock_sha256']==sha(R/'lock.json');check(folder2,c['files_sha256']);s=read(folder2/'state.json')
                assert (s['setting'],s['source'],s['inner'])==(setting,source,j)
                assert s['train_indices']==tr.tolist() and s['valid_indices']==va.tolist()
                assert s['reference_transform_sha256']==sha(pf) and s['reference_arrays_sha256']==sha(af)
                with np.load(folder2/'model.npz',allow_pickle=False) as a:
                    bb=a['candidate_beta'].copy();means=a['interaction_mean'].copy();center=float(a['intercept'])
                    saved_prediction=a['prediction'].copy();saved_route=a['onehot_route'].copy()
                close(center,y[tr].mean(),1e-12)
                expected=[(-1,-1,0.,0.,[n,0])];minimum=max(10,int(np.ceil(.2*n)))
                for feature in range(24):
                    thresholds=np.percentile(z[tr,feature],[25.,50.,75.],method='linear');seen=set()
                    for qi,t in enumerate(thresholds):
                        if float(t) in seen:continue
                        seen.add(float(t));mask=z[tr,feature]>t;counts=[int((~mask).sum()),int(mask.sum())]
                        if min(counts)<minimum:continue
                        expected.append((feature,qi,float(t),float(mask.mean()),counts))
                assert len(expected)==len(s['candidates']) and bb.shape==(len(expected),p+q) and means.shape==(len(expected),q)
                assert s['new_linear_solves']==len(expected)-1 and s['reference_reused']==1
                objectives=[];all_block=True;selected=int(s['selected']);selected_prediction=None
                penalty=np.r_[np.full(p,100.),np.full(q,400.)]
                for k,(feature,qi,t,prob,counts) in enumerate(expected):
                    cert=s['candidates'][k]
                    assert cert['feature']==feature and cert['quantile_index']==qi and cert['counts']==counts
                    close(cert['threshold'],t,0);close(cert['right_fraction'],prob,0)
                    assert cert['reused_reference']==(k==0)
                    if k==0:
                        close(bb[k,:p],reference_beta[j,:p],0);assert np.count_nonzero(bb[k,p:])==0
                        interaction=np.zeros_like(w)
                    else:
                        interaction=((z[:,feature]>t).astype(float)-prob)[:,None]*w
                    mean=interaction[tr].mean(0);close(means[k],mean,1e-11)
                    x=np.c_[z[tr],interaction[tr]-mean];target=x.T@(y[tr]-center)
                    normal=float(abs((x.T@x+np.diag(penalty))@bb[k]-target).max())
                    limit=1e-9*(1+float(abs(target).max()));numeric=normal<=limit
                    residual=y[tr]-center-x@bb[k];objective=float(residual@residual+np.dot(penalty,bb[k]*bb[k]))
                    close(cert['normal_error'],normal,1e-11);close(cert['normal_limit'],limit,1e-11)
                    close(cert['objective'],objective,1e-11);assert cert['certified']==numeric
                    objectives.append(objective);all_block &= numeric;certificates+=1;maxnormal=max(maxnormal,normal)
                    if k==selected:
                        aug=np.diag(np.sqrt(penalty))
                        independent=np.linalg.lstsq(np.r_[x,aug],np.r_[y[tr]-center,np.zeros(p+q)],rcond=None)[0]
                        close(independent,bb[k],1e-9);maxcoef=max(maxcoef,float(abs(independent-bb[k]).max()))
                        pp=np.c_[z[va],interaction[va]-mean]@bb[k]+center
                        route=np.zeros(len(va),int) if feature==-1 else (z[va,feature]>t).astype(int)
                        left=bb[k,:p].copy();right=left.copy();localstart=0 if p==24 else 24
                        left[localstart:]-=prob*bb[k,p:];right[localstart:]+=(1-prob)*bb[k,p:]
                        deployed=np.where(route==0,z[va]@left,z[va]@right)+center-mean@bb[k,p:]
                        close(pp,deployed,1e-10);close(saved_prediction,pp,1e-10)
                        np.testing.assert_array_equal(saved_route,route)
                        selected_prediction=pp;maxpred=max(maxpred,float(abs(saved_prediction-pp).max()))
                best=min(range(len(expected)),key=lambda k:(round(objectives[k],10),expected[k][0],expected[k][1],k))
                assert best==selected and s['all_certified']==all_block and selected_prediction is not None
                prediction[va]=saved_prediction;covered+=va.tolist();count+=1;solves+=len(expected)-1;reused+=1;all_numeric &= all_block
                chosen=expected[selected];block_states.append(dict(feature=chosen[0],quantile_index=chosen[1],threshold=chosen[2],
                    valid_right_fraction=float(saved_route.mean()),candidates=len(expected),new_linear_solves=len(expected)-1,
                    all_numerically_qualified=bool(all_block)))
                gates.append(dict(setting=setting,source=source,inner=j,**block_states[-1]))
            assert sorted(covered)==list(range(len(ids))) and np.isfinite(prediction).all()
            part=[r for r in details if r['setting']==setting and r['source']==source]
            assert [r['sample_id'] for r in part]==ids
            close([float(r['label']) for r in part],y,0);close([float(r['prediction']) for r in part],prediction,0)
            close([float(r['reference_prediction']) for r in part],ref,0);coverage+=len(part)
            m,rr=metric(y,prediction),metric(y,ref);delta=m['PCC']-rr['PCC'];red=100*(1-m['MSE']/rr['MSE'])
            rec=next(r for r in d['records'] if r['setting']==setting and r['source']==source);assert rec['n']==len(ids)
            for key in m:close(rec['metrics'][key],m[key],1e-11);close(rec['reference'][key],rr[key],1e-11)
            close(rec['delta_PCC_vs_R4'],delta,1e-11);close(rec['MSE_reduction_vs_R4_pct'],red,1e-10)
            for j,split in enumerate(item['inner']):
                va=np.array(split['valid']);block=rec['inner_blocks'][j];assert block['inner']==j and block['n']==len(va)
                for key,value in block_states[j].items():close(block[key],value,1e-11)
                for key,values in [('metrics',metric(y[va],prediction[va])),('reference',metric(y[va],ref[va]))]:
                    for name,value in values.items():close(block[key][name],value,1e-11)
            records.append(dict(setting=setting,source=source,delta=delta,reduction=red,PCC=m['PCC'],MSE=m['MSE']))
    assert count==60==d['inner_estimators'] and coverage==9196==len(details)==d['new_internal_OOF_rows']
    assert reused==60==d['reused_null_heads'] and solves==d['new_linear_solves']<=4320 and certificates==solves+reused
    core=[r for r in records if r['source']=='core_real']
    perf=sum(r['delta']>=.005 and r['reduction']>=0 for r in core)>=2 and all(r['delta']>=-.005 for r in core)
    paired=[]
    for r in core:
        td=next(a for a in records if a['source']=='TD_only' and a['setting']==r['setting'])
        dp=r['PCC']-td['PCC'];red=100*(1-r['MSE']/td['MSE']);paired.append((dp,red))
        p=next(a for a in d['paired_gated_TD'] if a['setting']==r['setting'])
        close(p['delta_PCC'],dp,1e-11);close(p['MSE_reduction_pct'],red,1e-10)
    inc=sum(a>=0 for a,b in paired)>=2 and np.mean([b for a,b in paired])>=0
    qualified=bool(perf and inc and all_numeric)
    assert d['performance_gate']==perf and d['source_increment_gate']==inc and d['qualified']==qualified
    assert d['all_numerically_qualified']==all_numeric
    result=dict(passed=True,qualified=qualified,verified_estimators=count,new_science_linear_solves=solves,
        reused_null_heads=reused,candidate_normal_equations=certificates,independent_selected_lstsq=count,internal_OOF_rows=coverage,
        max_normal_equation_error=maxnormal,max_independent_coefficient_difference=maxcoef,max_prediction_difference=maxpred,
        all_numerically_qualified=bool(all_numeric),selected_gates=gates,new_source_forward=0,new_PCA=0,GPU_hours=0,
        outer_test_metrics_used=False,lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),
        CPU_wall_seconds=time.monotonic()-started)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='selected_gates'}),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0),f,indent=2)
