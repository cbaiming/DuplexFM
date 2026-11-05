"""Independent quadratic expansion, deployment and reused-control audit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4';Q=R.parent/'round5'
sp=importlib.util.spec_from_file_location('_r28_audit4',P/'verify.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close


def metric(y,p):
    yc=y-y.mean();pc=p-p.mean();den=np.sqrt((yc@yc)*(pc@pc))
    return dict(PCC=float(yc@pc/den) if den>1e-14 else -2.,MSE=float(np.mean((y-p)**2)))


def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for name,h in lock['dependencies'].items():assert sha(Path(name))==h,name
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json')
    assert d['new_source_forward']==d['new_PCA']==d['GPU_hours']==0 and not d['outer_test_metrics_used']
    assert all(t['num_threads']==1 for t in d['threadpool'])
    check(R/'pilot',d['pilot_files_sha256']);assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    executions=[read(p) for p in R.glob('execution_*.json')];assert len(executions)==1 and executions[0]['returncode']==0
    plan=read(P/'plan.json');assert read(Q/'plan.json')['folds']==plan['folds']
    detail=table(R/'diagnostic_rows.csv');count=coverage=tdcount=tdcoverage=0
    maxcoef=maxpred=maxnormal=maxTDnormal=maxTDpred=0.;records=[];all_numeric=True
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids']
        assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        rows={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')}
        td=np.array([[float(v) for v in rows[s]['td'].split(',')] for s in ids]);y=np.array([float(rows[s]['label']) for s in ids])
        tdroot=Q/'results/folds'/setting/'fold0'/'TDQuadratic';tdcomplete=read(tdroot/'complete.json')
        check(tdroot,tdcomplete['files_sha256']);assert tdcomplete['lock_sha256']==sha(Q/'lock.json')
        tt=read(tdroot/'transforms.json')
        with np.load(tdroot/'arrays.npz',allow_pickle=False) as a:
            tdref=a['inner_oof'][:,18].copy();tdcoef=a['inner_coef'][:,18].copy();tdcenter=a['inner_intercept'].copy()
        tdcover=[]
        for j,split in enumerate(item['inner']):
            tr,va=np.array(split['train']),np.array(split['valid']);st=tt['inner'][j]
            assert st['kind']=='td_quadratic'
            mean,sd=old.scale(td[tr],False);close(mean,st['projection']['td_mean'],1e-11);close(sd,st['projection']['td_scale'],1e-11)
            z=(td-mean)/sd;qq=np.stack([z[:,i]*z[:,k] for i in range(24) for k in range(i,24)],axis=1)
            qm,qs=old.scale(qq[tr],False);close(qm,st['second_mean'],1e-10);close(qs,st['second_scale'],1e-10)
            x=np.c_[z,(qq-qm)/qs*np.sqrt(16/300)]
            close(tdcenter[j],y[tr].mean(),1e-12)
            normal=float(abs((x[tr].T@x[tr]+100*np.eye(324))@tdcoef[j]-x[tr].T@(y[tr]-tdcenter[j])).max())
            assert normal<=1e-9*(1+abs(x[tr].T@(y[tr]-tdcenter[j])).max())
            pp=x[va]@tdcoef[j]+tdcenter[j];close(pp,tdref[va],1e-10)
            maxTDnormal=max(maxTDnormal,normal);maxTDpred=max(maxTDpred,float(abs(pp-tdref[va]).max()))
            tdcover+=va.tolist();tdcount+=1
        assert sorted(tdcover)==list(range(len(ids)));tdcoverage+=len(ids)
        for source in old.SOURCES:
            folder=old.EXPORT/source;assert d['source_identities'][source]==read(folder/'complete.json')['source_identity']
            meta=table(folder/'row_identity.csv');idx={r['sample_id']:i for i,r in enumerate(meta)}
            for s in ids:assert all(rows[s][k]==meta[idx[s]][k] for k in ('sample_id','cohort','guide','target57'))
            rawsource=np.load(folder/'source.npy',allow_pickle=False)[[idx[s] for s in ids]].astype(float)
            raw=np.c_[td,rawsource];pf=P/'results/folds'/setting/'fold0'/source/'transforms.json';states=read(pf)
            with np.load(pf.parent/'arrays.npz',allow_pickle=False) as a:ref=a['inner_oof'][:,18].copy()
            prediction=np.full(len(ids),np.nan);covered=[];numeric_blocks=[]
            for j,split in enumerate(item['inner']):
                tr,va=np.array(split['train']),np.array(split['valid']);z=old.project(raw[tr],raw,states['inner'][j])
                out=R/'pilot'/setting/source/f'inner{j}';c=read(out/'complete.json');check(out,c['files_sha256'])
                assert c['lock_sha256']==sha(R/'lock.json');state=read(out/'state.json');t=read(out/'map.json')
                assert (state['setting'],state['source'],state['inner'])==(setting,source,j)
                assert state['train_indices']==tr.tolist() and state['valid_indices']==va.tolist()
                assert state['reference_transform_sha256']==sha(pf) and state['TD_reference_sha256']==sha(tdroot/'arrays.npz')
                pairs=[(i,k) for i in range(16) for k in range(i,16)]
                assert t['source_dimension']==16 and t['pairs']==[list(p) for p in pairs] and t['factor']==float(np.sqrt(16/136))
                q=np.stack([z[:,24+i]*z[:,24+k] for i,k in pairs],axis=1)
                mean,sd=old.scale(q[tr],False);close(mean,t['mean'],1e-10);close(sd,t['scale'],1e-10)
                x=np.c_[z,(q-mean)/sd*np.sqrt(16/136)]
                with np.load(out/'model.npz',allow_pickle=False) as a:
                    b=a['beta'].copy();center=float(a['intercept']);saved=a['prediction'].copy()
                assert b.shape==(176,);close(center,y[tr].mean(),1e-12)
                h=x[tr].T@x[tr]+100*np.eye(176);target=x[tr].T@(y[tr]-center)
                normal=float(abs(h@b-target).max());limit=1e-9*(1+float(abs(target).max()));numeric=normal<=limit
                independent=np.linalg.lstsq(np.r_[x[tr],10*np.eye(176)],np.r_[y[tr]-center,np.zeros(176)],rcond=None)[0]
                close(independent,b,1e-9);maxcoef=max(maxcoef,float(abs(b-independent).max()))
                residual=y[tr]-center-x[tr]@b;objective=float(residual@residual+100*(b@b));cert=state['certificate']
                assert cert['alpha']==100 and cert['features']==176 and cert['numerically_qualified']==numeric
                for key,value in [('normal_error',normal),('normal_limit',limit),('objective',objective)]:close(cert[key],value,1e-10)
                pp=x[va]@b+center;close(saved,pp,1e-10)
                weights=b[40:]*np.sqrt(16/136)/sd;matrix=np.zeros((16,16))
                for w,(i,k) in zip(weights,pairs):
                    matrix[i,k]+=w if i==k else w/2
                    if i!=k:matrix[k,i]+=w/2
                deployed=z[va]@b[:40]+np.sum((z[va,24:]@matrix)*z[va,24:],axis=1)+center-mean@weights
                close(pp,deployed,1e-10);maxpred=max(maxpred,float(abs(pp-saved).max()),float(abs(pp-deployed).max()))
                maxnormal=max(maxnormal,normal);all_numeric &= numeric;numeric_blocks.append(numeric)
                prediction[va]=saved;covered+=va.tolist();count+=1
            assert sorted(covered)==list(range(len(ids))) and np.isfinite(prediction).all()
            part=[r for r in detail if r['setting']==setting and r['source']==source];assert [r['sample_id'] for r in part]==ids
            for key,values in [('label',y),('prediction',prediction),('reference_prediction',ref),('TD_quadratic_prediction',tdref)]:close([float(r[key]) for r in part],values,0)
            coverage+=len(part);m,rr,tm=metric(y,prediction),metric(y,ref),metric(y,tdref)
            delta=m['PCC']-rr['PCC'];reduction=100*(1-m['MSE']/rr['MSE'])
            rec=next(r for r in d['records'] if r['setting']==setting and r['source']==source);assert rec['n']==len(ids)
            for key,values in [('metrics',m),('reference',rr),('TD_quadratic_metrics',tm)]:
                for name,value in values.items():close(rec[key][name],value,1e-11)
            close(rec['delta_PCC_vs_R4'],delta,1e-11);close(rec['MSE_reduction_vs_R4_pct'],reduction,1e-10)
            for j,split in enumerate(item['inner']):
                va=np.array(split['valid']);bl=rec['inner_blocks'][j];assert bl['inner']==j and bl['n']==len(va) and bl['numerically_qualified']==numeric_blocks[j]
                for key,values in [('metrics',metric(y[va],prediction[va])),('reference',metric(y[va],ref[va])),('TD_quadratic_metrics',metric(y[va],tdref[va]))]:
                    for name,value in values.items():close(bl[key][name],value,1e-11)
            records.append(dict(setting=setting,source=source,delta=delta,reduction=reduction,PCC=m['PCC'],MSE=m['MSE'],TD_PCC=tm['PCC'],TD_MSE=tm['MSE']))
    assert count==45==d['new_heads']==d['new_maps'] and coverage==6897==d['new_internal_OOF_rows']==len(detail)
    assert tdcount==15==d['reused_TD_quadratic_heads'] and tdcoverage==2299==d['reused_unique_TD_OOF_rows'] and d['reused_linear_heads']==45
    core=[r for r in records if r['source']=='core_real']
    perf=sum(r['delta']>=.005 and r['reduction']>=0 for r in core)>=2 and all(r['delta']>=-.005 for r in core)
    paired=[]
    for r in core:
        dp=r['PCC']-r['TD_PCC'];red=100*(1-r['MSE']/r['TD_MSE']);paired.append((dp,red))
        row=next(a for a in d['paired_TD_quadratic'] if a['setting']==r['setting'])
        close(row['delta_PCC'],dp,1e-11);close(row['MSE_reduction_pct'],red,1e-10)
    inc=sum(a>=0 for a,b in paired)>=2 and np.mean([b for a,b in paired])>=0;qualified=bool(perf and inc and all_numeric)
    assert d['performance_gate']==perf and d['source_increment_gate']==inc and d['qualified']==qualified and d['all_numerically_qualified']==all_numeric
    result=dict(passed=True,qualified=qualified,verified_new_heads=count,independent_augmented_lstsq=count,verified_reused_TD_heads=tdcount,
        internal_OOF_rows=coverage,unique_reused_TD_rows=tdcoverage,max_normal_equation_error=maxnormal,
        max_independent_coefficient_difference=maxcoef,max_prediction_difference=maxpred,max_TD_normal_error=maxTDnormal,
        max_TD_prediction_difference=maxTDpred,all_numerically_qualified=bool(all_numeric),
        new_source_forward=0,new_PCA=0,GPU_hours=0,outer_test_metrics_used=False,
        lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-started)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    started=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-started,GPU_hours=0,new_source_forward=0),f,indent=2)
