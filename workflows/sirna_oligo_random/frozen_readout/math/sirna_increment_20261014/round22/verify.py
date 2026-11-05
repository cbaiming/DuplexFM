"""Independent 216D equations and all four-setting selections/metrics."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4'
sp=importlib.util.spec_from_file_location('_r22_verify4',P/'verify.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close
ALPHAS=(.1,1.,10.,100.,1000.);MODES=('dense','sparse');SOURCES=old.SOURCES

def raw_project(raw,t):
    return (raw-np.r_[t['td_mean'],t['src_mean']])/np.r_[t['td_scale'],np.asarray(t['src_scale'])*np.sqrt(12.)]

def check_head(z,y,b,c,cert,alpha,mode):
    close(c,y.mean(),1e-12);g=z.T@z+alpha*np.eye(216);q=z.T@(y-y.mean())
    td=np.linalg.solve(g[:24,:24],q[:24]);cq=q[24:]-g[24:,:24]@td
    lmax=float(max(abs(cq)));lam=.1*lmax if mode=='sparse' else 0.;lim=1e-8*(1+lmax)
    close(cert['lambda_max'],lmax,1e-9);close(cert['lambda_value'],lam,1e-10);close(cert['KKT_threshold'],lim,1e-12)
    grad=g@b-q;tail=np.where(b[24:]!=0,grad[24:]+lam*np.sign(b[24:]),np.sign(grad[24:])*np.maximum(abs(grad[24:])-lam,0))
    err=float(max(abs(tail)));td_err=float(max(abs(grad[:24])))
    close(cert['source_KKT'],err,1e-9);close(cert['TD_KKT'],td_err,1e-9)
    assert max(err,td_err)<=lim+1e-10 and cert['numerically_qualified'] and not cert['warnings'] and cert['iterations']<100000
    assert cert['alpha']==alpha and cert['mode']==mode and cert['source_nonzero']==int(np.count_nonzero(b[24:]))
    close(cert['suboptimality_upper_bound'],np.r_[grad[:24],tail]@np.r_[grad[:24],tail]/(2*alpha),1e-10)
    dense_error=0.
    if mode=='dense':
        direct=np.linalg.solve(g,q);close(b,direct,1e-8);dense_error=float(max(abs(b-direct)));assert cert['iterations']==0
    return max(err,td_err),dense_error

def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    executions=[read(p) for p in R.glob('execution_*.json')];assert len(executions)==1 and executions[0]['returncode']==0
    complete=read(R/'results/complete.json');assert complete['lock_sha256']==sha(R/'lock.json');check(R/'results',complete['files_sha256'])
    assert complete['inner_fits']==2400 and complete['final_heads']==96 and complete['rows']==22788
    assert complete['source_forward']==complete['new_PCA']==complete['GPU_hours']==0 and all(t['num_threads']==1 for t in complete['threadpool'])
    plan=read(R/'plan.json');assert plan['folds']==read(P/'plan.json')['folds'] and plan['alphas']==list(ALPHAS) and plan['primary']=='core_real_sparse'
    rows=sum([table(old.BENCH/'data'/f'{s}.csv') for s in old.S[:3]],[]);assert len(rows)==3329
    idx={r['sample_id']:i for i,r in enumerate(rows)};td=np.array([[float(v) for v in r['td'].split(',')] for r in rows]);xx={}
    y=np.array([float(r['label']) for r in rows]);binary=np.array([int(r['y']) for r in rows])
    for source in SOURCES:
        folder=old.EXPORT/source;meta=table(folder/'row_identity.csv');imap={r['sample_id']:i for i,r in enumerate(meta)}
        for r in rows:assert all(r[k]==meta[imap[r['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
        xx[source]=np.c_[td,np.load(folder/'source.npy',allow_pickle=False)[[imap[r['sample_id']] for r in rows]].astype(float)]
    predictions=table(R/'results/oof_predictions.csv');summary=table(R/'results/summary.csv');selection_rows=table(R/'results/selection.csv')
    assert len(predictions)==22788 and len(summary)==24 and len(selection_rows)==96
    count=0;final=0;maxkkt=0.;maxdense=0.;maxpred=0.;maxTD=0.;canonical={};supports=[]
    for item in plan['folds']:
        setting,fold=item['setting'],item['fold'];fi=np.array([idx[s] for s in item['fit_ids']]);ti=np.array([idx[s] for s in item['test_ids']])
        assert set(fi).isdisjoint(ti)
        for source in SOURCES:
            out=R/'results/folds'/setting/f'fold{fold}'/source;ref=P/'results/folds'/setting/f'fold{fold}'/source
            receipt=read(out/'complete.json');assert receipt['lock_sha256']==sha(R/'lock.json');check(out,receipt['files_sha256'])
            reference=read(out/'references.json');assert reference['transform_sha256']==sha(ref/'transforms.json')
            assert reference['source_identity']==read(old.EXPORT/source/'complete.json')['source_identity']
            t=read(ref/'transforms.json');a=np.load(out/'arrays.npz',allow_pickle=False);sel=read(out/'selection.json');x=xx[source]
            oof=np.full((len(fi),2,5),np.nan);covered=[]
            with np.load(ref/'arrays.npz',allow_pickle=False) as r4:tdcoef=r4['inner_coef'][:,:5,:24].copy();tdcenter=r4['inner_intercept'].copy();tdref=r4['inner_oof'][:,:5].copy()
            for j,split in enumerate(item['inner']):
                tr=np.array(split['train']);va=np.array(split['valid']);covered+=va.tolist();old.project(x[fi[tr]],x[fi[tr]],t['inner'][j])
                z=raw_project(x[fi[tr]],t['inner'][j]);zv=raw_project(x[fi[va]],t['inner'][j]);close(z.mean(0),np.zeros(216),1e-10)
                tdp=zv[:,:24]@tdcoef[j].T+tdcenter[j];close(tdp,tdref[va],1e-10);maxTD=max(maxTD,float(max(abs(tdp-tdref[va]).ravel())))
                for m,mode in enumerate(MODES):
                    for k,alpha in enumerate(ALPHAS):
                        b,c=a['inner_coef'][j,m,k],float(a['inner_intercept'][j,m,k]);cert=read(out/f'inner{j}_{mode}_alpha{k}_certificate.json')
                        err,de=check_head(z,y[fi[tr]],b,c,cert,alpha,mode);maxkkt=max(maxkkt,err);maxdense=max(maxdense,de);count+=1
                        oof[va,m,k]=zv@b+c
            assert sorted(covered)==list(range(len(fi)));close(oof,a['inner_oof'],1e-9)
            assert sel['fit_ids']==item['fit_ids'] and sel['test_ids']==item['test_ids'] and sel['outer_validation_unused']
            for m,mode in enumerate(MODES):
                canonical_oof=a['inner_oof'][:,m,:];yc=y[fi]-y[fi].mean();pc=canonical_oof-canonical_oof.mean(0)
                pcc=yc@pc/np.sqrt((yc@yc)*np.sum(pc*pc,axis=0));mse=np.mean((canonical_oof-y[fi,None])**2,axis=0)
                close(sel['PCC'][m],pcc,1e-10);close(sel['MSE'][m],mse,1e-11)
                k=min(range(5),key=lambda k:(-pcc[k],mse[k],-ALPHAS[k],k));assert sel['selected'][mode]==k
                sr=next(r for r in selection_rows if r['setting']==setting and int(r['fold'])==fold and r['condition']==source+'_'+mode)
                assert int(sr['candidate'])==k and float(sr['alpha'])==ALPHAS[k];close(float(sr['inner_PCC']),pcc[k],1e-10)
                old.project(x[fi],x[fi],t['final']);z=raw_project(x[fi],t['final']);zt=raw_project(x[ti],t['final'])
                b,c=a['final_coef'][m],float(a['final_intercept'][m]);cert=read(out/f'final_{mode}_certificate.json')
                err,de=check_head(z,y[fi],b,c,cert,ALPHAS[k],mode);maxkkt=max(maxkkt,err);maxdense=max(maxdense,de);final+=1
                pp=zt@b+c;close(pp,a['test_pred'][:,m],1e-9);maxpred=max(maxpred,float(max(abs(pp-a['test_pred'][:,m]))))
                close(x[ti]@a['deployment_coef'][m]+a['deployment_intercept'][m],pp,1e-9)
                assert (out/'selection.json').stat().st_mtime_ns<=(out/f'final_{mode}_certificate.json').stat().st_mtime_ns
                part=[r for r in predictions if r['setting']==setting and int(r['fold'])==fold and r['condition']==source+'_'+mode]
                assert [r['sample_id'] for r in part]==item['test_ids'];p=np.array([float(r['prediction']) for r in part]);close(p,pp,1e-9)
                for r,i in zip(part,ti):assert float(r['label'])==y[i] and int(r['binary'])==binary[i]
                canonical.setdefault((setting,source+'_'+mode),[]).extend(zip(ti,p))
                supports.append(dict(setting=setting,fold=fold,source=source,mode=mode,alpha=ALPHAS[k],source_nonzero=cert['source_nonzero'],iterations=cert['iterations']))
            a.close()
        print('VERIFIED',setting,fold,flush=True)
    assert count==2400 and final==96
    baseline={(r['setting'],r['condition']):r for r in table(old.OLD/'results/summary.csv')}
    for r in summary:
        pairs=canonical[r['setting'],r['condition']];ii=np.array([i for i,p in pairs]);p=np.array([p for i,p in pairs])
        cohort='Mixset' if r['setting']=='Inter_Hu_to_Mix' else r['setting'];expected={i for i,rw in enumerate(rows) if rw['cohort']==cohort}
        assert len(ii)==len(expected) and set(ii)==expected
        metrics=old.metric(y[ii],p,binary[ii])
        for k,v in metrics.items():close(float(r[k]),v,1e-9)
        base=baseline[r['setting'],'TD24_Ridge_CV'];bp=float(base['PCC']);close(float(r['TD24_PCC']),bp,0)
        close(float(r['PCC_delta']),metrics['PCC']-bp,1e-10);close(float(r['PCC_relative_gain_pct']),100*(metrics['PCC']/bp-1),1e-9)
        close(float(r['MSE_relative_reduction_pct']),100*(1-metrics['MSE']/float(base['MSE'])),1e-9)
    for r in read(R/'results/goal_summary.json'):
        part=[s for s in summary if s['condition']==r['condition']];gain=float(np.mean([float(s['PCC_relative_gain_pct']) for s in part]))
        close(r['mean_relative_PCC_gain_pct'],gain,1e-9);assert r['reaches_development_5pct']==(gain>=5) and r['primary']==(r['condition']=='core_real_sparse')
        assert r['negative_settings']==[s['setting'] for s in part if float(s['PCC_delta'])<0]
    result=dict(passed=True,inner_heads=count,final_heads=final,independent_dense_solves=1248,OOF_rows=22788,
        max_KKT=maxkkt,max_dense_coef_difference=maxdense,max_prediction_difference=maxpred,max_TD_baseline_difference=maxTD,
        support_records=support,new_source_forward=0,GPU_hours=0,lock_sha256=sha(R/'lock.json'),complete_sha256=sha(R/'results/complete.json'),CPU_wall_seconds=time.monotonic()-started)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='support_records'}),flush=True)

if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0),f,indent=2)
