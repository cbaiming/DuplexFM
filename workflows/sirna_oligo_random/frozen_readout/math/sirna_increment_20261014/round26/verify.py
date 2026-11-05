"""Independent audit of every full-system certificate and four-setting output."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4';Q=R.parent/'round25';D=R.parent/'round22/results_v2'
sp=importlib.util.spec_from_file_location('_r26_verify4',P/'verify.py');old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close
ALPHAS=(.1,1.,10.,100.,1000.)


def project(x,t):
    return (x-np.r_[t['td_mean'],t['src_mean']])/np.r_[t['td_scale'],np.array(t['src_scale'])*np.sqrt(12.)]


def check_head(z,y,b,center,s,alpha,gram,rhs,reused=False):
    close(center,y.mean(),1e-12);g=gram+alpha*np.eye(216)
    td0=np.linalg.solve(g[:24,:24],rhs[:24]);coupling=np.linalg.solve(g[:24,:24],g[:24,24:])
    h=g[24:,24:]-g[24:,:24]@coupling;h=(h+h.T)/2;q=rhs[24:]-g[24:,:24]@td0
    critical=float(np.linalg.norm(q.reshape(6,32),2));lam=.1*critical
    close(s['lambda_critical'],critical,1e-9);close(s['lambda_value'],lam,1e-9)
    assert s['alpha']==alpha and s['lambda_ratio']==.1 and s['matrix_shape']==[6,32] and s['mode']=='native_trace'
    normal=float(abs((g@b-rhs)[:24]).max());assert normal<1e-8
    singular=np.linalg.svd(b[24:].reshape(6,32),compute_uv=False);close(s['singular_values'],singular,1e-10)
    assert s['rank']==int(sum(singular>1e-8))
    dual=q-h@b[24:];norm=float(np.linalg.norm(dual.reshape(6,32),2))
    if norm>lam:dual*=lam/norm
    close(s['dual'],dual,1e-8);assert np.linalg.norm(dual.reshape(6,32),2)<=lam+1e-9
    primal=.5*b@g@b-rhs@b+lam*singular.sum();dv=np.r_[np.zeros(24),dual]
    dualvalue=-.5*(rhs-dv)@np.linalg.solve(g,rhs-dv);gap=float(primal-dualvalue);constant=.5*rhs[:24]@td0
    close(s['primal_profile'],primal+constant,1e-9);close(s['dual_profile'],dualvalue+constant,1e-9)
    scale=max(1.,.5*q@np.linalg.solve(h,q));close(s['gap_scale'],scale,1e-9)
    assert gap>=-1e-8;close(s['gap'],max(0.,gap),1e-9)
    assert s['certified'] and s['gap']<=1e-9*s['gap_scale'] and gap<=1e-9*scale+1e-9
    cap=5000 if reused else 20000;assert 0<=s['iterations']<=cap
    if not reused:assert s['algorithm']=='strongly_convex_proximal_gradient' and s['maximum_iterations']==20000
    if s['iterations']:
        assert int(s['history'][-1][0])==s['iterations'];close(s['history'][-1][1],s['gap'],1e-12)
    else:assert critical==0
    close(s['train_MSE'],np.mean((z@b+center-y)**2),1e-11)
    return normal,abs(gap)/scale


def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for path,h in lock['dependencies'].items():assert sha(Path(path))==h,path
    root=R/'results';complete=read(root/'complete.json');check(root,complete['files_sha256']);assert complete['lock_sha256']==sha(R/'lock.json')
    assert complete['source_forward']==complete['new_PCA']==complete['GPU_hours']==0 and all(t['num_threads']==1 for t in complete['threadpool'])
    executions=[read(p) for p in R.glob('execution_*.json')];assert len(executions)==1 and executions[0]['returncode']==0
    plan=read(R/'plan.json');assert plan['folds']==read(P/'plan.json')['folds'] and plan['alphas']==list(ALPHAS)
    rows=[]
    for name,n in zip(old.S[:3],(2196,469,664)):
        part=table(old.BENCH/'data'/f'{name}.csv');assert len(part)==n;rows+=part
    idx={r['sample_id']:i for i,r in enumerate(rows)};assert len(idx)==3329
    td=np.array([[float(v) for v in r['td'].split(',')] for r in rows]);y=np.array([float(r['label']) for r in rows]);binary=np.array([int(r['y']) for r in rows]);xx={}
    for source in old.SOURCES:
        folder=old.EXPORT/source;meta=table(folder/'row_identity.csv');imap={r['sample_id']:i for i,r in enumerate(meta)}
        for row in rows:assert all(row[k]==meta[imap[row['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
        xx[source]=np.c_[td,np.load(folder/'source.npy',allow_pickle=False)[[imap[r['sample_id']] for r in rows]].astype(float)]
    predictions=table(root/'oof_predictions.csv');summary=table(root/'summary.csv');selection_rows=table(root/'selection.csv')
    assert len(predictions)==11394 and len(summary)==12 and len(selection_rows)==48
    count=final=reused_count=0;maxnormal=maxgap=maxpred=maxTD=0.;canonical={};ranks=[]
    for item in plan['folds']:
        setting,fold=item['setting'],item['fold'];fi=np.array([idx[s] for s in item['fit_ids']]);ti=np.array([idx[s] for s in item['test_ids']]);assert set(fi).isdisjoint(ti)
        for source in old.SOURCES:
            out=root/'folds'/setting/f'fold{fold}'/source;ref=P/'results/folds'/setting/f'fold{fold}'/source
            receipt=read(out/'complete.json');assert receipt['lock_sha256']==sha(R/'lock.json');check(out,receipt['files_sha256'])
            reference=read(out/'references.json');assert reference['transform_sha256']==sha(ref/'transforms.json') and reference['source_identity']==read(old.EXPORT/source/'complete.json')['source_identity']
            t=read(ref/'transforms.json');sel=read(out/'selection.json');reuse=read(out/'reuse.json');x=xx[source]
            expected_reuse=5 if fold==0 and setting in old.S[:3] else 0
            assert len(reuse)==expected_reuse and receipt['total_inner_heads']==25 and receipt['new_inner_heads']==25-expected_reuse and receipt['reused_inner_heads']==expected_reuse and receipt['final_heads']==1
            with np.load(out/'arrays.npz',allow_pickle=False) as aa:a={k:aa[k].copy() for k in aa.files}
            oof=np.full((len(fi),5),np.nan);covered=[]
            with np.load(ref/'arrays.npz',allow_pickle=False) as r4:tdcoef=r4['inner_coef'][:,:5,:24].copy();tdcenter=r4['inner_intercept'].copy();tdref=r4['inner_oof'][:,:5].copy()
            for j,split in enumerate(item['inner']):
                tr,va=np.array(split['train']),np.array(split['valid']);covered+=va.tolist();old.project(x[fi[tr]],x[fi[tr]],t['inner'][j])
                z=project(x[fi[tr]],t['inner'][j]);zv=project(x[fi[va]],t['inner'][j]);close(z.mean(0),np.zeros(216),1e-10)
                td_pred=zv[:,:24]@tdcoef[j].T+tdcenter[j];close(td_pred,tdref[va],1e-10);maxTD=max(maxTD,float(abs(td_pred-tdref[va]).max()))
                gram=z.T@z;rhs=z.T@(y[fi[tr]]-y[fi[tr]].mean())
                for k,alpha in enumerate(ALPHAS):
                    b,c=a['inner_coef'][j,k],float(a['inner_intercept'][j,k]);cert=read(out/f'inner{j}_alpha{k}_certificate.json')
                    is_reused=bool(expected_reuse and alpha==100.)
                    ne,gap=check_head(z,y[fi[tr]],b,c,cert,alpha,gram,rhs,is_reused);maxnormal=max(maxnormal,ne);maxgap=max(maxgap,gap);count+=1
                    oof[va,k]=zv@b+c
                    if is_reused:
                        cached=Q/'pilot'/setting/source/f'inner{j}';rr=next(r for r in reuse if r['inner']==j and r['candidate']==k)
                        assert Path(rr['origin'])==cached.resolve() and rr['alpha']==100.
                        for n in ('model','state','complete'):assert rr[n+'_sha256']==sha(cached/(n+'.npz' if n=='model' else n+'.json'))
                        state=read(cached/'state.json');assert state['train_indices']==tr.tolist() and state['valid_indices']==va.tolist() and state['certificate']==cert
                        with np.load(cached/'model.npz',allow_pickle=False) as cm:
                            np.testing.assert_array_equal(b,cm['beta']);assert c==float(cm['intercept']);np.testing.assert_array_equal(a['inner_oof'][va,k],cm['prediction'])
                        reused_count+=1
            assert sorted(covered)==list(range(len(fi)));close(oof,a['inner_oof'],1e-9)
            assert sel['fit_ids']==item['fit_ids'] and sel['test_ids']==item['test_ids'] and sel['outer_validation_unused'] and sel['source']==source
            saved_oof=a['inner_oof'];yc=y[fi]-y[fi].mean();pc=saved_oof-saved_oof.mean(0)
            pcc=yc@pc/np.sqrt((yc@yc)*np.sum(pc*pc,axis=0));mse=np.mean((saved_oof-y[fi,None])**2,axis=0)
            close(sel['PCC'],pcc,1e-10);close(sel['MSE'],mse,1e-11)
            k=min(range(5),key=lambda k:(-pcc[k],mse[k],-ALPHAS[k],k));assert sel['selected']==k and sel['alpha']==ALPHAS[k]
            sr=next(r for r in selection_rows if r['setting']==setting and int(r['fold'])==fold and r['source']==source)
            assert int(sr['candidate'])==k and float(sr['alpha'])==ALPHAS[k];close(float(sr['inner_PCC']),pcc[k],1e-10)
            old.project(x[fi],x[fi],t['final']);z=project(x[fi],t['final']);zt=project(x[ti],t['final'])
            b,c=a['final_coef'],float(a['final_intercept']);cert=read(out/'final_certificate.json')
            ne,gap=check_head(z,y[fi],b,c,cert,ALPHAS[k],z.T@z,z.T@(y[fi]-y[fi].mean()));maxnormal=max(maxnormal,ne);maxgap=max(maxgap,gap);final+=1
            pp=zt@b+c;close(pp,a['test_pred'],1e-9);maxpred=max(maxpred,float(abs(pp-a['test_pred']).max()))
            mean=np.r_[t['final']['td_mean'],t['final']['src_mean']];scale=np.r_[t['final']['td_scale'],np.array(t['final']['src_scale'])*np.sqrt(12.)]
            close(a['deployment_coef'],b/scale,1e-12);close(a['deployment_intercept'],c-mean@(b/scale),1e-12)
            close(x[ti]@a['deployment_coef']+a['deployment_intercept'],pp,1e-9)
            assert (out/'selection.json').stat().st_mtime_ns<=(out/'final_certificate.json').stat().st_mtime_ns
            part=[r for r in predictions if r['setting']==setting and int(r['fold'])==fold and r['source']==source]
            assert [r['sample_id'] for r in part]==item['test_ids'];p=np.array([float(r['prediction']) for r in part]);close(p,pp,1e-9)
            for r,i in zip(part,ti):assert r['condition']==source+'_native_trace' and float(r['label'])==y[i] and int(r['binary'])==binary[i]
            canonical.setdefault((setting,source),[]).extend(zip(ti,p));ranks.append(dict(setting=setting,fold=fold,source=source,alpha=ALPHAS[k],rank=cert['rank'],iterations=cert['iterations']))
        print('VERIFIED',setting,fold,flush=True)
    assert count==1200 and reused_count==45 and final==48
    assert complete['total_inner_heads']==1200 and complete['new_inner_heads']==1155 and complete['reused_inner_heads']==45 and complete['new_final_heads']==48 and complete['new_scientific_heads']==1203 and complete['rows']==11394 and complete['outer_folds']==16
    baseline={(r['setting'],r['condition']):r for r in table(old.OLD/'results/summary.csv')}
    dense_summary={(r['setting'],r['condition']):r for r in table(D/'summary.csv')};dense_pred=table(D/'oof_predictions.csv');paired=table(root/'paired_dense.csv')
    assert len(paired)==12
    for r in summary:
        pairs=canonical[r['setting'],r['source']];ii=np.array([i for i,p in pairs]);p=np.array([p for i,p in pairs])
        cohort='Mixset' if r['setting']=='Inter_Hu_to_Mix' else r['setting'];expected={i for i,rw in enumerate(rows) if rw['cohort']==cohort}
        assert len(ii)==len(expected) and set(ii)==expected;metrics=old.metric(y[ii],p,binary[ii])
        for k,v in metrics.items():close(float(r[k]),v,1e-9)
        base=baseline[r['setting'],'TD24_Ridge_CV'];bp=float(base['PCC']);close(float(r['TD24_PCC']),bp,0)
        close(float(r['PCC_delta']),metrics['PCC']-bp,1e-10);close(float(r['PCC_relative_gain_pct']),100*(metrics['PCC']/bp-1),1e-9)
        close(float(r['MSE_relative_reduction_pct']),100*(1-metrics['MSE']/float(base['MSE'])),1e-9)
        dc=r['source']+'_dense';dp=[v for v in dense_pred if v['setting']==r['setting'] and v['condition']==dc]
        assert len(dp)==len(ii) and {v['sample_id'] for v in dp}=={rows[i]['sample_id'] for i in ii}
        di=np.array([idx[v['sample_id']] for v in dp]);dd=np.array([float(v['prediction']) for v in dp])
        close([float(v['label']) for v in dp],y[di],0);dm=old.metric(y[di],dd,binary[di]);ds=dense_summary[r['setting'],dc]
        for k,v in dm.items():close(float(ds[k]),v,1e-9)
        pair=next(v for v in paired if v['setting']==r['setting'] and v['source']==r['source'])
        for k,v in dict(new_PCC=metrics['PCC'],dense_PCC=dm['PCC'],delta_PCC=metrics['PCC']-dm['PCC'],MSE_reduction_pct=100*(1-metrics['MSE']/dm['MSE'])).items():close(float(pair[k]),v,1e-9)
    goals=read(root/'goal_summary.json')
    for r in goals:
        part=[s for s in summary if s['condition']==r['condition']];assert len(part)==4
        gain=float(np.mean([float(s['PCC_relative_gain_pct']) for s in part]));close(r['mean_relative_PCC_gain_pct'],gain,1e-9)
        assert r['reaches_development_5pct']==(gain>=5) and r['primary']==(r['condition']=='core_real_native_trace')
        assert r['negative_settings']==[s['setting'] for s in part if float(s['PCC_delta'])<0]
    result=dict(passed=True,total_inner_heads=count,new_inner_heads=count-reused_count,reused_R25_heads=reused_count,new_final_heads=final,new_scientific_heads=1203,
        full_joint_certificates=count+final,new_OOF_rows=11394,reused_dense_OOF_rows=11394,no_new_dense_fit=True,
        max_TD_normal_equation_error=maxnormal,max_relative_dual_gap=maxgap,max_prediction_difference=maxpred,max_TD_baseline_difference=maxTD,
        rank_records=ranks,goal_summary=goals,new_source_forward=0,new_PCA=0,GPU_hours=0,
        lock_sha256=sha(R/'lock.json'),complete_sha256=sha(root/'complete.json'),CPU_wall_seconds=time.monotonic()-started)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rank_records'}),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0),f,indent=2)
