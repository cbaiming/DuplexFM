"""Independent Fenchel certificate, original PCA, atan2 kernel and deployment."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
R=Path(__file__).resolve().parent;P=R.parent/'round4';K=R.parent/'round33'
spec=importlib.util.spec_from_file_location('_r39_verify33',K/'verify.py');prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
old=prior.old;read,sha,table,check=old.read,old.sha,old.table,old.check
close=prior.close;metric=prior.metric


def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    assert lock['CPU_only'] and lock['source_seed']==42 and lock['alpha']==100 and lock['epsilon_fraction']==.1
    assert lock['free_intercept'] and lock['loss']=='half_squared_epsilon_insensitive' and lock['kernel_unchanged_from']=='round33'
    assert lock['maximum_iterations']==5000 and lock['gap_tolerance']==1e-9
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and not d['outer_test_metrics_used']
    assert d['new_source_forward']==d['new_PCA']==d['GPU_hours']==0 and all(t['num_threads']==1 for t in d['threadpool'])
    check(R/'pilot',d['pilot_files_sha256']);assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    executions=[read(p) for p in R.glob('execution_*.json')];assert len(executions)==1 and executions[0]['returncode']==0
    plan=read(P/'plan.json');rowsout=table(R/'diagnostic_rows.csv');records=[]
    count=covered=updates=training_rows=0;numeric=True
    maxgap=maxkkt=maxfeasible=maxpred=maxintercept=maxoldnormal=0.
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids']
        assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        rows={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')}
        y=np.array([float(rows[s]['label']) for s in ids]);td=np.array([[float(v) for v in rows[s]['td'].split(',')] for s in ids])
        for source in (*old.SOURCES,'TD_only'):
            src='core_real' if source=='TD_only' else source;folder=old.EXPORT/src;identity=read(folder/'complete.json')
            assert identity['source_identity']==d['source_identities'][src]
            meta=table(folder/'row_identity.csv');lookup={r['sample_id']:i for i,r in enumerate(meta)}
            for s in ids:assert all(rows[s][k]==meta[lookup[s]][k] for k in ('sample_id','cohort','guide','target57'))
            raw=np.c_[td,np.load(folder/'source.npy',allow_pickle=False)[[lookup[s] for s in ids]].astype(float)]
            pf=P/'results/folds'/setting/'fold0'/src/'transforms.json';af=pf.parent/'arrays.npz';states=read(pf)
            with np.load(af,allow_pickle=False) as a:
                index=3 if source=='TD_only' else 18;r4=a['inner_oof'][:,index].copy();beta4=a['inner_coef'][:,index].copy()
            pred=np.full(len(ids),np.nan);r33=np.full(len(ids),np.nan);coverage=[];blocks=[]
            for j,split in enumerate(item['inner']):
                if time.monotonic()-started>900:raise TimeoutError('Independent CPU audit cap')
                tr,va=np.array(split['train']),np.array(split['valid']);out=R/'pilot'/setting/source/f'inner{j}'
                receipt=read(out/'complete.json');check(out,receipt['files_sha256']);assert receipt['lock_sha256']==sha(R/'lock.json')
                st=read(out/'state.json');cert=st['certificate']
                assert (st['setting'],st['source'],st['inner'])==(setting,source,j)
                assert st['train_indices']==tr.tolist() and st['valid_indices']==va.tolist()
                rp=K/'pilot'/setting/source/f'inner{j}'/'model.npz'
                assert st['transform_sha256']==sha(pf) and st['R4_reference_sha256']==sha(af) and st['R33_reference_sha256']==sha(rp)
                with np.load(out/'model.npz',allow_pickle=False) as a:m={k:a[k].copy() for k in a.files}
                with np.load(rp,allow_pickle=False) as a:ref={k:a[k].copy() for k in a.files}
                assert st['saved_float_values']==sum(a.size for a in m.values())
                z=old.project(raw[tr],raw[tr],states['inner'][j]);zv=old.project(raw[tr],raw[va],states['inner'][j])
                if source=='TD_only':z=z[:,:24];zv=zv[:,:24]
                n,p=z.shape;mean=float(y[tr].mean());yc=y[tr]-mean
                close((z.T@z+100*np.eye(p))@beta4[j,:p],z.T@yc,1e-8);close(zv@beta4[j,:p]+mean,r4[va],1e-10)
                u,uv=prior.embed(z),prior.embed(zv)
                close(m['train_u'],u,1e-12);close(m['train_td'],z[:,:24],1e-12)
                for name in ('train_u','train_td','raw_row_mean','raw_grand_mean','aux_scale'):close(m[name],ref[name],0)
                rawk=prior.kernel(u,u);np.fill_diagonal(rawk,np.einsum('ij,ij->i',u,u));rawk=(rawk+rawk.T)/2
                means=rawk.mean(1);grand=float(means.mean());kc=rawk-means[:,None]-means[None,:]+grand
                variance=float(np.trace(kc)/n);scale=16/variance if variance>1e-12 else 0.
                gram=z[:,:24]@z[:,:24].T+scale*kc;gram=(gram+gram.T)/2
                close(m['raw_row_mean'],means);close(m['raw_grand_mean'],grand);close(m['aux_scale'],scale)
                olderr=float(abs((gram+100*np.eye(n))@ref['dual']-yc).max());assert olderr<1e-8
                maxoldnormal=max(maxoldnormal,olderr);close(ref['intercept'],mean,1e-12)
                v=m['epsilon_dual'];close(m['dual'],v/100,0);close(m['td_coef'],z[:,:24].T@v/100,1e-10)
                sd=float(np.sqrt(np.mean((y[tr]-mean)**2)));eps=.1*sd
                close(st['training_label_population_SD'],sd,1e-12);close(cert['epsilon'],eps,1e-12)
                f=gram@v/100
                def excess(r):return np.maximum(r-eps,0)-np.maximum(-r-eps,0)
                b=brentq(lambda b:float(excess(y[tr]-f-b).sum()),float((y[tr]-f).min()-eps),float((y[tr]-f).max()+eps),xtol=5e-15)
                maxintercept=max(maxintercept,abs(b-float(m['intercept'])));close(m['intercept'],b,1e-10)
                residual=y[tr]-f-float(m['intercept']);s=excess(residual);norm=float(v@f)
                primal=float(.5*(norm+s@s));dual=float(y[tr]@v-.5*(v@v)-eps*abs(v).sum()-.5*norm)
                gap=primal-dual;rel=max(0.,gap)/max(1.,primal);feasible=float(abs(v.sum()));kkt=float(abs(v-s).max());limit=1e-7*(1+sd)
                valid=bool(rel<=1e-9 and gap>=-1e-9*max(1.,primal) and feasible<=1e-10*(1+abs(v).sum()) and kkt<=limit)
                expected=dict(primal=primal,dual_objective=dual,gap=gap,relative_gap=rel,sum_dual=float(v.sum()),
                    max_loss_gradient_difference=kkt,loss_gradient_limit=limit,intercept_gradient=float(s.sum()),
                    train_MSE=float(np.mean(residual**2)),RKHS_squared_norm=norm/100,
                    inactive_fraction=float(np.mean(abs(residual)<=eps)),intercept=float(m['intercept']),
                    lipschitz_upper_bound=1+float(abs(gram).sum(1).max())/100)
                for key,value in expected.items():close(cert[key],value,1e-8)
                assert valid==cert['numerically_qualified'];numeric &= valid
                assert cert['alpha']==100 and cert['epsilon_fraction']==.1 and cert['gap_tolerance']==1e-9
                assert cert['maximum_iterations']==5000 and 1<=cert['iterations']<=5000 and st['training_rows']==n
                kv=prior.kernel(uv,u);kcv=kv-kv.mean(1)[:,None]-means[None,:]+grand
                pp=zv[:,:24]@m['td_coef']+scale*kcv@(v/100)+float(m['intercept'])
                pold=zv[:,:24]@ref['td_coef']+scale*kcv@ref['dual']+float(ref['intercept'])
                close(pp,m['prediction'],1e-9);close(pold,ref['prediction'],1e-9)
                maxpred=max(maxpred,float(abs(pp-m['prediction']).max()));pred[va]=m['prediction'];r33[va]=ref['prediction']
                maxgap=max(maxgap,rel);maxkkt=max(maxkkt,kkt);maxfeasible=max(maxfeasible,feasible)
                count+=1;updates+=cert['iterations'];training_rows+=n;coverage+=va.tolist()
                blocks.append(dict(inner=j,n=len(va),metrics=metric(y[va],pred[va]),R4_reference=metric(y[va],r4[va]),
                    R33_reference=metric(y[va],r33[va]),epsilon=eps,inactive_fraction=expected['inactive_fraction'],
                    relative_gap=rel,iterations=cert['iterations'],numerically_qualified=valid))
            assert sorted(coverage)==list(range(len(ids))) and np.isfinite(pred).all()
            part=[r for r in rowsout if r['setting']==setting and r['source']==source];assert [r['sample_id'] for r in part]==ids
            for key,val in (('label',y),('prediction',pred),('R4_reference_prediction',r4),('R33_reference_prediction',r33)):close([float(r[key]) for r in part],val,0)
            covered+=len(part);m,ref4,ref33=metric(y,pred),metric(y,r4),metric(y,r33)
            rec=next(r for r in d['records'] if r['setting']==setting and r['source']==source);assert rec['n']==len(ids)
            for name,val in (('metrics',m),('R4_reference',ref4),('R33_reference',ref33)):
                for k,v in val.items():close(rec[name][k],v,1e-11)
            expected=dict(delta_PCC_vs_R4=m['PCC']-ref4['PCC'],MSE_reduction_vs_R4_pct=100*(1-m['MSE']/ref4['MSE']),
                delta_PCC_vs_R33=m['PCC']-ref33['PCC'],MSE_reduction_vs_R33_pct=100*(1-m['MSE']/ref33['MSE']))
            for k,v in expected.items():close(rec[k],v,1e-10)
            assert len(rec['inner_blocks'])==5
            for a,b in zip(rec['inner_blocks'],blocks):
                for k,v in b.items():
                    if isinstance(v,dict):
                        for key,value in v.items():close(a[k][key],value,1e-9)
                    else:close(a[k],v,1e-9)
            records.append(dict(setting=setting,source=source,metrics=m,**expected))
    assert count==60==d['new_heads']==d['reused_R33_heads']==d['reused_R4_heads']
    assert covered==len(rowsout)==9196==d['new_internal_OOF_rows'];assert updates==d['new_proximal_updates']
    assert training_rows==d['stored_training_row_occurrences']
    core=[r for r in records if r['source']=='core_real']
    performance=sum(r['delta_PCC_vs_R4']>=.005 and r['MSE_reduction_vs_R4_pct']>=0 for r in core)>=2 and all(r['delta_PCC_vs_R4']>=-.005 for r in core)
    loss=sum(r['delta_PCC_vs_R33']>=0 and r['MSE_reduction_vs_R33_pct']>=0 for r in core)>=2 and all(r['delta_PCC_vs_R33']>=-.005 for r in core)
    pairs=[]
    for r in core:
        td=next(t for t in records if t['setting']==r['setting'] and t['source']=='TD_only')
        delta=r['metrics']['PCC']-td['metrics']['PCC'];red=100*(1-r['metrics']['MSE']/td['metrics']['MSE'])
        saved=next(t for t in d['paired_kernel_TD'] if t['setting']==r['setting']);close(saved['delta_PCC'],delta,1e-11);close(saved['MSE_reduction_pct'],red,1e-10)
        pairs.append((delta,red))
    inc=sum(p>=0 for p,r in pairs)>=2 and np.mean([r for p,r in pairs])>=0
    qualified=bool(performance and loss and inc and numeric)
    assert d['performance_gate']==performance and d['loss_factor_gate']==loss and d['source_increment_gate']==inc
    assert d['all_numerically_qualified']==numeric and d['qualified']==qualified
    check(R,lock['files'])
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    result=dict(passed=True,qualified=qualified,verified_new_convex_heads=60,verified_R33_reference_heads=60,verified_R4_reference_heads=60,
        new_internal_OOF_rows=9196,new_proximal_updates=updates,stored_training_row_occurrences=training_rows,
        max_relative_primal_dual_gap=maxgap,max_loss_gradient_difference=maxkkt,max_zero_sum_error=maxfeasible,
        max_independent_intercept_difference=maxintercept,max_deployment_difference=maxpred,max_R33_normal_error=maxoldnormal,
        all_numerically_qualified=bool(numeric),new_source_forward=0,new_PCA=0,GPU_hours=0,outer_test_metrics_used=False,
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
