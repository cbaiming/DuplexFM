"""Independent complete beta-system certificate plus kernel stationary equations."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4';N=R.parent/'round25';K=R.parent/'round40';D=R.parent/'round21'
sp=importlib.util.spec_from_file_location('_r41_v33',R.parent/'round33/verify.py');kv=importlib.util.module_from_spec(sp);sp.loader.exec_module(kv)
old=kv.old;read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close;metric=kv.metric


def embed(z):return z/np.sqrt(24) if z.shape[1]==24 else np.c_[z[:,:24]/np.sqrt(24),z[:,24:]/4]
def even(u,v):return kv.kernel(u,v)-.5*np.einsum('ik,jk->ij',u,v,optimize=False)


def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    assert lock['CPU_only'] and lock['source_seed']==42 and lock['alpha']==100 and lock['lambda_ratio']==.1
    assert lock['kernel']=='arc1_minus_half_linear' and lock['lambda_definition']=='exact_original_R25_without_kernel'
    assert lock['joint_objective'] and not lock['reference_predictions_used_for_training'] and lock['aux_average_trace']==16
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and not d['outer_test_metrics_used']
    assert d['new_source_forward']==d['new_PCA']==d['new_scaler_fits']==d['GPU_hours']==0 and all(t['num_threads']==1 for t in d['threadpool'])
    check(R/'pilot',d['pilot_files_sha256']);assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    runs=[read(p) for p in R.glob('execution_*.json')];assert len(runs)==1 and runs[0]['returncode']==0
    plan=read(P/'plan.json');output=table(R/'diagnostic_rows.csv');records=[];count=covered=updates=training_rows=0;numeric=True
    maxgap=maxkernel=maxnormal=maxdeploy=maxobjective=maxoldlambda=0.
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
                idx=3 if source=='TD_only' else 18;r4=a['inner_oof'][:,idx].copy();b4=a['inner_coef'][:,idx].copy()
            pred=np.full(len(ids),np.nan);r25=pred.copy();r40=pred.copy();coverage=[];blocks=[]
            for j,s in enumerate(item['inner']):
                if time.monotonic()-started>900:raise TimeoutError('Independent CPU cap')
                tr,va=np.array(s['train']),np.array(s['valid']);out=R/'pilot'/setting/source/f'inner{j}'
                receipt=read(out/'complete.json');check(out,receipt['files_sha256']);assert receipt['lock_sha256']==sha(R/'lock.json')
                st=read(out/'state.json');cert=st['certificate'];assert (st['setting'],st['source'],st['inner'])==(setting,source,j)
                assert st['train_indices']==tr.tolist() and st['valid_indices']==va.tolist()
                sf=D/'pilot'/setting/src/'dense'/f'inner{j}'/'transform.json';scale_state=read(sf);assert st['transform_sha256']==sha(sf)
                for key,val in scale_state.items():close(val,states['inner'][j][key],0)
                zpc=old.project(raw[tr],raw[tr],states['inner'][j]);zpcv=old.project(raw[tr],raw[va],states['inner'][j])
                mu=np.r_[scale_state['td_mean'],scale_state['src_mean']];sd=np.r_[scale_state['td_scale'],np.array(scale_state['src_scale'])*np.sqrt(12)]
                x=(raw[tr]-mu)/sd;xv=(raw[va]-mu)/sd
                if source=='TD_only':x=x[:,:24];xv=xv[:,:24];zpc=zpc[:,:24];zpcv=zpcv[:,:24]
                n,p=x.shape;center=float(y[tr].mean());yc=y[tr]-center
                close((zpc.T@zpc+100*np.eye(zpc.shape[1]))@b4[j,:zpc.shape[1]],zpc.T@yc,1e-8);close(zpcv@b4[j,:zpc.shape[1]]+center,r4[va],1e-10)
                assert st['reference_sha256']['R4']==sha(af)
                with np.load(out/'model.npz',allow_pickle=False) as a:m={k:a[k].copy() for k in a.files}
                assert st['saved_float_values']==sum(a.size for a in m.values())
                u=embed(x);rawk=even(u,u);np.fill_diagonal(rawk,.5*np.einsum('ij,ij->i',u,u));rawk=(rawk+rawk.T)/2
                means=rawk.mean(1);grand=float(means.mean());kc=rawk-means[:,None]-means[None,:]+grand
                var=float(np.trace(kc)/n);scale=16/var if var>1e-12 else 0.;gram=scale*kc;gram=(gram+gram.T)/2
                for key,val in (('train_u',u),('raw_row_mean',means),('raw_grand_mean',grand),('aux_variance',var),('aux_scale',scale),('aux_trace',16),('intercept',center)):close(m[key],val,1e-10)
                system=gram+100*np.eye(n);sol=np.linalg.solve(system,np.c_[x,yc]);px=100*sol[:,:p];py=100*sol[:,p]
                g=x.T@px+100*np.eye(p);g=(g+g.T)/2;q=x.T@py;b=m['beta'];c=m['dual']
                if p==216:
                    original=x.T@x+100*np.eye(p);oq=x.T@yc
                    oqs=oq[24:]-original[24:,:24]@np.linalg.solve(original[:24,:24],oq[:24])
                    critical=float(np.linalg.norm(oqs.reshape(6,32),2));lam=.1*critical
                    for folder,name,dest in ((N,'R25',r25),(K,'R40',r40)):
                        path=folder/'pilot'/setting/source/f'inner{j}'/'model.npz';assert st['reference_sha256'][name]==sha(path)
                        rs=read(path.parent/'state.json');assert rs['train_indices']==tr.tolist() and rs['valid_indices']==va.tolist()
                        with np.load(path,allow_pickle=False) as a:
                            dest[va]=a['prediction']
                            if name=='R25':
                                close(raw[va]@a['deployment_coef']+a['deployment_intercept'],a['prediction'],1e-9)
                                maxoldlambda=max(maxoldlambda,abs(lam-rs['certificate']['lambda_value']));close(lam,rs['certificate']['lambda_value'],1e-10)
                    td0=np.linalg.solve(g[:24,:24],q[:24]);coupling=np.linalg.solve(g[:24,:24],g[:24,24:])
                    h=g[24:,24:]-g[24:,:24]@coupling;h=(h+h.T)/2;rhs=q[24:]-g[24:,:24]@td0
                    sv=np.linalg.svd(b[24:].reshape(6,32),compute_uv=False);nuclear=float(sv.sum());penalty=lam*nuclear
                    v=rhs-h@b[24:];norm=float(np.linalg.norm(v.reshape(6,32),2))
                    if norm>lam:v*=lam/norm
                    feasible=np.r_[np.zeros(24),v];gap_scale=max(1.,.5*float(rhs@np.linalg.solve(h,rhs)))
                    profile=cert['profile'];close(profile['dual'],v,1e-9);close(profile['singular_values'],sv,1e-10)
                    assert profile['rank']==int(sum(sv>1e-8));close(profile['gap_scale'],gap_scale,1e-9)
                    assert cert['matrix_shape']==[6,32] and cert['lambda_ratio']==.1
                    if lam>0:
                        assert 0<profile['iterations']<=5000 and int(profile['history'][-1][0])==profile['iterations']
                    else:assert profile['iterations']==0
                else:
                    critical=lam=penalty=0.;feasible=np.zeros(p);gap_scale=max(1.,.5*float(q@np.linalg.solve(g,q)))
                    assert cert['matrix_shape'] is None and cert['lambda_ratio']==0 and cert['profile']['iterations']==0
                    close(g@b,q,1e-8);assert set(st['reference_sha256'])=={'R4'}
                close(cert['lambda_critical'],critical,1e-9);close(cert['lambda_value'],lam,1e-9)
                primal=float(.5*b@g@b-q@b+penalty);dv=q-feasible;dual=float(-.5*dv@np.linalg.solve(g,dv));gap=primal-dual
                close(cert['profiled_joint_primal'],primal,1e-9);close(cert['profiled_joint_dual'],dual,1e-9);close(cert['full_gap'],gap,1e-9);close(cert['gap_scale'],gap_scale,1e-9)
                if p==216:
                    const=.5*q[:24]@td0;close(cert['profile']['primal_profile'],primal+const,1e-9);close(cert['profile']['dual_profile'],dual+const,1e-9)
                    close(cert['profile']['gap'],max(0.,gap),1e-9)
                kernel_error=float(abs(system@c-(yc-x@b)).max());kernel_limit=1e-9*(1+abs(yc-x@b).max())
                normal=float(abs((g@b-q)[:24]).max());normal_limit=1e-8*(1+abs(q[:24]).max())
                residual=x@b+gram@c-yc;rkhs=float(c@gram@c);objective=float(.5*(residual@residual+100*(b@b+rkhs))+penalty)
                alternate=primal+.5*float(yc@py);maxobjective=max(maxobjective,abs(objective-alternate));close(objective,alternate,1e-9)
                expected=dict(TD_normal_error=normal,TD_normal_limit=normal_limit,kernel_stationarity_error=kernel_error,
                    kernel_stationarity_limit=kernel_limit,full_objective=objective,linear_squared_norm=float(b@b),nonlinear_RKHS_squared_norm=rkhs,
                    train_MSE=float(np.mean(residual**2)),training_prediction_mean=float((residual+y[tr]).mean()),sum_kernel_dual=float(c.sum()))
                for key,val in expected.items():close(cert[key],val,1e-9)
                assert abs(c.sum())<1e-9 and abs(expected['training_prediction_mean']-center)<1e-9
                valid=bool(cert['profile']['certified'] and -1e-8<=gap<=1e-9*gap_scale+1e-11 and kernel_error<=kernel_limit and normal<=normal_limit)
                assert valid==cert['numerically_qualified'] and cert['joint_fitting'] and not cert['other_model_predictions_used']
                assert cert['training_rows']==n and cert['input_dim']==p and cert['alpha']==100;numeric &= valid
                rawv=even(embed(xv),u);kcv=rawv-rawv.mean(1)[:,None]-means[None,:]+grand;pp=xv@b+scale*kcv@c+center
                close(pp,m['prediction'],1e-9);pred[va]=m['prediction'];maxdeploy=max(maxdeploy,float(abs(pp-m['prediction']).max()))
                maxgap=max(maxgap,abs(gap)/gap_scale);maxkernel=max(maxkernel,kernel_error);maxnormal=max(maxnormal,normal)
                count+=1;updates+=cert['profile']['iterations'];training_rows+=n;coverage+=va.tolist()
                block=dict(inner=j,n=len(va),metrics=metric(y[va],pred[va]),R4_reference=metric(y[va],r4[va]),relative_gap=max(0.,gap)/gap_scale,
                    iterations=cert['profile']['iterations'],numerically_qualified=valid)
                if p==216:block.update(R25_reference=metric(y[va],r25[va]),R40_reference=metric(y[va],r40[va]))
                blocks.append(block)
            assert sorted(coverage)==list(range(len(ids))) and np.isfinite(pred).all()
            part=[r for r in output if r['setting']==setting and r['source']==source];assert [r['sample_id'] for r in part]==ids
            for key,val in (('label',y),('prediction',pred),('R4_reference_prediction',r4)):close([float(r[key]) for r in part],val,0)
            if source!='TD_only':
                for key,val in (('R25_reference_prediction',r25),('R40_reference_prediction',r40)):close([float(r[key]) for r in part],val,0)
            else:assert all(r['R25_reference_prediction']==r['R40_reference_prediction']=='' for r in part)
            m=metric(y,pred);refs=dict(R4=metric(y,r4))
            if source!='TD_only':refs.update(R25=metric(y,r25),R40=metric(y,r40))
            rec=next(r for r in d['records'] if r['setting']==setting and r['source']==source);assert rec['n']==len(ids)
            for key,val in m.items():close(rec['metrics'][key],val,1e-11)
            values={}
            for name,ref in refs.items():
                for key,val in ref.items():close(rec[name+'_reference'][key],val,1e-11)
                for key,val in (('delta_PCC_vs_'+name,m['PCC']-ref['PCC']),('MSE_reduction_vs_'+name+'_pct',100*(1-m['MSE']/ref['MSE']))):close(rec[key],val,1e-10);values[key]=val
            assert len(rec['inner_blocks'])==5
            for saved,computed in zip(rec['inner_blocks'],blocks):
                for key,val in computed.items():
                    if isinstance(val,dict):
                        for k,v in val.items():close(saved[key][k],v,1e-9)
                    else:close(saved[key],val,1e-9)
            records.append(dict(setting=setting,source=source,metrics=m,**values));covered+=len(ids)
    assert count==60==d['new_heads'] and covered==len(output)==9196==d['new_internal_OOF_rows']
    assert d['new_nuclear_heads']==45 and d['new_TD_heads']==15 and d['reused_R25_source_heads']==d['reused_R40_source_heads']==45 and d['reused_R4_TD_heads']==15
    assert updates==d['new_proximal_updates'] and training_rows==d['stored_training_row_occurrences']
    core=[r for r in records if r['source']=='core_real']
    primary=sum(r['delta_PCC_vs_R25']>=.005 and r['MSE_reduction_vs_R25_pct']>=0 for r in core)>=2 and all(r['delta_PCC_vs_R25']>=-.005 for r in core)
    paired=sum(r['MSE_reduction_vs_R40_pct']>=0 for r in core)>=2 and np.mean([r['MSE_reduction_vs_R40_pct'] for r in core])>=0 and all(r['delta_PCC_vs_R40']>=-.005 for r in core)
    pairs=[]
    for r in core:
        td=next(t for t in records if t['setting']==r['setting'] and t['source']=='TD_only');p=r['metrics']['PCC']-td['metrics']['PCC'];red=100*(1-r['metrics']['MSE']/td['metrics']['MSE'])
        orig=next(t for t in d['paired_TD'] if t['setting']==r['setting']);close(orig['delta_PCC'],p,1e-11);close(orig['MSE_reduction_pct'],red,1e-10);pairs.append((p,red))
    inc=sum(p>=0 for p,r in pairs)>=2 and np.mean([r for p,r in pairs])>=0;qualified=bool(primary and paired and inc and numeric)
    assert d['primary_native_gate']==primary and d['paired_kernel_gate']==paired and d['source_increment_gate']==inc and d['all_numerically_qualified']==numeric and d['qualified']==qualified
    check(R,lock['files'])
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    result=dict(passed=True,qualified=qualified,verified_new_joint_heads=60,independent_full_convex_certificates=60,independent_kernel_stationarity_checks=60,
        verified_R25_deployments=45,verified_R40_reference_readbacks=45,verified_R4_reference_heads=60,new_internal_OOF_rows=9196,new_proximal_updates=updates,
        max_relative_joint_gap=maxgap,max_kernel_stationarity_error=maxkernel,max_TD_normal_error=maxnormal,max_deployment_difference=maxdeploy,
        max_full_vs_profiled_objective_difference=maxobjective,max_lambda_difference_from_original_R25=maxoldlambda,all_numerically_qualified=bool(numeric),
        new_source_forward=0,new_PCA=0,new_scaler_fits=0,GPU_hours=0,outer_test_metrics_used=False,
        lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-started)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    started=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-started,GPU_hours=0,new_source_forward=0),f,indent=2)
