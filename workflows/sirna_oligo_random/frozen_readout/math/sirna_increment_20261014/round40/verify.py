"""Independent native scaling, atan2 Gram, LU and frozen reference readback."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4';K=R.parent/'round33';N=R.parent/'round25';D=R.parent/'round21'
sp=importlib.util.spec_from_file_location('_r40_v33',K/'verify.py');kv=importlib.util.module_from_spec(sp);sp.loader.exec_module(kv)
old=kv.old;read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close
metric=kv.metric


def embed(z):
    if z.shape[1]==216:return np.c_[z[:,:24]/np.sqrt(24),z[:,24:]/np.sqrt(192)]
    return kv.embed(z)


def gram_parts(z):
    u=embed(z);raw=kv.kernel(u,u);np.fill_diagonal(raw,np.einsum('ij,ij->i',u,u));raw=(raw+raw.T)/2
    means=raw.mean(1);grand=float(means.mean());centered=raw-means[:,None]-means[None,:]+grand
    variance=float(np.trace(centered)/len(z));scale=16/variance if variance>1e-12 else 0.
    gram=z[:,:24]@z[:,:24].T+scale*centered;gram=(gram+gram.T)/2
    return u,means,grand,variance,scale,gram,centered


def deploy(zv,z,m):
    u,means,grand,variance,scale,gram,kc=gram_parts(z)
    raw=kv.kernel(embed(zv),u);centered=raw-raw.mean(1)[:,None]-means[None,:]+grand
    return zv[:,:24]@m['td_coef']+scale*centered@m['dual']+float(m['intercept'])


def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    assert lock['CPU_only'] and lock['source_seed']==42 and lock['alpha']==100 and lock['source_coordinates']==192
    assert lock['aux_average_trace']==16 and lock['kernel_depth']==lock['kernel_degree']==1 and lock['no_kernel_bias']
    close(lock['source_divisor'],np.sqrt(192),0);close(lock['TD_divisor'],np.sqrt(24),0)
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and not d['outer_test_metrics_used']
    assert d['new_source_forward']==d['new_PCA']==d['new_scaler_fits']==d['GPU_hours']==0
    assert all(t['num_threads']==1 for t in d['threadpool']);check(R/'pilot',d['pilot_files_sha256'])
    assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    executions=[read(p) for p in R.glob('execution_*.json')];assert len(executions)==1 and executions[0]['returncode']==0
    plan=read(P/'plan.json');rowsout=table(R/'diagnostic_rows.csv');records=[];td_records=[];count=td_count=covered=training_rows=0
    numeric=True;maxnormal=maxdual=maxpred=maxsumdual=maxref=0.
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids']
        assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        rows={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')}
        y=np.array([float(rows[s]['label']) for s in ids]);td=np.array([[float(v) for v in rows[s]['td'].split(',')] for s in ids])
        td_prediction=np.full(len(ids),np.nan)
        for source in old.SOURCES:
            folder=old.EXPORT/source;identity=read(folder/'complete.json');assert identity['source_identity']==d['source_identities'][source]
            meta=table(folder/'row_identity.csv');lookup={r['sample_id']:i for i,r in enumerate(meta)}
            for s in ids:assert all(rows[s][k]==meta[lookup[s]][k] for k in ('sample_id','cohort','guide','target57'))
            raw=np.c_[td,np.load(folder/'source.npy',allow_pickle=False)[[lookup[s] for s in ids]].astype(float)]
            pf=P/'results/folds'/setting/'fold0'/source/'transforms.json';af=pf.parent/'arrays.npz';states=read(pf)
            with np.load(af,allow_pickle=False) as a:r4=a['inner_oof'][:,18].copy();beta4=a['inner_coef'][:,18].copy()
            pred=np.full(len(ids),np.nan);r33=pred.copy();r25=pred.copy();coverage=[];blocks=[]
            for j,s in enumerate(item['inner']):
                if time.monotonic()-started>900:raise TimeoutError('Independent audit cap')
                tr,va=np.array(s['train']),np.array(s['valid']);out=R/'pilot'/setting/source/f'inner{j}'
                receipt=read(out/'complete.json');check(out,receipt['files_sha256']);assert receipt['lock_sha256']==sha(R/'lock.json')
                state=read(out/'state.json');cert=state['certificate'];assert (state['setting'],state['source'],state['inner'])==(setting,source,j)
                assert state['train_indices']==tr.tolist() and state['valid_indices']==va.tolist()
                kp=K/'pilot'/setting/source/f'inner{j}'/'model.npz';npth=N/'pilot'/setting/source/f'inner{j}'/'model.npz'
                for key,path in (('transform_sha256',pf),('R4_reference_sha256',af),('R33_reference_sha256',kp),('R25_reference_sha256',npth)):assert state[key]==sha(path)
                zpc=old.project(raw[tr],raw[tr],states['inner'][j]);zpcv=old.project(raw[tr],raw[va],states['inner'][j])
                mu,sd=old.scale(raw[tr][:,:24],False);sm,ss=old.scale(raw[tr][:,24:],True)
                z=np.c_[(raw[tr,:24]-mu)/sd,(raw[tr,24:]-sm)/ss];zv=np.c_[(raw[va,:24]-mu)/sd,(raw[va,24:]-sm)/ss]
                n=len(tr);mean=float(y[tr].mean());yc=y[tr]-mean
                close((zpc.T@zpc+100*np.eye(40))@beta4[j],zpc.T@yc,1e-8);close(zpcv@beta4[j]+mean,r4[va],1e-10)
                with np.load(out/'model.npz',allow_pickle=False) as a:m={k:a[k].copy() for k in a.files}
                assert state['saved_float_values']==sum(a.size for a in m.values())
                u,means,grand,variance,scale,gram,kc=gram_parts(z);close(m['train_u'],u,1e-12);close(m['train_td'],z[:,:24],1e-12)
                system=gram+100*np.eye(n);dual=np.linalg.solve(system,yc);close(m['dual'],dual,1e-9)
                err=float(abs(system@m['dual']-yc).max());limit=1e-9*(1+abs(yc).max());valid=bool(err<=limit);numeric &= valid
                assert valid==cert['numerically_qualified'];close(cert['normal_error'],err);close(cert['normal_limit'],limit,1e-12)
                close(m['raw_row_mean'],means);close(m['raw_grand_mean'],grand);close(m['aux_scale'],scale);close(m['intercept'],mean,1e-12)
                close(m['td_coef'],z[:,:24].T@m['dual']);close(cert['aux_variance'],variance);close(cert['mean_aux_diagonal'],scale*variance)
                close(cert['centered_kernel_row_mean_max'],abs(kc.mean(1)).max());close(cert['sum_dual'],m['dual'].sum())
                assert abs(m['dual'].sum())<1e-9 and cert['alpha']==100 and cert['new_linear_solves']==1 and cert['training_rows']==n and cert['embedding_dim']==216
                residual=gram@m['dual']-yc;norm=float(m['dual']@gram@m['dual'])
                close(cert['train_MSE'],np.mean(residual**2));close(cert['RKHS_squared_norm'],norm);close(cert['objective'],.5*(residual@residual+100*norm))
                pp=deploy(zv,z,m);close(pp,m['prediction'],1e-9);pred[va]=m['prediction']
                with np.load(kp,allow_pickle=False) as a:ref33={k:a[k].copy() for k in a.files}
                u33,_,_,_,_,g33,_=gram_parts(zpc);close(ref33['train_u'],u33,1e-11)
                close((g33+100*np.eye(n))@ref33['dual'],yc,1e-8);p33=deploy(zpcv,zpc,ref33);close(p33,ref33['prediction'],1e-9);r33[va]=ref33['prediction']
                ns=read(N/'pilot'/setting/source/f'inner{j}'/'state.json');assert ns['train_indices']==tr.tolist() and ns['valid_indices']==va.tolist()
                nt=read(D/'pilot'/setting/source/'dense'/f'inner{j}'/'transform.json')
                for name,val in (('td_mean',mu),('td_scale',sd),('src_mean',sm),('src_scale',ss)):close(nt[name],val,1e-11)
                with np.load(npth,allow_pickle=False) as a:ref25={k:a[k].copy() for k in a.files}
                p25=raw[va]@ref25['deployment_coef']+float(ref25['deployment_intercept']);close(p25,ref25['prediction'],1e-9);r25[va]=ref25['prediction']
                maxref=max(maxref,float(abs(p25-ref25['prediction']).max()),float(abs(p33-ref33['prediction']).max()))
                if source=='core_real':
                    with np.load(K/'pilot'/setting/'TD_only'/f'inner{j}'/'model.npz',allow_pickle=False) as a:mt={k:a[k].copy() for k in a.files}
                    _,_,_,_,_,gt,_=gram_parts(z[:,:24]);close((gt+100*np.eye(n))@mt['dual'],yc,1e-8)
                    tp=deploy(zv[:,:24],z[:,:24],mt);close(tp,mt['prediction'],1e-9);td_prediction[va]=mt['prediction'];td_count+=1
                maxnormal=max(maxnormal,err);maxdual=max(maxdual,float(abs(m['dual']-dual).max()));maxpred=max(maxpred,float(abs(pp-m['prediction']).max()));maxsumdual=max(maxsumdual,float(abs(m['dual'].sum())))
                count+=1;training_rows+=n;coverage+=va.tolist()
                blocks.append(dict(inner=j,n=len(va),training_rows=n,metrics=metric(y[va],pred[va]),R4_reference=metric(y[va],r4[va]),
                    R33_reference=metric(y[va],r33[va]),R25_reference=metric(y[va],r25[va]),normal_error=err,numerically_qualified=valid))
            assert sorted(coverage)==list(range(len(ids))) and np.isfinite(pred).all()
            part=[r for r in rowsout if r['setting']==setting and r['source']==source];assert [r['sample_id'] for r in part]==ids and all(r['new_fit']=='True' for r in part)
            for k,val in (('label',y),('prediction',pred),('R4_reference_prediction',r4),('R33_reference_prediction',r33),('R25_reference_prediction',r25)):close([float(r[k]) for r in part],val,0)
            m=metric(y,pred);refs=dict(R4=metric(y,r4),R33=metric(y,r33),R25=metric(y,r25));rec=next(r for r in d['records'] if r['setting']==setting and r['source']==source)
            assert rec['n']==len(ids)
            for k,v in m.items():close(rec['metrics'][k],v,1e-11)
            values={}
            for name,ref in refs.items():
                for k,v in ref.items():close(rec[name+'_reference'][k],v,1e-11)
                for key,val in (('delta_PCC_vs_'+name,m['PCC']-ref['PCC']),('MSE_reduction_vs_'+name+'_pct',100*(1-m['MSE']/ref['MSE']))):close(rec[key],val,1e-10);values[key]=val
            for saved,computed in zip(rec['inner_blocks'],blocks):
                for key,val in computed.items():
                    if isinstance(val,dict):
                        for k,v in val.items():close(saved[key][k],v,1e-9)
                    else:close(saved[key],val,1e-9)
            assert len(rec['inner_blocks'])==5;records.append(dict(setting=setting,source=source,metrics=m,**values));covered+=len(ids)
        assert np.isfinite(td_prediction).all();tm=metric(y,td_prediction);tr=next(r for r in d['TD_records'] if r['setting']==setting)
        assert tr['new_fit'] is False and tr['n']==len(ids)
        for k,v in tm.items():close(tr['metrics'][k],v,1e-11)
        part=[r for r in rowsout if r['setting']==setting and r['source']=='TD_only'];assert [r['sample_id'] for r in part]==ids and all(r['new_fit']=='False' for r in part)
        for key,val in (('label',y),('prediction',td_prediction),('R33_reference_prediction',td_prediction)):close([float(r[key]) for r in part],val,0)
        assert all(r['R4_reference_prediction']==r['R25_reference_prediction']=='' for r in part)
        covered+=len(ids);td_records.append(dict(setting=setting,metrics=tm))
    assert count==45==d['new_heads']==d['new_linear_solves']==d['reused_R33_source_heads']==d['reused_R4_source_heads']==d['reused_R25_source_heads']
    assert td_count==15==d['reused_TD_heads'] and covered==len(rowsout)==9196==d['combined_internal_OOF_rows']
    assert d['new_internal_OOF_rows']==6897 and d['reused_TD_OOF_rows']==2299 and training_rows==d['stored_training_row_occurrences']
    core=[r for r in records if r['source']=='core_real']
    performance=sum(r['delta_PCC_vs_R4']>=.005 and r['MSE_reduction_vs_R4_pct']>=0 for r in core)>=2 and all(r['delta_PCC_vs_R4']>=-.005 for r in core)
    compression=sum(r['delta_PCC_vs_R33']>=0 and r['MSE_reduction_vs_R33_pct']>=0 for r in core)>=2 and all(r['delta_PCC_vs_R33']>=-.005 for r in core)
    native=sum(r['MSE_reduction_vs_R25_pct']>=0 for r in core)>=2 and np.mean([r['MSE_reduction_vs_R25_pct'] for r in core])>=0 and all(r['delta_PCC_vs_R25']>=-.005 for r in core)
    pairs=[]
    for r in core:
        td=next(t for t in td_records if t['setting']==r['setting']);delta=r['metrics']['PCC']-td['metrics']['PCC'];red=100*(1-r['metrics']['MSE']/td['metrics']['MSE'])
        orig=next(t for t in d['paired_kernel_TD'] if t['setting']==r['setting']);close(orig['delta_PCC'],delta,1e-11);close(orig['MSE_reduction_pct'],red,1e-10);pairs.append((delta,red))
    inc=sum(p>=0 for p,r in pairs)>=2 and np.mean([r for p,r in pairs])>=0
    qualified=bool(performance and compression and native and inc and numeric)
    assert d['performance_gate']==performance and d['compression_pair_gate']==compression and d['native_reference_gate']==native and d['source_increment_gate']==inc and d['all_numerically_qualified']==numeric and d['qualified']==qualified
    check(R,lock['files'])
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    result=dict(passed=True,qualified=qualified,verified_new_heads=45,independent_new_LU_solves=45,verified_reused_TD_heads=15,
        verified_R33_source_heads=45,verified_R4_source_heads=45,verified_R25_deployments=45,new_internal_OOF_rows=6897,reused_TD_OOF_rows=2299,
        combined_internal_OOF_rows=9196,stored_training_row_occurrences=training_rows,max_normal_error=maxnormal,max_dual_difference=maxdual,
        max_deployment_difference=maxpred,max_reference_deployment_difference=maxref,max_sum_dual=maxsumdual,all_numerically_qualified=bool(numeric),
        new_source_forward=0,new_PCA=0,new_scaler_fits=0,GPU_hours=0,outer_test_metrics_used=False,lock_sha256=sha(R/'lock.json'),
        diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-started)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    started=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-started,GPU_hours=0,new_source_forward=0),f,indent=2)
