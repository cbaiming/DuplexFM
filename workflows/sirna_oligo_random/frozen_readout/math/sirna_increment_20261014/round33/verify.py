"""Independent atan2 kernel, LU coefficients, old PCA and result readback."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4'
spec=importlib.util.spec_from_file_location('_r33_verify4',P/'verify.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,sha,table,check=old.read,old.sha,old.table,old.check


def close(a,b,tol=1e-9):np.testing.assert_allclose(a,b,rtol=tol,atol=tol)


def kernel(x,y):
    norm=np.outer(np.sqrt(np.einsum('ij,ij->i',x,x)),np.sqrt(np.einsum('ij,ij->i',y,y)))
    dot=np.einsum('ik,jk->ij',x,y,optimize=False);dot=np.maximum(-norm,np.minimum(dot,norm))
    perpendicular=np.sqrt(np.maximum(norm*norm-dot*dot,0.));theta=np.arctan2(perpendicular,dot)
    out=(perpendicular+(np.pi-theta)*dot)/np.pi;out[norm==0]=0.
    return out


def embed(z):return np.c_[z[:,:24]/np.sqrt(24.),z[:,24:]/np.sqrt(16.)] if z.shape[1]==40 else z/np.sqrt(24.)
def metric(y,p):return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))


def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for path,h in lock['dependencies'].items():assert sha(Path(path))==h,path
    assert lock['CPU_only'] and lock['source_seed']==42 and lock['alpha']==100 and lock['aux_average_trace']==16
    assert lock['kernel_degree']==lock['kernel_depth']==1 and lock['no_kernel_bias'] and lock['exact_training_Gram']
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and not d['outer_test_metrics_used']
    assert d['new_source_forward']==d['new_PCA']==d['GPU_hours']==0 and all(t['num_threads']==1 for t in d['threadpool'])
    check(R/'pilot',d['pilot_files_sha256']);assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    ex=[read(p) for p in R.glob('execution_*.json')];assert len(ex)==1 and ex[0]['returncode']==0
    plan=read(P/'plan.json');rowsout=table(R/'diagnostic_rows.csv');records=[];allvalid=True
    count=covered=training_rows=0;maxnormal=maxdual=maxpred=maxsumdual=0.
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
                k=3 if source=='TD_only' else 18;reference=a['inner_oof'][:,k].copy();refbeta=a['inner_coef'][:,k].copy()
            pred=np.full(len(ids),np.nan);coverage=[];blocks=[]
            for j,split in enumerate(item['inner']):
                if time.monotonic()-started>900:raise TimeoutError('Independent audit cap')
                tr,va=np.array(split['train']),np.array(split['valid']);out=R/'pilot'/setting/source/f'inner{j}'
                receipt=read(out/'complete.json');check(out,receipt['files_sha256']);assert receipt['lock_sha256']==sha(R/'lock.json')
                state=read(out/'state.json');cert=state['certificate']
                assert (state['setting'],state['source'],state['inner'])==(setting,source,j)
                assert state['train_indices']==tr.tolist() and state['valid_indices']==va.tolist()
                assert state['transform_sha256']==sha(pf) and state['reference_arrays_sha256']==sha(af)
                z=old.project(raw[tr],raw[tr],states['inner'][j]);zv=old.project(raw[tr],raw[va],states['inner'][j])
                if source=='TD_only':z=z[:,:24];zv=zv[:,:24]
                n,p=z.shape;center=float(y[tr].mean());yc=y[tr]-center
                b0=refbeta[j,:p];close((z.T@z+100*np.eye(p))@b0,z.T@yc,1e-8)
                close(zv@b0+center,reference[va],1e-10)
                with np.load(out/'model.npz',allow_pickle=False) as a:m={k:a[k].copy() for k in a.files}
                assert state['saved_float_values']==sum(a.size for a in m.values())
                u,uv=embed(z),embed(zv);close(m['train_u'],u,1e-12);close(m['train_td'],z[:,:24],1e-12)
                rawk=kernel(u,u);np.fill_diagonal(rawk,np.einsum('ij,ij->i',u,u));rawk=(rawk+rawk.T)/2
                means=rawk.mean(1);grand=float(means.mean());kc=rawk-means[:,None]-means[None,:]+grand
                variance=float(np.trace(kc)/n);scale=16/variance if variance>1e-12 else 0.
                gram=z[:,:24]@z[:,:24].T+scale*kc;gram=(gram+gram.T)/2;system=gram+100*np.eye(n)
                chol=np.linalg.cholesky(system);assert np.all(np.diag(chol)>0)
                dual=np.linalg.solve(system,yc);close(m['dual'],dual,1e-9)
                maxdual=max(maxdual,float(abs(m['dual']-dual).max()))
                close(m['raw_row_mean'],means);close(m['raw_grand_mean'],grand);close(m['aux_scale'],scale);close(m['intercept'],center,1e-12)
                close(m['td_coef'],z[:,:24].T@m['dual']);close(cert['aux_variance'],variance)
                normal=float(abs(system@m['dual']-yc).max());limit=float(1e-9*(1+abs(yc).max()))
                close(cert['normal_error'],normal);close(cert['normal_limit'],limit,1e-12)
                valid=bool(normal<=limit);assert valid==cert['numerically_qualified'];allvalid &= valid
                assert cert['alpha']==100 and cert['new_linear_solves']==1 and cert['training_rows']==n and cert['embedding_dim']==p
                close(cert['mean_aux_diagonal'],scale*variance);close(cert['centered_kernel_row_mean_max'],abs(kc.mean(1)).max())
                close(cert['sum_dual'],m['dual'].sum());maxsumdual=max(maxsumdual,float(abs(m['dual'].sum())))
                assert abs(m['dual'].sum())<1e-9
                residual=gram@m['dual']-yc;norm=float(m['dual']@gram@m['dual'])
                close(cert['train_MSE'],np.mean(residual**2));close(cert['RKHS_squared_norm'],norm)
                close(cert['objective'],.5*(residual@residual+100*norm))
                rawv=kernel(uv,u);centeredv=rawv-rawv.mean(1)[:,None]-means[None,:]+grand
                pp=zv[:,:24]@(z[:,:24].T@m['dual'])+scale*centeredv@m['dual']+center
                close(pp,m['prediction']);maxpred=max(maxpred,float(abs(pp-m['prediction']).max()));maxnormal=max(maxnormal,normal)
                pred[va]=m['prediction'];coverage+=va.tolist();count+=1;training_rows+=n
                blocks.append(dict(inner=j,n=len(va),training_rows=n,metrics=metric(y[va],pred[va]),reference=metric(y[va],reference[va]),
                    normal_error=normal,numerically_qualified=valid))
            assert sorted(coverage)==list(range(len(ids))) and np.isfinite(pred).all()
            part=[r for r in rowsout if r['setting']==setting and r['source']==source];assert [r['sample_id'] for r in part]==ids
            for key,values in (('label',y),('prediction',pred),('reference_prediction',reference)):close([float(r[key]) for r in part],values,0)
            covered+=len(part);m,rr=metric(y,pred),metric(y,reference)
            rec=next(r for r in d['records'] if r['setting']==setting and r['source']==source);assert rec['n']==len(ids)
            for name,actual in (('metrics',m),('reference',rr)):
                for k,v in actual.items():close(rec[name][k],v,1e-11)
            delta=m['PCC']-rr['PCC'];red=100*(1-m['MSE']/rr['MSE'])
            close(rec['delta_PCC_vs_R4'],delta,1e-11);close(rec['MSE_reduction_vs_R4_pct'],red,1e-10)
            assert len(rec['inner_blocks'])==5
            for actual,expected in zip(rec['inner_blocks'],blocks):
                for key,value in expected.items():
                    if isinstance(value,dict):
                        for k,v in value.items():close(actual[key][k],v)
                    else:close(actual[key],value)
            records.append(dict(setting=setting,source=source,delta=delta,reduction=red,metrics=m))
    assert count==60==d['new_heads']==d['new_linear_solves']==d['reused_reference_heads']
    assert len(rowsout)==covered==9196==d['new_internal_OOF_rows'] and training_rows==d['stored_training_row_occurrences']
    core=[r for r in records if r['source']=='core_real'];performance=sum(r['delta']>=.005 and r['reduction']>=0 for r in core)>=2 and all(r['delta']>=-.005 for r in core)
    pairs=[]
    for r in core:
        td=next(t for t in records if t['source']=='TD_only' and t['setting']==r['setting'])
        delta=r['metrics']['PCC']-td['metrics']['PCC'];red=100*(1-r['metrics']['MSE']/td['metrics']['MSE'])
        original=next(x for x in d['paired_kernel_TD'] if x['setting']==r['setting'])
        close(original['delta_PCC'],delta,1e-11);close(original['MSE_reduction_pct'],red,1e-10);pairs.append((delta,red))
    incremental=sum(p>=0 for p,r in pairs)>=2 and np.mean([r for p,r in pairs])>=0;qualified=bool(performance and incremental and allvalid)
    assert d['performance_gate']==performance and d['source_increment_gate']==incremental and d['all_numerically_qualified']==allvalid and d['qualified']==qualified
    result=dict(passed=True,qualified=qualified,verified_new_heads=60,verified_reference_heads=60,new_internal_OOF_rows=9196,
        independent_LU_solutions=60,positive_Cholesky_systems=60,stored_training_row_occurrences=training_rows,
        max_normal_equation_error=maxnormal,max_dual_coefficient_difference=maxdual,max_prediction_difference=maxpred,
        max_absolute_sum_dual=maxsumdual,all_numerically_qualified=bool(allvalid),new_source_forward=0,new_PCA=0,GPU_hours=0,
        outer_test_metrics_used=False,lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-started)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0),f,indent=2)
