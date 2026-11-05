"""Independent SVD input metric reconstruction and full-system head audits."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,math,time,traceback
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;P=R.parent/'round4';D=R.parent/'round21';N=R.parent/'round25'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


old=load('_r48_independent4',P/'verify.py');certify=load('_r48_independent36',R.parent/'round36/verify.py').certify
read,sha,table,check=old.read,old.sha,old.table,old.check


def close(a,b,tol=1e-9):np.testing.assert_allclose(a,b,rtol=tol,atol=tol)
def metric(y,p):
    assert np.isfinite(p).all() and np.std(p)>1e-14 and np.std(y)>1e-14
    return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))


def geo_stats(h,w):
    return dict(n=len(h),leverage_min=float(min(h)),leverage_max=float(max(h)),leverage_mean=float(sum(h)/len(h)),
        weight_min=float(min(w)),weight_mean=float(sum(w)/len(w)),clipped_count=int(sum(v<1 for v in w)))


def compare_mapping(actual,expected):
    assert set(actual)==set(expected)
    for k,value in expected.items():
        if isinstance(value,dict):compare_mapping(actual[k],value)
        elif isinstance(value,(str,bool)) or value is None:assert actual[k]==value,k
        else:close(actual[k],value)


def gate_audit(x,xv,m,state):
    n,p=x.shape;_,singular,vt=np.linalg.svd(x,full_matrices=False)
    precision=np.eye(p)/100+vt.T@((1/(singular**2+100)-.01)[:,None]*vt)
    close((x.T@x+100*np.eye(p))@precision,np.eye(p),1e-9)
    close(m['gate_precision'],precision,1e-10)
    h=np.array([float(row@precision@row) for row in x]);hv=np.array([float(row@precision@row) for row in xv])
    assert h.min()>=-1e-12 and hv.min()>=-1e-12;h=np.maximum(h,0);hv=np.maximum(hv,0)
    index=math.ceil(.9*(n-1));threshold=float(sorted(h)[index])
    def calc_weight(values):return np.array([1. if v<=threshold else math.sqrt(threshold/v) for v in values])
    w,wv=calc_weight(h),calc_weight(hv);mean=np.sum(w[:,None]*x,axis=0)/n
    close(m['gate_mean'],mean,1e-10)
    expected=dict(threshold=threshold,quantile=.9,quantile_method='higher',quantile_index=index,metric_ridge=100.,
        training_n=n,feature_dim=p,training_leverage_sum=float(sum(h)),training_leverage_min=float(min(h)),training_leverage_max=float(max(h)),
        training_clipped_count=int(sum(v<1 for v in w)),training_weight_min=float(min(w)),training_weight_mean=float(sum(w)/n),
        input_only=True,calibrated_predictive_uncertainty=False)
    compare_mapping(state,expected)
    for name,value in (('train_leverage',h),('train_weight',w),('valid_leverage',hv),('valid_weight',wv)):close(m[name],value,1e-10)
    close(h.sum(),np.sum(singular**2/(singular**2+100)),1e-10)
    assert np.max(w*w*h)<=threshold+1e-10 and np.max(wv*wv*hv)<=threshold+1e-10
    z=w[:,None]*x-mean;zv=wv[:,None]*xv-mean;close(z.mean(0),np.zeros(p),1e-12)
    # Single-row replay must not depend on the composition of the validation batch.
    one=np.array([(1. if float(r@precision@r)<=threshold else math.sqrt(threshold/float(r@precision@r)))*r-mean for r in xv])
    close(zv,one,1e-11)
    return z,zv,h,w,hv,wv,float(np.max(abs(m['gate_precision']-precision)))


def audit_TD(z,y,b,c,cert):
    assert cert['mode']=='TD_leverage_gated_Ridge' and cert['alpha']==100
    close(c,y.mean(),1e-12)
    expected=np.linalg.lstsq(np.r_[z,10*np.eye(24)],np.r_[y-y.mean(),np.zeros(24)],rcond=None)[0]
    close(b,expected,1e-10);rhs=z.T@(y-y.mean());normal=float(max(abs((z.T@z+100*np.eye(24))@b-rhs)))
    threshold=1e-8*(1+float(max(abs(rhs))));valid=normal<=threshold
    close(cert['normal_max'],normal,1e-10);close(cert['normal_threshold'],threshold,1e-10)
    close(cert['train_MSE'],np.mean((z@b+c-y)**2),1e-11);assert cert['certified']==valid
    return valid,normal,float(np.max(abs(b-expected)))


def main():
    start=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    assert all(t['num_threads']==1 for t in threadpool_info())
    lock=read(R/'lock.json');check(R,lock['files'])
    for path,h in lock['dependencies'].items():assert sha(Path(path))==h,path
    assert lock['alpha']==lock['metric_ridge']==100 and lock['lambda_ratio']==.1 and lock['quantile']==.9 and lock['quantile_method']=='higher'
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and not d['outer_test_metrics_used']
    assert d['new_source_forward']==d['new_PCA']==d['new_original_scalers']==d['GPU_hours']==0
    assert all(t['num_threads']==1 for t in d['threadpool'])
    check(R/'pilot',d['pilot_files_sha256']);assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    runs=[read(p) for p in R.glob('execution_*.json')];assert len(runs)==1 and runs[0]['returncode']==0
    rowsout=table(R/'diagnostic_rows.csv');plan=read(P/'plan.json');records=[]
    counts=dict(new_source=0,new_TD=0,old_source=0,old_TD=0,rows=0,input_geometries=0)
    numeric=True;maxnormal=maxgap=maxpred=maxdeploy=maxlstsq=maxprecision=0.
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids']
        assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        rows={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')};y=np.array([float(rows[s]['label']) for s in ids])
        td=np.array([[float(v) for v in rows[s]['td'].split(',')] for s in ids]);tdpath=P/'results/folds'/setting/'fold0/core_real/arrays.npz'
        with np.load(tdpath,allow_pickle=False) as a:
            tdpred=a['inner_oof'][:,3].copy();tdcoef=a['inner_coef'][:,3,:24].copy();tdintercept=a['inner_intercept'].copy()
        assert tdintercept.shape==(5,)
        for j,sp in enumerate(item['inner']):
            tr,va=np.array(sp['train']),np.array(sp['valid']);tm,ts=old.scale(td[tr],False);z=(td[tr]-tm)/ts;zv=(td[va]-tm)/ts
            close(tdintercept[j],y[tr].mean(),1e-12)
            close((z.T@z+100*np.eye(24))@tdcoef[j],z.T@(y[tr]-y[tr].mean()),1e-8)
            close(zv@tdcoef[j]+tdintercept[j],tdpred[va],1e-10);counts['old_TD']+=1
        for source in ('TD_only',)+tuple(old.SOURCES):
            is_source=source!='TD_only';origin=source if is_source else 'core_real'
            folder=old.EXPORT/origin;identity=read(folder/'complete.json')
            assert d['source_identities'][origin]==identity['source_identity'];check(folder,identity['files_sha256'])
            meta=table(folder/'row_identity.csv');idx={r['sample_id']:i for i,r in enumerate(meta)}
            for s in ids:assert all(rows[s][k]==meta[idx[s]][k] for k in ('sample_id','cohort','guide','target57'))
            raw=np.c_[td,np.load(folder/'source.npy',allow_pickle=False)[[idx[s] for s in ids]].astype(float)]
            pred=np.full(len(ids),np.nan);reference=pred.copy();allh=pred.copy();allw=pred.copy();coverage=[];blocks=[]
            for j,sp in enumerate(item['inner']):
                if time.monotonic()-start>900:raise TimeoutError('CPU audit cap')
                tr,va=np.array(sp['train']),np.array(sp['valid']);assert set(tr).isdisjoint(va)
                out=R/'pilot'/setting/source/f'inner{j}';receipt=read(out/'complete.json');check(out,receipt['files_sha256'])
                assert receipt['lock_sha256']==sha(R/'lock.json');state=read(out/'state.json');cert=state['certificate']
                control=D/'pilot'/setting/origin/'dense'/f'inner{j}';native=N/'pilot'/setting/origin/f'inner{j}'
                assert (state['setting'],state['source'],state['inner'])==(setting,source,j)
                assert state['train_indices']==tr.tolist() and state['valid_indices']==va.tolist()
                assert state['transform_sha256']==sha(control/'transform.json') and state['TD_reference_sha256']==sha(tdpath)
                t=read(control/'transform.json');tm,ts=old.scale(raw[tr,:24],False);sm,ss=old.scale(raw[tr,24:],True)
                for key,value in (('td_mean',tm),('td_scale',ts),('src_mean',sm),('src_scale',ss)):close(t[key],value,1e-11)
                mean=np.r_[tm,sm];scale=np.r_[ts,ss*np.sqrt(12.)];z=(raw[tr]-mean)/scale;zv=(raw[va]-mean)/scale
                if is_source:
                    assert state['native_reference_sha256']==sha(native/'model.npz')
                    nr=read(native/'complete.json');check(native,nr['files_sha256']);assert nr['lock_sha256']==sha(N/'lock.json')
                    ns=read(native/'state.json');assert ns['train_indices']==tr.tolist() and ns['valid_indices']==va.tolist()
                    assert ns['transform_sha256']==sha(control/'transform.json')
                    with np.load(native/'model.npz',allow_pickle=False) as a:nb=a['beta'].copy();nc=float(a['intercept']);npred=a['prediction'].copy()
                    nv,normal,gap,_=certify(z,y[tr],nb,nc,ns['certificate'],False);assert nv
                    close(zv@nb+nc,npred,1e-10);reference[va]=npred;counts['old_source']+=1
                    maxnormal=max(maxnormal,normal);maxgap=max(maxgap,gap)
                else:
                    assert state['native_reference_sha256'] is None;reference[va]=tdpred[va]
                with np.load(out/'model.npz',allow_pickle=False) as a:m={k:a[k].copy() for k in a.files}
                x=z[:,24:] if is_source else z[:,:24];xv=zv[:,24:] if is_source else zv[:,:24]
                gx,gxv,h,w,hv,wv,precision_error=gate_audit(x,xv,m,state['gate']);counts['input_geometries']+=1
                design=np.c_[z[:,:24],gx] if is_source else gx
                valid_design=np.c_[zv[:,:24],gxv] if is_source else gxv
                b=m['beta'];c=float(m['intercept'])
                if is_source:
                    valid,normal,gap,_=certify(design,y[tr],b,c,cert,False);maxgap=max(maxgap,gap);counts['new_source']+=1
                else:
                    valid,normal,lstsq_error=audit_TD(design,y[tr],b,c,cert);maxlstsq=max(maxlstsq,lstsq_error);counts['new_TD']+=1
                numeric &= valid;assert valid==state['numerically_qualified']
                pp=valid_design@b+c;close(m['prediction'],pp,1e-10)
                # Explicit saved-state deployment, not an incorrect single raw linear coefficient vector.
                actual_h=np.einsum('ni,ni->n',xv,xv@m['gate_precision'])
                stored_tau=state['gate']['threshold'];actual_w=np.array([1. if v<=stored_tau else math.sqrt(stored_tau/v) for v in actual_h])
                if is_source:rawp=((raw[va,:24]-tm)/ts)@b[:24]+actual_w*(((raw[va,24:]-sm)/(ss*np.sqrt(12)))@b[24:])-m['gate_mean']@b[24:]+c
                else:rawp=actual_w*(((raw[va,:24]-tm)/ts)@b)-m['gate_mean']@b+c
                close(rawp,pp,1e-9)
                maxnormal=max(maxnormal,normal);maxprecision=max(maxprecision,precision_error)
                maxpred=max(maxpred,float(abs(m['prediction']-pp).max()));maxdeploy=max(maxdeploy,float(abs(rawp-pp).max()))
                pred[va]=m['prediction'];allh[va]=m['valid_leverage'];allw[va]=m['valid_weight'];coverage+=va.tolist()
                blocks.append(dict(inner=j,n=len(va),metrics=metric(y[va],pp),reference=metric(y[va],reference[va]),TD_metrics=metric(y[va],tdpred[va]),
                    threshold=state['gate']['threshold'],training_geometry=geo_stats(h,w),validation_geometry=geo_stats(hv,wv),numerically_qualified=valid))
            assert sorted(coverage)==list(range(len(ids))) and np.isfinite(pred).all()
            part=[r for r in rowsout if r['setting']==setting and r['source']==source];assert [r['sample_id'] for r in part]==ids
            for key,value in (('label',y),('prediction',pred),('reference_prediction',reference),('TD_prediction',tdpred),('leverage',allh),('gate_weight',allw)):
                close([float(r[key]) for r in part],value,0)
            counts['rows']+=len(part);m,rr,tm=metric(y,pred),metric(y,reference),metric(y,tdpred)
            rec=next(r for r in d['records'] if r['setting']==setting and r['source']==source);assert rec['n']==len(ids)
            for field,mm in (('metrics',m),('reference',rr),('TD_metrics',tm)):compare_mapping(rec[field],mm)
            delta=m['PCC']-rr['PCC'];reduction=100*(1-m['MSE']/rr['MSE'])
            close(rec['delta_PCC_vs_reference'],delta,1e-11);close(rec['MSE_reduction_vs_reference_pct'],reduction,1e-10)
            assert len(rec['inner_blocks'])==5
            for actual,expected in zip(rec['inner_blocks'],blocks):compare_mapping(actual,expected)
            records.append(dict(setting=setting,source=source,metrics=m,delta=delta,reduction=reduction))
    assert counts==dict(new_source=45,new_TD=15,old_source=45,old_TD=15,rows=9196,input_geometries=60) and len(rowsout)==9196
    assert len(d['records'])==12 and len(d['pilot_files_sha256'])==180
    assert d['new_heads']==d['new_input_geometries']==d['new_gate_centering_means']==60
    assert d['new_source_heads']==d['reused_native_heads']==45
    assert d['new_TD_gated_heads']==d['reused_unique_TD_heads']==15 and d['reused_unique_TD_rows']==2299 and d['new_internal_OOF_rows']==9196
    core=[r for r in records if r['source']=='core_real']
    performance=sum(r['delta']>=.005 and r['reduction']>=0 for r in core)>=2 and all(r['delta']>=-.005 for r in core)
    pairs=[]
    for r in core:
        td=next(a for a in records if a['setting']==r['setting'] and a['source']=='TD_only')
        p=dict(setting=r['setting'],delta_PCC=r['metrics']['PCC']-td['metrics']['PCC'],MSE_reduction_pct=100*(1-r['metrics']['MSE']/td['metrics']['MSE']),
            delta_gain_above_TD_gate=r['delta']-td['delta'])
        expected=next(x for x in d['paired_gated_TD'] if x['setting']==r['setting']);compare_mapping(expected,p);pairs.append(p)
    increment=sum(p['delta_PCC']>=0 for p in pairs)>=2 and np.mean([p['MSE_reduction_pct'] for p in pairs])>=0
    qualified=bool(performance and increment and numeric)
    assert d['qualified']==qualified and d['performance_gate']==performance and d['source_increment_vs_gated_TD']==increment and d['all_numerically_qualified']==numeric
    result=dict(passed=True,qualified=qualified,all_numerically_qualified=bool(numeric),counts=counts,
        independent_SVD_geometry_checks=60,independent_new_full_joint_certificates=45,independent_old_full_joint_certificates=45,
        independent_TD_augmented_lstsq=15,old_TD_normal_equations=15,max_normal_error=maxnormal,max_relative_dual_gap=maxgap,
        max_independent_TD_lstsq_difference=maxlstsq,max_precision_difference=maxprecision,max_prediction_difference=maxpred,max_deployment_difference=maxdeploy,
        GPU_hours=0,new_source_forward=0,new_PCA=0,new_original_scalers=0,outer_test_metrics_used=False,
        lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-start)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0),f,indent=2)
