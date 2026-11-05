"""Independent full-system dual certificates and exact64-mask expectation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,itertools,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4';D=R.parent/'round21';N=R.parent/'round25'
spec=importlib.util.spec_from_file_location('_r36_audit4',P/'verify.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,sha,table,check=old.read,old.sha,old.table,old.check


def close(a,b,tol=1e-9):np.testing.assert_allclose(a,b,rtol=tol,atol=tol)
def metric(y,p):return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))


def certify(z,y,b,c,cert,drop):
    g=z.T@z+100*np.eye(216);rhs=z.T@(y-y.mean());keep=.9
    extra=np.zeros((216,216))
    if drop:
        for j in range(6):
            first=24+32*j;last=first+32;part=z[:,first:last]
            extra[first:last,first:last]=((1-keep)/keep)*np.einsum('ni,nj->ij',part,part)
        g+=extra
        close(cert['dropout_penalty_trace'],np.trace(extra));assert cert['keep_probability']==keep
    close(c,y.mean(),1e-12);td0=np.linalg.solve(g[:24,:24],rhs[:24]);coupling=np.linalg.solve(g[:24,:24],g[:24,24:])
    h=g[24:,24:]-g[24:,:24]@coupling;h=(h+h.T)/2;q=rhs[24:]-g[24:,:24]@td0
    critical=float(np.linalg.norm(q.reshape(6,32),2));lam=.1*critical
    close(cert['lambda_critical'],critical,1e-10);close(cert['lambda_value'],lam,1e-10)
    assert cert['alpha']==100 and cert['lambda_ratio']==.1 and cert['matrix_shape']==[6,32]
    assert cert['mode']==('native_trace_pool_dropout' if drop else 'native_trace')
    singular=np.linalg.svd(b[24:].reshape(6,32),compute_uv=False);close(cert['singular_values'],singular,1e-10)
    assert cert['rank']==int(sum(singular>1e-8))
    normal=float(abs((g@b-rhs)[:24]).max());assert normal<1e-8
    dual=q-h@b[24:];norm=float(np.linalg.norm(dual.reshape(6,32),2))
    if norm>lam:dual*=lam/norm
    close(cert['dual'],dual,1e-9);assert np.linalg.norm(dual.reshape(6,32),2)<=lam+1e-9
    primal=.5*b@g@b-rhs@b+lam*singular.sum();dv=np.r_[np.zeros(24),dual];dualvalue=-.5*(rhs-dv)@np.linalg.solve(g,rhs-dv)
    gap=float(primal-dualvalue);scale=max(1.,.5*q@np.linalg.solve(h,q));constant=.5*rhs[:24]@td0
    close(cert['primal_profile'],primal+constant);close(cert['dual_profile'],dualvalue+constant);close(cert['gap_scale'],scale)
    assert gap>=-1e-8;close(cert['gap'],max(0.,gap));assert cert['certified']==(cert['gap']<=1e-9*cert['gap_scale'])
    valid=bool(cert['certified'] and gap<=1e-9*scale+1e-11)
    assert 0<=cert['iterations']<=5000
    if not cert['certified']:assert cert['iterations']==5000
    mse=float(np.mean((z@b+c-y)**2));close(cert['train_MSE'],mse,1e-11)
    enum_error=0.
    if drop:
        outputs=np.array([z[:,24+32*j:24+32*(j+1)]@b[24+32*j:24+32*(j+1)] for j in range(6)]).T
        close(cert['block_output_RMS'],np.sqrt(np.mean(outputs**2,axis=0)),1e-11)
        expected_extra=float(b@extra@b);close(cert['dropout_extra_SSE'],expected_extra,1e-10)
        meanSSE=0.;prob=0.
        for bits in itertools.product((0,1),repeat=6):
            weight=keep**sum(bits)*(1-keep)**(6-sum(bits));prob+=weight
            pred=z[:,:24]@b[:24]+c+outputs@np.array(bits)/keep
            meanSSE+=weight*float(np.sum((pred-y)**2))
        close(prob,1,1e-12);expected=mse*len(y)+expected_extra;close(meanSSE,expected,1e-10)
        close(cert['expected_train_MSE'],meanSSE/len(y),1e-11);enum_error=abs(meanSSE-expected)
    return valid,normal,abs(gap)/scale,enum_error


def main():
    start=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for path,h in lock['dependencies'].items():assert sha(Path(path))==h,path
    assert lock['alpha']==100 and lock['lambda_ratio']==.1 and lock['keep_probability']==.9 and lock['exact_expectation']
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and not d['outer_test_metrics_used']
    assert d['new_source_forward']==d['new_PCA']==d['GPU_hours']==0 and all(t['num_threads']==1 for t in d['threadpool'])
    check(R/'pilot',d['pilot_files_sha256']);assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    runs=[read(p) for p in R.glob('execution_*.json')];assert len(runs)==1 and runs[0]['returncode']==0
    rowsout=table(R/'diagnostic_rows.csv');plan=read(P/'plan.json');records=[];counts=dict(new=0,old=0,TD=0,rows=0)
    numeric=True;maxnormal=maxgap=maxpred=maxdeploy=maxenum=0.
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids']
        assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        rows={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')};y=np.array([float(rows[s]['label']) for s in ids])
        td=np.array([[float(v) for v in rows[s]['td'].split(',')] for s in ids]);tdpath=P/'results/folds'/setting/'fold0/core_real/arrays.npz'
        with np.load(tdpath,allow_pickle=False) as a:tdpred=a['inner_oof'][:,3].copy();tdcoef=a['inner_coef'][:,3,:24].copy()
        for j,sp in enumerate(item['inner']):
            tr,va=np.array(sp['train']),np.array(sp['valid']);tm,ts=old.scale(td[tr],False);z=(td[tr]-tm)/ts;zv=(td[va]-tm)/ts
            close((z.T@z+100*np.eye(24))@tdcoef[j],z.T@(y[tr]-y[tr].mean()),1e-8)
            close(zv@tdcoef[j]+y[tr].mean(),tdpred[va],1e-10);counts['TD']+=1
        for source in old.SOURCES:
            folder=old.EXPORT/source;identity=read(folder/'complete.json');assert d['source_identities'][source]==identity['source_identity'];check(folder,identity['files_sha256'])
            meta=table(folder/'row_identity.csv');idx={r['sample_id']:i for i,r in enumerate(meta)}
            for s in ids:assert all(rows[s][k]==meta[idx[s]][k] for k in ('sample_id','cohort','guide','target57'))
            raw=np.c_[td,np.load(folder/'source.npy',allow_pickle=False)[[idx[s] for s in ids]].astype(float)]
            pred=np.full(len(ids),np.nan);reference=pred.copy();coverage=[];blocks=[]
            for j,sp in enumerate(item['inner']):
                if time.monotonic()-start>900:raise TimeoutError('CPU audit cap')
                tr,va=np.array(sp['train']),np.array(sp['valid']);out=R/'pilot'/setting/source/f'inner{j}'
                receipt=read(out/'complete.json');check(out,receipt['files_sha256']);assert receipt['lock_sha256']==sha(R/'lock.json')
                state=read(out/'state.json');cert=state['certificate'];control=D/'pilot'/setting/source/'dense'/f'inner{j}';native=N/'pilot'/setting/source/f'inner{j}'
                assert (state['setting'],state['source'],state['inner'])==(setting,source,j)
                assert state['train_indices']==tr.tolist() and state['valid_indices']==va.tolist()
                assert state['transform_sha256']==sha(control/'transform.json') and state['native_reference_sha256']==sha(native/'model.npz') and state['TD_reference_sha256']==sha(tdpath)
                nr=read(native/'complete.json');check(native,nr['files_sha256']);assert nr['lock_sha256']==sha(N/'lock.json')
                ns=read(native/'state.json');assert ns['train_indices']==tr.tolist() and ns['valid_indices']==va.tolist()
                t=read(control/'transform.json');tm,ts=old.scale(raw[tr,:24],False);sm,ss=old.scale(raw[tr,24:],True)
                for key,value in (('td_mean',tm),('td_scale',ts),('src_mean',sm),('src_scale',ss)):close(t[key],value,1e-11)
                mean=np.r_[tm,sm];scale=np.r_[ts,ss*np.sqrt(12.)];z=(raw[tr]-mean)/scale;zv=(raw[va]-mean)/scale
                with np.load(native/'model.npz',allow_pickle=False) as a:nb=a['beta'].copy();nc=float(a['intercept']);npred=a['prediction'].copy()
                nv,_,_,_=certify(z,y[tr],nb,nc,ns['certificate'],False);assert nv;close(zv@nb+nc,npred,1e-10);reference[va]=npred;counts['old']+=1
                with np.load(out/'model.npz',allow_pickle=False) as a:m={k:a[k].copy() for k in a.files}
                b=m['beta'];center=float(m['intercept']);valid,normal,gap,enum=certify(z,y[tr],b,center,cert,True)
                assert valid==state['numerically_qualified'];numeric &= valid
                pp=zv@b+center;close(m['prediction'],pp,1e-10);close(m['deployment_coef'],b/scale,1e-12);close(m['deployment_intercept'],center-mean@(b/scale),1e-11)
                rawp=raw[va]@m['deployment_coef']+m['deployment_intercept'];close(rawp,pp,1e-9)
                maxnormal=max(maxnormal,normal);maxgap=max(maxgap,gap);maxenum=max(maxenum,enum)
                maxpred=max(maxpred,float(abs(m['prediction']-pp).max()));maxdeploy=max(maxdeploy,float(abs(rawp-pp).max()))
                pred[va]=m['prediction'];coverage+=va.tolist();counts['new']+=1
                blocks.append(dict(inner=j,n=len(va),metrics=metric(y[va],pp),reference=metric(y[va],npred),TD_metrics=metric(y[va],tdpred[va]),
                    dropout_extra_SSE=cert['dropout_extra_SSE'],expected_train_MSE=cert['expected_train_MSE'],iterations=cert['iterations'],rank=cert['rank'],numerically_qualified=valid))
            assert sorted(coverage)==list(range(len(ids))) and np.isfinite(pred).all()
            part=[r for r in rowsout if r['setting']==setting and r['source']==source];assert [r['sample_id'] for r in part]==ids
            for key,value in (('label',y),('prediction',pred),('native_reference_prediction',reference),('TD_prediction',tdpred)):
                close([float(r[key]) for r in part],value,0)
            counts['rows']+=len(part);m,rr,tm=metric(y,pred),metric(y,reference),metric(y,tdpred)
            rec=next(r for r in d['records'] if r['setting']==setting and r['source']==source);assert rec['n']==len(ids)
            for field,mm in (('metrics',m),('reference',rr),('TD_metrics',tm)):
                for k,v in mm.items():close(rec[field][k],v,1e-11)
            delta=m['PCC']-rr['PCC'];reduction=100*(1-m['MSE']/rr['MSE'])
            close(rec['delta_PCC_vs_R25'],delta,1e-11);close(rec['MSE_reduction_vs_R25_pct'],reduction,1e-10)
            assert len(rec['inner_blocks'])==5
            for actual,expected in zip(rec['inner_blocks'],blocks):
                for field,value in expected.items():
                    if isinstance(value,dict):
                        for k,v in value.items():close(actual[field][k],v)
                    else:close(actual[field],value)
            records.append(dict(setting=setting,source=source,metrics=m,TD_metrics=tm,delta=delta,reduction=reduction))
    assert counts==dict(new=45,old=45,TD=15,rows=6897) and len(rowsout)==6897
    assert d['new_heads']==d['reused_native_heads']==45 and d['reused_unique_TD_heads']==15 and d['reused_unique_TD_rows']==2299
    core=[r for r in records if r['source']=='core_real'];performance=sum(r['delta']>=.005 and r['reduction']>=0 for r in core)>=2 and all(r['delta']>=-.005 for r in core)
    pairs=[]
    for r in core:
        p=dict(setting=r['setting'],delta_PCC=r['metrics']['PCC']-r['TD_metrics']['PCC'],MSE_reduction_pct=100*(1-r['metrics']['MSE']/r['TD_metrics']['MSE']))
        expected=next(x for x in d['paired_TD'] if x['setting']==r['setting'])
        for field in ('delta_PCC','MSE_reduction_pct'):close(expected[field],p[field],1e-10)
        pairs.append(p)
    increment=sum(p['delta_PCC']>=0 for p in pairs)>=2 and np.mean([p['MSE_reduction_pct'] for p in pairs])>=0
    qualified=bool(performance and increment and numeric)
    assert d['qualified']==qualified and d['performance_gate']==performance and d['source_increment_gate']==increment and d['all_numerically_qualified']==numeric
    result=dict(passed=True,qualified=qualified,all_numerically_qualified=bool(numeric),counts=counts,
        independent_new_full_joint_certificates=45,independent_old_full_joint_certificates=45,TD_normal_equations=15,
        exact_mask_combinations_per_new_head=64,total_mask_loss_evaluations=2880,max_expected_SSE_identity_error=maxenum,
        max_TD_normal_error=maxnormal,max_relative_dual_gap=maxgap,max_prediction_difference=maxpred,max_deployment_difference=maxdeploy,
        GPU_hours=0,new_source_forward=0,new_PCA=0,outer_test_metrics_used=False,
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
