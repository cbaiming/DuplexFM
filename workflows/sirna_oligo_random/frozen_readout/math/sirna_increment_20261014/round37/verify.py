"""Independent input alignment, missing-energy map, and full-system certificates."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4';D=R.parent/'round21';N=R.parent/'round25'
s=importlib.util.spec_from_file_location('_r37_audit4',P/'verify.py');old=importlib.util.module_from_spec(s);s.loader.exec_module(old)
s=importlib.util.spec_from_file_location('_r37_oldcert',R.parent/'round36/verify.py');previous=importlib.util.module_from_spec(s);s.loader.exec_module(previous)
read,sha,table,check=old.read,old.sha,old.table,old.check


def close(a,b,tol=1e-9):np.testing.assert_allclose(a,b,rtol=tol,atol=tol)
def metric(y,p):return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))


def independent_map(e,tr,t):
    assert e.shape[1]==4 and np.isfinite(e).all() and np.isin(e[:,3],[0.,1.]).all()
    valid=e[:,3].astype(bool);selected=tr[valid[tr]];n=len(selected);f=e[tr,3]
    mean=e[selected,:3].mean(0) if n else np.zeros(3)
    scale=e[selected,:3].std(0) if n else np.ones(3);scale=np.where(scale<1e-12,1.,scale)
    flagmean=float(f.mean());flagsd=float(f.std());flags=flagsd if flagsd>=1e-12 else 1.
    assert t['enabled']==(n>0) and t['valid_training_rows']==n and t['flag_enabled']==(flagsd>=1e-12)
    for key,value in (('mean',mean),('scale',scale),('flag_mean',flagmean),('flag_scale',flags)):close(t[key],value,1e-12)
    z=np.zeros_like(e)
    if n:
        z[valid,:3]=(e[valid,:3]-mean)/scale
        if flagsd>=1e-12:z[:,3]=(e[:,3]-flagmean)/flags
    close(z[tr].mean(0),0,1e-12);assert not z[~valid,:3].any()
    return z


def certify(z,y,b,c,cert):
    assert z.shape[1]==220 and len(b)==220;close(z.mean(0),0,1e-9);close(c,y.mean(),1e-12)
    g=z.T@z+100*np.eye(220);rhs=z.T@(y-y.mean());a=g[:28,:28]
    stat0=np.linalg.solve(a,rhs[:28]);coupling=np.linalg.solve(a,g[:28,28:])
    h=g[28:,28:]-g[28:,:28]@coupling;h=(h+h.T)/2;q=rhs[28:]-g[28:,:28]@stat0
    critical=float(np.linalg.norm(q.reshape(6,32),2));lam=.1*critical
    close(cert['lambda_critical'],critical,1e-10);close(cert['lambda_value'],lam,1e-10)
    assert cert['mode']=='native_trace_energy' and cert['alpha']==100 and cert['lambda_ratio']==.1 and cert['matrix_shape']==[6,32]
    singular=np.linalg.svd(b[28:].reshape(6,32),compute_uv=False);close(cert['singular_values'],singular,1e-10)
    assert cert['rank']==int(sum(singular>1e-8));normal=float(abs((g@b-rhs)[:28]).max());assert normal<1e-8
    dual=q-h@b[28:];norm=float(np.linalg.norm(dual.reshape(6,32),2))
    if norm>lam:dual*=lam/norm
    close(cert['dual'],dual);assert np.linalg.norm(dual.reshape(6,32),2)<=lam+1e-9
    primal=.5*b@g@b-rhs@b+lam*singular.sum();dv=np.r_[np.zeros(28),dual]
    dualvalue=-.5*(rhs-dv)@np.linalg.solve(g,rhs-dv);gap=float(primal-dualvalue)
    scale=max(1.,.5*q@np.linalg.solve(h,q));constant=.5*rhs[:28]@stat0
    close(cert['primal_profile'],primal+constant);close(cert['dual_profile'],dualvalue+constant);close(cert['gap_scale'],scale)
    assert gap>=-1e-8;close(cert['gap'],max(0.,gap));assert cert['certified']==(cert['gap']<=1e-9*cert['gap_scale'])
    valid=bool(cert['certified'] and gap<=1e-9*scale+1e-11)
    assert 0<=cert['iterations']<=5000
    if not cert['certified']:assert cert['iterations']==5000
    close(cert['train_MSE'],np.mean((z@b+c-y)**2),1e-11)
    return valid,normal,abs(gap)/scale


def audit_deployment(raw,e,st,t,b,c,m,z):
    coef=np.zeros_like(b);coef[:24]=b[:24]/st['td_scale'];offset=c-np.asarray(st['td_mean'])@coef[:24]
    if t['enabled']:
        coef[24:27]=b[24:27]/t['scale'];coef[27]=-np.asarray(t['mean'])@coef[24:27]
        if t['flag_enabled']:
            coef[27]+=b[27]/t['flag_scale'];offset-=t['flag_mean']*b[27]/t['flag_scale']
    if len(b)==220:
        coef[28:]=b[28:]/np.asarray(st['src_scale'])/np.sqrt(12.);offset-=np.asarray(st['src_mean'])@coef[28:]
    close(m['deployment_coef'],coef,1e-12);close(m['deployment_intercept'],offset,1e-11)
    data=np.c_[raw[:,:24],e[:,:3]*e[:,3,None],e[:,3],raw[:,24:]][:,:len(b)]
    predicted=data@m['deployment_coef']+m['deployment_intercept'];expected=z[:,:len(b)]@b+c
    close(predicted,expected,1e-9);return float(abs(predicted-expected).max())


def main():
    start=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    assert lock['alpha']==100 and lock['lambda_ratio']==.1 and lock['native_dimensions']==220
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and not d['outer_test_metrics_used']
    assert d['new_source_forward']==d['new_PCA']==d['GPU_hours']==0 and all(t['num_threads']==1 for t in d['threadpool'])
    check(R/'pilot',d['pilot_files_sha256']);assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    runs=[read(p) for p in R.glob('execution_*.json')];assert len(runs)==1 and runs[0]['returncode']==0
    rowsout=table(R/'diagnostic_rows.csv');plan=read(P/'plan.json');records=[];tdrecords=[]
    counts=dict(new_native=0,new_TD28=0,old_native=0,old_TD24=0,rows=0)
    numeric=True;maxnormal=maxgap=maxpred=maxdeploy=maxtd=0.
    static_hashes=set()
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids']
        assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        rows={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')};y=np.array([float(rows[s]['label']) for s in ids])
        td=np.array([[float(v) for v in rows[s]['td'].split(',')] for s in ids]);tdpath=P/'results/folds'/setting/'fold0/core_real/arrays.npz'
        with np.load(tdpath,allow_pickle=False) as a:td24=a['inner_oof'][:,3].copy();tdcoef=a['inner_coef'][:,3,:24].copy()
        td28=np.full(len(ids),np.nan)
        for j,sp in enumerate(item['inner']):
            tr,va=np.array(sp['train']),np.array(sp['valid']);tm,ts=old.scale(td[tr],False);z=(td[tr]-tm)/ts;zv=(td[va]-tm)/ts
            close((z.T@z+100*np.eye(24))@tdcoef[j],z.T@(y[tr]-y[tr].mean()),1e-8)
            close(zv@tdcoef[j]+y[tr].mean(),td24[va],1e-10);counts['old_TD24']+=1
        same_static=None
        for source in old.SOURCES:
            folder=old.EXPORT/source;identity=read(folder/'complete.json');assert d['source_identities'][source]==identity['source_identity'];check(folder,identity['files_sha256'])
            static_hashes.add(sha(folder/'static28.npy'));assert sha(folder/'static28.npy')==d['static28_sha256'][source]
            meta=table(folder/'row_identity.csv');idx={r['sample_id']:i for i,r in enumerate(meta)}
            for sid in ids:assert all(rows[sid][k]==meta[idx[sid]][k] for k in ('sample_id','cohort','guide','target57'))
            take=[idx[sid] for sid in ids];raw=np.c_[td,np.load(folder/'source.npy',allow_pickle=False)[take].astype(float)]
            static=np.load(folder/'static28.npy',allow_pickle=False)[take];assert np.array_equal(static[:,:24],td)
            if same_static is None:same_static=static
            else:assert np.array_equal(static,same_static)
            e=static[:,24:];assert not e[e[:,3]==0,:3].any()
            assert d['energy_coverage'][setting]==dict(rows=len(ids),valid=int(e[:,3].sum()),invalid=int((1-e[:,3]).sum()))
            pred=np.full(len(ids),np.nan);reference=pred.copy();coverage=[];blocks=[]
            for j,sp in enumerate(item['inner']):
                if time.monotonic()-start>900:raise TimeoutError('CPU audit cap')
                tr,va=np.array(sp['train']),np.array(sp['valid']);out=R/'pilot'/setting/source/f'inner{j}'
                receipt=read(out/'complete.json');check(out,receipt['files_sha256']);assert receipt['lock_sha256']==sha(R/'lock.json')
                state=read(out/'state.json');cert=state['certificate'];control=D/'pilot'/setting/source/'dense'/f'inner{j}';native=N/'pilot'/setting/source/f'inner{j}'
                assert (state['setting'],state['source'],state['inner'])==(setting,source,j)
                assert state['train_indices']==tr.tolist() and state['valid_indices']==va.tolist()
                assert state['transform_sha256']==sha(control/'transform.json') and state['native_reference_sha256']==sha(native/'model.npz')
                tout=R/'pilot'/setting/'TD_energy28'/f'inner{j}'
                assert state['energy_transform_sha256']==sha(tout/'energy_transform.json') and state['TD28_reference_sha256']==sha(tout/'model.npz')
                nr=read(native/'complete.json');check(native,nr['files_sha256']);assert nr['lock_sha256']==sha(N/'lock.json')
                ns=read(native/'state.json');assert ns['train_indices']==tr.tolist() and ns['valid_indices']==va.tolist()
                st=read(control/'transform.json');tm,ts=old.scale(raw[tr,:24],False);sm,ss=old.scale(raw[tr,24:],True)
                for key,value in (('td_mean',tm),('td_scale',ts),('src_mean',sm),('src_scale',ss)):close(st[key],value,1e-11)
                base=np.c_[(raw[:,:24]-tm)/ts,(raw[:,24:]-sm)/(ss*np.sqrt(12.))]
                t=read(tout/'energy_transform.json');emap=independent_map(e,tr,t);z=np.c_[base[:,:24],emap,base[:,24:]]
                with np.load(native/'model.npz',allow_pickle=False) as a:nb=a['beta'].copy();nc=float(a['intercept']);npred=a['prediction'].copy()
                nv,_,_,_=previous.certify(base[tr],y[tr],nb,nc,ns['certificate'],False);assert nv
                close(base[va]@nb+nc,npred,1e-10);reference[va]=npred;counts['old_native']+=1
                if source=='core_real':
                    tc=read(tout/'complete.json');check(tout,tc['files_sha256']);assert tc['lock_sha256']==sha(R/'lock.json')
                    state28=read(tout/'state.json');assert state28['train_indices']==tr.tolist() and state28['valid_indices']==va.tolist()
                    assert state28['transform_sha256']==sha(control/'transform.json') and state28['energy_transform_sha256']==sha(tout/'energy_transform.json')
                    assert (state28['setting'],state28['source'],state28['inner'])==(setting,'TD_energy28',j)
                    with np.load(tout/'model.npz',allow_pickle=False) as a:m28={k:a[k].copy() for k in a.files}
                    expected=np.linalg.lstsq(np.r_[z[tr,:28],10*np.eye(28)],np.r_[y[tr]-y[tr].mean(),np.zeros(28)],rcond=None)[0]
                    close(m28['beta'],expected,1e-11);close(m28['intercept'],y[tr].mean(),1e-12)
                    error=float(abs(expected-m28['beta']).max());maxtd=max(maxtd,error)
                    pp28=z[va,:28]@expected+y[tr].mean();close(m28['prediction'],pp28,1e-10);td28[va]=m28['prediction']
                    c28=state28['certificate'];assert c28['mode']=='TD_energy28' and c28['alpha']==100 and c28['certified'] and state28['numerically_qualified']
                    residual=(z[tr,:28].T@z[tr,:28]+100*np.eye(28))@m28['beta']-z[tr,:28].T@(y[tr]-y[tr].mean())
                    assert abs(residual).max()<1e-8;close(c28['normal_error'],abs(residual).max(),1e-10)
                    close(c28['train_MSE'],np.mean((z[tr,:28]@expected+y[tr].mean()-y[tr])**2),1e-11)
                    maxdeploy=max(maxdeploy,audit_deployment(raw,e,st,t,m28['beta'],float(m28['intercept']),m28,z));counts['new_TD28']+=1
                with np.load(out/'model.npz',allow_pickle=False) as a:m={k:a[k].copy() for k in a.files}
                b,c=m['beta'],float(m['intercept']);valid,normal,gap=certify(z[tr],y[tr],b,c,cert)
                assert valid==state['numerically_qualified'];numeric &= valid
                pp=z[va]@b+c;close(m['prediction'],pp,1e-10)
                maxdeploy=max(maxdeploy,audit_deployment(raw,e,st,t,b,c,m,z));maxnormal=max(maxnormal,normal);maxgap=max(maxgap,gap)
                maxpred=max(maxpred,float(abs(m['prediction']-pp).max()));pred[va]=m['prediction'];coverage+=va.tolist();counts['new_native']+=1
                blocks.append(dict(inner=j,n=len(va),metrics=metric(y[va],pp),reference=metric(y[va],npred),TD28_metrics=metric(y[va],td28[va]),iterations=cert['iterations'],rank=cert['rank'],numerically_qualified=valid))
            assert sorted(coverage)==list(range(len(ids))) and np.isfinite(pred).all()
            part=[r for r in rowsout if r['setting']==setting and r['source']==source];assert [r['sample_id'] for r in part]==ids and all(r['reference_kind']=='R25' for r in part)
            for key,value in (('label',y),('prediction',pred),('reference_prediction',reference),('TD28_prediction',td28),('TD24_prediction',td24)):close([float(r[key]) for r in part],value,0)
            counts['rows']+=len(part);m,rr,tm=metric(y,pred),metric(y,reference),metric(y,td28)
            rec=next(r for r in d['records'] if r['setting']==setting and r['source']==source);assert rec['n']==len(ids)
            for field,mm in (('metrics',m),('reference',rr),('TD28_metrics',tm),('TD24_metrics',metric(y,td24))):
                for k,val in mm.items():close(rec[field][k],val,1e-11)
            delta=m['PCC']-rr['PCC'];reduction=100*(1-m['MSE']/rr['MSE'])
            close(rec['delta_PCC_vs_R25'],delta,1e-11);close(rec['MSE_reduction_vs_R25_pct'],reduction,1e-10);assert len(rec['inner_blocks'])==5
            for actual,expected in zip(rec['inner_blocks'],blocks):
                for field,value in expected.items():
                    if isinstance(value,dict):
                        for k,val in value.items():close(actual[field][k],val)
                    else:close(actual[field],value)
            records.append(dict(setting=setting,source=source,metrics=m,TD28_metrics=tm,delta=delta,reduction=reduction))
        part=[r for r in rowsout if r['setting']==setting and r['source']=='TD_energy28'];assert [r['sample_id'] for r in part]==ids and all(r['reference_kind']=='TD24' for r in part)
        for key,value in (('label',y),('prediction',td28),('reference_prediction',td24),('TD28_prediction',td28),('TD24_prediction',td24)):close([float(r[key]) for r in part],value,0)
        counts['rows']+=len(part);t28,t24=metric(y,td28),metric(y,td24);rec=next(r for r in d['TD28_records'] if r['setting']==setting)
        assert rec['n']==len(ids) and rec['source']=='TD_energy28'
        for field,mm in (('metrics',t28),('TD24_metrics',t24)):
            for k,val in mm.items():close(rec[field][k],val,1e-11)
        close(rec['delta_PCC'],t28['PCC']-t24['PCC']);close(rec['MSE_reduction_pct'],100*(1-t28['MSE']/t24['MSE']))
    assert counts==dict(new_native=45,new_TD28=15,old_native=45,old_TD24=15,rows=9196) and len(rowsout)==9196 and len(static_hashes)==1
    assert d['new_native_heads']==45 and d['new_TD28_heads']==15 and d['new_heads']==60 and d['new_internal_OOF_rows']==9196
    assert d['reused_native_heads']==45 and d['reused_unique_TD24_heads']==15
    core=[r for r in records if r['source']=='core_real'];performance=sum(r['delta']>=.005 and r['reduction']>=0 for r in core)>=2 and all(r['delta']>=-.005 for r in core)
    pairs=[]
    for r in core:
        pair=dict(setting=r['setting'],delta_PCC=r['metrics']['PCC']-r['TD28_metrics']['PCC'],MSE_reduction_pct=100*(1-r['metrics']['MSE']/r['TD28_metrics']['MSE']))
        expected=next(x for x in d['paired_TD28'] if x['setting']==r['setting'])
        for field in ('delta_PCC','MSE_reduction_pct'):close(expected[field],pair[field],1e-10)
        pairs.append(pair)
    increment=sum(p['delta_PCC']>=0 for p in pairs)>=2 and np.mean([p['MSE_reduction_pct'] for p in pairs])>=0;qualified=bool(performance and increment and numeric)
    assert d['qualified']==qualified and d['performance_gate']==performance and d['source_increment_gate']==increment and d['all_numerically_qualified']==numeric
    result=dict(passed=True,qualified=qualified,all_numerically_qualified=bool(numeric),counts=counts,independent_new_joint_certificates=45,
        independent_old_joint_certificates=45,independent_TD28_augmented_lstsq=15,old_TD24_normal_equations=15,
        max_static_normal_error=maxnormal,max_relative_dual_gap=maxgap,max_prediction_difference=maxpred,max_deployment_difference=maxdeploy,
        max_TD28_lstsq_difference=maxtd,GPU_hours=0,new_source_forward=0,new_PCA=0,outer_test_metrics_used=False,
        lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-start)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0),f,indent=2)
