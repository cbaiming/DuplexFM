"""Independent explicit-indicator profiling and augmented nuisance least squares."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib,importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent;B=R.parent;A=B/'round49';D=B/'round21';P=B/'round4'
s=importlib.util.spec_from_file_location('_r54_audit49',A/'verify.py');ref=importlib.util.module_from_spec(s);s.loader.exec_module(ref)
old=ref.old;read,sha,table,check=ref.read,ref.sha,ref.table,ref.check
close,metric,compare=ref.close,ref.metric,ref.compare_mapping


def indicator_geometry(labels):
    n=len(labels);k=int(max(labels))+1;G=np.eye(k)[labels];sizes=G.sum(0);rho=n/k
    inverse=1/(rho+sizes);w=np.ones(n)-G@(inverse*(G.T@np.ones(n)))
    gamma=np.array([w[np.flatnonzero(labels==j)[0]] for j in range(k)])
    trace=float(n-np.trace(np.diag(inverse)@(G.T@G))-(w@w)/w.sum())
    c=(n-1)/trace
    meta=dict(n=n,K=k,sizes=sizes.astype(int).tolist(),common_group_penalty=float(rho),group_gamma=gamma.tolist(),
        centered_metric_trace=trace,normalizer=float(c),reference_equal=bool(len(set(sizes))==1),
        gamma_min=float(gamma.min()),gamma_max=float(gamma.max()),training_inputs_only=True,
        variance_estimated=False,confirmed_gene_or_assay=False)
    return G,inverse,w,meta


def profile(z,y,labels):
    G,inverse,w,meta=indicator_geometry(labels);mz=z-G@(inverse[:,None]*(G.T@z));my=y-G@(inverse*(G.T@y))
    zz=z.T@mz-np.outer(z.T@w,z.T@w)/w.sum()
    rhs=z.T@my-(z.T@w)*(w@y)/w.sum();c=meta['normalizer']
    return c*(zz+zz.T)/2,c*rhs,float(c*(y@my-(w@y)**2/w.sum())),(w@z)/w.sum(),float((w@y)/w.sum()),G,w,meta


def certify_source(z,y,g,b,center,cert,reused=False):
    data,rhs,quad,zm,ym,G,w,meta=profile(z,y,g);gram=data+100*np.eye(216)
    close(center,ym-zm@b,1e-11);close(w@(y-z@b-center),0,1e-9)
    td0=np.linalg.solve(gram[:24,:24],rhs[:24]);coupling=np.linalg.solve(gram[:24,:24],gram[:24,24:])
    h=gram[24:,24:]-gram[24:,:24]@coupling;h=(h+h.T)*.5;q=rhs[24:]-gram[24:,:24]@td0
    critical=float(np.linalg.norm(q.reshape(6,32),2));lam=.1*critical
    assert cert['alpha']==100 and cert['lambda_ratio']==.1 and list(cert['matrix_shape'])==[6,32]
    assert cert['mode']==('native_trace' if reused else 'native_trace_common_target_intercept')
    close(cert['lambda_critical'],critical,1e-10);close(cert['lambda_value'],lam,1e-10)
    singular=np.linalg.svd(b[24:].reshape(6,32),compute_uv=False);close(cert['singular_values'],singular,1e-10)
    assert cert['rank']==int(sum(singular>1e-8))
    normal=float(np.max(abs((gram@b-rhs)[:24])));assert normal<1e-8
    dual=q-h@b[24:];norm=float(np.linalg.norm(dual.reshape(6,32),2))
    if norm>lam:dual*=lam/norm
    close(cert['dual'],dual,1e-9);assert np.linalg.norm(dual.reshape(6,32),2)<=lam+1e-9
    primal=.5*b@gram@b-rhs@b+lam*singular.sum();dv=np.r_[np.zeros(24),dual]
    dualvalue=-.5*(rhs-dv)@np.linalg.solve(gram,rhs-dv);gap=float(primal-dualvalue)
    scale=max(1.,.5*q@np.linalg.solve(h,q));constant=.5*rhs[:24]@td0
    close(cert['primal_profile'],primal+constant);close(cert['dual_profile'],dualvalue+constant)
    close(cert['gap_scale'],scale);close(cert['gap'],max(0.,gap));assert gap>=-1e-8
    assert cert['certified']==(cert['gap']<=1e-9*cert['gap_scale']) and 0<=cert['iterations']<=5000
    if not cert['certified']:assert cert['iterations']==5000
    close(cert['train_MSE'],(quad-2*rhs@b+b@data@b)/len(y),1e-10)
    return bool(cert['certified'] and gap<=1e-9*scale+1e-11),normal,abs(gap)/scale


def audit_TD(z,y,g,b,center,cert,reused=False):
    data,rhs,quad,zm,ym,G,w,meta=profile(z,y,g);n,k=G.shape;c=meta['normalizer'];rho=meta['common_group_penalty']
    aug=np.zeros((n+24+k,1+24+k));aug[:n,0]=np.sqrt(c);aug[:n,1:25]=np.sqrt(c)*z;aug[:n,25:]=np.sqrt(c)*G
    aug[n:n+24,1:25]=10*np.eye(24);aug[n+24:,25:]=np.sqrt(c*rho)*np.eye(k)
    target=np.r_[np.sqrt(c)*y,np.zeros(24+k)];solution=np.linalg.lstsq(aug,target,rcond=None)[0]
    close(center,solution[0],1e-10);close(b,solution[1:25],1e-10)
    close(center,ym-zm@b,1e-11);close(w@(y-z@b-center),0,1e-9)
    normal=float(np.max(abs((data+100*np.eye(24))@b-rhs)));threshold=1e-8*(1+float(np.max(abs(rhs))))
    assert cert['mode']==('TD_partial_target_group_Ridge' if reused else 'TD_common_target_intercept_Ridge') and cert['alpha']==100
    close(cert['normal_max'],normal,1e-10);close(cert['normal_threshold'],threshold,1e-10)
    close(cert['train_MSE'],(quad-2*rhs@b+b@data@b)/n,1e-10);assert cert['certified']==(normal<=threshold)
    return bool(normal<=threshold),normal,float(np.max(abs(b-solution[1:25])))


def residual_audit(z,y,b,c,g,source):
    x=z if source else z[:,:24];r=y-x@b-c;G,inverse,w,m=indicator_geometry(g)
    u=inverse*(G.T@r);profiled=r@(r-G@u);augmented=np.sum((r-G@u)**2)+m['common_group_penalty']*(u@u)
    close(profiled,augmented,1e-10)
    return dict(raw_training_MSE=float(r@r/len(r)),normalized_profiled_SSE=float(m['normalizer']*profiled),
        normalized_augmented_SSE=float(m['normalizer']*augmented),weighted_intercept_normal=float(w@r),
        training_group_effect_norm=float(np.linalg.norm(u)),group_effects_deployed=False)


def main():
    began=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    assert all(t['num_threads']==1 for t in threadpool_info())
    lock=read(R/'lock.json');check(R,lock['files'])
    for path,h in lock['dependencies'].items():assert sha(Path(path))==h,path
    assert lock['alpha']==100 and lock['lambda_ratio']==.1 and lock['common_group_penalty']=='n/K'
    assert lock['weighted_intercept_profiled'] and lock['centered_metric_trace_normalization'] and lock['no_group_effects_at_prediction']
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and not d['outer_test_metrics_used']
    check(R/'pilot',d['pilot_files_sha256']);assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    assert d['new_source_forward']==d['new_PCA']==d['new_original_scalers']==d['GPU_hours']==d['new_training_partitions']==0
    runs=[read(p) for p in R.glob('execution_*.json')];assert len(runs)==1 and runs[0]['returncode']==0
    rowsout=table(R/'diagnostic_rows.csv');plan=read(P/'plan.json')
    refrows={(r['setting'],r['source'],r['sample_id']):r for r in table(A/'diagnostic_rows.csv')}
    counts=dict(new_source=0,new_TD=0,equal_reused=0,geometries=0,rows=0)
    records=[];numeric=True;maxnormal=maxgap=maxlstsq=maxdeploy=maxintercept=0.;reference_heads=0
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids']
        assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids']) and set(ids).isdisjoint(item['validation_ids'])
        rows={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')};y=np.array([float(rows[s]['label']) for s in ids])
        td=np.array([[float(v) for v in rows[s]['td'].split(',')] for s in ids]);grouping=[];td_current=None
        for j,sp in enumerate(item['inner']):
            tr=np.array(sp['train']);target=[rows[ids[i]]['target57'] for i in tr];labels,gm=ref.independent_groups(target)
            gp=A/'pilot/groups'/setting/f'inner{j}/groups.json';gr=read(gp);close(labels,gr['labels'],0);compare(gr['metadata'],gm)
            assert gr['train_indices']==tr.tolist() and gr['train_ids']==[ids[i] for i in tr]
            assert gr['target_strings_sha256']==hashlib.sha256('\n'.join(target).encode()).hexdigest()
            mp=R/'pilot/geometry'/setting/f'inner{j}';receipt=read(mp/'complete.json');check(mp,receipt['files_sha256'])
            assert receipt['lock_sha256']==sha(R/'lock.json');saved=read(mp/'geometry.json');meta=indicator_geometry(labels)[3]
            assert saved['groups_sha256']==sha(gp) and saved['train_indices']==tr.tolist() and saved['train_ids']==[ids[i] for i in tr]
            assert saved['setting']==setting and saved['inner']==j;compare(saved['metadata'],meta)
            compare(next(m for m in d['geometries'] if (m['setting'],m['inner'])==(setting,j)),dict(setting=setting,inner=j,**meta))
            grouping.append((labels,meta,gp,mp/'geometry.json'));counts['geometries']+=1
        for source in ('TD_only',)+tuple(old.SOURCES):
            is_source=source!='TD_only';origin=source if is_source else 'core_real';folder=old.EXPORT/origin
            identity=read(folder/'complete.json');check(folder,identity['files_sha256']);assert d['source_identities'][origin]==identity['source_identity']
            metadata=table(folder/'row_identity.csv');index={r['sample_id']:i for i,r in enumerate(metadata)}
            for sid in ids:assert all(rows[sid][k]==metadata[index[sid]][k] for k in ('sample_id','cohort','guide','target57'))
            raw=np.c_[td,np.load(folder/'source.npy',allow_pickle=False)[[index[s] for s in ids]].astype(float)]
            pred=np.full(len(ids),np.nan);reference=pred.copy();coverage=[];blocks=[]
            for j,sp in enumerate(item['inner']):
                if time.monotonic()-began>900:raise TimeoutError('CPU audit cap')
                tr,va=np.array(sp['train']),np.array(sp['valid']);assert set(tr).isdisjoint(va)
                labels,meta,gp,mp=grouping[j];reused=meta['reference_equal'];out=R/'pilot'/setting/source/f'inner{j}'
                receipt=read(out/'complete.json');check(out,receipt['files_sha256']);assert receipt['lock_sha256']==sha(R/'lock.json')
                state=read(out/'state.json');cert=state['certificate'];stpath=D/'pilot'/setting/origin/'dense'/f'inner{j}/transform.json';st=read(stpath)
                assert (state['setting'],state['source'],state['inner'])==(setting,source,j)
                assert state['train_indices']==tr.tolist() and state['valid_indices']==va.tolist() and state['equal_reused']==reused
                assert state['transform_sha256']==sha(stpath) and state['groups_sha256']==sha(gp) and state['geometry_sha256']==sha(mp)
                tm,ts=old.scale(raw[tr,:24],False);sm,ss=old.scale(raw[tr,24:],True)
                for key,value in (('td_mean',tm),('td_scale',ts),('src_mean',sm),('src_scale',ss)):close(st[key],value,1e-11)
                mean=np.r_[tm,sm];scale=np.r_[ts,ss*np.sqrt(12.)];z=(raw[tr]-mean)/scale;zv=(raw[va]-mean)/scale
                previous=A/'pilot'/setting/source/f'inner{j}';rr=read(previous/'complete.json');check(previous,rr['files_sha256']);assert rr['lock_sha256']==sha(A/'lock.json')
                ns=read(previous/'state.json');assert ns['numerically_qualified'] and ns['train_indices']==tr.tolist() and ns['valid_indices']==va.tolist()
                assert ns['transform_sha256']==sha(stpath) and ns['groups_sha256']==sha(gp)
                assert state['reference_model_sha256']==sha(previous/'model.npz') and state['reference_state_sha256']==sha(previous/'state.json')
                with np.load(previous/'model.npz',allow_pickle=False) as a:nb=a['beta'].copy();nc=float(a['intercept']);npred=a['prediction'].copy()
                reference[va]=npred;reference_heads+=1;tx=ref.weighted_design(z,labels);ty=ref.weighted_design(y[tr]-y[tr].mean(),labels)
                if is_source:valid,normal,gap,_=ref.certify(tx,ty+y[tr].mean(),nb,nc,ns['certificate'],False);maxgap=max(maxgap,gap)
                else:valid,normal,err=ref.audit_TD(tx[:,:24],ty,nb,ns['certificate']);maxlstsq=max(maxlstsq,err)
                assert valid;close((zv if is_source else zv[:,:24])@nb+nc,npred,1e-10)
                with np.load(out/'model.npz',allow_pickle=False) as a:m={k:a[k].copy() for k in a.files}
                beta=m['beta'];center=float(m['intercept'])
                if reused:close(beta,nb,0);close(center,nc,0);counts['equal_reused']+=1
                else:counts['new_source' if is_source else 'new_TD']+=1
                if is_source:valid,normal,gap=certify_source(z,y[tr],labels,beta,center,cert,reused);maxgap=max(maxgap,gap)
                else:valid,normal,err=audit_TD(z[:,:24],y[tr],labels,beta,center,cert,reused);maxlstsq=max(maxlstsq,err)
                numeric &= valid;assert valid==state['numerically_qualified'];maxnormal=max(maxnormal,normal)
                pp=(zv if is_source else zv[:,:24])@beta+center;close(pp,m['prediction'],1e-10)
                if reused:close(pp,npred,1e-10)
                full=beta if is_source else np.r_[beta,np.zeros(192)];coef=full/scale;offset=center-mean@coef
                close(coef,m['deployment_coef'],1e-11);close(offset,m['deployment_intercept'],1e-10)
                deployed=raw[va]@m['deployment_coef']+m['deployment_intercept'];close(deployed,pp,1e-9)
                maxdeploy=max(maxdeploy,float(np.max(abs(deployed-pp))))
                res=residual_audit(z,y[tr],beta,center,labels,is_source);compare(state['training_residual'],res)
                close(cert['train_MSE']*len(tr),res['normalized_profiled_SSE'],1e-9)
                maxintercept=max(maxintercept,abs(res['weighted_intercept_normal']))
                pred[va]=m['prediction'];coverage+=va.tolist()
                blocks.append(dict(inner=j,n=len(va),metrics=metric(y[va],pp),reference=metric(y[va],npred),K=meta['K'],
                    equal_reused=reused,training_residual=res,numerically_qualified=valid))
            assert sorted(coverage)==list(range(len(ids))) and np.isfinite(pred).all()
            if not is_source:td_current=pred.copy()
            part=[r for r in rowsout if (r['setting'],r['source'])==(setting,source)];assert [r['sample_id'] for r in part]==ids
            for key,value in (('label',y),('prediction',pred),('reference_prediction',reference),('TD_prediction',td_current)):
                close([float(r[key]) for r in part],value,0)
            close(reference,[float(refrows[(setting,source,s)]['prediction']) for s in ids],0)
            counts['rows']+=len(ids);m,rr=metric(y,pred),metric(y,reference);delta=m['PCC']-rr['PCC'];reduction=100*(1-m['MSE']/rr['MSE'])
            actual=next(r for r in d['records'] if (r['setting'],r['source'])==(setting,source))
            compare({k:v for k,v in actual.items() if k!='inner_blocks'},dict(setting=setting,source=source,n=len(ids),metrics=m,reference=rr,TD_metrics=metric(y,td_current),delta_PCC_vs_reference=delta,MSE_reduction_vs_reference_pct=reduction))
            assert len(actual['inner_blocks'])==5
            for a,b in zip(actual['inner_blocks'],blocks):compare(a,b)
            records.append(dict(setting=setting,source=source,metrics=m,delta=delta,reduction=reduction))
    assert counts==d['counts']==dict(new_source=39,new_TD=13,equal_reused=8,geometries=15,rows=9196)
    assert len(d['pilot_files_sha256'])==210 and len(rowsout)==9196 and len(d['records'])==12 and len(d['geometries'])==15
    assert d['condition_heads']==reference_heads==d['reused_reference_heads']==60 and d['reused_training_partitions']==15
    core=[r for r in records if r['source']=='core_real'];pairs=[]
    performance=sum(r['delta']>=.005 and r['reduction']>=0 for r in core)>=2 and all(r['delta']>=-.005 for r in core)
    for r in core:
        td=next(a for a in records if a['setting']==r['setting'] and a['source']=='TD_only')
        pair=dict(setting=r['setting'],delta_PCC=r['metrics']['PCC']-td['metrics']['PCC'],MSE_reduction_pct=100*(1-r['metrics']['MSE']/td['metrics']['MSE']),delta_gain_above_TD=r['delta']-td['delta'])
        compare(next(p for p in d['paired_TD'] if p['setting']==r['setting']),pair);pairs.append(pair)
    increment=sum(p['delta_PCC']>=0 for p in pairs)>=2 and np.mean([p['MSE_reduction_pct'] for p in pairs])>=0
    qualified=bool(performance and increment and numeric)
    assert (d['qualified'],d['performance_gate'],d['source_increment_gate'],d['all_numerically_qualified'])==(qualified,performance,increment,numeric)
    result=dict(passed=True,qualified=qualified,counts=counts,independent_input_graphs=15,independent_group_metrics=15,
        independent_condition_source_certificates=45,independent_explicit_nuisance_TD_lstsq=15,
        independent_reference_source_certificates=45,independent_reference_TD_lstsq=15,
        max_normal_error=maxnormal,max_relative_dual_gap=maxgap,max_TD_lstsq_difference=maxlstsq,
        max_deployment_difference=maxdeploy,max_intercept_normal_error=maxintercept,
        all_numerically_qualified=bool(numeric),GPU_hours=0,new_source_forward=0,outer_test_metrics_used=False,
        lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-began)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0),f,indent=2)
