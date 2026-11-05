"""Independent metric quadratic forms, certificates, and deployment readback."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import hashlib, importlib.util, json, time, traceback
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
R=Path(__file__).resolve().parent; B=R.parent; A=B/'round49'; D=B/'round21'; P=B/'round4'
s=importlib.util.spec_from_file_location('_r53_audit49',A/'verify.py')
ref=importlib.util.module_from_spec(s); s.loader.exec_module(ref)
old=ref.old; read,sha,table,check=ref.read,ref.sha,ref.table,ref.check
close,metric=ref.close,ref.metric


def compare(actual,expected):
    if isinstance(expected,dict):
        assert set(actual)==set(expected)
        for key,value in expected.items(): compare(actual[key],value)
    elif isinstance(expected,(list,tuple)):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected): compare(a,b)
    elif isinstance(expected,(str,bool,np.bool_)) or expected is None:
        assert actual==expected
    else: close(actual,expected)


def independent_scaling(z,labels):
    """Compute column quadratic forms of c(I-.5(P-J)); not fit_scaling."""
    n=len(z); g=np.asarray(labels,int); k=int(g.max())+1; counts=np.bincount(g)
    q=np.zeros((n,k)); q[np.arange(n),g]=1/np.sqrt(counts[g])
    v=np.einsum('ij,ij->j',z,z); between=np.sum((q.T@z)**2,axis=0)-n*z.mean(0)**2
    identity=k in (1,n)
    m=v.copy() if identity else (n-1)/(n-1-.5*(k-1))*(v-.5*between)
    factor=np.ones(216); blocks=[]
    for first,last in ((0,24),(24,216)):
        vv,mm=v[first:last],m[first:last]; active=vv>0
        ratio=float(np.sum(mm)/np.sum(vv)) if np.any(active) else None
        if not identity and np.any(active):
            ix=np.flatnonzero(active)+first
            factor[ix]=np.sqrt(ratio/(m[ix]/v[ix]))
        f=factor[first:last]
        blocks.append(dict(first=first,last=last,active_columns=int(active.sum()),original_energy=float(vv.sum()),
            projected_energy=float(mm.sum()),adjusted_energy=float(np.sum(mm*f*f)),block_metric_ratio=ratio,
            factor_min=float(f.min()),factor_max=float(f.max()),factor_RMS=float(np.sqrt(np.mean(f*f)))))
    assert np.isfinite(factor).all() and np.min(factor)>=np.sqrt(.5)-1e-12 and np.max(factor)<=np.sqrt(2)+1e-12
    return factor,dict(identity=identity,blocks=blocks,finite=True,train_inputs_only=True,response_used=False,separate_TD_and_source=True)


def main():
    began=time.monotonic(); assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    assert all(t['num_threads']==1 for t in threadpool_info())
    lock=read(R/'lock.json'); check(R,lock['files'])
    for path,h in lock['dependencies'].items(): assert sha(Path(path))==h,path
    assert lock['alpha']==100 and lock['lambda_ratio']==.1 and lock['gamma']==.5 and lock['reference_round']==49
    assert lock['blocks']==[[0,24],[24,216]] and lock['block_energy_preserved'] and lock['metric_scaling_no_strength_parameter']
    d=read(R/'diagnostic.json'); assert d['lock_sha256']==sha(R/'lock.json') and not d['outer_test_metrics_used']
    assert d['new_source_forward']==d['new_PCA']==d['new_original_scalers']==d['GPU_hours']==0
    assert d['new_training_partitions']==0 and d['reused_training_partitions']==15 and d['reused_reference_heads']==60
    check(R/'pilot',d['pilot_files_sha256']); assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    runs=[read(p) for p in R.glob('execution_*.json')]; assert len(runs)==1 and runs[0]['returncode']==0
    rowsout=table(R/'diagnostic_rows.csv'); plan=read(P/'plan.json')
    refrows={(r['setting'],r['source'],r['sample_id']):r for r in table(A/'diagnostic_rows.csv')}
    counts=dict(new_source=0,new_TD=0,identity_reused=0,scalings=0,nonidentity_scalings=0,rows=0)
    graphs=0; reference_heads=0; numeric=True; records=[]; scale_seen=set(); td_factors={}
    maxnormal=maxgap=maxscale=maxlstsq=maxdeploy=maxenergy=0.
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0); ids=item['fit_ids']
        assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids']) and set(ids).isdisjoint(item['validation_ids'])
        rows={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')}
        y=np.array([float(rows[s]['label']) for s in ids]); td=np.array([[float(v) for v in rows[s]['td'].split(',')] for s in ids])
        grouping=[]; td_current=None
        for j,sp in enumerate(item['inner']):
            tr=np.array(sp['train']); target=[rows[ids[i]]['target57'] for i in tr]
            labels,gm=ref.independent_groups(target); gp=A/'pilot/groups'/setting/f'inner{j}/groups.json'; state=read(gp)
            close(labels,state['labels'],0); compare(state['metadata'],gm)
            assert state['train_indices']==tr.tolist() and state['train_ids']==[ids[i] for i in tr]
            assert state['target_strings_sha256']==hashlib.sha256('\n'.join(target).encode()).hexdigest()
            grouping.append((labels,gm,gp)); graphs+=1
        for source in ('TD_only',)+tuple(old.SOURCES):
            is_source=source!='TD_only'; origin=source if is_source else 'core_real'; folder=old.EXPORT/origin
            identity=read(folder/'complete.json'); check(folder,identity['files_sha256'])
            assert d['source_identities'][origin]==identity['source_identity']
            metadata=table(folder/'row_identity.csv'); index={r['sample_id']:i for i,r in enumerate(metadata)}
            for sid in ids: assert all(rows[sid][k]==metadata[index[sid]][k] for k in ('sample_id','cohort','guide','target57'))
            raw=np.c_[td,np.load(folder/'source.npy',allow_pickle=False)[[index[s] for s in ids]].astype(float)]
            pred=np.full(len(ids),np.nan); reference=pred.copy(); coverage=[]; blocks=[]
            for j,sp in enumerate(item['inner']):
                if time.monotonic()-began>900: raise TimeoutError('CPU audit cap')
                tr,va=np.array(sp['train']),np.array(sp['valid']); assert set(tr).isdisjoint(va)
                labels,gm,gp=grouping[j]; reused=gm['identity_on_centered_space']; out=R/'pilot'/setting/source/f'inner{j}'
                receipt=read(out/'complete.json'); check(out,receipt['files_sha256']); assert receipt['lock_sha256']==sha(R/'lock.json')
                state=read(out/'state.json'); cert=state['certificate']
                stpath=D/'pilot'/setting/origin/'dense'/f'inner{j}/transform.json'; st=read(stpath)
                assert (state['setting'],state['source'],state['inner'])==(setting,source,j)
                assert state['train_indices']==tr.tolist() and state['valid_indices']==va.tolist() and state['identity_reused']==reused
                assert state['transform_sha256']==sha(stpath) and state['groups_sha256']==sha(gp)
                tm,ts=old.scale(raw[tr,:24],False); sm,ss=old.scale(raw[tr,24:],True)
                for key,value in (('td_mean',tm),('td_scale',ts),('src_mean',sm),('src_scale',ss)): close(st[key],value,1e-11)
                mean=np.r_[tm,sm]; scale=np.r_[ts,ss*np.sqrt(12.)]
                z=(raw[tr]-mean)/scale; zv=(raw[va]-mean)/scale
                sf=R/'pilot/scales'/setting/origin/f'inner{j}'; sc=read(sf/'complete.json'); check(sf,sc['files_sha256'])
                assert sc['lock_sha256']==sha(R/'lock.json') and state['scaling_sha256']==sha(sf/'scaling.json')
                ssaved=read(sf/'scaling.json'); factor,sm=independent_scaling(z,labels); close(factor,ssaved['factor'],1e-11)
                compare(ssaved['metadata'],sm); assert ssaved['metadata']['identity']==reused
                assert ssaved['groups_sha256']==sha(gp) and ssaved['transform_sha256']==sha(stpath)
                assert ssaved['train_indices']==tr.tolist() and ssaved['train_ids']==[ids[i] for i in tr]
                assert (ssaved['setting'],ssaved['source'],ssaved['inner'])==(setting,origin,j)
                key=(setting,j)
                if key in td_factors: close(factor[:24],td_factors[key],1e-12)
                else: td_factors[key]=factor[:24].copy()
                if sf not in scale_seen:
                    scale_seen.add(sf); counts['scalings']+=1; counts['nonidentity_scalings']+=int(not reused)
                    entries=[a for a in d['scalings'] if (a['setting'],a['source'],a['inner'])==(setting,origin,j)]; assert len(entries)==1
                    compare(entries[0],dict(setting=setting,source=origin,inner=j,**sm))
                maxscale=max(maxscale,float(np.max(abs(factor-np.asarray(ssaved['factor'])))))
                x=z*factor; xv=zv*factor; weighted=ref.weighted_design(x,labels)
                oldweighted=ref.weighted_design(z,labels); yc=ref.weighted_design(y[tr]-y[tr].mean(),labels)
                for first,last in ((0,24),(24,216)):
                    before=float(np.sum(oldweighted[:,first:last]**2)); after=float(np.sum(weighted[:,first:last]**2))
                    close(before,after,1e-11); maxenergy=max(maxenergy,abs(before-after)/(1+before))
                previous=A/'pilot'/setting/source/f'inner{j}'; rc=read(previous/'complete.json'); check(previous,rc['files_sha256'])
                assert rc['lock_sha256']==sha(A/'lock.json')
                ns=read(previous/'state.json'); assert ns['numerically_qualified'] and ns['train_indices']==tr.tolist() and ns['valid_indices']==va.tolist()
                assert ns['transform_sha256']==sha(stpath) and ns['groups_sha256']==sha(gp)
                assert state['reference_model_sha256']==sha(previous/'model.npz') and state['reference_state_sha256']==sha(previous/'state.json')
                with np.load(previous/'model.npz',allow_pickle=False) as a: nb=a['beta'].copy(); nc=float(a['intercept']); npred=a['prediction'].copy()
                reference[va]=npred; reference_heads+=1
                if is_source: valid,normal,gap,_=ref.certify(oldweighted,yc+y[tr].mean(),nb,nc,ns['certificate'],False); maxgap=max(maxgap,gap)
                else: valid,normal,err=ref.audit_TD(oldweighted[:,:24],yc,nb,ns['certificate']); maxlstsq=max(maxlstsq,err)
                assert valid; close((zv if is_source else zv[:,:24])@nb+nc,npred,1e-10)
                with np.load(out/'model.npz',allow_pickle=False) as a: m={k:a[k].copy() for k in a.files}
                beta=m['beta']; center=float(m['intercept']); close(center,y[tr].mean(),1e-12)
                if reused:
                    close(factor,np.ones(216),0); close(beta,nb,0); close(center,nc,0); counts['identity_reused']+=1
                else: counts['new_source' if is_source else 'new_TD']+=1
                if is_source: valid,normal,gap,_=ref.certify(weighted,yc+y[tr].mean(),beta,center,cert,False); maxgap=max(maxgap,gap)
                else: valid,normal,err=ref.audit_TD(weighted[:,:24],yc,beta,cert); maxlstsq=max(maxlstsq,err)
                numeric &= valid; assert valid==state['numerically_qualified']; maxnormal=max(maxnormal,normal)
                pp=(xv if is_source else xv[:,:24])@beta+center; close(pp,m['prediction'],1e-10)
                if reused: close(pp,npred,1e-10)
                full=beta if is_source else np.r_[beta,np.zeros(192)]; coef=(factor*full)/scale; offset=center-mean@coef
                close(coef,m['deployment_coef'],1e-11); close(offset,m['deployment_intercept'],1e-10)
                deployed=raw[va]@m['deployment_coef']+m['deployment_intercept']; close(deployed,pp,1e-9)
                maxdeploy=max(maxdeploy,float(np.max(abs(deployed-pp))))
                residual=ref.residual_audit(x,y[tr],beta,center,labels,is_source); compare(state['training_residual'],residual)
                close(cert['train_MSE']*len(tr),residual['transformed_residual_SSE'],1e-9)
                pred[va]=m['prediction']; coverage+=va.tolist()
                blocks.append(dict(inner=j,n=len(va),metrics=metric(y[va],pp),reference=metric(y[va],npred),K=gm['K'],
                    identity_reused=reused,training_residual=residual,numerically_qualified=valid))
            assert sorted(coverage)==list(range(len(ids))) and np.isfinite(pred).all()
            if not is_source: td_current=pred.copy()
            part=[r for r in rowsout if (r['setting'],r['source'])==(setting,source)]; assert [r['sample_id'] for r in part]==ids
            for key,value in (('label',y),('prediction',pred),('reference_prediction',reference),('TD_prediction',td_current)):
                close([float(r[key]) for r in part],value,0)
            close(reference,[float(refrows[(setting,source,s)]['prediction']) for s in ids],0)
            counts['rows']+=len(ids); m,rr=metric(y,pred),metric(y,reference)
            delta=m['PCC']-rr['PCC']; improvement=100*(1-m['MSE']/rr['MSE'])
            actual=next(r for r in d['records'] if (r['setting'],r['source'])==(setting,source))
            compare({k:v for k,v in actual.items() if k!='inner_blocks'},dict(setting=setting,source=source,n=len(ids),metrics=m,reference=rr,TD_metrics=metric(y,td_current),
                delta_PCC_vs_reference=delta,MSE_reduction_vs_reference_pct=improvement))
            assert len(actual['inner_blocks'])==5
            for a,b in zip(actual['inner_blocks'],blocks): compare(a,b)
            records.append(dict(setting=setting,source=source,metrics=m,delta=delta,reduction=improvement))
    assert counts==d['counts']==dict(new_source=39,new_TD=13,identity_reused=8,scalings=45,nonidentity_scalings=39,rows=9196)
    assert d['condition_heads']==reference_heads==60 and graphs==15 and len(d['scalings'])==45
    assert len(d['pilot_files_sha256'])==270 and len(rowsout)==9196 and len(d['records'])==12
    core=[r for r in records if r['source']=='core_real']; pairs=[]
    performance=sum(r['delta']>=.005 and r['reduction']>=0 for r in core)>=2 and all(r['delta']>=-.005 for r in core)
    for r in core:
        td=next(a for a in records if a['setting']==r['setting'] and a['source']=='TD_only')
        pair=dict(setting=r['setting'],delta_PCC=r['metrics']['PCC']-td['metrics']['PCC'],
            MSE_reduction_pct=100*(1-r['metrics']['MSE']/td['metrics']['MSE']),delta_gain_above_TD=r['delta']-td['delta'])
        compare(next(p for p in d['paired_TD'] if p['setting']==r['setting']),pair); pairs.append(pair)
    increment=sum(p['delta_PCC']>=0 for p in pairs)>=2 and np.mean([p['MSE_reduction_pct'] for p in pairs])>=0
    qualified=bool(performance and increment and numeric)
    assert (d['qualified'],d['performance_gate'],d['source_increment_gate'],d['all_numerically_qualified'])==(qualified,performance,increment,numeric)
    result=dict(passed=True,qualified=qualified,counts=counts,independent_input_graphs=graphs,independent_scale_records=45,
        independent_new_condition_source_certificates=45,independent_new_condition_TD_lstsq=15,
        independent_reference_source_certificates=45,independent_reference_TD_lstsq=15,
        max_normal_error=maxnormal,max_relative_dual_gap=maxgap,max_scaling_difference=maxscale,
        max_TD_lstsq_difference=maxlstsq,max_deployment_difference=maxdeploy,max_relative_energy_difference=maxenergy,
        all_numerically_qualified=bool(numeric),GPU_hours=0,new_source_forward=0,outer_test_metrics_used=False,
        lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-began)
    with (R/'verification.json').open('x',encoding='utf-8') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    start=time.monotonic(); rc=1; error=None
    try: main(); rc=0
    except Exception: error=traceback.format_exc(); raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0),f,indent=2)
