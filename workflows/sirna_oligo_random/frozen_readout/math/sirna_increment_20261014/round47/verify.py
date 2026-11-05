"""Independent augmented BVLS solutions and conditional KKT; no new runner calls."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
R=Path(__file__).resolve().parent;P=R.parent/'round4';D=R.parent/'round21';N=R.parent/'round25'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


old=load('_r47_independent4',P/'verify.py');oldcert=load('_r47_independent36',R.parent/'round36/verify.py').certify
read,sha,table,check=old.read,old.sha,old.table,old.check


def close(a,b,tol=1e-9):np.testing.assert_allclose(a,b,rtol=tol,atol=tol)
def metric(y,p):return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))


def audit_new(z,y,previous,m,c):
    C=previous[24:].reshape(6,32);u=m['left_vectors'];v=m['right_rows'];s=m['old_singular_values'];rank=len(s)
    assert rank==c['selected_rank'] and u.shape==(6,rank) and v.shape==(rank,32)
    close(u.T@u,np.eye(rank),1e-11);close(v@v.T,np.eye(rank),1e-11)
    close(C@v.T,u*s,1e-10);close(C.T@u,v.T*s,1e-10)
    allsing=np.linalg.svd(C,compute_uv=False);close(s,allsing[allsing>1e-8],1e-10)
    basis=np.empty((192,rank))
    for j in range(rank):
        for a in range(6):
            for b in range(32):basis[a*32+b,j]=u[a,j]*v[j,b]
    close(m['basis'],basis,1e-12);close(basis.T@basis,np.eye(rank),1e-11)
    close(c['old_basis_reconstruction_error'],np.max(abs(basis@s-previous[24:])),1e-12)
    x=np.c_[z[:,:24],z[:,24:]@basis];p=24+rank;yc=y-y.mean()
    augmented=np.r_[x,10*np.eye(p)];response=np.r_[yc,np.zeros(p)]
    fit=lsq_linear(augmented,response,bounds=(np.r_[np.full(24,-np.inf),np.zeros(rank)],np.full(p,np.inf)),
                   method='bvls',tol=1e-12,max_iter=500)
    assert fit.success and fit.optimality<1e-8
    w=m['reduced_coef'];a=m['amplitudes'];close(w,fit.x,1e-9);close(w[24:],a,1e-12)
    assert np.all(a>=0);close(m['beta'],np.r_[w[:24],basis@a],1e-11);close(m['intercept'],y.mean(),1e-12)
    h=x.T@x+100*np.eye(p);q=x.T@yc;g=h@w-q
    minimum=np.r_[g[:24],np.where(a>0,g[24:],np.minimum(g[24:],0))]
    error=float(max(abs(minimum)));threshold=1e-8*(1+float(max(abs(q))))
    obj=.5*np.sum((x@w-yc)**2)+50*(w@w);oldw=np.r_[previous[:24],s]
    oldobj=.5*np.sum((x@oldw-yc)**2)+50*(oldw@oldw)
    t0=np.linalg.solve(h[:24,:24],q[:24]);coupling=np.linalg.solve(h[:24,:24],h[:24,24:])
    hs=h[24:,24:]-h[24:,:24]@coupling;qs=q[24:]-h[24:,:24]@t0;pg=hs@a-qs
    pmin=np.where(a>0,pg,np.minimum(pg,0));profile_error=float(max(abs(pmin))) if rank else 0.
    for key,value in (('KKT_max',error),('KKT_threshold',threshold),('conditional_suboptimality_upper',minimum@minimum/200),
                      ('objective_without_trace',obj),('reference_objective_without_trace',oldobj),
                      ('profile_KKT_max',profile_error),('profile_objective',.5*a@hs@a-qs@a),
                      ('train_MSE',np.mean((z@m['beta']+float(m['intercept'])-y)**2))):close(c[key],value,1e-9)
    assert c['alpha']==100 and c['rank_threshold']==1e-8 and c['mode']=='fixed_singular_amplitude_refit'
    assert c['enumerated_active_sets']==1<<rank and c['active_system_solves']==(1<<rank)-1
    assert 1<=c['feasible_face_solutions']<=1<<rank
    assert c['selected_parameter_count']==25+rank and c['deployment_parameter_count']==217
    assert c['supervised_basis_from_old_training_head'] and not c['new_direction_search']
    assert c['active_amplitudes']==np.count_nonzero(a)
    if rank:close(c['minimum_amplitude'],a.min(),1e-12)
    else:assert c['minimum_amplitude'] is None
    certified=bool(error<=threshold and np.all(a>=0) and obj<=oldobj+1e-9*max(1.,oldobj))
    assert c['certified']==certified
    return certified,error,float(max(abs(w-fit.x))),basis


def main():
    start=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for path,h in lock['dependencies'].items():assert sha(Path(path))==h,path
    assert lock['alpha']==100 and lock['previous_lambda_ratio']==.1 and lock['rank_threshold']==1e-8 and lock['nonnegative_amplitudes']
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and not d['outer_test_metrics_used']
    assert d['new_source_forward']==d['new_PCA']==d['new_original_scalers']==d['GPU_hours']==0
    assert all(t['num_threads']==1 for t in d['threadpool'])
    check(R/'pilot',d['pilot_files_sha256']);assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    runs=[read(p) for p in R.glob('execution_*.json')];assert len(runs)==1 and runs[0]['returncode']==0
    rowsout=table(R/'diagnostic_rows.csv');plan=read(P/'plan.json');records=[]
    counts=dict(new=0,old=0,TD=0,rows=0,supervised_direction_decompositions=0,active_system_solves=0)
    numeric=True;maxnormal=maxoldgap=maxpred=maxdeploy=maxBVLS=0.
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
            folder=old.EXPORT/source;identity=read(folder/'complete.json')
            assert d['source_identities'][source]==identity['source_identity'];check(folder,identity['files_sha256'])
            meta=table(folder/'row_identity.csv');idx={r['sample_id']:i for i,r in enumerate(meta)}
            for s in ids:assert all(rows[s][k]==meta[idx[s]][k] for k in ('sample_id','cohort','guide','target57'))
            raw=np.c_[td,np.load(folder/'source.npy',allow_pickle=False)[[idx[s] for s in ids]].astype(float)]
            pred=np.full(len(ids),np.nan);reference=pred.copy();coverage=[];blocks=[]
            for j,sp in enumerate(item['inner']):
                if time.monotonic()-start>900:raise TimeoutError('CPU audit cap')
                tr,va=np.array(sp['train']),np.array(sp['valid']);assert set(tr).isdisjoint(va)
                out=R/'pilot'/setting/source/f'inner{j}';receipt=read(out/'complete.json');check(out,receipt['files_sha256'])
                assert receipt['lock_sha256']==sha(R/'lock.json');state=read(out/'state.json');cert=state['certificate']
                control=D/'pilot'/setting/source/'dense'/f'inner{j}';native=N/'pilot'/setting/source/f'inner{j}'
                assert (state['setting'],state['source'],state['inner'])==(setting,source,j)
                assert state['train_indices']==tr.tolist() and state['valid_indices']==va.tolist()
                assert state['transform_sha256']==sha(control/'transform.json') and state['native_reference_sha256']==sha(native/'model.npz')
                assert state['TD_reference_sha256']==sha(tdpath)
                nr=read(native/'complete.json');check(native,nr['files_sha256']);assert nr['lock_sha256']==sha(N/'lock.json')
                ns=read(native/'state.json');assert ns['train_indices']==tr.tolist() and ns['valid_indices']==va.tolist()
                t=read(control/'transform.json');tm,ts=old.scale(raw[tr,:24],False);sm,ss=old.scale(raw[tr,24:],True)
                for key,value in (('td_mean',tm),('td_scale',ts),('src_mean',sm),('src_scale',ss)):close(t[key],value,1e-11)
                mean=np.r_[tm,sm];scale=np.r_[ts,ss*np.sqrt(12.)];z=(raw[tr]-mean)/scale;zv=(raw[va]-mean)/scale
                with np.load(native/'model.npz',allow_pickle=False) as a:nb=a['beta'].copy();nc=float(a['intercept']);npred=a['prediction'].copy()
                nv,_,oldgap,_=oldcert(z,y[tr],nb,nc,ns['certificate'],False);assert nv
                close(zv@nb+nc,npred,1e-10);reference[va]=npred;counts['old']+=1;maxoldgap=max(maxoldgap,oldgap)
                with np.load(out/'model.npz',allow_pickle=False) as a:m={k:a[k].copy() for k in a.files}
                valid,normal,bvls_error,basis=audit_new(z,y[tr],nb,m,cert);numeric &= valid
                assert valid==state['numerically_qualified'] and cert['selected_rank']==ns['certificate']['rank']
                counts['supervised_direction_decompositions']+=1;counts['active_system_solves']+=cert['active_system_solves']
                pp=zv@m['beta']+float(m['intercept']);close(m['prediction'],pp,1e-10)
                close(m['deployment_coef'],m['beta']/scale,1e-11);close(m['deployment_intercept'],float(m['intercept'])-mean@(m['beta']/scale),1e-10)
                rawp=raw[va]@m['deployment_coef']+m['deployment_intercept'];close(rawp,pp,1e-9)
                maxnormal=max(maxnormal,normal);maxBVLS=max(maxBVLS,bvls_error)
                maxpred=max(maxpred,float(abs(m['prediction']-pp).max()));maxdeploy=max(maxdeploy,float(abs(rawp-pp).max()))
                pred[va]=m['prediction'];coverage+=va.tolist();counts['new']+=1
                blocks.append(dict(inner=j,n=len(va),metrics=metric(y[va],pp),reference=metric(y[va],npred),TD_metrics=metric(y[va],tdpred[va]),
                    selected_rank=cert['selected_rank'],active_amplitudes=cert['active_amplitudes'],
                    amplitude_change_norm=float(np.linalg.norm(m['amplitudes']-m['old_singular_values'])),
                    objective_reduction=cert['reference_objective_without_trace']-cert['objective_without_trace'],numerically_qualified=valid))
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
    assert counts==dict(new=45,old=45,TD=15,rows=6897,supervised_direction_decompositions=45,active_system_solves=2707)
    assert len(rowsout)==6897 and d['new_active_system_solves']==counts['active_system_solves'] and d['new_TD_profile_factors']==45
    assert d['new_heads']==d['reused_native_heads']==d['new_supervised_direction_decompositions']==45
    assert d['reused_unique_TD_heads']==15 and d['reused_unique_TD_rows']==2299 and d['new_internal_OOF_rows']==6897
    core=[r for r in records if r['source']=='core_real']
    performance=sum(r['delta']>=.005 and r['reduction']>=0 for r in core)>=2 and all(r['delta']>=-.005 for r in core)
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
        independent_augmented_BVLS_fits=45,independent_old_full_joint_certificates=45,TD_normal_equations=15,
        max_conditional_KKT_error=maxnormal,max_old_relative_dual_gap=maxoldgap,max_independent_BVLS_difference=maxBVLS,
        max_prediction_difference=maxpred,max_deployment_difference=maxdeploy,
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
