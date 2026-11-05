"""Independent weighted full-system certificates and learning-curve readback."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4';Q=R.parent/'round25'
spec=importlib.util.spec_from_file_location('_r35_audit4',P/'verify.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,sha,table,check=old.read,old.sha,old.table,old.check
CONDITIONS=(*old.SOURCES,'TD_only');FRACTIONS=(.5,.75);DRAWS=(42,43,44)


def close(a,b,tol=1e-9):np.testing.assert_allclose(a,b,rtol=tol,atol=tol)
def metric(y,p):return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))


def projection(raw,train,state):
    tm,ts=old.scale(train[:,:24],False);sm,ss=old.scale(train[:,24:],True)
    for key,value in (('td_mean',tm),('td_scale',ts),('src_mean',sm),('src_scale',ss)):close(state[key],value,1e-11)
    mu=np.r_[tm,sm];sd=np.r_[ts,ss*np.sqrt(12.)]
    return (raw-mu)/sd,mu,sd


def native_certificate(z,y,b,c,cert,weight):
    close(c,y.mean(),1e-12);g=weight*z.T@z+100*np.eye(216);rhs=weight*z.T@(y-c)
    td0=np.linalg.solve(g[:24,:24],rhs[:24]);coupling=np.linalg.solve(g[:24,:24],g[:24,24:])
    h=g[24:,24:]-g[24:,:24]@coupling;h=(h+h.T)/2;q=rhs[24:]-g[24:,:24]@td0
    critical=float(np.linalg.norm(q.reshape(6,32),2));lam=.1*critical
    close(cert['lambda_critical'],critical,1e-9);close(cert['lambda_value'],lam,1e-9)
    assert cert['alpha']==100 and cert['lambda_ratio']==.1 and cert['matrix_shape']==[6,32] and cert['mode']=='native_trace'
    normal=float(abs((g@b-rhs)[:24]).max());assert normal<1e-8
    singular=np.linalg.svd(b[24:].reshape(6,32),compute_uv=False);close(cert['singular_values'],singular,1e-10)
    assert cert['rank']==int(sum(singular>1e-8))
    dual=q-h@b[24:];norm=np.linalg.norm(dual.reshape(6,32),2)
    if norm>lam:dual*=lam/norm
    close(cert['dual'],dual,1e-9);assert np.linalg.norm(dual.reshape(6,32),2)<=lam+1e-9
    primal=.5*b@g@b-rhs@b+lam*singular.sum();dv=np.r_[np.zeros(24),dual]
    dual_value=-.5*(rhs-dv)@np.linalg.solve(g,rhs-dv);gap=float(primal-dual_value)
    constant=.5*rhs[:24]@td0;close(cert['primal_profile'],primal+constant,1e-9);close(cert['dual_profile'],dual_value+constant,1e-9)
    scale=max(1.,.5*q@np.linalg.solve(h,q));close(cert['gap_scale'],scale,1e-9)
    assert gap>=-1e-8;close(cert['gap'],max(0.,gap),1e-9)
    valid=bool(cert['certified'] and gap<=1e-9*scale+1e-10)
    assert cert['certified']==(cert['gap']<=1e-9*cert['gap_scale'])
    close(cert['train_MSE'],weight*np.mean((z@b+c-y)**2),1e-10)
    return valid,normal,abs(gap)/scale


def main():
    start=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for path,h in lock['dependencies'].items():assert sha(Path(path))==h,path
    assert lock['CPU_only'] and not lock['model_selection'] and not lock['outer_test_evaluation']
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and not d['model_selection'] and not d['outer_test_metrics_used']
    assert d['new_source_forward']==d['GPU_hours']==d['new_PCA']==0 and all(t['num_threads']==1 for t in d['threadpool'])
    check(R/'pilot',d['pilot_files_sha256']);check(R,d['output_sha256'])
    executions=[read(p) for p in R.glob('execution_*.json')];assert len(executions)==1 and executions[0]['returncode']==0
    plan=read(P/'plan.json');sampling=read(R/'subsets.json');rowsout=table(R/'diagnostic_rows.csv');refout=table(R/'reference_rows.csv')
    counts=dict(new_native=0,new_TD=0,old_native=0,old_TD=0,new_rows=0,old_rows=0)
    maxnormal=maxgap=maxpred=maxdeploy=0.;allvalid=True;records=[];allpred={};training_counts=[]
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids']
        assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        rows={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')}
        y=np.array([float(rows[s]['label']) for s in ids]);td=np.array([[float(v) for v in rows[s]['td'].split(',')] for s in ids])
        raw={};ts={};refs={c:np.full(len(ids),np.nan) for c in CONDITIONS}
        for source in old.SOURCES:
            folder=old.EXPORT/source;identity=read(folder/'complete.json');assert d['source_identities'][source]==identity['source_identity'];check(folder,identity['files_sha256'])
            meta=table(folder/'row_identity.csv');idx={r['sample_id']:i for i,r in enumerate(meta)}
            for s in ids:assert all(rows[s][k]==meta[idx[s]][k] for k in ('sample_id','cohort','guide','target57'))
            raw[source]=np.c_[td,np.load(folder/'source.npy',allow_pickle=False)[[idx[s] for s in ids]].astype(float)]
            ts[source]=read(P/'results/folds'/setting/'fold0'/source/'transforms.json')
        # Audit each100% control once; no repeated fit or artificial draw uncertainty.
        for j,sp in enumerate(item['inner']):
            tr,va=np.array(sp['train']),np.array(sp['valid'])
            for source in old.SOURCES:
                out=Q/'pilot'/setting/source/f'inner{j}';receipt=read(out/'complete.json');check(out,receipt['files_sha256'])
                assert receipt['lock_sha256']==sha(Q/'lock.json');state=read(out/'state.json')
                assert state['train_indices']==tr.tolist() and state['valid_indices']==va.tolist() and state['source']==source
                z,_,_=projection(raw[source][tr],raw[source][tr],ts[source]['inner'][j]);zv,_,_=projection(raw[source][va],raw[source][tr],ts[source]['inner'][j])
                with np.load(out/'model.npz',allow_pickle=False) as a:b=a['beta'].copy();c=float(a['intercept']);p=a['prediction'].copy()
                valid,normal,gap=native_certificate(z,y[tr],b,c,state['certificate'],1.)
                assert valid;close(zv@b+c,p,1e-10);refs[source][va]=p;counts['old_native']+=1
            with np.load(P/'results/folds'/setting/'fold0/core_real/arrays.npz',allow_pickle=False) as a:
                b=a['inner_coef'][j,3,:24].copy();c=float(a['inner_intercept'][j]);p=a['inner_oof'][va,3].copy()
            z,_,_=projection(raw['core_real'][tr],raw['core_real'][tr],ts['core_real']['inner'][j]);zv,_,_=projection(raw['core_real'][va],raw['core_real'][tr],ts['core_real']['inner'][j])
            close(c,y[tr].mean(),1e-12);close((z[:,:24].T@z[:,:24]+100*np.eye(24))@b,z[:,:24].T@(y[tr]-c),1e-8)
            close(zv[:,:24]@b+c,p,1e-10);refs['TD_only'][va]=p;counts['old_TD']+=1
        for condition in CONDITIONS:
            part=[r for r in refout if r['setting']==setting and r['condition']==condition];assert [r['sample_id'] for r in part]==ids
            close([float(r['prediction']) for r in part],refs[condition],0);close([float(r['label']) for r in part],y,0);counts['old_rows']+=len(part)
            allpred[setting,1.,None,condition]=refs[condition]
        pred={(f,s,c):np.full(len(ids),np.nan) for f in FRACTIONS for s in DRAWS for c in CONDITIONS};covered={k:[] for k in pred}
        blocks=[b for b in sampling['blocks'] if b['setting']==setting];assert len(blocks)==30
        for block in blocks:
            if time.monotonic()-start>900:raise TimeoutError('CPU audit cap')
            j,draw,fraction=block['inner'],block['draw'],block['fraction'];original=item['inner'][j]
            eligible=[ids[i] for i in original['train']]
            order=sorted(eligible,key=lambda s:(hashlib.sha256((str(draw)+chr(0)+s).encode('utf-8')).hexdigest(),s))
            selected=set(order[:int(np.ceil(fraction*len(eligible)))]);expected=[i for i in original['train'] if ids[i] in selected]
            assert block['train_indices']==expected and block['train_ids']==[ids[i] for i in expected]
            assert block['valid_indices']==original['valid'] and block['eligible_indices']==original['train'] and block['full_n']==len(eligible)
            tr,va=np.array(expected),np.array(original['valid']);assert not set(tr)&set(va)
            training_counts.append(len(tr))
            for condition in CONDITIONS:
                source='core_real' if condition=='TD_only' else condition
                out=R/'pilot'/setting/f'p{int(fraction*100)}'/f'draw{draw}'/condition/f'inner{j}'
                receipt=read(out/'complete.json');check(out,receipt['files_sha256']);assert receipt['lock_sha256']==sha(R/'lock.json')
                state=read(out/'state.json');t=read(out/'transform.json')
                for k,v in block.items():assert state[k]==v,k
                assert state['condition']==condition and state['source']==source
                z,mu,sd=projection(raw[source][tr],raw[source][tr],t);zv,_,_=projection(raw[source][va],raw[source][tr],t)
                close(z.mean(0),0,1e-10)
                with np.load(out/'model.npz',allow_pickle=False) as a:m={k:a[k].copy() for k in a.files}
                b=m['beta'];c=float(m['intercept']);cert=state['certificate'];weight=len(eligible)/len(tr)
                close(c,y[tr].mean(),1e-12);close(cert['row_weight'],weight,1e-12);close(cert['equivalent_unweighted_alpha'],100/weight,1e-12)
                assert cert['full_training_rows']==len(eligible) and cert['training_rows']==len(tr)
                if condition=='TD_only':
                    assert cert['mode']=='TD' and cert['alpha']==100 and np.count_nonzero(b[24:])==0
                    design=np.sqrt(weight)*z[:,:24];target=np.sqrt(weight)*(y[tr]-c)
                    solution=np.linalg.lstsq(np.r_[design,10*np.eye(24)],np.r_[target,np.zeros(24)],rcond=None)[0]
                    close(b[:24],solution,1e-10)
                    normal=float(abs((design.T@design+100*np.eye(24))@b[:24]-design.T@target).max()/len(tr))
                    close(cert['normal_error'],normal,1e-10);valid=normal<1e-8;gap=0.;counts['new_TD']+=1
                else:
                    valid,normal,gap=native_certificate(z,y[tr],b,c,cert,weight);counts['new_native']+=1
                    assert cert['maximum_iterations']==20000 and 0<=cert['iterations']<=20000
                assert state['numerically_qualified']==valid;allvalid &= valid
                p=zv@b+c;close(p,m['prediction'],1e-10);close(m['raw_coef'],b/sd,1e-12);close(m['raw_intercept'],c-mu@(b/sd),1e-11)
                rawp=raw[source][va]@m['raw_coef']+m['raw_intercept'];close(rawp,p,1e-9)
                maxnormal=max(maxnormal,normal);maxgap=max(maxgap,gap);maxpred=max(maxpred,float(abs(p-m['prediction']).max()));maxdeploy=max(maxdeploy,float(abs(rawp-p).max()))
                pred[fraction,draw,condition][va]=m['prediction'];covered[fraction,draw,condition]+=va.tolist()
        for fraction in FRACTIONS:
            for draw in DRAWS:
                small=next(b for b in blocks if b['inner']==0 and b['fraction']==.5 and b['draw']==draw)
                large=next(b for b in blocks if b['inner']==0 and b['fraction']==.75 and b['draw']==draw)
                assert set(small['train_indices'])<=set(large['train_indices'])
                for condition in CONDITIONS:
                    key=(fraction,draw,condition);p=pred[key];assert sorted(covered[key])==list(range(len(ids))) and np.isfinite(p).all()
                    part=[r for r in rowsout if r['setting']==setting and r['condition']==condition and float(r['fraction'])==fraction and int(r['draw'])==draw]
                    assert [r['sample_id'] for r in part]==ids;close([float(r['prediction']) for r in part],p,0);close([float(r['label']) for r in part],y,0)
                    counts['new_rows']+=len(part);allpred[setting,fraction,draw,condition]=p
        for fraction in (.5,.75,1.):
            for draw in (DRAWS if fraction<1 else (None,)):
                tdmetrics=metric(y,allpred[setting,fraction,draw,'TD_only'])
                for condition in CONDITIONS:
                    m=metric(y,allpred[setting,fraction,draw,condition])
                    result=dict(setting=setting,condition=condition,fraction=fraction,draw=draw,n=len(ids),**m,TD_PCC=tdmetrics['PCC'],TD_MSE=tdmetrics['MSE'],
                        PCC_delta=m['PCC']-tdmetrics['PCC'],MSE_reduction_vs_TD_pct=100*(1-m['MSE']/tdmetrics['MSE']))
                    actual=next(r for r in d['records'] if all(r[k]==result[k] for k in ('setting','condition','fraction','draw')))
                    for k,v in result.items():
                        if k not in ('setting','condition','draw'):close(actual[k],v,1e-10)
                    records.append(result)
    assert counts==dict(new_native=270,new_TD=90,old_native=45,old_TD=15,new_rows=55176,old_rows=9196)
    assert len(rowsout)==55176 and len(refout)==9196 and len(records)==len(d['records'])==84
    assert d['new_heads']==360 and d['new_scalers']==270 and d['reused_reference_heads']==60 and d['all_numerically_qualified']==allvalid
    assert d['new_internal_OOF_rows']==55176 and d['reused_internal_OOF_rows']==9196
    agg_csv=table(R/'aggregate.csv');assert len(agg_csv)==len(d['aggregate'])==36
    for actual in d['aggregate']:
        keys={k:actual[k] for k in ('setting','condition','fraction')};rr=[r for r in records if all(r[k]==v for k,v in keys.items())]
        assert actual['n_draws']==len(rr) and actual['n_validation']==rr[0]['n']
        values={}
        for field in ('PCC','MSE','PCC_delta','MSE_reduction_vs_TD_pct'):
            vv=np.array([r[field] for r in rr]);values[field+'_mean']=float(vv.sum()/len(vv));values[field+'_SD']=float(np.sqrt(np.sum((vv-vv.mean())**2)/(len(vv)-1))) if len(vv)>1 else None
        if keys['fraction']<1:
            pp=np.stack([allpred[keys['setting'],keys['fraction'],s,keys['condition']] for s in DRAWS])
            # Independent central-moment identity; this mean is never scored as a predictor.
            values['subset_perturbation_variance']=float(np.mean((pp-pp.mean(0))**2))
        else:values['subset_perturbation_variance']=None
        csvrow=next(r for r in agg_csv if r['setting']==keys['setting'] and r['condition']==keys['condition'] and float(r['fraction'])==keys['fraction'])
        for field,v in values.items():
            if v is None:assert actual[field] is None and csvrow[field]==''
            else:close(actual[field],v,1e-10);close(float(csvrow[field]),v,1e-10)
    result=dict(passed=True,all_numerically_qualified=bool(allvalid),counts=counts,
        new_full_joint_certificates=270,independent_TD_augmented_LS=90,old_full_joint_certificates=45,old_TD_normal_equations=15,
        new_OOF_rows=55176,reused_OOF_rows=9196,aggregate_rows=36,draw_records=84,minimum_subset_training_rows=min(training_counts),maximum_subset_training_rows=max(training_counts),
        max_TD_normal_error=maxnormal,max_relative_dual_gap=maxgap,max_prediction_difference=maxpred,max_deployment_difference=maxdeploy,
        GPU_hours=0,new_source_forward=0,new_PCA=0,model_selection=False,outer_test_metrics_used=False,
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
