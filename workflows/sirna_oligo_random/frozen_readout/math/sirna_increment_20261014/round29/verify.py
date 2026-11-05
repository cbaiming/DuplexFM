"""Independent augmented-system certificate for a fixed TD coefficient anchor."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4';D=R.parent/'round21';N=R.parent/'round25'
spec=importlib.util.spec_from_file_location('_r29_independent4',P/'verify.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close


def metric(y,p):
    return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))


def compare_metrics(actual,expected):
    for name,value in expected.items():close(actual[name],value,1e-11)


def main():
    started=time.monotonic()
    assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for name,h in lock['dependencies'].items():assert sha(Path(name))==h,name
    assert lock['alpha']==lock['kappa']==100 and lock['lambda_ratio']==.1
    assert lock['CPU_only'] and lock['source_seed']==42 and lock['matrix_shape']==[6,32]
    assert lock['settings']==list(old.S[:3]) and lock['sources']==list(old.SOURCES)
    d=read(R/'diagnostic.json')
    assert d['lock_sha256']==sha(R/'lock.json') and not d['outer_test_metrics_used']
    assert d['new_source_forward']==d['new_PCA']==d['GPU_hours']==0
    assert all(t['num_threads']==1 for t in d['threadpool'])
    check(R/'pilot',d['pilot_files_sha256'])
    assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    executions=[read(p) for p in R.glob('execution_*.json')]
    assert len(executions)==1 and executions[0]['returncode']==0
    assert executions[0]['GPU_hours']==executions[0]['new_source_forward']==0
    plan=read(P/'plan.json');details=table(R/'diagnostic_rows.csv')
    records=[];counts=dict(new=0,old_native=0,unique_TD=0,rows=0,unique_TD_rows=0)
    maxpred=maxgap=maxnormal=maxtd=maxreference=0.;allvalid=True;block_records=[]
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids'];assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        rows={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')}
        y=np.array([float(rows[s]['label']) for s in ids])
        td=np.array([[float(v) for v in rows[s]['td'].split(',')] for s in ids])
        tdpath=P/'results/folds'/setting/'fold0'/'core_real'/'arrays.npz'
        with np.load(tdpath,allow_pickle=False) as a:
            tdcoef=a['inner_coef'][:,3,:24].copy();tdpred=a['inner_oof'][:,3].copy()
            tdcenter=a['inner_intercept'].copy()
            assert np.count_nonzero(a['inner_coef'][:,3,24:])==0
        for j,split in enumerate(item['inner']):
            tr,va=np.array(split['train']),np.array(split['valid'])
            tm,ts=old.scale(td[tr],False);t=(td[tr]-tm)/ts;tv=(td[va]-tm)/ts
            err=float(abs((t.T@t+100*np.eye(24))@tdcoef[j]-t.T@(y[tr]-y[tr].mean())).max())
            assert err<1e-8;maxtd=max(maxtd,err)
            close(tdcenter[j],y[tr].mean(),1e-12)
            close(tv@tdcoef[j]+tdcenter[j],tdpred[va],1e-10)
            counts['unique_TD']+=1
        counts['unique_TD_rows']+=len(ids)
        for source in old.SOURCES:
            export=old.EXPORT/source;identity=read(export/'complete.json')
            assert d['source_identities'][source]==identity['source_identity']
            meta=table(export/'row_identity.csv');lookup={r['sample_id']:i for i,r in enumerate(meta)}
            for sample in ids:
                assert all(rows[sample][k]==meta[lookup[sample]][k] for k in ('sample_id','cohort','guide','target57'))
            raw=np.c_[td,np.load(export/'source.npy',allow_pickle=False)[[lookup[s] for s in ids]].astype(float)]
            pred=np.full(len(ids),np.nan);reference=np.full(len(ids),np.nan);coverage=[];blocks=[]
            for j,split in enumerate(item['inner']):
                tr,va=np.array(split['train']),np.array(split['valid']);out=R/'pilot'/setting/source/f'inner{j}'
                c=read(out/'complete.json');check(out,c['files_sha256']);assert c['lock_sha256']==sha(R/'lock.json')
                s=read(out/'state.json');cert=s['certificate']
                assert (s['setting'],s['source'],s['inner'])==(setting,source,j)
                assert s['train_indices']==tr.tolist() and s['valid_indices']==va.tolist()
                control=D/'pilot'/setting/source/'dense'/f'inner{j}';native=N/'pilot'/setting/source/f'inner{j}'
                nc=read(native/'complete.json');check(native,nc['files_sha256']);assert nc['lock_sha256']==sha(N/'lock.json')
                ns=read(native/'state.json');st=read(control/'transform.json')
                assert ns['train_indices']==tr.tolist() and ns['valid_indices']==va.tolist()
                assert ns['transform_sha256']==s['transform_sha256']==sha(control/'transform.json')
                assert s['native_reference_sha256']==sha(native/'model.npz') and s['TD_reference_sha256']==sha(tdpath)
                tm,ts=old.scale(raw[tr,:24],False);sm,ss=old.scale(raw[tr,24:],True)
                for actual,expected in ((st['td_mean'],tm),(st['td_scale'],ts),(st['src_mean'],sm),(st['src_scale'],ss)):
                    close(actual,expected,1e-11)
                mean=np.r_[tm,sm];scale=np.r_[ts,ss*np.sqrt(12.)]
                z=(raw[tr]-mean)/scale;zv=(raw[va]-mean)/scale;close(z.mean(0),np.zeros(216),1e-10)
                with np.load(out/'model.npz',allow_pickle=False) as a:
                    beta=a['beta'].copy();center=float(a['intercept']);pp=a['prediction'].copy()
                    b0=a['TD_reference'].copy();dc=a['deployment_coef'].copy();di=float(a['deployment_intercept'])
                with np.load(native/'model.npz',allow_pickle=False) as a:
                    nb=a['beta'].copy();nc0=float(a['intercept']);npred=a['prediction'].copy()
                assert beta.shape==(216,) and b0.shape==(24,)
                close(b0,tdcoef[j],0);close(center,y[tr].mean(),1e-12);close(nc0,center,0)
                close(npred,zv@nb+nc0,1e-10)
                maxreference=max(maxreference,float(abs(npred-(zv@nb+nc0)).max()))
                assert ns['numerically_qualified'] and ns['certificate']['certified']
                g=z.T@z+100*np.eye(216);rhs=z.T@(y[tr]-center)
                referr=float(abs(g[:24,:24]@b0-rhs[:24]).max())
                close(cert['TD_reference_error'],referr,1e-10)
                assert referr<=1e-9*(1+abs(rhs[:24]).max())
                g[:24,:24]+=100*np.eye(24);rhs[:24]+=100*b0
                t0=np.linalg.solve(g[:24,:24],rhs[:24]);close(t0,b0,1e-11)
                coupling=np.linalg.solve(g[:24,:24],g[:24,24:])
                h=g[24:,24:]-g[24:,:24]@coupling;h=(h+h.T)/2
                q=rhs[24:]-g[24:,:24]@t0
                critical=float(np.linalg.norm(q.reshape(6,32),2));lam=.1*critical
                close(cert['lambda_critical'],critical,1e-10);close(cert['lambda_value'],lam,1e-10)
                close(ns['certificate']['lambda_critical'],critical,1e-10)
                assert cert['alpha']==cert['kappa']==100 and cert['lambda_ratio']==.1
                assert cert['matrix_shape']==[6,32] and cert['mode']=='TD_anchored_native_trace'
                normal=float(abs((g@beta-rhs)[:24]).max());assert normal<1e-8;maxnormal=max(maxnormal,normal)
                singular=np.linalg.svd(beta[24:].reshape(6,32),compute_uv=False)
                rank=int(sum(singular>1e-8));close(cert['singular_values'],singular,1e-11);assert cert['rank']==rank
                dual=q-h@beta[24:];norm=float(np.linalg.norm(dual.reshape(6,32),2))
                if norm>lam:dual*=lam/norm
                close(cert['dual'],dual,1e-10);assert np.linalg.norm(dual.reshape(6,32),2)<=lam+1e-10
                primal=.5*beta@g@beta-rhs@beta+lam*singular.sum();dv=np.r_[np.zeros(24),dual]
                dual_value=-.5*(rhs-dv)@np.linalg.solve(g,rhs-dv);gap=float(primal-dual_value)
                constant=.5*rhs[:24]@t0
                close(cert['primal_profile'],primal+constant,1e-10)
                close(cert['dual_profile'],dual_value+constant,1e-10)
                gapscale=max(1.,.5*q@np.linalg.solve(h,q));close(cert['gap_scale'],gapscale,1e-10)
                assert gap>=-1e-8;close(cert['gap'],max(0.,gap),1e-10)
                assert cert['certified']==(cert['gap']<=1e-9*cert['gap_scale'])
                valid=bool(cert['certified'] and gap<=1e-9*gapscale+1e-11)
                assert valid==s['numerically_qualified'];allvalid &= valid
                assert 0<=cert['iterations']<=5000
                if cert['iterations']:
                    assert int(cert['history'][-1][0])==cert['iterations']
                    close(cert['history'][-1][1],cert['gap'],1e-12)
                else:assert critical==0
                if not cert['certified']:assert cert['iterations']==5000
                shift=float(np.linalg.norm(beta[:24]-b0));oldshift=float(np.linalg.norm(nb[:24]-b0))
                close(cert['TD_shift_norm'],shift,1e-11);close(s['reference_TD_shift_norm'],oldshift,1e-11)
                close(cert['anchor_penalty'],50*np.sum((beta[:24]-b0)**2),1e-12)
                train_mse=float(np.mean((z@beta+center-y[tr])**2));close(cert['train_MSE'],train_mse,1e-12)
                complete_objective=.5*len(tr)*train_mse+50*(beta@beta)+50*np.sum((beta[:24]-b0)**2)+lam*singular.sum()
                close(complete_objective,primal+.5*np.sum((y[tr]-center)**2)+50*(b0@b0),1e-10)
                reconstruction=zv@beta+center;close(pp,reconstruction,1e-10)
                maxpred=max(maxpred,float(abs(pp-reconstruction).max()))
                close(dc,beta/scale,1e-12);close(di,center-mean@(beta/scale),1e-12)
                close(raw[va]@dc+di,pp,1e-9)
                pred[va]=pp;reference[va]=npred;coverage+=va.tolist()
                counts['new']+=1;counts['old_native']+=1;maxgap=max(maxgap,abs(gap)/gapscale)
                br=dict(inner=j,n=len(va),metrics=metric(y[va],pp),reference=metric(y[va],npred),
                    TD_metrics=metric(y[va],tdpred[va]),TD_shift_norm=shift,reference_TD_shift_norm=oldshift,
                    rank=rank,iterations=cert['iterations'],numerically_qualified=valid)
                blocks.append(br)
                block_records.append(dict(setting=setting,source=source,**br,relative_gap=gap/gapscale))
            assert sorted(coverage)==list(range(len(ids))) and np.isfinite(pred).all() and np.isfinite(reference).all()
            part=[r for r in details if r['setting']==setting and r['source']==source]
            assert [r['sample_id'] for r in part]==ids
            for key,values in [('label',y),('prediction',pred),('native_reference_prediction',reference),('TD_prediction',tdpred)]:
                close([float(r[key]) for r in part],values,0)
            counts['rows']+=len(part);m,rr,tm=metric(y,pred),metric(y,reference),metric(y,tdpred)
            rec=next(r for r in d['records'] if r['setting']==setting and r['source']==source)
            assert rec['n']==len(ids) and len(rec['inner_blocks'])==5
            for name,expected in [('metrics',m),('reference',rr),('TD_metrics',tm)]:compare_metrics(rec[name],expected)
            delta=m['PCC']-rr['PCC'];reduction=100*(1-m['MSE']/rr['MSE'])
            close(rec['delta_PCC_vs_R25'],delta,1e-11);close(rec['MSE_reduction_vs_R25_pct'],reduction,1e-10)
            for actual,expected in zip(rec['inner_blocks'],blocks):
                for key,value in expected.items():
                    if isinstance(value,dict):compare_metrics(actual[key],value)
                    else:close(actual[key],value,1e-11)
            records.append(dict(setting=setting,source=source,delta=delta,reduction=reduction,metrics=m,TD_metrics=tm))
    assert counts==dict(new=45,old_native=45,unique_TD=15,rows=6897,unique_TD_rows=2299)
    assert d['new_heads']==lock['new_heads']==45 and d['reused_native_heads']==45 and len(details)==d['new_internal_OOF_rows']==6897
    assert d['reused_unique_TD_heads']==15 and d['reused_unique_TD_rows']==2299
    assert len(d['records'])==9
    core=[r for r in records if r['source']=='core_real']
    performance=sum(r['delta']>=.005 and r['reduction']>=0 for r in core)>=2 and all(r['delta']>=-.005 for r in core)
    paired=[]
    for rec in core:
        pair=dict(setting=rec['setting'],delta_PCC=rec['metrics']['PCC']-rec['TD_metrics']['PCC'],
            MSE_reduction_pct=100*(1-rec['metrics']['MSE']/rec['TD_metrics']['MSE']))
        actual=next(r for r in d['paired_TD'] if r['setting']==rec['setting'])
        close(actual['delta_PCC'],pair['delta_PCC'],1e-11);close(actual['MSE_reduction_pct'],pair['MSE_reduction_pct'],1e-10)
        paired.append(pair)
    incremental=sum(r['delta_PCC']>=0 for r in paired)>=2 and np.mean([r['MSE_reduction_pct'] for r in paired])>=0
    qualified=bool(performance and incremental and allvalid)
    assert d['performance_gate']==performance and d['source_increment_gate']==incremental
    assert d['qualified']==qualified and d['all_numerically_qualified']==allvalid
    result=dict(passed=True,qualified=qualified,independent_full_joint_certificates=45,
        independent_unique_TD_normal_equations=15,verified_reused_native_heads=45,new_heads=45,new_internal_OOF_rows=6897,
        unique_reused_TD_rows=2299,max_prediction_difference=maxpred,max_reference_prediction_difference=maxreference,
        max_relative_dual_gap=maxgap,max_TD_normal_equation_error=maxnormal,max_reference_TD_error=maxtd,
        all_numerically_qualified=bool(allvalid),block_records=block_records,no_new_reference_fit=True,
        new_source_forward=0,new_PCA=0,GPU_hours=0,outer_test_metrics_used=False,
        lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-started)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='block_records'}),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0),f,indent=2)
