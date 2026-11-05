"""Independent covariance spectra, original controls and augmented LS audit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
from scipy.linalg import eigvalsh
R=Path(__file__).resolve().parent;P=R.parent/'round4'
spec=importlib.util.spec_from_file_location('_r34_audit4',P/'verify.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,sha,table,check=old.read,old.sha,old.table,old.check
CONDITIONS=('core_contrast','core_pooled','initial_contrast')


def close(a,b,tol=1e-9):np.testing.assert_allclose(a,b,rtol=tol,atol=tol)
def metric(y,p):return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))


def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for path,h in lock['dependencies'].items():assert sha(Path(path))==h,path
    assert lock['CPU_only'] and lock['source_seed']==42 and lock['alpha']==100 and lock['dimensions']==16
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and not d['outer_test_metrics_used']
    assert d['new_source_forward']==d['GPU_hours']==0 and all(t['num_threads']==1 for t in d['threadpool'])
    check(R/'pilot',d['pilot_files_sha256']);assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    executions=[read(p) for p in R.glob('execution_*.json')];assert len(executions)==1 and executions[0]['returncode']==0
    identities={s:read(old.EXPORT/s/'complete.json') for s in ('core_real','initial_core')}
    f,b=[identities[s]['source_identity'] for s in ('core_real','initial_core')]
    assert f['initial_sha256']==b['checkpoint_sha256']==b['initial_sha256']
    assert f['source_receipt_sha256']==b['source_receipt_sha256'] and f['variant']==b['variant']=='core_real'
    assert identities['core_real']['site_layer']==identities['initial_core']['site_layer']
    assert identities['core_real']['new_input_manifest_sha256']==identities['initial_core']['new_input_manifest_sha256']
    for source,identity in identities.items():
        assert d['source_identities'][source]==identity['source_identity'];check(old.EXPORT/source,identity['files_sha256'])
    rowsout=table(R/'diagnostic_rows.csv');plan=read(P/'plan.json');records=[];allvalid=True
    heads=references=covered=spectra=0;maxnormal=maxpred=maxraw=maxbeta=maxeigen=0.
    positive=[]
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids']
        assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        rows={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')}
        y=np.array([float(rows[s]['label']) for s in ids]);td=np.array([[float(v) for v in rows[s]['td'].split(',')] for s in ids])
        raw={};states={};references_oof={};ref_betas={};refsha={}
        for source in identities:
            folder=old.EXPORT/source;meta=table(folder/'row_identity.csv');lookup={r['sample_id']:i for i,r in enumerate(meta)}
            for s in ids:assert all(rows[s][k]==meta[lookup[s]][k] for k in ('sample_id','cohort','guide','target57'))
            raw[source]=np.c_[td,np.load(folder/'source.npy',allow_pickle=False)[[lookup[s] for s in ids]].astype(float)]
            pf=P/'results/folds'/setting/'fold0'/source/'transforms.json';af=pf.parent/'arrays.npz'
            states[source]=read(pf);refsha[source]=dict(transforms=sha(pf),arrays=sha(af))
            with np.load(af,allow_pickle=False) as a:
                references_oof[source]=a['inner_oof'][:,18].copy();ref_betas[source]=a['inner_coef'][:,18].copy()
                if source=='core_real':tdref=a['inner_oof'][:,3].copy();tdbeta=a['inner_coef'][:,3,:24].copy()
        predictions={c:np.full(len(ids),np.nan) for c in CONDITIONS};blocks={c:[] for c in CONDITIONS};coverage=[]
        for j,split in enumerate(item['inner']):
            if time.monotonic()-started>900:raise TimeoutError('Independent audit cap')
            tr,va=np.array(split['train']),np.array(split['valid']);coverage+=va.tolist()
            # Validate old controls independently, without fitting old heads again.
            for source in raw:
                z0=old.project(raw[source][tr],raw[source][tr],states[source]['inner'][j])
                v0=old.project(raw[source][tr],raw[source][va],states[source]['inner'][j])
                beta=ref_betas[source][j];close((z0.T@z0+100*np.eye(40))@beta,z0.T@(y[tr]-y[tr].mean()),1e-8)
                close(v0@beta+y[tr].mean(),references_oof[source][va],1e-10);references+=1
                if source=='core_real':
                    close((z0[:,:24].T@z0[:,:24]+100*np.eye(24))@tdbeta[j],z0[:,:24].T@(y[tr]-y[tr].mean()),1e-8)
                    close(v0[:,:24]@tdbeta[j]+y[tr].mean(),tdref[va],1e-10);references+=1
            tm,ts=old.scale(raw['core_real'][tr,:24],False)
            fm,fs=old.scale(raw['core_real'][tr,24:],True)
            fz=(raw['core_real'][tr,24:]-fm)/fs
            bm=raw['initial_core'][tr,24:].mean(0);bz=(raw['initial_core'][tr,24:]-bm)/fs
            cf=np.cov(fz,rowvar=False,bias=True);cb=np.cov(bz,rowvar=False,bias=True)
            tf,tb=float(np.trace(cf)),float(np.trace(cb));gamma=tf/tb
            cache={}
            for mode,sign in (('contrast',-1),('pooled',1)):
                a=cf+sign*gamma*cb;a=(a+a.T)/2
                e=eigvalsh(a,driver='evr');limit=1e-10*max(float(abs(e).max()),1.)
                cache[mode]=(a,e,limit,int(np.count_nonzero(e>limit)));spectra+=1
            for condition in CONDITIONS:
                source='initial_core' if condition.startswith('initial') else 'core_real';mode='pooled' if condition=='core_pooled' else 'contrast'
                out=R/'pilot'/setting/condition/f'inner{j}';receipt=read(out/'complete.json')
                check(out,receipt['files_sha256']);assert receipt['lock_sha256']==sha(R/'lock.json')
                state=read(out/'state.json');trstate=read(out/'transform.json');t,mapping=trstate['projection'],trstate['mapping']
                assert (state['setting'],state['condition'],state['source'],state['inner'])==(setting,condition,source,j)
                assert state['train_indices']==tr.tolist() and state['valid_indices']==va.tolist() and state['reference_sha256']==refsha
                assert mapping['mode']==mode
                a,e,limit,npositive=cache[mode];c=np.array(mapping['components']);ev=np.array(mapping['eigenvalues'])
                close(c@c.T,np.eye(16),1e-10);close(ev,e[-16:][::-1],1e-9);close(a@c.T,c.T*ev,1e-8)
                maxeigen=max(maxeigen,float(abs(a@c.T-c.T*ev).max()))
                for row in c:assert row[np.argmax(abs(row))]>=0
                for name,value in (('gamma',gamma),('trace_foreground',tf),('trace_background',tb),('covariance_trace',np.trace(a)),('positive_limit',limit)):
                    close(mapping[name],value,1e-9)
                assert mapping['positive_eigenvalues']==npositive
                spectral_ok=bool(ev[-1]>limit);assert mapping['spectral_qualified']==spectral_ok
                close(mapping['foreground_scale'],fs,1e-11)
                if mode=='contrast':assert abs(np.trace(a))<1e-8
                mean=raw[source][tr,24:].mean(0);z=(raw[source][tr,24:]-mean)/fs;pm=z.mean(0)
                pcm,pcs=old.scale((z-pm)@c.T,True)
                for key,value in (('td_mean',tm),('td_scale',ts),('src_mean',mean),('src_scale',fs),('pca_mean',pm),('components',c),('pc_mean',pcm),('pc_scale',pcs)):
                    close(t[key],value,1e-10)
                z=np.c_[(raw[source][tr,:24]-tm)/ts,((z-pm)@c.T-pcm)/pcs]
                zv=np.c_[(raw[source][va,:24]-tm)/ts,(((raw[source][va,24:]-mean)/fs-pm)@c.T-pcm)/pcs]
                close(z.mean(0),0,1e-10)
                with np.load(out/'model.npz',allow_pickle=False) as saved:model={k:saved[k].copy() for k in saved.files}
                beta=model['beta'];center=float(y[tr].mean());close(model['intercept'],center,1e-12)
                reference=np.linalg.lstsq(np.r_[z,10*np.eye(40)],np.r_[y[tr]-center,np.zeros(40)],rcond=None)[0]
                close(beta,reference,1e-10);maxbeta=max(maxbeta,float(abs(beta-reference).max()))
                normal=float(abs((z.T@z+100*np.eye(40))@beta-z.T@(y[tr]-center)).max()/len(tr))
                cert=state['certificate'];close(cert['normal_error'],normal,1e-10);assert cert['alpha']==100
                ok=bool(normal<1e-8);assert cert['numerically_qualified']==ok
                valid=bool(ok and spectral_ok);assert state['numerically_qualified']==valid;allvalid &= valid
                residual=z@beta+center-y[tr];close(cert['objective'],.5*(residual@residual+100*(beta@beta)))
                pp=zv@beta+center;close(pp,model['prediction'],1e-10)
                coeff=np.r_[beta[:24]/ts,(c.T@(beta[24:]/pcs))/fs]
                intercept=center-tm@coeff[:24]-mean@coeff[24:]-pm@(c.T@(beta[24:]/pcs))-pcm@(beta[24:]/pcs)
                close(coeff,model['raw_coef'],1e-10);close(intercept,model['raw_intercept'],1e-10)
                rawpred=raw[source][va]@model['raw_coef']+model['raw_intercept'];close(rawpred,pp,1e-10)
                maxnormal=max(maxnormal,normal);maxpred=max(maxpred,float(abs(pp-model['prediction']).max()));maxraw=max(maxraw,float(abs(rawpred-pp).max()))
                predictions[condition][va]=model['prediction'];heads+=1
                blocks[condition].append(dict(inner=j,n=len(va),metrics=metric(y[va],pp),reference=metric(y[va],references_oof[source][va]),
                    gamma=gamma,positive_eigenvalues=npositive,smallest_selected_eigenvalue=float(ev[-1]),normal_error=normal,numerically_qualified=valid))
                if condition=='core_contrast':positive.append(npositive)
        assert sorted(coverage)==list(range(len(ids)))
        for condition in CONDITIONS:
            source='initial_core' if condition.startswith('initial') else 'core_real';pred=predictions[condition]
            assert np.isfinite(pred).all();part=[r for r in rowsout if r['setting']==setting and r['condition']==condition]
            assert [r['sample_id'] for r in part]==ids
            for key,values in (('label',y),('prediction',pred),('reference_prediction',references_oof[source]),('TD_prediction',tdref)):
                close([float(r[key]) for r in part],values,0)
            covered+=len(part);m,ref,tdm=metric(y,pred),metric(y,references_oof[source]),metric(y,tdref)
            rec=next(r for r in d['records'] if r['setting']==setting and r['condition']==condition)
            assert rec['n']==len(ids) and rec['source']==source
            for name,value in (('metrics',m),('reference',ref),('TD_reference',tdm)):
                for k,v in value.items():close(rec[name][k],v,1e-11)
            delta=m['PCC']-ref['PCC'];red=100*(1-m['MSE']/ref['MSE'])
            close(rec['delta_PCC_vs_R4'],delta,1e-11);close(rec['MSE_reduction_vs_R4_pct'],red,1e-10)
            for actual,expected in zip(rec['inner_blocks'],blocks[condition]):
                for key,value in expected.items():
                    if isinstance(value,dict):
                        for k,v in value.items():close(actual[key][k],v)
                    else:close(actual[key],value)
            records.append(dict(setting=setting,condition=condition,metrics=m,TD_reference=tdm,delta=delta,reduction=red))
    assert heads==45==d['new_heads'] and references==45==d['reused_reference_heads'] and spectra==30==d['new_spectral_decompositions']
    assert covered==len(rowsout)==6897==d['new_internal_OOF_rows'] and d['new_transform_standardizations']==45
    primary=[r for r in records if r['condition']=='core_contrast'];performance=sum(r['delta']>=.005 and r['reduction']>=0 for r in primary)>=2 and all(r['delta']>=-.005 for r in primary)
    pairs=[]
    for r in primary:
        pooled=next(x for x in records if x['setting']==r['setting'] and x['condition']=='core_pooled')
        pair=dict(setting=r['setting'],MSE_reduction_vs_pooled_pct=100*(1-r['metrics']['MSE']/pooled['metrics']['MSE']),
            delta_PCC_vs_pooled=r['metrics']['PCC']-pooled['metrics']['PCC'],delta_PCC_vs_TD=r['metrics']['PCC']-r['TD_reference']['PCC'],
            MSE_reduction_vs_TD_pct=100*(1-r['metrics']['MSE']/r['TD_reference']['MSE']))
        expected=next(x for x in d['pairs'] if x['setting']==r['setting'])
        for key in pair:
            if key!='setting':close(expected[key],pair[key],1e-10)
        pairs.append(pair)
    contrastive=sum(x['MSE_reduction_vs_pooled_pct']>=0 for x in pairs)>=2 and np.mean([x['MSE_reduction_vs_pooled_pct'] for x in pairs])>=0
    increment=sum(x['delta_PCC_vs_TD']>=0 for x in pairs)>=2 and np.mean([x['MSE_reduction_vs_TD_pct'] for x in pairs])>=0
    qualified=bool(performance and contrastive and increment and allvalid)
    assert d['qualified']==qualified and d['performance_gate']==performance and d['contrastive_gate']==contrastive and d['source_increment_gate']==increment and d['all_numerically_qualified']==allvalid
    result=dict(passed=True,qualified=qualified,verified_new_heads=heads,verified_reference_heads=references,
        independent_spectra=spectra,independent_augmented_LS_solutions=heads,internal_OOF_rows=covered,
        max_normal_equation_error=maxnormal,max_coefficient_difference=maxbeta,max_prediction_difference=maxpred,
        max_single_vector_deployment_difference=maxraw,max_eigen_equation_error=maxeigen,
        minimum_contrast_positive_directions=min(positive),maximum_contrast_positive_directions=max(positive),
        all_numerically_qualified=bool(allvalid),GPU_hours=0,new_source_forward=0,outer_test_metrics_used=False,
        lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-started)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0,new_source_forward=0),f,indent=2)
