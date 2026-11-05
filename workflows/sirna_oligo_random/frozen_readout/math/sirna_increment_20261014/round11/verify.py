"""Independent posterior/selection/coverage audit plus 64 native final replays."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback,warnings
from pathlib import Path
import numpy as np
from sklearn.linear_model import ARDRegression
R=Path(__file__).resolve().parent;P=R.parent/'round4';B=R.parents[1]
spec=importlib.util.spec_from_file_location('_r11_independent',P/'verify.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
read,sha,table,check,close=audit.read,audit.sha,audit.table,audit.check,audit.close
VIEWS=('core_real','full_real','initial_core','TD_only');MODES=('AlwaysARD','AdaptiveARD')
SETTINGS=('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix')
PARAM=dict(max_iter=300,tol=.001,alpha_1=1e-6,alpha_2=1e-6,lambda_1=1e-6,lambda_2=1e-6,
    threshold_lambda=10000.,fit_intercept=True,compute_score=True,copy_X=True,verbose=False)
ALPHA=(.1,1.,10.,100.,1000.)

def project_fixed(x,t):
    td=(x[:,:24]-t['td_mean'])/t['td_scale'];src=(x[:,24:]-t['src_mean'])/t['src_scale']
    pc=(src-t['pca_mean'])@np.asarray(t['components']).T
    return np.c_[td,(pc-t['pc_mean'])/t['pc_scale']]

def posterior(z,y,m):
    n,p=z.shape;coef=np.asarray(m['coef']);lam=np.asarray(m['weight_precision']);noise=m['noise_precision']
    assert coef.shape==(p,) and lam.shape==(p,) and noise>0 and np.all(lam>0)
    mean=z.mean(0);close(mean,m['offset'],1e-12);x=z-mean;yc=y-y.mean()
    active=np.flatnonzero(lam<10000.);assert active.tolist()==m['active_indices']
    assert np.count_nonzero(coef[lam>=10000.])==0
    cov=np.asarray(m['posterior_covariance']).reshape(len(active),len(active))
    assert m['active_TD']==int(np.count_nonzero(active<24)) and m['active_source']==int(np.count_nonzero(active>=24))
    close(m['intercept'],y.mean()-mean@coef,1e-11)
    if len(active):
        xa=x[:,active];precision=noise*(xa.T@xa)+np.diag(lam[active])
        residual=precision@coef[active]-noise*xa.T@yc
        err=float(np.max(abs(residual))/max(1.,noise*n));assert err<1e-8,err
        close(precision@cov,np.eye(len(active)),1e-7);close(cov,cov.T,1e-12)
        assert np.linalg.eigvalsh(cov).min()>0
    else:err=0.
    assert m['input_width']==p and 1<=m['iterations']<=300 and m['iteration_cap_reached']==(m['iterations']==300)
    return err

def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    c=read(R/'results/complete.json');assert c['lock_sha256']==sha(R/'lock.json');check(R/'results',c['files_sha256'])
    runs=[read(p) for p in sorted(R.glob('execution_*.json'))];assert runs[-1]['returncode']==0
    assert all(v['GPU_hours']==v['new_source_forward']==v['source_updates']==0 for v in runs)
    plan=read(R/'plan.json');assert plan['folds']==read(P/'plan.json')['folds'] and plan['parameters']==PARAM
    assert plan['views']==list(VIEWS) and plan['primary_condition']=='core_real_AdaptiveARD'
    benchmark=B/'core_six_settings_20261011/benchmark/data';export=B/'core_six_settings_20261011/results/exports'
    old=B/'expert_banded_20261014/results'
    rows=sum([table(benchmark/f'{s}.csv') for s in SETTINGS[:3]],[]);assert len(rows)==3329
    ids={r['sample_id']:i for i,r in enumerate(rows)};td=np.array([[float(v) for v in r['td'].split(',')] for r in rows]);xx={}
    for view in VIEWS[:3]:
        meta=table(export/view/'row_identity.csv');index={r['sample_id']:i for i,r in enumerate(meta)}
        for r in rows:assert all(r[k]==meta[index[r['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
        src=np.load(export/view/'source.npy',allow_pickle=False)[[index[r['sample_id']] for r in rows]].astype(float)
        xx[view]=np.c_[td,src]
    y=np.array([float(r['label']) for r in rows]);binary=np.array([int(r['y']) for r in rows])
    pred=table(R/'results/oof_predictions.csv');summary=table(R/'results/summary.csv')
    selections=table(R/'results/selection.csv');fit_audit=table(R/'results/fit_audit.csv')
    assert len(pred)==30384 and len(summary)==32 and len(selections)==128 and len(fit_audit)==384
    canonical={};count=0;final=0;replay=0;maxerr=0.;maxpredict=0.;maxreplay=0.;capped=0
    for item in plan['folds']:
        setting,fold=item['setting'],item['fold'];fi=np.array([ids[s] for s in item['fit_ids']]);ti=np.array([ids[s] for s in item['test_ids']])
        assert not set(fi)&set(ti)
        basepath=old/'folds'/setting/f'fold{fold}'
        base_sel=read(basepath/'selection.json')['selected']['TD24_Ridge_CV']-5
        with np.load(basepath/'arrays.npz',allow_pickle=False) as base:
            baseinner=base['inner_oof'][:,5:10].copy();basetest=base['test_pred'][:,1].copy()
        for view in VIEWS:
            source='core_real' if view=='TD_only' else view;width=24 if view=='TD_only' else 40;x=xx[source]
            folder=R/'results/folds'/setting/f'fold{fold}'/view
            bc=read(folder/'complete.json');assert bc['lock_sha256']==sha(R/'lock.json');check(folder,bc['files_sha256'])
            models=read(folder/'models.json');sel=read(folder/'selection.json')
            path=P/'results/folds'/setting/f'fold{fold}'/source/'transforms.json'
            assert models['transform_path']==str(path.resolve()) and models['transform_sha256']==sha(path)
            assert models['source']==source and models['input_width']==width
            t=read(path);arrays=np.load(folder/'arrays.npz',allow_pickle=False);oof=arrays['inner_oof'];assert oof.shape==(len(fi),6)
            np.testing.assert_array_equal(oof[:,:5],baseinner)
            coverage=[]
            for j,part in enumerate(item['inner']):
                tr=np.array(part['train']);va=np.array(part['valid']);coverage+=va.tolist();assert not set(tr)&set(va)
                independent=audit.project(x[fi[tr]],x[fi[va]],t['inner'][j])
                z=project_fixed(x[fi[tr]],t['inner'][j])[:,:width];zv=project_fixed(x[fi[va]],t['inner'][j])[:,:width]
                close(zv,independent[:,:width],1e-11)
                m=models['inner'][j];maxerr=max(maxerr,posterior(z,y[fi[tr]],m));count+=1;capped+=int(m['iteration_cap_reached'])
                close(zv@np.asarray(m['coef'])+m['intercept'],oof[va,5],1e-10)
            assert sorted(coverage)==list(range(len(fi)))
            yc=y[fi]-y[fi].mean();op=oof-oof.mean(0);den=np.sqrt((yc@yc)*np.sum(op*op,axis=0))
            pc=np.divide(yc@op,den,out=np.full(6,-2.),where=den>1e-14);ms=np.mean((oof-y[fi,None])**2,axis=0)
            close(pc,sel['PCC']);close(ms,sel['MSE'])
            key=lambda k:(-pc[k],ms[k],int(k==5),-ALPHA[k] if k<5 else 0,k)
            assert min(range(5),key=key)==base_sel
            chosen={'AlwaysARD':5,'AdaptiveARD':min(range(6),key=key)}
            assert chosen==sel['selected'] and sel['fit_ids']==item['fit_ids'] and sel['test_ids']==item['test_ids']
            assert sel['outer_validation_unused'] and sel['TD_original_candidate']==base_sel
            independent=audit.project(x[fi],x[ti],t['final']);z=project_fixed(x[fi],t['final'])[:,:width];zt=project_fixed(x[ti],t['final'])[:,:width]
            close(zt,independent[:,:width],1e-11);m=models['final'];maxerr=max(maxerr,posterior(z,y[fi],m));final+=1;capped+=int(m['iteration_cap_reached'])
            with warnings.catch_warnings(record=True) as warning:
                warnings.simplefilter('always');native=ARDRegression(**PARAM).fit(z,y[fi])
            replay+=1;diff=float(np.max(abs(native.coef_-np.asarray(m['coef']))));maxreplay=max(maxreplay,diff)
            close(native.coef_,m['coef'],1e-10);close(native.intercept_,m['intercept'],1e-10)
            close(native.alpha_,m['noise_precision'],1e-8);close(native.lambda_,m['weight_precision'],1e-8)
            close(native.sigma_,np.asarray(m['posterior_covariance']).reshape(native.sigma_.shape),1e-10)
            assert native.n_iter_==m['iterations'] and [str(w.message) for w in warning]==m['warnings']
            coeff=np.pad(np.asarray(m['coef']),(0,40-width));state=t['final']
            bt=coeff[:24]/np.asarray(state['td_scale']);bp=coeff[24:]/np.asarray(state['pc_scale'])
            bs=(bp@np.asarray(state['components']))/np.asarray(state['src_scale'])
            bi=m['intercept']-bt@np.asarray(state['td_mean'])-bs@np.asarray(state['src_mean'])
            bi-=bp@(np.asarray(state['components'])@np.asarray(state['pca_mean'])+np.asarray(state['pc_mean']))
            close(np.r_[bt,bs],arrays['collapsed_coef'],1e-10);close(bi,arrays['collapsed_intercept'],1e-10)
            ard=zt@np.asarray(m['coef'])+m['intercept'];close(x[ti]@np.r_[bt,bs]+bi,ard,1e-10)
            for k,mode in enumerate(MODES):
                pp=ard if chosen[mode]==5 else basetest;close(arrays['test_pred'][:,k],pp,1e-10)
                if chosen[mode]!=5:np.testing.assert_array_equal(arrays['test_pred'][:,k],basetest)
                name=view+'_'+mode;part=[r for r in pred if r['setting']==setting and int(r['fold'])==fold and r['condition']==name]
                assert [r['sample_id'] for r in part]==item['test_ids']
                actual=np.array([float(r['prediction']) for r in part]);maxpredict=max(maxpredict,float(np.max(abs(actual-pp))))
                close(actual,pp,1e-10)
                for row,i in zip(part,ti):assert float(row['label'])==y[i] and int(row['binary'])==binary[i]
                canonical.setdefault((setting,name),[]).extend(zip(ti,actual))
                selected=[r for r in selections if r['setting']==setting and int(r['fold'])==fold and r['condition']==name]
                assert len(selected)==1 and int(selected[0]['candidate'])==chosen[mode]
            for j,m in enumerate(models['inner']+[models['final']]):
                partition='final' if j==5 else f'inner{j}'
                record=[r for r in fit_audit if r['setting']==setting and int(r['fold'])==fold and r['view']==view and r['partition']==partition]
                assert len(record)==1;record=record[0]
                assert int(record['iterations'])==m['iterations'] and (record['cap_reached']=='True')==m['iteration_cap_reached']
                assert int(record['active_TD'])==m['active_TD'] and int(record['active_source'])==m['active_source']
            arrays.close()
        print('VERIFIED',setting,fold,flush=True)
    assert (count,final,replay)==(320,64,64) and c['iteration_capped_fits']==capped
    assert c['inner_fits']==320 and c['final_fits']==64 and c['rows']==30384 and c['new_PCA_fits']==0
    baseline={(r['setting'],r['condition']):r for r in table(old/'summary.csv')}
    for row in summary:
        pairs=canonical[row['setting'],row['condition']];ii=np.array([i for i,v in pairs]);p=np.array([v for i,v in pairs])
        cohort='Mixset' if row['setting']=='Inter_Hu_to_Mix' else row['setting'];expected={i for i,r in enumerate(rows) if r['cohort']==cohort}
        assert len(ii)==len(expected) and set(ii)==expected
        metric=audit.metric(y[ii],p,binary[ii])
        for k,v in metric.items():close(float(row[k]),v)
        base=baseline[row['setting'],'TD24_Ridge_CV'];bpc=float(base['PCC'])
        close(float(row['TD24_PCC']),bpc);close(float(row['PCC_delta']),metric['PCC']-bpc)
        close(float(row['PCC_relative_gain_pct']),100*(metric['PCC']/bpc-1))
        close(float(row['MSE_relative_reduction_pct']),100*(1-metric['MSE']/float(base['MSE'])))
    goals=read(R/'results/goal_summary.json');assert len(goals)==8
    for row in goals:
        part=[s for s in summary if s['condition']==row['condition']];assert len(part)==4
        gain=float(np.mean([float(s['PCC_relative_gain_pct']) for s in part]));close(gain,row['mean_relative_PCC_gain_pct'])
        assert row['reaches_development_5pct']==(gain>=5) and row['primary']==(row['condition']==plan['primary_condition'])
        assert row['negative_settings']==[s['setting'] for s in part if float(s['PCC_delta'])<0]
    result=dict(passed=True,inner_posterior_equations=count,final_posterior_equations=final,native_final_replay_fits=replay,
        rows=30384,selections=128,iteration_capped_fits=capped,max_normalized_posterior_residual=maxerr,
        max_final_prediction_error=maxpredict,max_native_replay_coefficient_error=maxreplay,
        SHA_inputs_PCA_selection_metrics_coverage_verified=True,original_TD_predictions_unchanged=True,
        lock_sha256=sha(R/'lock.json'),complete_sha256=sha(R/'results/complete.json'),
        verifier_sha256=sha(Path(__file__)),CPU_wall_seconds=time.monotonic()-started,GPU_hours=0,new_source_forward=0)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0),f,indent=2)
