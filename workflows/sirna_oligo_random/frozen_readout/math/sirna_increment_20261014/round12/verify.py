"""Independent R12 audit: reconstruct features and normal equations, no refitting."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback,sys
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4';B=R.parents[1]
spec=importlib.util.spec_from_file_location('_previous_audit',P/'verify.py')
audit4=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit4)
sha,read,table,close,check=audit4.sha,audit4.read,audit4.table,audit4.close,audit4.check
SETTINGS=('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix')
VIEWS=('core_real','full_real','initial_core');MODES=('Uniform','Adaptive')
OLD=B/'expert_banded_20261014';BENCH=B/'core_six_settings_20261011/benchmark/data'
EXPORT=B/'core_six_settings_20261011/results/exports'

def condition(view,mode):return view+'_'+mode+'Spectral'

sys.path.insert(0,str(R))
from audit_spectral import expand as reconstruct
def expand(train,other,t,view):
    assert view in VIEWS
    return reconstruct(train,other,t)


def equations(z,y,b,configs):
    close(z.mean(0),np.zeros(z.shape[1]),1e-10)
    gram=z.T@z;rhs=z.T@(y-y.mean());maxerr=0
    for coef,cfg in zip(b,configs):
        n=len(coef) if cfg['weight']>0 else 24
        assert np.count_nonzero(coef[n:])==0
        pen=np.r_[np.full(24,cfg['alpha']),np.full(n-24,cfg['alpha']/cfg['weight'] if cfg['weight'] else 0)]
        residual=gram[:n,:n]@coef[:n]+pen*coef[:n]-rhs[:n]
        err=float(np.max(abs(residual))/len(y));assert err<1e-8,err;maxerr=max(maxerr,err)
        assert abs(np.mean(z@coef+y.mean()-y))<1e-10
    return maxerr

def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    complete=read(R/'results/complete.json');assert complete['lock_sha256']==sha(R/'lock.json');check(R/'results',complete['files_sha256'])
    execution=[read(p) for p in sorted(R.glob('execution_*.json'))];assert execution[-1]['returncode']==0
    assert all(r['GPU_hours']==0 and r['source_updates']==0 and r['new_source_forward']==0 for r in execution)
    plan=read(R/'plan.json');assert plan['folds']==read(P/'plan.json')['folds'] and plan['views']==list(VIEWS)
    cfg=[dict(weight=w,alpha=a) for w in (0.,.01,.1,1.,10.) for a in (.1,1.,10.,100.,1000.)]
    assert cfg==plan['configs'] and plan['primary_condition']=='core_real_AdaptiveSpectral'
    rows=sum([table(BENCH/f'{s}.csv') for s in SETTINGS[:3]],[]);assert len(rows)==3329
    ids={r['sample_id']:i for i,r in enumerate(rows)};td=np.array([[float(v) for v in r['td'].split(',')] for r in rows]);xx={}
    for view in VIEWS[:3]:
        meta=table(EXPORT/view/'row_identity.csv');lookup={r['sample_id']:i for i,r in enumerate(meta)}
        for r in rows:assert all(r[k]==meta[lookup[r['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
        source=np.load(EXPORT/view/'source.npy',allow_pickle=False)[[lookup[r['sample_id']] for r in rows]].astype(float)
        xx[view]=np.c_[td,source]
    y=np.array([float(r['label']) for r in rows]);binary=np.array([int(r['y']) for r in rows])
    pred=table(R/'results/oof_predictions.csv');summary=table(R/'results/summary.csv');selection=table(R/'results/selection.csv')
    assert len(pred)==22788 and len(summary)==24 and len(selection)==96
    canonical={};count=0;final=0;maxres=0.;maxTD=0.;maxPrediction=0.
    for item in plan['folds']:
        setting,fold=item['setting'],item['fold'];fi=np.array([ids[a] for a in item['fit_ids']]);ti=np.array([ids[a] for a in item['test_ids']])
        assert not set(fi)&set(ti)
        for view in VIEWS:
            path=R/'results/folds'/setting/f'fold{fold}'/view;x=xx[view]
            c=read(path/'complete.json');assert c['lock_sha256']==sha(R/'lock.json');check(path,c['files_sha256'])
            a=np.load(path/'arrays.npz',allow_pickle=False);t=read(path/'transforms.json');sel=read(path/'selection.json')
            width=40
            assert sel['feature_width']==width and a['inner_coef'].shape==(5,25,width) and a['final_coef'].shape==(2,width)
            prior=read(P/'results/folds'/setting/f'fold{fold}'/view/'transforms.json')
            for nt,ot in zip(t['inner']+[t['final']],prior['inner']+[prior['final']]):
                for key in ot:close(nt['projection'][key],ot[key],1e-12)
            oof=np.full((len(fi),25),np.nan);coverage=[]
            for j,sp in enumerate(item['inner']):
                tr=np.array(sp['train']);va=np.array(sp['valid']);coverage+=va.tolist()
                z=expand(x[fi[tr]],x[fi[tr]],t['inner'][j],view);zv=expand(x[fi[tr]],x[fi[va]],t['inner'][j],view)
                close(a['inner_intercept'][j],y[fi[tr]].mean(),1e-12)
                maxres=max(maxres,equations(z,y[fi[tr]],a['inner_coef'][j],cfg));count+=25
                oof[va]=zv@a['inner_coef'][j].T+a['inner_intercept'][j]
            assert sorted(coverage)==list(range(len(fi)));close(oof,a['inner_oof'],1e-10)
            with np.load(OLD/'results/folds'/setting/f'fold{fold}'/'arrays.npz',allow_pickle=False) as old:
                err=float(np.max(abs(oof[:,:5]-old['inner_oof'][:,5:10])));maxTD=max(maxTD,err);assert err<1e-10
            yc=y[fi]-y[fi].mean();pc=oof-oof.mean(0);den=np.sqrt((yc@yc)*np.sum(pc*pc,axis=0))
            corr=np.divide(yc@pc,den,out=np.full(25,-2.),where=den>1e-14);mse=np.mean((oof-y[fi,None])**2,axis=0)
            close(corr,sel['PCC']);close(mse,sel['MSE'])
            key=lambda k:(-corr[k],mse[k],cfg[k]['weight'],-cfg[k]['alpha'],k)
            chosen=dict(zip(MODES,(min(range(15,20),key=key),min(range(25),key=key))))
            assert chosen==sel['selected'] and sel['fit_ids']==item['fit_ids'] and sel['test_ids']==item['test_ids'] and sel['outer_validation_unused']
            z=expand(x[fi],x[fi],t['final'],view);zt=expand(x[fi],x[ti],t['final'],view)
            close(a['final_intercept'],y[fi].mean(),1e-12)
            maxres=max(maxres,equations(z,y[fi],a['final_coef'],[cfg[chosen[m]] for m in MODES]));final+=2
            pstate=t['final']['projection'];coef=a['final_coef'];source_coef=coef[:,24:]/t['final']['common_pc_scale']
            td_coef=coef[:,:24]/np.asarray(pstate['td_scale'])
            src_coef=(source_coef@np.asarray(pstate['components']))/np.asarray(pstate['src_scale'])
            offset=np.full(2,float(a['final_intercept']))
            offset-=td_coef@np.asarray(pstate['td_mean'])+src_coef@np.asarray(pstate['src_mean'])
            offset-=source_coef@(np.asarray(pstate['components'])@np.asarray(pstate['pca_mean'])+np.asarray(pstate['pc_mean']))
            close(np.c_[td_coef,src_coef],a['collapsed_coef'],1e-10);close(offset,a['collapsed_intercept'],1e-10)
            for k,mode in enumerate(MODES):
                pp=zt@a['final_coef'][k]+a['final_intercept'];close(pp,a['test_pred'][:,k],1e-10)
                maxPrediction=max(maxPrediction,float(np.max(abs(pp-a['test_pred'][:,k]))))
                close(x[ti]@a['collapsed_coef'][k]+a['collapsed_intercept'][k],pp,1e-10)
                cond=condition(view,mode);part=[r for r in pred if r['setting']==setting and int(r['fold'])==fold and r['condition']==cond]
                assert [r['sample_id'] for r in part]==item['test_ids'];close([float(r['prediction']) for r in part],pp,1e-10)
                for r,i in zip(part,ti):assert float(r['label'])==y[i] and int(r['binary'])==binary[i]
                sr=[r for r in selection if r['setting']==setting and int(r['fold'])==fold and r['condition']==cond];assert len(sr)==1
                assert int(sr[0]['candidate'])==chosen[mode]
                for f in ('weight','alpha'):close(float(sr[0][f]),cfg[chosen[mode]][f])
                canonical.setdefault((setting,cond),[]).extend(zip(ti,[float(r['prediction']) for r in part]))
            a.close()
        print('VERIFIED',setting,fold,flush=True)
    assert count==6000 and final==96
    diagnostic=read(R/'diagnostic.json');assert diagnostic['qualified'] and not diagnostic['real_labels_used']
    assert diagnostic['lock_sha256']==sha(R/'diagnostic_lock.json') and len(diagnostic['rows'])==9
    dv=read(R/'diagnostic_verification.json')
    assert dv['passed'] and dv['qualified'] and dv['independent_half_SVDs']==18
    assert not dv['real_labels_used'] and dv['new_efficiency_fits']==0
    assert dv['diagnostic_sha256']==sha(R/'diagnostic.json') and dv['lock_sha256']==sha(R/'diagnostic_lock.json')
    assert dv['code_sha256']==sha(R/'audit_spectral.py')
    diagnostics_checked=9
    assert complete['new_PCA_fits']==0 and complete['inner_scale_blocks']==240 and complete['final_scale_blocks']==48
    assert complete['inner_fits']==6000 and complete['final_heads']==96 and complete['rows']==22788
    baseline={(r['setting'],r['condition']):r for r in table(OLD/'results/summary.csv')}
    for row in summary:
        pairs=canonical[row['setting'],row['condition']];ii=np.array([i for i,p in pairs]);p=np.array([p for i,p in pairs])
        cohort='Mixset' if row['setting']=='Inter_Hu_to_Mix' else row['setting'];expected={i for i,r in enumerate(rows) if r['cohort']==cohort}
        assert len(ii)==len(expected) and set(ii)==expected
        m=audit4.metric(y[ii],p,binary[ii])
        for k,v in m.items():close(float(row[k]),v)
        base=baseline[row['setting'],'TD24_Ridge_CV'];bp=float(base['PCC'])
        close(float(row['TD24_PCC']),bp);close(float(row['PCC_delta']),m['PCC']-bp)
        close(float(row['PCC_relative_gain_pct']),100*(m['PCC']/bp-1));close(float(row['MSE_relative_reduction_pct']),100*(1-m['MSE']/float(base['MSE'])))
    goals=read(R/'results/goal_summary.json');assert len(goals)==6
    for r in goals:
        part=[x for x in summary if x['condition']==r['condition']];assert len(part)==4
        g=float(np.mean([float(x['PCC_relative_gain_pct']) for x in part]));close(r['mean_relative_PCC_gain_pct'],g)
        assert r['reaches_development_5pct']==(g>=5) and r['primary']==(r['condition']==plan['primary_condition'])
    result=dict(passed=True,inner_normal_equations=count,final_heads=final,OOF_rows=22788,source_models=3,views=3,feature_only_diagnostics_checked=diagnostics_checked,
        max_normal_residual=maxres,max_TD24_candidate_prediction_difference=maxTD,max_final_prediction_reconstruction_difference=maxPrediction,
        all_PCA_common_scale_deployment_selection_metrics_checked=True,source_checkpoint_hashes_verified=True,
        lock_sha256=sha(R/'lock.json'),complete_sha256=sha(R/'results/complete.json'),verifier_sha256=sha(Path(__file__)),
        GPU_hours=0,new_source_forward=0,CPU_wall_seconds=time.monotonic()-started)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':
    begin=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-begin,GPU_hours=0),f,indent=2)

