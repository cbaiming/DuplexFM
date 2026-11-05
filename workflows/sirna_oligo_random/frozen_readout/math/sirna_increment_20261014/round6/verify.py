"""Independent consensus selection and reused/new head provenance audit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.stats import pearsonr
R=Path(__file__).resolve().parent;P=R.parent/'round5';B=R.parents[1]
spec=importlib.util.spec_from_file_location('_r5_verified_math',P/'verify_artifact_metrics.py')
a5=importlib.util.module_from_spec(spec);spec.loader.exec_module(a5)
read,sha,table,close,check=a5.read,a5.sha,a5.table,a5.close,a5.check
VIEWS=('core_real','full_real','initial_core','TDQuadratic');MODES=('Uniform','Adaptive')
def condition(view,mode):return view+'_'+mode+('Quadratic' if view=='TDQuadratic' else 'Bilinear')+'_Consensus'

def selection(y,p,cfg,blocks,allowed,stored):
    assert sorted(int(i) for b in blocks for i in b)==list(range(len(y)))
    def order(ii):
        pc=np.array([pearsonr(y[ii],p[ii,k]).statistic if np.std(p[ii,k])>1e-14 else -2 for k in range(25)])
        ms=np.array([np.mean((y[ii]-p[ii,k])**2) for k in range(25)])
        return sorted(allowed,key=lambda k:(-pc[k],ms[k],cfg[k]['weight'],-cfg[k]['alpha'],k)),pc,ms
    original,pc,ms=order(np.arange(len(y)));winners=[]
    for block in blocks:
        omitted=set(int(i) for i in block);ii=np.array([i for i in range(len(y)) if i not in omitted]);winners.append(order(ii)[0][0])
    counts=Counter(winners);rank={k:i for i,k in enumerate(original)}
    chosen=min(allowed,key=lambda k:(-counts[k],rank[k]))
    assert stored['original']==original[0] and stored['selected']==chosen and stored['winners']==winners
    assert stored['votes']==[counts[k] for k in range(25)] and stored['original_support']==counts[original[0]]
    assert stored['unique_winners']==len(counts);close(stored['PCC'],pc);close(stored['MSE'],ms)
    close(stored['original_margin'],pc[original[0]]-pc[original[1]])
    return chosen

def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    c=read(R/'results/complete.json');assert c['lock_sha256']==sha(R/'lock.json');check(R/'results',c['files_sha256'])
    executions=[read(p) for p in sorted(R.glob('execution_*.json'))];assert executions[-1]['returncode']==0
    assert all(e['GPU_hours']==0 and e['new_source_forward']==0 for e in executions)
    plan=read(R/'plan.json');assert plan['folds']==read(P/'plan.json')['folds'];cfg=plan['configs']
    assert cfg==[dict(weight=w,alpha=a) for w in (0.,.01,.1,1.,10.) for a in (.1,1.,10.,100.,1000.)]
    rows=sum([table(a5.BENCH/f'{s}.csv') for s in a5.SETTINGS[:3]],[]);ids={r['sample_id']:i for i,r in enumerate(rows)}
    td=np.array([[float(v) for v in r['td'].split(',')] for r in rows]);xx={'TDQuadratic':td}
    for view in VIEWS[:3]:
        meta=table(a5.EXPORT/view/'row_identity.csv');ind={r['sample_id']:i for i,r in enumerate(meta)}
        for row in rows:assert all(row[k]==meta[ind[row['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
        source=np.load(a5.EXPORT/view/'source.npy',allow_pickle=False)[[ind[r['sample_id']] for r in rows]].astype(float)
        xx[view]=np.c_[td,source]
    y=np.array([float(r['label']) for r in rows]);binary=np.array([int(r['y']) for r in rows])
    pred=table(R/'results/oof_predictions.csv');summary=table(R/'results/summary.csv');selected=table(R/'results/selection.csv')
    assert len(pred)==34182 and len(summary)==36 and len(selected)==144
    new=0;reused=0;matchedTD=0;maxerr=0.;maxpred=0.;canon={};votechecks=0
    for item in plan['folds']:
        s,f=item['setting'],item['fold'];fi=np.array([ids[i] for i in item['fit_ids']]);ti=np.array([ids[i] for i in item['test_ids']]);blocks=[v['valid'] for v in item['inner']]
        assert not set(fi)&set(ti)
        for view in VIEWS:
            parent=P/'results/folds'/s/f'fold{f}'/view;out=R/'results/folds'/s/f'fold{f}'/view
            cc=read(out/'complete.json');assert cc['lock_sha256']==sha(R/'lock.json');check(out,cc['files_sha256'])
            with np.load(parent/'arrays.npz',allow_pickle=False) as p:
                inner=p['inner_oof'].copy();bc=p['final_coef'].copy();bp=p['test_pred'].copy();bi=float(p['final_intercept'])
            a=np.load(out/'arrays.npz',allow_pickle=False);meta=read(out/'selection.json');ps=read(parent/'selection.json');t=read(parent/'transforms.json')['final']
            assert meta['fit_ids']==item['fit_ids'] and meta['test_ids']==item['test_ids'] and meta['outer_validation_unused']
            assert meta['parent_arrays_sha256']==sha(parent/'arrays.npz') and meta['parent_transforms_sha256']==sha(parent/'transforms.json')
            assert float(a['final_intercept'])==bi;z=zt=None
            for k,mode in enumerate(MODES):
                d=meta['diagnostics'][mode];chosen=selection(y[fi],inner,cfg,blocks,list(range(15,20) if mode=='Uniform' else range(25)),d);votechecks+=1
                assert d['original']==ps['selected'][mode];origin=meta['origin'][mode]
                matches=[m for m in MODES if ps['selected'][m]==chosen]
                if matches:
                    oldmode=matches[0];j=MODES.index(oldmode)
                    assert origin['kind']=='reuse_R5_final' and origin['mode']==oldmode and origin['parent_array_sha256']==sha(parent/'arrays.npz')
                    assert a['final_coef'][k].tobytes()==bc[j].tobytes() and a['test_pred'][:,k].tobytes()==bp[:,j].tobytes()
                    pp=bp[:,j];reused+=1
                else:
                    assert origin['kind']=='new_R6_final' and origin['parent_transform_sha256']==sha(parent/'transforms.json')
                    if z is None:z=a5.expand(xx[view][fi],xx[view][fi],t,view);zt=a5.expand(xx[view][fi],xx[view][ti],t,view)
                    maxerr=max(maxerr,a5.equations(z,y[fi],a['final_coef'][k:k+1],[cfg[chosen]]))
                    pp=zt@a['final_coef'][k]+bi;close(pp,a['test_pred'][:,k],1e-10);maxpred=max(maxpred,float(np.max(abs(pp-a['test_pred'][:,k]))));new+=1
                cond=condition(view,mode);part=[r for r in pred if r['setting']==s and int(r['fold'])==f and r['condition']==cond]
                assert [r['sample_id'] for r in part]==item['test_ids'];actual=np.array([float(r['prediction']) for r in part]);close(actual,pp,1e-10)
                for r,i in zip(part,ti):assert float(r['label'])==y[i] and int(r['binary'])==binary[i]
                record=[r for r in selected if r['setting']==s and int(r['fold'])==f and r['condition']==cond];assert len(record)==1 and int(record[0]['candidate'])==chosen
                assert (record[0]['new_fit']=='True')==(not bool(matches))
                canon.setdefault((s,cond),[]).extend(zip(ti,actual))
            a.close()
        out=R/'results/folds'/s/f'fold{f}'/'TD24_Consensus';tdmeta=read(out/'selection.json');check(out,read(out/'complete.json')['files_sha256'])
        parent=a5.OLD/'results/folds'/s/f'fold{f}'
        with np.load(P/'results/folds'/s/f'fold{f}'/'core_real/arrays.npz',allow_pickle=False) as z:inner=z['inner_oof']
        chosen=selection(y[fi],inner,cfg,blocks,list(range(5)),tdmeta['diagnostics']);votechecks+=1
        assert chosen==read(parent/'selection.json')['selected']['TD24_Ridge_CV']-5==tdmeta['original_candidate']
        assert tdmeta['exact_original_baseline'] and tdmeta['parent_prediction_sha256']==sha(parent/'predictions.csv')
        oldrows=[r for r in table(parent/'predictions.csv') if r['condition']=='TD24_Ridge_CV']
        part=[r for r in pred if r['setting']==s and int(r['fold'])==f and r['condition']=='TD24_Consensus']
        assert [r['sample_id'] for r in part]==item['test_ids']==[r['sample_id'] for r in oldrows]
        for oldrow,row in zip(oldrows,part):assert all(float(row[k])==float(oldrow[k]) for k in ('label','binary','prediction'))
        canon.setdefault((s,'TD24_Consensus'),[]).extend(zip(ti,[float(r['prediction']) for r in part]));matchedTD+=1
        print('VERIFIED',s,f,flush=True)
    assert new+reused==128 and matchedTD==16 and votechecks==144
    assert c['new_final_heads']==new and c['reused_R5_final_heads']==reused and c['reused_matched_TD_heads']==matchedTD and c['new_inner_fits']==0
    baseline={(r['setting'],r['condition']):r for r in table(a5.OLD/'results/summary.csv')}
    for row in summary:
        pairs=canon[row['setting'],row['condition']];ii=np.array([i for i,p in pairs]);pp=np.array([p for i,p in pairs]);cohort='Mixset' if row['setting']=='Inter_Hu_to_Mix' else row['setting']
        expected={i for i,r in enumerate(rows) if r['cohort']==cohort};assert len(ii)==len(expected) and set(ii)==expected
        met=a5.audit4.metric(y[ii],pp,binary[ii]);base=baseline[row['setting'],'TD24_Ridge_CV']
        for k,value in met.items():close(value,float(row[k]),1e-12)
        close(float(row['TD24_PCC']),float(base['PCC']),1e-12)
        close(float(row['PCC_delta']),met['PCC']-float(base['PCC']),1e-12)
        close(float(row['PCC_relative_gain_pct']),100*(met['PCC']/float(base['PCC'])-1),1e-10)
        close(float(row['MSE_relative_reduction_pct']),100*(1-met['MSE']/float(base['MSE'])),1e-10)
        if row['condition']=='TD24_Consensus':
            for k,value in met.items():close(value,float(base[k]),1e-12)
    goals=read(R/'results/goal_summary.json');assert len(goals)==9
    for g in goals:
        values=[float(r['PCC_relative_gain_pct']) for r in summary if r['condition']==g['condition']];assert len(values)==4
        gain=float(np.mean(values));close(g['mean_relative_PCC_gain_pct'],gain);assert g['reaches_development_5pct']==(gain>=5)
        assert g['primary']==(g['condition']==plan['primary_condition'])
    result=dict(passed=True,consensus_selections_checked=votechecks,new_final_heads=new,reused_R5_final_heads=reused,
        matched_TD_heads_identical=matchedTD,new_inner_fits=0,OOF_rows=34182,max_new_head_normal_residual=maxerr,
        max_new_prediction_reconstruction_difference=maxpred,all_stored_metrics_exactly_recomputed=True,
        original_baseline_unchanged=True,lock_sha256=sha(R/'lock.json'),complete_sha256=sha(R/'results/complete.json'),
        verifier_sha256=sha(Path(__file__)),GPU_hours=0,new_source_forward=0,CPU_wall_seconds=time.monotonic()-started)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':
    begin=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-begin,GPU_hours=0),f,indent=2)
