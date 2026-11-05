"""Independent SVD screen, PCA spectrum, head equations and deployment audit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4'
spec=importlib.util.spec_from_file_location('_r19_independent4',P/'verify.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close


def scores(td,source,y):
    t=td-td.mean(0);s=source-source.mean(0);yc=y-y.mean()
    u,v,_=np.linalg.svd(t,full_matrices=False);rank=int(np.sum(v>v[0]*1e-10)) if v[0]>0 else 0;q=u[:,:rank]
    s-=q@(q.T@s);yc-=q@(q.T@yc);yn=float(yc@yc);sn=np.sum(s*s,axis=0);den=np.sqrt(sn*yn)
    return np.clip(np.divide(s.T@yc,den,out=np.zeros(192),where=den>1e-12),-1,1),rank,yn,sn


def project_selected(raw,ref,st):
    td=(raw[:,:24]-ref['td_mean'])/ref['td_scale'];source=(raw[:,24:]-ref['src_mean'])/ref['src_scale']
    selected=np.asarray(st['selected'],int);pc=(source[:,selected]-st['pca_mean'])@np.asarray(st['components']).T
    return np.c_[td,(pc-st['pc_mean'])/st['pc_scale']]


def metric(y,p):return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))


def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for path,h in lock['dependencies'].items():assert sha(Path(path))==h,path
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and d['new_efficiency_fits']==d['new_PCA_fits']==90
    assert d['new_source_forward']==d['GPU_hours']==0 and not d['outer_test_metrics_used']
    assert all(t['num_threads']==1 for t in d['threadpool']);check(R/'pilot',d['pilot_files_sha256'])
    assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    execution=[read(p) for p in R.glob('execution_*.json')];assert len(execution)==1 and execution[0]['returncode']==0
    plan=read(P/'plan.json');details=table(R/'diagnostic_rows.csv');count=0;covered=0;allvalid=True
    max_prediction=0.;max_score_difference=0.;max_normal=0.;selection_records=[]
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids'];assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        lookup={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')};rows=[lookup[s] for s in ids]
        y=np.array([float(r['label']) for r in rows]);td=np.array([[float(v) for v in r['td'].split(',')] for r in rows])
        for source in ('core_real','full_real','initial_core'):
            folder=old.EXPORT/source;meta=table(folder/'row_identity.csv');idx={r['sample_id']:i for i,r in enumerate(meta)}
            for row in rows:assert all(row[k]==meta[idx[row['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
            raw=np.c_[td,np.load(folder/'source.npy',allow_pickle=False)[[idx[s] for s in ids]].astype(float)]
            pf=P/'results/folds'/setting/'fold0'/source/'transforms.json';projections=read(pf)
            with np.load(P/'results/folds'/setting/'fold0'/source/'arrays.npz',allow_pickle=False) as a:reference=a['inner_oof'][:,18].copy()
            for mode in ('conditional','random'):
                pred=np.full(len(y),np.nan);coverage=[];blockvalid=[]
                for j,sp in enumerate(item['inner']):
                    tr=np.array(sp['train']);va=np.array(sp['valid']);coverage+=va.tolist();out=R/'pilot'/setting/source/mode/f'inner{j}'
                    c=read(out/'complete.json');assert c['lock_sha256']==sha(R/'lock.json');check(out,c['files_sha256'])
                    st=read(out/'state.json');t=read(out/'transform.json');ref=projections['inner'][j]
                    assert st['projection_sha256']==sha(pf) and st['train_indices']==tr.tolist() and st['valid_indices']==va.tolist()
                    assert st['setting']==setting and st['source']==source and st['mode']==t['mode']==mode and st['inner']==j
                    old.project(raw[tr],raw[tr],ref)
                    for k in ('td_mean','td_scale','src_mean','src_scale'):close(t[k],ref[k],0)
                    tt=(raw[tr,:24]-ref['td_mean'])/ref['td_scale'];ss=(raw[tr,24:]-ref['src_mean'])/ref['src_scale']
                    corr,rank,yn,sn=scores(tt,ss,y[tr]);close(t['partial_correlation'],corr,1e-10)
                    max_score_difference=max(max_score_difference,float(abs(corr-t['partial_correlation']).max()))
                    assert t['td_rank']==rank;close(t['residual_y_norm_squared'],yn,1e-10);close(t['residual_source_norm_squared'],sn,1e-8)
                    assert t['response_residual_defined']==(yn>1e-12)
                    if mode=='conditional':selected=np.sort(sorted(range(192),key=lambda k:(-abs(corr[k]),k))[:32])
                    else:selected=np.sort(np.random.default_rng(42).choice(192,32,replace=False))
                    assert selected.tolist()==t['selected'] and len(set(t['selected']))==32
                    z=ss[:,selected];mean=z.mean(0);z-=mean;close(t['pca_mean'],mean,1e-12)
                    gram=z.T@z/len(z);components=np.array(t['components']);eig=np.array(t['eigenvalues'])
                    close(components@components.T,np.eye(16),1e-10);close(gram@components.T,components.T*eig,1e-9)
                    close(eig,np.maximum(np.linalg.eigvalsh(gram)[-16:][::-1],0),1e-9)
                    for row in components:assert row[np.argmax(abs(row))]>=0
                    pcm,pcs=old.scale(z@components.T,True);close(t['pc_mean'],pcm,1e-12);close(t['pc_scale'],pcs,1e-10)
                    rankvalid=bool(eig[0]>0 and eig[-1]>1e-10*eig[0]);assert t['rank16_valid']==rankvalid
                    valid=bool(rankvalid and yn>1e-12);assert st['rank_and_response_valid']==valid;allvalid &= valid;blockvalid.append(valid)
                    x=project_selected(raw[tr],ref,t);xv=project_selected(raw[va],ref,t);a=np.load(out/'model.npz',allow_pickle=False)
                    b,center=a['beta'],float(a['intercept']);close(center,y[tr].mean(),1e-12)
                    norm=float(abs((x.T@x+100*np.eye(40))@b-x.T@(y[tr]-y[tr].mean())).max()/len(tr));assert norm<1e-8;max_normal=max(max_normal,norm)
                    pp=xv@b+center;close(pp,a['prediction'],1e-10);max_prediction=max(max_prediction,float(abs(pp-a['prediction']).max()))
                    close(raw[va]@a['deployment_coef']+a['deployment_intercept'],pp,1e-9)
                    used=set((np.flatnonzero(a['deployment_coef'][24:])).tolist());assert used.issubset(set(selected.tolist()))
                    assert (out/'transform.json').stat().st_mtime_ns<=(out/'model.npz').stat().st_mtime_ns
                    pred[va]=a['prediction'];a.close();count+=1
                    selection_records.append(dict(setting=setting,source=source,mode=mode,inner=j,selected=selected.tolist(),td_rank=rank,rank_and_response_valid=valid))
                assert sorted(coverage)==list(range(len(y)))
                part=[r for r in details if r['setting']==setting and r['source']==source and r['mode']==mode];assert [r['sample_id'] for r in part]==ids
                close([float(r['label']) for r in part],y,0);close([float(r['reference_prediction']) for r in part],reference,0)
                close([float(r['prediction']) for r in part],pred,0);covered+=len(part)
                record=next(r for r in d['records'] if r['setting']==setting and r['source']==source and r['mode']==mode);assert record['n']==len(y)
                m,refmetrics=metric(y,pred),metric(y,reference)
                for name,expected in (('metrics',m),('reference',refmetrics)):
                    for key,value in expected.items():close(record[name][key],value,1e-11)
                delta=m['PCC']-refmetrics['PCC'];reduction=100*(1-m['MSE']/refmetrics['MSE'])
                close(record['delta_PCC_vs_R4'],delta,1e-11);close(record['MSE_reduction_vs_R4_pct'],reduction,1e-10)
                assert record['pool_qualified']==(delta>=.005 and reduction>=0)
                for j,sp in enumerate(item['inner']):
                    va=np.array(sp['valid']);b=record['inner_blocks'][j];assert b['inner']==j and b['n']==len(va) and b['rank_and_response_valid']==blockvalid[j]
                    for name,expected in (('metrics',metric(y[va],pred[va])),('reference',metric(y[va],reference[va]))):
                        for k,value in expected.items():close(b[name][k],value,1e-11)
            assert d['source_identities'][source]==read(folder/'complete.json')['source_identity']
    assert count==90 and covered==len(details)==13794
    core=[r for r in d['records'] if r['source']=='core_real' and r['mode']=='conditional']
    gate=sum(r['pool_qualified'] for r in core)>=2 and all(r['delta_PCC_vs_R4']>=-.005 for r in core);paired=[]
    for a in core:
        b=next(r for r in d['records'] if r['source']=='core_real' and r['mode']=='random' and r['setting']==a['setting'])
        p=next(r for r in d['paired_core_records'] if r['setting']==a['setting']);dp=a['metrics']['PCC']-b['metrics']['PCC'];red=100*(1-a['metrics']['MSE']/b['metrics']['MSE'])
        close(p['delta_PCC'],dp,1e-11);close(p['MSE_reduction_pct'],red,1e-10);paired.append(red)
    pg=sum(r>=0 for r in paired)>=2 and np.mean(paired)>=0
    assert d['performance_gate']==gate and d['paired_gate']==pg and d['all_rank_and_response_valid']==allvalid and d['qualified']==bool(gate and pg and allvalid)
    result=dict(passed=True,qualified=bool(gate and pg and allvalid),all_rank_and_response_valid=bool(allvalid),verified_heads=count,
        verified_new_PCA_fits=90,internal_OOF_rows=covered,max_prediction_difference=max_prediction,
        max_partial_score_difference=max_score_difference,max_normal_equation_residual=max_normal,selection_records=selection_records,
        new_source_forward=0,GPU_hours=0,outer_test_metrics_used=False,
        lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-started)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='selection_records'}),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0),f,indent=2)
