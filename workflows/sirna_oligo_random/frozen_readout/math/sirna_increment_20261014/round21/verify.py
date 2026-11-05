"""Independent joint-system KKT, raw deployment and exact coverage readback."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4'
spec=importlib.util.spec_from_file_location('_r21_independent4',P/'verify.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close

def metric(y,p):return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))

def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for path,h in lock['dependencies'].items():assert sha(Path(path))==h,path
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and d['new_efficiency_fits']==90
    assert d['new_source_forward']==d['new_PCA_fits']==d['GPU_hours']==0 and not d['outer_test_metrics_used']
    assert all(t['num_threads']==1 for t in d['threadpool']);check(R/'pilot',d['pilot_files_sha256'])
    assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    execution=[read(p) for p in R.glob('execution_*.json')];assert len(execution)==1 and execution[0]['returncode']==0
    plan=read(P/'plan.json');details=table(R/'diagnostic_rows.csv');count=0;covered=0;allvalid=True;converged=0
    maxpred=0.;maxkkt=0.;maxdense=0.;maxcert=0.;support=[]
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids'];assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        lookup={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')};rows=[lookup[s] for s in ids]
        y=np.array([float(r['label']) for r in rows]);td=np.array([[float(v) for v in r['td'].split(',')] for r in rows])
        for source in ('core_real','full_real','initial_core'):
            folder=old.EXPORT/source;meta=table(folder/'row_identity.csv');idx={r['sample_id']:i for i,r in enumerate(meta)}
            for row in rows:assert all(row[k]==meta[idx[row['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
            raw=np.c_[td,np.load(folder/'source.npy',allow_pickle=False)[[idx[s] for s in ids]].astype(float)]
            pf=P/'results/folds'/setting/'fold0'/source/'transforms.json';refs=read(pf)
            with np.load(P/'results/folds'/setting/'fold0'/source/'arrays.npz',allow_pickle=False) as a:reference=a['inner_oof'][:,18].copy()
            for mode in ('dense','sparse'):
                pred=np.full(len(y),np.nan);coverage=[];blockvalid=[];blocksupport=[]
                for j,sp in enumerate(item['inner']):
                    tr=np.array(sp['train']);va=np.array(sp['valid']);coverage+=va.tolist();out=R/'pilot'/setting/source/mode/f'inner{j}'
                    c=read(out/'complete.json');assert c['lock_sha256']==sha(R/'lock.json');check(out,c['files_sha256'])
                    state=read(out/'state.json');t=read(out/'transform.json');ref=refs['inner'][j];cert=state['certificate']
                    assert state['reference_transform_sha256']==sha(pf) and state['train_indices']==tr.tolist() and state['valid_indices']==va.tolist()
                    assert state['setting']==setting and state['source']==source and state['mode']==mode==cert['mode'] and state['inner']==j
                    old.project(raw[tr],raw[tr],ref)
                    for k in ('td_mean','td_scale','src_mean','src_scale'):close(t[k],ref[k],0)
                    mean=np.r_[t['td_mean'],t['src_mean']];scale=np.r_[t['td_scale'],np.asarray(t['src_scale'])*np.sqrt(12.)]
                    z=(raw[tr]-mean)/scale;zv=(raw[va]-mean)/scale;close(z.mean(0),np.zeros(216),1e-10)
                    assert np.var(z[:,24:],axis=0).sum()<=16+1e-10
                    a=np.load(out/'model.npz',allow_pickle=False);b=a['beta'];center=float(a['intercept']);close(center,y[tr].mean(),1e-12)
                    g=z.T@z+100*np.eye(216);q=z.T@(y[tr]-center)
                    td0=np.linalg.solve(g[:24,:24],q[:24]);condition_q=q[24:]-g[24:,:24]@td0
                    lmax=float(max(abs(condition_q)));lam=.1*lmax if mode=='sparse' else 0.
                    close(cert['lambda_max'],lmax,1e-10);close(cert['lambda_value'],lam,1e-10)
                    assert cert['alpha']==100 and cert['lambda_ratio']==(.1 if mode=='sparse' else 0.) and cert['source_coordinates']==192
                    gradient=g@b-q;tail=np.where(b[24:]!=0,gradient[24:]+lam*np.sign(b[24:]),np.sign(gradient[24:])*np.maximum(abs(gradient[24:])-lam,0.))
                    residual=float(max(abs(tail)));maxkkt=max(maxkkt,residual);threshold=1e-8*(1+lmax)
                    close(gradient[:24],np.zeros(24),1e-9);close(cert['profiled_KKT_max'],residual,1e-10);close(cert['KKT_threshold'],threshold,1e-12)
                    bound=float(tail@tail/200);close(cert['profiled_suboptimality_upper_bound'],bound,1e-12);maxcert=max(maxcert,bound)
                    valid=bool(cert['converged'] and residual<=threshold+1e-11);assert state['numerically_qualified']==valid;allvalid &= valid
                    if cert['converged']:converged+=1
                    else:assert cert['iterations']==5000
                    assert 0<=cert['iterations']<=5000 and cert['source_nonzero']==int(np.count_nonzero(b[24:]))
                    nactive=cert['source_nonzero'];blocksupport.append(nactive);blockvalid.append(valid)
                    if mode=='dense':
                        direct=np.linalg.solve(g,q);close(b,direct,1e-10);maxdense=max(maxdense,float(max(abs(b-direct))))
                    close(cert['train_MSE'],np.mean((z@b+center-y[tr])**2),1e-12)
                    pp=zv@b+center;close(pp,a['prediction'],1e-10);maxpred=max(maxpred,float(max(abs(pp-a['prediction']))))
                    close(a['deployment_coef'],b/scale,1e-12);close(a['deployment_intercept'],center-mean@(b/scale),1e-12)
                    close(raw[va]@a['deployment_coef']+a['deployment_intercept'],pp,1e-9)
                    assert (out/'transform.json').stat().st_mtime_ns<=(out/'model.npz').stat().st_mtime_ns
                    pred[va]=a['prediction'];a.close();count+=1
                    support.append(dict(setting=setting,source=source,mode=mode,inner=j,nonzero_source=nactive,KKT=residual,iterations=cert['iterations']))
                assert sorted(coverage)==list(range(len(y)))
                part=[r for r in details if r['setting']==setting and r['source']==source and r['mode']==mode];assert [r['sample_id'] for r in part]==ids
                close([float(r['label']) for r in part],y,0);close([float(r['prediction']) for r in part],pred,0);close([float(r['reference_prediction']) for r in part],reference,0);covered+=len(part)
                record=next(r for r in d['records'] if r['setting']==setting and r['source']==source and r['mode']==mode);assert record['n']==len(y)
                m,r=metric(y,pred),metric(y,reference)
                for name,expected in (('metrics',m),('reference',r)):
                    for k,v in expected.items():close(record[name][k],v,1e-11)
                delta=m['PCC']-r['PCC'];red=100*(1-m['MSE']/r['MSE'])
                close(record['delta_PCC_vs_R4'],delta,1e-11);close(record['MSE_reduction_vs_R4_pct'],red,1e-10)
                assert record['pool_qualified']==(delta>=.005 and red>=0)
                for j,sp in enumerate(item['inner']):
                    va=np.array(sp['valid']);bl=record['inner_blocks'][j]
                    assert bl['inner']==j and bl['n']==len(va) and bl['source_nonzero']==blocksupport[j] and bl['numerically_qualified']==blockvalid[j]
                    for name,expected in (('metrics',metric(y[va],pred[va])),('reference',metric(y[va],reference[va]))):
                        for k,v in expected.items():close(bl[name][k],v,1e-11)
            assert d['source_identities'][source]==read(folder/'complete.json')['source_identity']
    assert count==90 and covered==len(details)==13794
    core=[r for r in d['records'] if r['source']=='core_real' and r['mode']=='sparse']
    perf=sum(r['pool_qualified'] for r in core)>=2 and all(r['delta_PCC_vs_R4']>=-.005 for r in core);paired=[]
    for a in core:
        b=next(r for r in d['records'] if r['source']=='core_real' and r['mode']=='dense' and r['setting']==a['setting'])
        p=next(r for r in d['paired_core_records'] if r['setting']==a['setting']);dp=a['metrics']['PCC']-b['metrics']['PCC'];red=100*(1-a['metrics']['MSE']/b['metrics']['MSE'])
        close(p['delta_PCC'],dp,1e-11);close(p['MSE_reduction_pct'],red,1e-10);paired.append(red)
    pg=sum(r>=0 for r in paired)>=2 and np.mean(paired)>=0
    assert d['performance_gate']==perf and d['paired_gate']==pg and d['all_numerically_qualified']==allvalid and d['qualified']==bool(perf and pg and allvalid)
    result=dict(passed=True,qualified=bool(perf and pg and allvalid),verified_heads=count,converged_heads=converged,all_numerically_qualified=bool(allvalid),
        verified_dense_independent_solves=45,internal_OOF_rows=covered,max_prediction_difference=maxpred,max_joint_KKT=maxkkt,
        max_dense_coefficient_difference=maxdense,max_profiled_suboptimality_upper_bound=maxcert,support_records=support,
        new_source_forward=0,GPU_hours=0,new_PCA_fits=0,outer_test_metrics_used=False,
        lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-started)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='support_records'}),flush=True)

if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0),f,indent=2)
