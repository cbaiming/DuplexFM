"""Independent sample-SVD whitening and fixed-projection readback."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4'
spec=importlib.util.spec_from_file_location('_r20_independent4',P/'verify.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close


def project(raw,ref,t):
    td=(raw[:,:24]-ref['td_mean'])/ref['td_scale'];source=(raw[:,24:]-ref['src_mean'])/ref['src_scale']
    return np.c_[td,(source-t['source_center'])@np.asarray(t['projection'])@np.asarray(t['whitening'])]


def metric(y,p):return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))


def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for path,h in lock['dependencies'].items():assert sha(Path(path))==h,path
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and d['new_efficiency_fits']==d['new_covariance_whitenings']==135
    assert d['new_source_forward']==d['GPU_hours']==d['new_PCA_fits']==0 and not d['outer_test_metrics_used'] and not d['prediction_averaging']
    assert all(t['num_threads']==1 for t in d['threadpool']);check(R/'pilot',d['pilot_files_sha256'])
    assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    execution=[read(p) for p in R.glob('execution_*.json')];assert len(execution)==1 and execution[0]['returncode']==0
    with np.load(R/'projections.npz',allow_pickle=False) as a:maps={s:a[f'seed{s}'].copy() for s in (42,43,44)}
    for seed,p in maps.items():
        g=np.random.default_rng(seed).standard_normal((192,16));q,r=np.linalg.qr(g,mode='reduced');q*=np.where(np.diag(r)<0,-1.,1.)
        close(p,q,0);close(p.T@p,np.eye(16),1e-12)
    plan=read(P/'plan.json');details=table(R/'diagnostic_rows.csv');count=0;covered=0;allvalid=True
    max_prediction=0.;max_whitening=0.;max_normal=0.;rank_records=[]
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
            for seed in (42,43,44):
                pred=np.full(len(y),np.nan);coverage=[];valids=[]
                for j,sp in enumerate(item['inner']):
                    tr=np.array(sp['train']);va=np.array(sp['valid']);coverage+=va.tolist();out=R/'pilot'/setting/source/f'map{seed}'/f'inner{j}'
                    c=read(out/'complete.json');assert c['lock_sha256']==sha(R/'lock.json');check(out,c['files_sha256'])
                    st=read(out/'state.json');t=read(out/'transform.json');ref=projections['inner'][j]
                    assert st['projection_sha256']==sha(pf) and st['fixed_maps_sha256']==sha(R/'projections.npz') and st['train_indices']==tr.tolist() and st['valid_indices']==va.tolist()
                    assert st['setting']==setting and st['source']==source and st['map_seed']==seed and st['inner']==j
                    old.project(raw[tr],raw[tr],ref)
                    for k in ('td_mean','td_scale','src_mean','src_scale'):close(t[k],ref[k],0)
                    close(t['projection'],maps[seed],0);ss=(raw[tr,24:]-ref['src_mean'])/ref['src_scale'];mu=ss.mean(0);close(mu,t['source_center'],1e-12)
                    z=(ss-mu)@maps[seed];cov=z.T@z/len(z);close(cov,t['covariance'],1e-11)
                    _,singular,right=np.linalg.svd(z,full_matrices=False);sd=singular/np.sqrt(len(z));eig=sd**2
                    close(t['eigenvalues'],eig[::-1],1e-10)
                    positive=sd[sd>0];floor=.01*(float(np.median(positive)) if len(positive) else 1.);close(floor,t['floor'],1e-12)
                    scales=np.where(sd>0,np.maximum(sd,floor),1.);white=(right.T/scales)@right
                    close(white,t['whitening'],1e-9);max_whitening=max(max_whitening,float(abs(white-t['whitening']).max()))
                    assert t['floor_active']==bool(np.any((sd>0)&(sd<floor)))
                    valid=bool(eig[0]>0 and eig[-1]>1e-10*eig[0]);assert t['rank16_valid']==valid;valids.append(valid);allvalid &= valid
                    x=project(raw[tr],ref,t);xv=project(raw[va],ref,t);a=np.load(out/'model.npz',allow_pickle=False)
                    b,center=a['beta'],float(a['intercept']);close(center,y[tr].mean(),1e-12)
                    norm=float(abs((x.T@x+100*np.eye(40))@b-x.T@(y[tr]-center)).max()/len(tr));assert norm<1e-8;max_normal=max(max_normal,norm)
                    pp=xv@b+center;close(pp,a['prediction'],1e-10);max_prediction=max(max_prediction,float(abs(pp-a['prediction']).max()))
                    close(raw[va]@a['deployment_coef']+a['deployment_intercept'],pp,1e-9)
                    assert (out/'transform.json').stat().st_mtime_ns<=(out/'model.npz').stat().st_mtime_ns
                    pred[va]=a['prediction'];a.close();count+=1
                    rank_records.append(dict(setting=setting,source=source,map_seed=seed,inner=j,rank16_valid=valid,floor_active=t['floor_active']))
                assert sorted(coverage)==list(range(len(y)))
                part=[r for r in details if r['setting']==setting and r['source']==source and int(r['map_seed'])==seed];assert [r['sample_id'] for r in part]==ids
                close([float(r['label']) for r in part],y,0);close([float(r['reference_prediction']) for r in part],reference,0)
                close([float(r['prediction']) for r in part],pred,0);covered+=len(part)
                rec=next(r for r in d['records'] if r['setting']==setting and r['source']==source and r['map_seed']==seed);assert rec['n']==len(y)
                m,refm=metric(y,pred),metric(y,reference)
                for name,expected in (('metrics',m),('reference',refm)):
                    for key,value in expected.items():close(rec[name][key],value,1e-11)
                delta=m['PCC']-refm['PCC'];reduction=100*(1-m['MSE']/refm['MSE'])
                close(rec['delta_PCC_vs_R4'],delta,1e-11);close(rec['MSE_reduction_vs_R4_pct'],reduction,1e-10)
                assert rec['pool_qualified']==(delta>=.005 and reduction>=0)
                for j,sp in enumerate(item['inner']):
                    va=np.array(sp['valid']);b=rec['inner_blocks'][j];assert b['inner']==j and b['n']==len(va) and b['rank16_valid']==valids[j]
                    for name,expected in (('metrics',metric(y[va],pred[va])),('reference',metric(y[va],reference[va]))):
                        for k,value in expected.items():close(b[name][k],value,1e-11)
            assert d['source_identities'][source]==read(folder/'complete.json')['source_identity']
    assert count==135 and covered==len(details)==20691;gates={}
    for seed in (42,43,44):
        core=[r for r in d['records'] if r['source']=='core_real' and r['map_seed']==seed]
        gates[str(seed)]=bool(sum(r['pool_qualified'] for r in core)>=2 and all(r['delta_PCC_vs_R4']>=-.005 for r in core))
    assert d['core_map_gates']==gates and d['all_rank16_valid']==allvalid and d['qualified']==bool(all(gates.values()) and allvalid)
    result=dict(passed=True,qualified=bool(all(gates.values()) and allvalid),core_map_gates=gates,all_rank16_valid=bool(allvalid),
        verified_heads=count,verified_covariance_whitenings=135,internal_OOF_rows=covered,max_prediction_difference=max_prediction,
        max_whitening_difference=max_whitening,max_normal_equation_residual=max_normal,rank_records=rank_records,
        new_source_forward=0,GPU_hours=0,outer_test_metrics_used=False,prediction_averaging=False,
        lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-started)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rank_records'}),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0),f,indent=2)
