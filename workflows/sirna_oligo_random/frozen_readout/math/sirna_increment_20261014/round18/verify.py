"""Independent equations, covariance spectrum and finite difference audit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4'
spec=importlib.util.spec_from_file_location('_r18_independent4',P/'verify.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close


def independent(x,y,theta,widths):
    z=x-x.mean(0);yc=y-y.mean();lam=np.exp(theta);diag=np.repeat(lam,widths);g=z.T@z
    k=g+np.diag(diag);b=np.linalg.solve(k,z.T@yc);inverse=np.linalg.inv(k)
    whitened=g/np.sqrt(np.outer(diag,diag));eigen=np.linalg.eigvalsh((whitened+whitened.T)/2)
    assert eigen.min()>-1e-6
    logdet=float(np.log1p(np.maximum(eigen,0)).sum())
    residual=yc-z@b;rss=float(residual@residual+(diag*b)@b);nu=len(y)-1
    objective=.5*(nu*np.log(rss/nu)+logdet);gradient=[];off=0
    for precision,d in zip(lam,widths):
        bb=b[off:off+d];trace=np.trace(inverse[off:off+d,off:off+d])
        gradient.append(.5*(nu*precision*(bb@bb)/rss+precision*trace-d));off+=d
    return dict(objective=objective,gradient=np.array(gradient),beta=b,rss=rss,logdet=logdet,noise_variance=rss/nu)


def metric(y,p):return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))


def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for path,h in lock['dependencies'].items():assert sha(Path(path))==h,path
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and d['new_efficiency_fits']==60
    assert d['new_PCA_fits']==d['new_source_forward']==d['GPU_hours']==0 and not d['outer_test_metrics_used']
    assert all(t['num_threads']==1 for t in d['threadpool']);check(R/'pilot',d['pilot_files_sha256'])
    assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    execution=[read(p) for p in R.glob('execution_*.json')];assert len(execution)==1 and execution[0]['returncode']==0
    plan=read(P/'plan.json');details=table(R/'diagnostic_rows.csv');count=0;covered=0;allgood=True;fdcount=0
    max_prediction=0.;max_gradient=0.;max_fd=0.;numerical=[]
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        ids=item['fit_ids'];assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        lookup={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')};rows=[lookup[s] for s in ids]
        y=np.array([float(r['label']) for r in rows]);td=np.array([[float(v) for v in r['td'].split(',')] for r in rows])
        for view in ('core_real','full_real','initial_core','TD24'):
            source='core_real' if view=='TD24' else view;folder=old.EXPORT/source;meta=table(folder/'row_identity.csv');idx={r['sample_id']:i for i,r in enumerate(meta)}
            for row in rows:assert all(row[k]==meta[idx[row['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
            raw=np.c_[td,np.load(folder/'source.npy',allow_pickle=False)[[idx[s] for s in ids]].astype(float)]
            pf=P/'results/folds'/setting/'fold0'/source/'transforms.json';projections=read(pf)
            with np.load(P/'results/folds'/setting/'fold0'/source/'arrays.npz',allow_pickle=False) as a:reference=a['inner_oof'][:,3 if view=='TD24' else 18].copy()
            widths=[24] if view=='TD24' else [24,16];pred=np.full(len(y),np.nan);coverage=[];blockaudit=[]
            for j,sp in enumerate(item['inner']):
                tr=np.array(sp['train']);va=np.array(sp['valid']);coverage+=va.tolist();out=R/'pilot'/setting/view/f'inner{j}'
                c=read(out/'complete.json');assert c['lock_sha256']==sha(R/'lock.json');check(out,c['files_sha256'])
                st=read(out/'state.json');s=st['model'];assert st['projection_sha256']==sha(pf) and st['train_indices']==tr.tolist() and st['valid_indices']==va.tolist()
                assert st['setting']==setting and st['view']==view and st['inner']==j and s['widths']==widths and s['nu']==len(tr)-1
                x=old.project(raw[tr],raw[tr],projections['inner'][j])[:,:sum(widths)]
                xv=old.project(raw[tr],raw[va],projections['inner'][j])[:,:sum(widths)]
                yt=y[tr];initial=np.load(out/'initial.npz',allow_pickle=False);theta0=np.full(len(widths),np.log(100.))
                close(initial['theta'],theta0,0);close(initial['mean'],x.mean(0),1e-12);close(initial['intercept'],yt.mean(),1e-12)
                initcheck=independent(x,yt,theta0,widths);close(initial['beta'],initcheck['beta'],1e-10);initial.close()
                assert (out/'initial.npz').stat().st_mtime_ns<=(out/'model.npz').stat().st_mtime_ns
                a=np.load(out/'model.npz',allow_pickle=False);theta=a['theta'];close(theta,s['theta'],0);close(np.exp(theta),s['lambdas'],1e-11)
                assert np.all(theta>=np.log(1e-6)-1e-12) and np.all(theta<=np.log(1e8)+1e-12)
                q=independent(x,yt,theta,widths);close(a['beta'],q['beta'],1e-9);close(a['mean'],x.mean(0),1e-12)
                close(s['intercept'],yt.mean(),1e-12)
                for key in ('rss','logdet','noise_variance','objective','gradient'):close(s[key],q[key],1e-7)
                residual=(x-x.mean(0))@a['beta']-(yt-yt.mean())
                normal=(x-x.mean(0)).T@residual+np.repeat(np.exp(theta),widths)*a['beta'];assert abs(normal).max()/len(tr)<1e-8
                pg=q['gradient'].copy()
                pg[(theta<=np.log(1e-6)+1e-10)&(pg>0)]=0.;pg[(theta>=np.log(1e8)-1e-10)&(pg<0)]=0.
                close(pg,s['projected_gradient'],1e-8);max_gradient=max(max_gradient,float(abs(pg).max()))
                good=bool(s['optimizer_success'] and abs(pg).max()<=1e-6);assert s['numerically_qualified']==good
                assert s['optimizer_success']==(s['status']==0);assert s['iterations']<=1000 and len(s['history'])==s['iterations']
                assert s['objective']<=initcheck['objective']+1e-7
                if s['history']:
                    close(s['history'][-1]['theta'],theta,1e-12);close(s['history'][-1]['objective'],s['objective'],1e-10)
                for k in range(len(widths)):
                    step=np.eye(len(widths))[k]*1e-4
                    fd=(independent(x,yt,theta+step,widths)['objective']-independent(x,yt,theta-step,widths)['objective'])/2e-4
                    error=abs(fd-q['gradient'][k]);assert error<2e-6;max_fd=max(max_fd,float(error));fdcount+=1
                pp=(xv-a['mean'])@a['beta']+s['intercept'];close(pp,a['prediction'],1e-10)
                max_prediction=max(max_prediction,float(abs(pp-a['prediction']).max()));pred[va]=a['prediction'];a.close()
                allgood &= good;count+=1
                numerical.append(dict(setting=setting,view=view,inner=j,lambdas=s['lambdas'],iterations=s['iterations'],
                    projected_gradient_inf=float(abs(pg).max()),optimizer_success=s['optimizer_success'],numerically_qualified=good))
                blockaudit.append(dict(lambdas=s['lambdas'],iterations=s['iterations'],numerically_qualified=good))
            assert sorted(coverage)==list(range(len(y)))
            part=[r for r in details if r['setting']==setting and r['view']==view];assert [r['sample_id'] for r in part]==ids
            close([float(r['label']) for r in part],y,0);close([float(r['reference_prediction']) for r in part],reference,0)
            close([float(r['prediction']) for r in part],pred,0);covered+=len(part)
            record=next(r for r in d['records'] if r['setting']==setting and r['view']==view);assert record['n']==len(y)
            m,ref=metric(y,pred),metric(y,reference)
            for name,expected in (('metrics',m),('reference',ref)):
                for key,value in expected.items():close(record[name][key],value,1e-11)
            delta=m['PCC']-ref['PCC'];reduction=100*(1-m['MSE']/ref['MSE'])
            close(record['delta_PCC_vs_R4'],delta,1e-11);close(record['MSE_reduction_vs_R4_pct'],reduction,1e-10)
            assert record['pool_qualified']==(delta>=.005 and reduction>=0)
            for j,sp in enumerate(item['inner']):
                va=np.array(sp['valid']);b=record['inner_blocks'][j];assert b['inner']==j and b['n']==len(va)
                for k,v in blockaudit[j].items():assert b[k]==v
                for name,expected in (('metrics',metric(y[va],pred[va])),('reference',metric(y[va],reference[va]))):
                    for k,v in expected.items():close(b[name][k],v,1e-11)
            assert d['source_identities'][source]==read(folder/'complete.json')['source_identity']
    assert count==60 and covered==len(details)==9196 and fdcount==105
    core=[r for r in d['records'] if r['view']=='core_real']
    gate=sum(r['pool_qualified'] for r in core)>=2 and all(r['delta_PCC_vs_R4']>=-.005 for r in core)
    assert d['performance_gate']==gate and d['all_numerically_qualified']==allgood and d['qualified']==bool(gate and allgood)
    result=dict(passed=True,qualified=bool(gate and allgood),all_numerically_qualified=bool(allgood),verified_heads=count,
        internal_OOF_rows=covered,gradient_difference_checks=fdcount,max_finite_difference_error=max_fd,
        max_prediction_difference=max_prediction,max_projected_gradient=max_gradient,numerical_records=numerical,
        new_source_forward=0,new_PCA_fits=0,GPU_hours=0,outer_test_metrics_used=False,
        lock_sha256=sha(R/'lock.json'),diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-started)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='numerical_records'}),flush=True)


if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0),f,indent=2)
