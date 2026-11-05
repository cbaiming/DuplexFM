"""Independent full-design normal equations and feasible dual certificates."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=R.parent/'round4'
spec=importlib.util.spec_from_file_location('_r17_audit4',P/'verify.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close

def products(z,td):
    left=z[:,:24];right=left if td else z[:,24:]
    return np.column_stack([left[:,i]*right[:,j] for i in range(24) for j in range(right.shape[1])]),(24,right.shape[1])

def metric(y,p):return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))

def main():
    start=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for path,h in lock['dependencies'].items():assert sha(Path(path))==h,path
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and d['models']==d['new_efficiency_fits']==120
    assert d['new_PCA_fits']==d['new_source_forward']==d['GPU_hours']==0 and not d['outer_test_metrics_used']
    assert all(t['num_threads']==1 for t in d['threadpool']);check(R/'pilot',d['pilot_files_sha256'])
    assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256'];ex=[read(p) for p in R.glob('execution_*.json')];assert len(ex)==1 and ex[0]['returncode']==0
    plan=read(P/'plan.json');details=table(R/'diagnostic_rows.csv');count=0;covered=0;max_pred=0.;max_gap=0.;allgood=True;ranks=[]
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids'];assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        lookup={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')};rows=[lookup[s] for s in ids]
        y=np.array([float(r['label']) for r in rows]);td=np.array([[float(v) for v in r['td'].split(',')] for r in rows])
        for view in ('core_real','full_real','initial_core','TD24'):
            source='core_real' if view=='TD24' else view;folder=old.EXPORT/source;meta=table(folder/'row_identity.csv');idx={r['sample_id']:i for i,r in enumerate(meta)}
            for row in rows:assert all(row[k]==meta[idx[row['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
            rawx=np.c_[td,np.load(folder/'source.npy',allow_pickle=False)[[idx[s] for s in ids]].astype(float)]
            pf=P/'results/folds'/setting/'fold0'/source/'transforms.json';projections=read(pf)
            with np.load(P/'results/folds'/setting/'fold0'/source/'arrays.npz',allow_pickle=False) as a:reference=a['inner_oof'][:,3 if view=='TD24' else 18].copy()
            predicted={m:np.full(len(y),np.nan) for m in ('Ridge','Nuclear')};coverage=[];blocks=[]
            for j,sp in enumerate(item['inner']):
                tr=np.array(sp['train']);va=np.array(sp['valid']);coverage+=va.tolist();out=R/'pilot'/setting/view/f'inner{j}'
                complete=read(out/'complete.json');assert complete['lock_sha256']==sha(R/'lock.json');check(out,complete['files_sha256'])
                st=read(out/'state.json');assert st['train_indices']==tr.tolist() and st['valid_indices']==va.tolist() and st['projection_sha256']==sha(pf)
                assert st['setting']==setting and st['view']==view and st['inner']==j
                z=old.project(rawx[tr],rawx[tr],projections['inner'][j]);zv=old.project(rawx[tr],rawx[va],projections['inner'][j])
                if view=='TD24':z=z[:,:24];zv=zv[:,:24]
                product,shape=products(z,view=='TD24');other,_=products(zv,view=='TD24');mean=product.mean(0);scale=np.sqrt(np.sum(product.var(0))/16);scale=scale if scale>=1e-12 else 1.
                t=st['map'];assert t['matrix_shape']==list(shape) and t['kind']==('TD24' if view=='TD24' else 'source')
                close(t['main_mean'],z.mean(0),1e-12);close(t['product_mean'],mean,1e-12);close(t['product_scale'],scale,1e-12)
                x=z-z.mean(0);xv=zv-z.mean(0);jx=(product-mean)/scale;jv=(other-mean)/scale;yc=y[tr]-y[tr].mean()
                a=x.T@x+100*np.eye(x.shape[1]);cross=x.T@jx;rhs=x.T@yc;Q=jx.T@jx+100*np.eye(jx.shape[1])-cross.T@np.linalg.solve(a,cross);Q=(Q+Q.T)/2
                q=jx.T@yc-cross.T@np.linalg.solve(a,rhs);e=np.linalg.eigvalsh(Q);assert e[0]>=100-1e-7
                critical=float(np.linalg.norm(q.reshape(shape),2));gapscale=max(1.,.5*float(q@np.linalg.solve(Q,q)))
                initial=np.load(out/'initial.npz',allow_pickle=False);close(initial['b'],np.linalg.solve(a,rhs),1e-11);close(initial['c'],np.zeros(jx.shape[1]),0);close(initial['intercept'],y[tr].mean(),1e-12);initial.close()
                assert (out/'initial.npz').stat().st_mtime_ns <= (out/'models.npz').stat().st_mtime_ns
                models=np.load(out/'models.npz',allow_pickle=False);full=np.c_[x,jx];ridge=np.linalg.solve(full.T@full+100*np.eye(full.shape[1]),full.T@yc);bm={}
                for mode in ('Ridge','Nuclear'):
                    b,c,dual=(models[mode+'_'+k] for k in ('b','c','dual'));info=st['models'][mode];lam=0. if mode=='Ridge' else .25*critical
                    close(info['intercept'],y[tr].mean(),1e-12);close(info['lambda_value'],lam,1e-11);close(info['lambda_critical'],critical,1e-11)
                    residual=x@b+jx@c-yc;close(x.T@residual+100*b,np.zeros_like(b),1e-8)
                    if mode=='Ridge':close(np.r_[b,c],ridge,1e-10);assert info['iterations']==0
                    singular=np.linalg.svd(c.reshape(shape),compute_uv=False);rank=int(np.sum(singular>1e-8));assert info['rank']==rank;close(info['singular_values'],singular,1e-10)
                    if view=='TD24':close(c.reshape(shape),c.reshape(shape).T,1e-9)
                    expected=q-Q@c;norm=np.linalg.norm(expected.reshape(shape),2)
                    if norm>lam:expected*=lam/norm
                    close(dual,expected,1e-9);assert np.linalg.norm(dual.reshape(shape),2)<=lam+1e-8
                    primal=.5*c@Q@c-q@c+lam*sum(singular);dualvalue=-.5*(q-dual)@np.linalg.solve(Q,q-dual);gap=primal-dualvalue
                    assert gap>=-1e-8;gap=max(0.,float(gap));close(info['primal_profile'],primal,1e-9);close(info['dual_profile'],dualvalue,1e-9);close(info['gap'],gap,1e-9);close(info['gap_scale'],gapscale,1e-9)
                    good=bool(gap<=1e-9*gapscale);assert info['certified']==good;allgood &= good;max_gap=max(max_gap,gap/gapscale)
                    if mode=='Nuclear' and lam>0:
                        close(info['L'],e[-1],1e-8);assert 1<=info['iterations']<=5000
                        history=np.asarray(info['history']);assert history[-1,0]==info['iterations'];close(history[-1,1],info['gap'],1e-12)
                        ranks.append(dict(setting=setting,view=view,inner=j,rank=rank,iterations=info['iterations']))
                    pp=xv@b+jv@c+info['intercept'];close(pp,models[mode+'_prediction'],1e-10);max_pred=max(max_pred,float(abs(pp-models[mode+'_prediction']).max()));predicted[mode][va]=models[mode+'_prediction'];count+=1
                    bm[mode]=dict(rank=rank,iterations=info['iterations'],certified=good)
                models.close();blocks.append(bm)
            assert sorted(coverage)==list(range(len(y)))
            part=[r for r in details if r['setting']==setting and r['view']==view];assert [r['sample_id'] for r in part]==ids
            close([float(r['label']) for r in part],y,0);close([float(r['reference_prediction']) for r in part],reference,0);covered+=len(part)
            record=next(r for r in d['records'] if r['setting']==setting and r['view']==view);assert record['n']==len(y)
            metrics={m:metric(y,predicted[m]) for m in predicted};ref=metric(y,reference)
            for m in predicted:
                close([float(r[m.lower()+'_prediction']) for r in part],predicted[m],0)
                for key,val in metrics[m].items():close(record['metrics'][m][key],val,1e-11)
            for key,val in ref.items():close(record['reference'][key],val,1e-11)
            delta=metrics['Nuclear']['PCC']-ref['PCC'];reduction=100*(1-metrics['Nuclear']['MSE']/ref['MSE'])
            close(record['delta_PCC_vs_R4'],delta,1e-11);close(record['MSE_reduction_vs_R4_pct'],reduction,1e-10);assert record['pool_qualified']==(delta>=.005 and reduction>=0)
            close(record['delta_PCC_vs_paired'],metrics['Nuclear']['PCC']-metrics['Ridge']['PCC'],1e-11);close(record['MSE_reduction_vs_paired_pct'],100*(1-metrics['Nuclear']['MSE']/metrics['Ridge']['MSE']),1e-10)
            for j,sp in enumerate(item['inner']):
                va=np.array(sp['valid']); block=record['inner_blocks'][j];assert block['inner']==j and block['n']==len(va)
                for mode in predicted:
                    for key,val in blocks[j][mode].items():assert block[mode][key]==val
                    for key,val in metric(y[va],predicted[mode][va]).items():close(block[mode]['metrics'][key],val,1e-11)
                for key,val in metric(y[va],reference[va]).items():close(block['reference'][key],val,1e-11)
            assert d['source_identities'][source]==read(folder/'complete.json')['source_identity']
    assert count==120 and covered==len(details)==4*sum(len(v) for v in d['fit_ids'].values()) and len(ranks)==60
    core=[r for r in d['records'] if r['view']=='core_real'];g1=sum(r['pool_qualified'] for r in core)>=2 and all(r['delta_PCC_vs_R4']>=-.005 for r in core)
    g2=sum(r['MSE_reduction_vs_paired_pct']>=0 for r in core)>=2 and np.mean([r['MSE_reduction_vs_paired_pct'] for r in core])>=0
    assert d['R4_gate']==g1 and d['paired_gate']==g2 and d['all_certified']==allgood and d['qualified']==(g1 and g2 and allgood)
    result=dict(passed=True,qualified=bool(g1 and g2 and allgood),all_certified=bool(allgood),verified_models=count,
        independent_direct_Ridge_solves=60,independent_convex_certificates=60,internal_OOF_rows=covered,prediction_values=2*covered,
        max_prediction_difference=max_pred,max_relative_dual_gap=max_gap,rank_records=ranks,
        new_source_forward=0,GPU_hours=0,outer_test_metrics_used=False,lock_sha256=sha(R/'lock.json'),
        diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-start)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rank_records'}),flush=True)

if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0),f,indent=2)
