"""Independent PCA, profiled equations, gate derivatives and deployment audit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
from scipy.special import expit
R=Path(__file__).resolve().parent;P=R.parent/'round4'
spec=importlib.util.spec_from_file_location('_r31_audit4',P/'verify.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close


def reconstruct(a,z,y,augmented=False):
    p=z.shape[1];w=z if p==24 else z[:,24:];q=w.shape[1]
    t=np.c_[z[:,:24],np.ones(len(z))];h=2*expit(2*(t@a))-1;hm=float(h.mean())
    j=(h-hm)[:,None]*w;jm=j.mean(0);x=np.c_[z,j-jm]
    penalty=np.r_[np.full(p,100.),np.full(q,400.)];yc=y-y.mean();scale=float(yc@yc)
    gram=x.T@x+np.diag(penalty);rhs=x.T@yc
    beta=(np.linalg.lstsq(np.r_[x,np.diag(np.sqrt(penalty))],np.r_[yc,np.zeros(p+q)],rcond=None)[0]
          if augmented else np.linalg.solve(gram,rhs))
    residual=x@beta-yc;value=float(.5*(residual@residual+penalty@(beta*beta))/scale+.005*(a@a))
    dh=(1-h*h)[:,None]*t;dh-=dh.mean(0)
    jacobian=dh*(w@beta[p:])[:,None];jacobian-=jacobian.mean(0)
    gradient=jacobian.T@residual/scale+.01*a
    return dict(x=x,beta=beta,h=h,h_mean=hm,j_mean=jm,penalty=penalty,gram=gram,rhs=rhs,
        loss_scale=scale,gamma=.01*scale,objective=value,gradient=gradient,train_MSE=float(np.mean(residual**2)))


def prediction(z,a,beta,center,hm,jm):
    p=z.shape[1];w=z if p==24 else z[:,24:];g=expit(2*(z[:,:24]@a[:24]+a[24]))
    base=z@beta[:p]+center-jm@beta[p:]
    left=base+(-1-hm)*(w@beta[p:]);right=base+(1-hm)*(w@beta[p:])
    return (1-g)*left+g*right,g


def metric(y,p):return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))
def gate_stats(g):return dict(mean=float(g.mean()),std=float(g.std()),minimum=float(g.min()),maximum=float(g.max()))
def compare(a,b,tol=1e-10):
    for k,v in b.items():close(a[k],v,tol)


def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for path,h in lock['dependencies'].items():assert sha(Path(path))==h,path
    assert lock['CPU_only'] and lock['source_seed']==lock['head_seed']==42
    assert lock['shared_alpha']==100 and lock['local_multiplier']==4 and lock['gate_fraction']==.01
    assert lock['maxiter']==500 and lock['maxfun']==10000 and lock['maxls']==50
    assert lock['ftol']==1e-12 and lock['gtol']==1e-8 and lock['audit_gradient_limit']==1e-6
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'lock.json') and not d['outer_test_metrics_used']
    assert d['new_source_forward']==d['new_PCA']==d['GPU_hours']==0 and all(t['num_threads']==1 for t in d['threadpool'])
    check(R/'pilot',d['pilot_files_sha256']);assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    exe=[read(p) for p in R.glob('execution_*.json')];assert len(exe)==1 and exe[0]['returncode']==0
    plan=read(P/'plan.json');details=table(R/'diagnostic_rows.csv');allvalid=True;count=covered=solves=audit_solves=0
    maxpred=maxfd=maxnormal=maxgradient=0.;records=[];blocks_all=[]
    rng=np.random.default_rng(42);initial=rng.normal(size=24);initial=np.r_[initial/np.linalg.norm(initial),0.]
    directions=np.random.default_rng(3100).normal(size=(3,25));directions/=np.linalg.norm(directions,axis=1)[:,None]
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids']
        assert ids==d['fit_ids'][setting] and set(ids).isdisjoint(item['test_ids'])
        rows={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')};y=np.array([float(rows[s]['label']) for s in ids])
        td=np.array([[float(v) for v in rows[s]['td'].split(',')] for s in ids])
        for source in (*old.SOURCES,'TD_only'):
            src='core_real' if source=='TD_only' else source;export=old.EXPORT/src
            identity=read(export/'complete.json');assert d['source_identities'][src]==identity['source_identity']
            meta=table(export/'row_identity.csv');lookup={r['sample_id']:i for i,r in enumerate(meta)}
            for sample in ids:assert all(rows[sample][k]==meta[lookup[sample]][k] for k in ('sample_id','cohort','guide','target57'))
            raw=np.c_[td,np.load(export/'source.npy',allow_pickle=False)[[lookup[s] for s in ids]].astype(float)]
            pf=P/'results/folds'/setting/'fold0'/src/'transforms.json';af=pf.parent/'arrays.npz';states=read(pf)
            with np.load(af,allow_pickle=False) as a:
                k=3 if source=='TD_only' else 18;reference=a['inner_oof'][:,k].copy();refbeta=a['inner_coef'][:,k].copy()
            pred=np.full(len(ids),np.nan);probs=np.full(len(ids),np.nan);coverage=[];blocks=[]
            for j,split in enumerate(item['inner']):
                tr,va=np.array(split['train']),np.array(split['valid']);out=R/'pilot'/setting/source/f'inner{j}'
                receipt=read(out/'complete.json');check(out,receipt['files_sha256']);assert receipt['lock_sha256']==sha(R/'lock.json')
                s=read(out/'state.json');cert=s['certificate']
                assert (s['setting'],s['source'],s['inner'])==(setting,source,j)
                assert s['train_indices']==tr.tolist() and s['valid_indices']==va.tolist()
                assert s['reference_transform_sha256']==sha(pf) and s['reference_arrays_sha256']==sha(af)
                z=old.project(raw[tr],raw[tr],states['inner'][j]);zv=old.project(raw[tr],raw[va],states['inner'][j])
                if source=='TD_only':z=z[:,:24];zv=zv[:,:24]
                p=z.shape[1];q=24 if p==24 else 16
                b0=refbeta[j,:p];center=float(y[tr].mean())
                close((z.T@z+100*np.eye(p))@b0,z.T@(y[tr]-center),1e-8)
                close(zv@b0+center,reference[va],1e-10)
                with np.load(out/'initial.npz',allow_pickle=False) as a:close(a['gate'],initial,0)
                with np.load(out/'model.npz',allow_pickle=False) as a:
                    gate=a['gate'].copy();beta=a['beta'].copy();c=float(a['intercept']);hm=float(a['h_mean']);jm=a['j_mean'].copy()
                    pp=a['prediction'].copy();gg=a['gate_probability'].copy();accepted=a['accepted_gates'].copy()
                assert gate.shape==(25,) and beta.shape==(p+q,) and accepted.shape==(cert['iterations'],25)
                close(gate,accepted[-1] if len(accepted) else initial,0);close(c,center,1e-12)
                actual=reconstruct(gate,z,y[tr],augmented=True);init=reconstruct(initial,z,y[tr]);audit_solves+=2
                close(beta,actual['beta'],1e-10);close(hm,actual['h_mean'],1e-11);close(jm,actual['j_mean'],1e-11)
                for key in ('loss_scale','gamma','objective','train_MSE'):close(cert[key],actual[key],1e-11)
                close(cert['initial_objective'],init['objective'],1e-11)
                assert cert['objective']<=cert['initial_objective']+1e-10
                close(cert['gradient'],actual['gradient'],1e-10);gradmax=float(abs(actual['gradient']).max());close(cert['gradient_max'],gradmax,1e-10)
                maxgradient=max(maxgradient,gradmax);normal=float(abs(actual['gram']@beta-actual['rhs']).max())
                limit=float(1e-9*(1+abs(actual['rhs']).max()))
                close(cert['normal_error'],normal,1e-10);close(cert['normal_limit'],limit,1e-12);maxnormal=max(maxnormal,normal)
                assert cert['shared_alpha']==100 and cert['local_multiplier']==4 and cert['gate_fraction']==.01
                valid=bool(cert['success'] and gradmax<=1e-6 and normal<=limit)
                assert valid==cert['numerically_qualified'];allvalid &= valid
                assert 0<=cert['iterations']<=500 and cert['new_linear_solves']==cert['function_evaluations']+2
                solves+=cert['new_linear_solves']
                for direction in directions:
                    step=1e-5*direction
                    fd=(reconstruct(gate+step,z,y[tr])['objective']-reconstruct(gate-step,z,y[tr])['objective'])/2e-5
                    err=abs(fd-direction@actual['gradient']);assert err<1e-7;maxfd=max(maxfd,float(err));audit_solves+=2
                reconstruction,g=prediction(zv,gate,beta,center,hm,jm)
                close(reconstruction,pp,1e-10);close(g,gg,1e-12);maxpred=max(maxpred,float(abs(pp-reconstruction).max()))
                train_gate=(1+actual['h'])/2;compare(s['train_gate'],gate_stats(train_gate));compare(s['valid_gate'],gate_stats(g))
                pred[va]=pp;probs[va]=gg;coverage+=va.tolist();count+=1
                blocks.append(dict(inner=j,n=len(va),metrics=metric(y[va],pp),reference=metric(y[va],reference[va]),
                    train_gate=gate_stats(train_gate),valid_gate=gate_stats(g),iterations=cert['iterations'],new_linear_solves=cert['new_linear_solves'],
                    gradient_max=gradmax,numerically_qualified=valid))
                blocks_all.append(dict(setting=setting,source=source,inner=j,gate_norm=float(np.linalg.norm(gate)),
                    local_coef_norm=float(np.linalg.norm(beta[p:])),normal_error=normal,gradient_max=gradmax,
                    iterations=cert['iterations'],success=cert['success'],numerically_qualified=valid))
            assert sorted(coverage)==list(range(len(ids))) and np.isfinite(pred).all()
            part=[r for r in details if r['setting']==setting and r['source']==source];assert [r['sample_id'] for r in part]==ids
            for key,values in [('label',y),('prediction',pred),('reference_prediction',reference),('gate_probability',probs)]:
                close([float(r[key]) for r in part],values,0)
            covered+=len(part);m,rr=metric(y,pred),metric(y,reference)
            rec=next(r for r in d['records'] if r['setting']==setting and r['source']==source);assert rec['n']==len(ids)
            compare(rec['metrics'],m);compare(rec['reference'],rr)
            delta=m['PCC']-rr['PCC'];red=100*(1-m['MSE']/rr['MSE'])
            close(rec['delta_PCC_vs_R4'],delta,1e-11);close(rec['MSE_reduction_vs_R4_pct'],red,1e-10)
            for actual,expected in zip(rec['inner_blocks'],blocks):
                for key,value in expected.items():
                    if isinstance(value,dict):compare(actual[key],value)
                    else:close(actual[key],value,1e-10)
            records.append(dict(setting=setting,source=source,delta=delta,reduction=red,metrics=m))
    assert count==60==d['new_heads']==d['reused_reference_heads'] and len(details)==covered==9196==d['new_internal_OOF_rows']
    assert solves==d['new_linear_solves'] and audit_solves==480 and len(d['records'])==12
    core=[r for r in records if r['source']=='core_real']
    performance=sum(r['delta']>=.005 and r['reduction']>=0 for r in core)>=2 and all(r['delta']>=-.005 for r in core)
    pairs=[]
    for r in core:
        td=next(t for t in records if t['source']=='TD_only' and t['setting']==r['setting'])
        delta=r['metrics']['PCC']-td['metrics']['PCC'];red=100*(1-r['metrics']['MSE']/td['metrics']['MSE'])
        pair=next(t for t in d['paired_gated_TD'] if t['setting']==r['setting'])
        close(pair['delta_PCC'],delta,1e-11);close(pair['MSE_reduction_pct'],red,1e-10);pairs.append((delta,red))
    incremental=sum(dp>=0 for dp,red in pairs)>=2 and np.mean([red for dp,red in pairs])>=0
    qualified=bool(performance and incremental and allvalid)
    assert d['performance_gate']==performance and d['source_increment_gate']==incremental
    assert d['qualified']==qualified and d['all_numerically_qualified']==allvalid
    result=dict(passed=True,qualified=qualified,verified_new_heads=60,verified_reference_heads=60,new_internal_OOF_rows=9196,
        new_scientific_linear_solves=solves,independent_final_augmented_lstsq=60,independent_audit_linear_solves=480,
        finite_difference_directions=180,max_finite_difference_error=maxfd,max_gate_gradient=maxgradient,
        max_normal_equation_error=maxnormal,max_prediction_difference=maxpred,all_numerically_qualified=bool(allvalid),
        block_records=blocks_all,new_source_forward=0,new_PCA=0,GPU_hours=0,outer_test_metrics_used=False,
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
