"""Independent natural CubicSpline reconstruction of the internal-only pilot."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
R=Path(__file__).resolve().parent;P=R.parent/'round4';B=R.parents[1]
spec=importlib.util.spec_from_file_location('_r13_audit4',P/'verify.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close

def quantiles(z):
    ordered=np.sort(z,axis=0);pos=(len(z)-1)*np.array([.1,.5,.9]);lo=np.floor(pos).astype(int);hi=np.ceil(pos).astype(int)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo)[:,None]

def cubic(z,knots,active):
    out=np.zeros_like(z)
    for j in np.flatnonzero(active):
        k=knots[:,j];a=(k[1]-k[0])/(k[2]-k[0]);u=(z[:,j]-k[0])/(k[2]-k[0])
        spline=CubicSpline([0,a,1],[0,a**3,2*a-a*a],bc_type='natural')
        h=spline(np.clip(u,0,1));h[u<0]=spline(0)+spline(0,1)*u[u<0]
        h[u>1]=spline(1)+spline(1,1)*(u[u>1]-1);out[:,j]=h
    return out

def expand(train,other,state):
    p=state['projection'];a=old.project(train,train,p);b=old.project(train,other,p)
    t=a[:,:24] if state['view']=='TD24' else a[:,24:];v=b[:,:24] if state['view']=='TD24' else b[:,24:]
    knots=quantiles(t);mean=t.mean(0);center=t-mean;variance=(center*center).mean(0)
    active=(np.diff(knots,axis=0).min(0)>1e-8)&(variance>1e-12)
    h=cubic(t,knots,active);hm=h.mean(0);sl=np.divide((center*(h-hm)).mean(0),variance,out=np.zeros(t.shape[1]),where=variance>1e-12)
    sd=(h-hm-center*sl).std(0);active&=(sd>1e-10);scale=np.where(active,sd,1.)
    expected=dict(knots=knots,mean=mean,h_mean=hm,slope=sl,scale=scale,active=active)
    for k in expected:close(expected[k],state['curve'][k],1e-9)
    q=(cubic(v,knots,active)-hm-(v-mean)*sl)/scale;q[:,~active]=0
    assert not state['new_PCA_fit']
    return np.c_[b[:,:24] if state['view']=='TD24' else b,q]

def stats(y,p):
    return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))

def main():
    start=time.monotonic();assert not (R/'diagnostic.mutex').exists() and not (R/'diagnostic_verification.json').exists()
    lock=read(R/'diagnostic_lock.json');check(R,lock['files'])
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    d=read(R/'diagnostic.json');assert d['lock_sha256']==sha(R/'diagnostic_lock.json')
    assert d['new_efficiency_fits']==60 and not d['outer_test_metrics_used'] and d['GPU_hours']==d['new_source_forward']==d['new_PCA_fits']==0
    check(R/'pilot',d['pilot_files_sha256']);assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    execution=[read(p) for p in R.glob('diagnostic_execution_*.json')];assert len(execution)==1 and execution[0]['returncode']==0
    plan=read(P/'plan.json');details=table(R/'diagnostic_rows.csv');count=0;maxerr=0.;maxpred=0.;covered=0
    for setting in ('Huesken','Mixset','Takayuki'):
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0);ids=item['fit_ids'];assert ids==d['fit_ids'][setting]
        allrows=table(B/'core_six_settings_20261011/benchmark/data'/f'{setting}.csv');lookup={r['sample_id']:r for r in allrows}
        rows=[lookup[i] for i in ids];y=np.array([float(r['label']) for r in rows]);td=np.array([[float(a) for a in r['td'].split(',')] for r in rows])
        for view in ('core_real','full_real','initial_core','TD24'):
            source='core_real' if view=='TD24' else view;folder=B/'core_six_settings_20261011/results/exports'/source
            meta=table(folder/'row_identity.csv');index={r['sample_id']:i for i,r in enumerate(meta)}
            for r in rows:assert all(r[k]==meta[index[r['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
            x=np.c_[td,np.load(folder/'source.npy',allow_pickle=False)[[index[i] for i in ids]].astype(float)]
            path=R/'pilot'/setting/view;states=read(path/'states.json');a=np.load(path/'arrays.npz',allow_pickle=False)
            c=read(path/'complete.json');assert c['lock_sha256']==sha(R/'diagnostic_lock.json') and c['new_fits']==5;check(path,c['files_sha256'])
            reference=read(P/'results/folds'/setting/'fold0'/source/'transforms.json')
            with np.load(P/'results/folds'/setting/'fold0'/source/'arrays.npz',allow_pickle=False) as prior:
                ref=prior['inner_oof'][:,3 if view=='TD24' else 18].copy()
            close(ref,a['reference_oof'],0);oof=np.full(len(y),np.nan);coverage=[]
            for j,split in enumerate(item['inner']):
                tr=np.array(split['train']);va=np.array(split['valid']);coverage+=va.tolist()
                assert not set(tr)&set(va)
                for key,value in reference['inner'][j].items():close(states[j]['projection'][key],value,0)
                assert states[j]['view']==view
                z=expand(x[tr],x[tr],states[j]);zv=expand(x[tr],x[va],states[j])
                b=a['inner_coef'][j];intercept=a['inner_intercept'][j];close(z.mean(0),0,1e-10);close(intercept,y[tr].mean(),1e-12)
                residual=(z.T@(z@b+intercept-y[tr])+100*b)/len(tr);err=float(abs(residual).max());assert err<1e-8;maxerr=max(maxerr,err)
                oof[va]=zv@b+intercept;count+=1
            assert sorted(coverage)==list(range(len(y)));close(oof,a['inner_oof'],1e-9);maxpred=max(maxpred,float(abs(oof-a['inner_oof']).max()))
            part=[r for r in details if r['setting']==setting and r['view']==view];assert [r['sample_id'] for r in part]==ids
            p=np.array([float(r['new_prediction']) for r in part]);close(p,oof,1e-9);close([float(r['label']) for r in part],y,0);close([float(r['reference_prediction']) for r in part],ref,0);covered+=len(part)
            row=next(r for r in d['records'] if r['setting']==setting and r['view']==view);m,pm=stats(y,p),stats(y,ref)
            assert row['n']==len(y) and row['feature_width']==(48 if view=='TD24' else 56)
            for key in m:close(row['new'][key],m[key],1e-12);close(row['old'][key],pm[key],1e-12)
            delta=m['PCC']-pm['PCC'];reduction=100*(1-m['MSE']/pm['MSE']);close(row['delta_PCC'],delta,1e-12);close(row['MSE_relative_reduction_pct'],reduction,1e-12)
            assert row['pool_qualified']==(delta>=.005 and reduction>=0)
            for j,split in enumerate(item['inner']):
                va=np.array(split['valid']);block=row['inner_blocks'][j];assert block['inner']==j and block['n']==len(va)
                for typ,prediction in (('new',p),('old',ref)):
                    met=stats(y[va],prediction[va])
                    for key in met:close(block[typ][key],met[key],1e-12)
            a.close()
    assert count==60 and covered==len(details)==4*sum(len(v) for v in d['fit_ids'].values()) and len(d['records'])==12
    core=[r for r in d['records'] if r['view']=='core_real'];qualified=sum(r['pool_qualified'] for r in core)>=2 and all(r['delta_PCC']>=-.005 for r in core)
    assert qualified==d['qualified']
    result=dict(passed=True,qualified=qualified,independent_normal_equations=count,verified_internal_OOF_rows=covered,
        max_normal_residual=maxerr,max_prediction_reconstruction_difference=maxpred,
        independent_CubicSpline_reconstruction=True,new_efficiency_fits=0,new_source_forward=0,GPU_hours=0,
        diagnostic_sha256=sha(R/'diagnostic.json'),lock_sha256=sha(R/'diagnostic_lock.json'),
        verifier_sha256=sha(Path(__file__)),CPU_wall_seconds=time.monotonic()-start)
    with (R/'diagnostic_verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    start=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0),f,indent=2)
