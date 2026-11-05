"""Independent augmented-matrix neural forward and gradient audit."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,traceback
from pathlib import Path
import numpy as np
from scipy.special import expit
R=Path(__file__).resolve().parent; P=R.parent/'round4'
spec=importlib.util.spec_from_file_location('_r16_audit4',P/'verify.py'); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
read,sha,table,check,close=old.read,old.sha,old.table,old.check,old.close


def evaluate(z,theta,y=None):
    d=z.shape[1]; h=16; cut=d*h; w=theta[:cut].reshape(d,h); b=theta[cut:cut+h]; a=theta[cut+h:cut+2*h]; c=theta[-1]
    augmented=np.column_stack([z,np.ones(len(z))]); matrix=np.vstack([w,b])
    logistic=expit(2*augmented@matrix); hidden=2*logistic-1
    pred=np.column_stack([hidden,np.ones(len(z))])@np.r_[a,c]
    if y is None: return pred
    residual=pred-y; n=len(y); value=(residual@residual+10*(np.sum(w*w)+a@a))/(2*n)
    hidden_error=residual[:,None]*a*(4*logistic*(1-logistic))/n
    matrix_grad=augmented.T@hidden_error; matrix_grad[:d]+=10*w/n
    output_grad=np.column_stack([hidden,np.ones(len(z))]).T@residual/n; output_grad[:h]+=10*a/n
    gradient=np.r_[matrix_grad[:d].ravel(),matrix_grad[d],output_grad]
    return float(value),gradient,pred


def metric(y,p): return dict(PCC=float(np.corrcoef(y,p)[0,1]),MSE=float(np.mean((y-p)**2)))


def main():
    start=time.monotonic(); assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json'); check(R,lock['files'])
    for p,h in lock['dependencies'].items(): assert sha(Path(p))==h,p
    d=read(R/'diagnostic.json'); assert d['lock_sha256']==sha(R/'lock.json') and d['new_efficiency_fits']==60
    assert d['new_PCA_fits']==d['new_source_forward']==d['GPU_hours']==0 and not d['outer_test_metrics_used']
    assert all(t['num_threads']==1 for t in d['threadpool'])
    check(R/'pilot',d['pilot_files_sha256']); assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    executions=[read(p) for p in R.glob('execution_*.json')]; assert len(executions)==1 and executions[0]['returncode']==0
    plan=read(P/'plan.json'); details=table(R/'diagnostic_rows.csv'); counts=0; covered=0; maxdiff=0.; maxfinite=0.; all_good=True
    for setting in old.S[:3]:
        item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0); ids=item['fit_ids']; assert ids==d['fit_ids'][setting]
        assert set(ids).isdisjoint(item['test_ids'])
        lookup={r['sample_id']:r for r in table(old.BENCH/'data'/f'{setting}.csv')}; rows=[lookup[i] for i in ids]
        y=np.array([float(r['label']) for r in rows]); td=np.array([[float(v) for v in r['td'].split(',')] for r in rows])
        for view in ('core_real','full_real','initial_core','TD24'):
            source='core_real' if view=='TD24' else view; folder=old.EXPORT/source
            meta=table(folder/'row_identity.csv'); index={r['sample_id']:i for i,r in enumerate(meta)}
            for row in rows: assert all(row[k]==meta[index[row['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
            src=np.load(folder/'source.npy',allow_pickle=False)[[index[i] for i in ids]].astype(float); x=np.c_[td,src]
            projection_file=P/'results/folds'/setting/'fold0'/source/'transforms.json'; transforms=read(projection_file)
            with np.load(P/'results/folds'/setting/'fold0'/source/'arrays.npz',allow_pickle=False) as reference:
                previous=reference['inner_oof'][:,3 if view=='TD24' else 18].copy()
            oof=np.full(len(y),np.nan); coverage=[]; view_good=True; states=[]
            for j,split in enumerate(item['inner']):
                tr=np.array(split['train']); va=np.array(split['valid']); coverage+=va.tolist(); out=R/'pilot'/setting/view/f'inner{j}'
                complete=read(out/'complete.json'); assert complete['lock_sha256']==sha(R/'lock.json'); check(out,complete['files_sha256'])
                state=read(out/'state.json'); states.append(state); initial=np.load(out/'initial.npz',allow_pickle=False); a=np.load(out/'model.npz',allow_pickle=False)
                t=transforms['inner'][j]; z=old.project(x[tr],x[tr],t); zv=old.project(x[tr],x[va],t)
                if view=='TD24': z=z[:,:24]; zv=zv[:,:24]
                dimension=z.shape[1]; assert state['d']==dimension and state['head_seed']==42 and state['view']==view and state['inner']==j
                assert state['source']==source and state['setting']==setting and state['parameter_count']==dimension*16+33
                assert state['train_indices']==tr.tolist() and state['valid_indices']==va.tolist()
                assert state['n_train']==len(tr) and state['n_valid']==len(va) and state['projection_sha256']==sha(projection_file)
                rng=np.random.default_rng(42); w=rng.uniform(-np.sqrt(6/56),np.sqrt(6/56),(40,16)); outw=rng.uniform(-np.sqrt(6/17),np.sqrt(6/17),16)
                expected=np.r_[w[:dimension].ravel(),np.zeros(16),outw,0.]; close(initial['theta'],expected,0)
                mean=y[tr].mean(); sd=y[tr].std(); scale=sd if sd>=1e-12 else 1.
                close(state['y_mean'],mean,1e-12); close(state['y_scale'],scale,1e-12)
                close(initial['y_mean'],mean,1e-12); close(initial['y_scale'],scale,1e-12)
                assert (out/'initial.npz').stat().st_mtime_ns <= (out/'model.npz').stat().st_mtime_ns
                target=(y[tr]-mean)/scale; objective,gradient,_=evaluate(z,a['theta'],target); initial_obj=evaluate(z,initial['theta'],target)[0]
                close(objective,state['objective'],1e-11); close(initial_obj,state['initial_objective'],1e-11)
                maximum=float(np.max(abs(gradient))); close(maximum,state['max_abs_gradient'],1e-10)
                good=bool(state['optimizer_success'] and maximum<=1e-4); assert state['qualified_numerically']==good
                if state['optimizer_success']: assert state['optimizer_status']==0
                all_good &= good; view_good &= good
                trace=a['trace']; assert trace.shape==(state['n_iter'],3); close(trace[:,0],np.arange(1,len(trace)+1),0)
                assert len(trace)>0 and state['n_iter']<=1000 and objective<=initial_obj+1e-10
                assert np.max(np.diff(np.r_[initial_obj,trace[:,1]]))<=1e-10
                close(trace[-1,1],objective,1e-11); close(trace[-1,2],maximum,1e-10)
                for frequency in (1.,2.,3.):
                    direction=np.sin(frequency*np.arange(1,len(a['theta'])+1)); direction/=np.linalg.norm(direction); epsilon=1e-5
                    fd=(evaluate(z,a['theta']+epsilon*direction,target)[0]-evaluate(z,a['theta']-epsilon*direction,target)[0])/(2*epsilon)
                    difference=abs(fd-direction@gradient); assert difference<2e-7; maxfinite=max(maxfinite,float(difference))
                pp=mean+scale*evaluate(zv,a['theta']); close(pp,a['prediction'],1e-10)
                maxdiff=max(maxdiff,float(abs(pp-a['prediction']).max())); oof[va]=a['prediction']; counts+=1
                initial.close(); a.close()
            assert sorted(coverage)==list(range(len(y)))
            part=[r for r in details if r['setting']==setting and r['view']==view]; assert [r['sample_id'] for r in part]==ids
            p=np.array([float(r['new_prediction']) for r in part]); close(p,oof,0); close([float(r['label']) for r in part],y,0)
            close([float(r['reference_prediction']) for r in part],previous,0); covered+=len(part)
            row=next(r for r in d['records'] if r['setting']==setting and r['view']==view); m,pm=metric(y,p),metric(y,previous)
            assert row['n']==len(y) and row['input_width']==dimension and row['parameters']==dimension*16+33
            for key in m: close(row['new'][key],m[key],1e-11); close(row['old'][key],pm[key],1e-11)
            delta=m['PCC']-pm['PCC']; reduction=100*(1-m['MSE']/pm['MSE']); close(row['delta_PCC'],delta,1e-11); close(row['MSE_relative_reduction_pct'],reduction,1e-10)
            assert row['pool_qualified']==(delta>=.005 and reduction>=0) and row['all_fits_numerically_qualified']==view_good
            for j,split in enumerate(item['inner']):
                va=np.array(split['valid']); block=row['inner_blocks'][j]; st=states[j]
                assert block['inner']==j and block['n']==len(va) and block['n_iter']==st['n_iter'] and block['qualified_numerically']==st['qualified_numerically']
                close(block['max_abs_gradient'],st['max_abs_gradient'],0)
                for typ,prediction in (('new',p),('old',previous)):
                    for key,value in metric(y[va],prediction[va]).items(): close(block[typ][key],value,1e-11)
            assert d['identities'][source]==read(folder/'complete.json')['source_identity']
    assert counts==60 and covered==len(details)==4*sum(len(v) for v in d['fit_ids'].values()) and len(d['records'])==12
    core=[r for r in d['records'] if r['view']=='core_real']; gate=sum(r['pool_qualified'] for r in core)>=2 and all(r['delta_PCC']>=-.005 for r in core)
    assert d['metric_gate']==gate and d['all_numerically_qualified']==all_good and d['qualified']==(gate and all_good)
    result=dict(passed=True,qualified=bool(gate and all_good),all_numerically_qualified=bool(all_good),verified_models=counts,
        verified_internal_OOF_rows=covered,initial_states=60,finite_difference_directions=180,
        max_prediction_difference=maxdiff,max_directional_derivative_difference=maxfinite,
        source_PCA_and_train_response_scaling_checked=True,outer_test_metrics_used=False,
        new_efficiency_fits=0,new_source_forward=0,GPU_hours=0,lock_sha256=sha(R/'lock.json'),
        diagnostic_sha256=sha(R/'diagnostic.json'),CPU_wall_seconds=time.monotonic()-start)
    with (R/'verification.json').open('x',encoding='utf-8') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    start=time.monotonic(); rc=1; error=None
    try: main(); rc=0
    except Exception: error=traceback.format_exc(); raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-start,GPU_hours=0),f,indent=2)
