"""Explicit convex Newton polishing; original code and objective stay locked."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
import run_router as pipeline
from router_core import objective
io=pipeline.io

def hessian(theta,z,q):
    k=q.shape[1];aug=np.c_[z,np.ones(len(z))];par=theta.reshape(z.shape[1]+1,k)
    logits=aug@par;p=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
    cov=p[:,:,None]*np.eye(k)[None,:,:]-p[:,:,None]*p[:,None,:]
    h=np.einsum('nd,ne,nkl->dkel',aug,aug,cov,optimize=True).reshape(theta.size,theta.size)/len(z)
    h[:z.shape[1]*k,:z.shape[1]*k]+=np.eye(z.shape[1]*k)
    return (h+h.T)/2

def polish(theta,z,q):
    theta=theta.copy();before=float(np.max(abs(objective(theta,z,q)[1])));steps=0
    gauge=np.r_[np.zeros(z.shape[1]*q.shape[1]),np.ones(q.shape[1])]
    for _ in range(20):
        loss,grad=objective(theta,z,q)
        if np.max(abs(grad))<1e-9:break
        h=hessian(theta,z,q);kkt=np.block([[h,gauge[:,None]],[gauge[None,:],np.zeros((1,1))]])
        step=np.linalg.solve(kkt,np.r_[-grad,0.])[:-1];accepted=False
        for backtrack in range(20):
            trial=theta+step*(.5**backtrack);val,newgrad=objective(trial,z,q)
            if val<=loss+1e-14 and np.max(abs(newgrad))<np.max(abs(grad)):
                theta=trial;accepted=True;steps+=1;break
        if not accepted:raise RuntimeError('Convex polish failed without changing tolerance')
    final=float(np.max(abs(objective(theta,z,q)[1])))
    if final>=1e-7:raise RuntimeError(f'Original threshold still unmet: {final}')
    return theta,dict(newton_steps=steps,gradient_before=before,gradient_after=final)

def fit_gate(z,q):
    z=np.asarray(z,float);q=np.asarray(q,float)
    if z.shape!=(len(q),20) or q.shape[1]!=7 or not np.isfinite(z).all() or not np.isfinite(q).all():raise ValueError('shape/finite')
    if np.any(q<0) or not np.allclose(q.sum(1),1,rtol=0,atol=1e-12):raise ValueError('targets')
    bias=np.log(np.maximum(q.mean(0),1e-15));bias-=bias.mean();initial=np.r_[np.zeros(140),bias]
    result=minimize(objective,initial,args=(z,q),jac=True,method='L-BFGS-B',options=dict(maxiter=1000,gtol=1e-9,ftol=1e-14))
    if not result.success:raise RuntimeError(str(result.message))
    theta=result.x;polishing=dict(newton_steps=0,gradient_before=float(np.max(abs(objective(theta,z,q)[1]))))
    if polishing['gradient_before']>=1e-7:theta,polishing=polish(theta,z,q)
    loss,grad=objective(theta,z,q);gn=float(np.max(abs(grad)));assert gn<1e-7
    return theta,dict(success=True,iterations=int(result.nit),function_evaluations=int(result.nfev),objective=loss,max_gradient=gn,
        regularization=1.,initial_bias=bias.tolist(),constant_route=int(np.argmax(q.mean(0))),
        numeric_compatibility='same-objective gauge-constrained Newton polish',polishing=polishing)

def register():
    import xml.etree.ElementTree as ET
    pipeline.validate();assert not (R/'results/complete.json').exists() and not (R/'execution.mutex').exists()
    suites=list(ET.parse(R/'pytest_numeric_compat.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==2
    assert not any(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites)
    receipts=sorted((R/'results/folds').glob('*/*/complete.json'));assert len(receipts)==15
    for p in receipts:io.check_files(p.parent,io.read(p)['files_sha256'])
    failure=sorted(R.glob('execution_*.json'))[-1];f=io.read(failure)
    assert f['returncode']==1 and 'grad=1.0884409412290595e-07' in f['error']
    files=('run_router_numeric_compat.py','test_numeric_compat.py','NUMERIC_COMPAT.md','pytest_numeric_compat.xml')
    io.write(R/'numeric_execution_lock.json',dict(original_lock_sha256=io.sha(R/'lock.json'),
        files_sha256={n:io.sha(R/n) for n in files},prior_failure={failure.name:io.sha(failure)},
        reused_folds={p.relative_to(R).as_posix():io.sha(p) for p in receipts},objective_changed=False,
        tolerance_changed=False,test_metrics_used=False,new_GPU_hours=0))
    print('NUMERIC_COMPAT_REGISTERED',io.sha(R/'numeric_execution_lock.json'))

def main():
    if len(sys.argv)==2 and sys.argv[1]=='register':register();return
    assert sys.argv[1:]==['run']
    lock=io.read(R/'numeric_execution_lock.json');io.check_files(R,lock['files_sha256']);io.check_files(R,lock['prior_failure']);io.check_files(R,lock['reused_folds'])
    assert io.sha(R/'lock.json')==lock['original_lock_sha256']
    original=pipeline.fit_gate;pipeline.fit_gate=fit_gate;started=time.monotonic();rc=1;error=None
    try:pipeline.main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        pipeline.fit_gate=original
        io.write(R/f'numeric_compat_execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-started,
            GPU_hours=0,numeric_execution_lock_sha256=io.sha(R/'numeric_execution_lock.json'),
            complete_sha256=io.sha(R/'results/complete.json') if (R/'results/complete.json').exists() else None,
            timing_note='Wrapper duration overlaps nested pipeline execution receipt; do not add both.'))
if __name__=='__main__':main()
