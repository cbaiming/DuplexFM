"""One smooth varying-coefficient head, profiled linear coefficients."""
import time
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
ALPHA=100.;LOCAL_MULTIPLIER=4.;GATE_FRACTION=.01
MAXITER=500;MAXFUN=10000;MAXLS=50;FTOL=1e-12;GTOL=1e-8;AUDIT_GTOL=1e-6


def inputs(z,y=None):
    z=np.asarray(z,float)
    if z.ndim!=2 or z.shape[1] not in (24,40) or not np.isfinite(z).all():raise ValueError('Finite TD24 or TD24+PCA16')
    if y is None:return z
    y=np.asarray(y,float)
    if y.shape!=(len(z),) or not np.isfinite(y).all() or np.ptp(y)==0 or len(y)<20:raise ValueError('Training response contract')
    if abs(z.mean(0)).max()>1e-8:raise ValueError('Training inputs must be centered')
    return z,y


def initial_gate():
    a=np.random.default_rng(42).normal(size=24);return np.r_[a/np.linalg.norm(a),0.]


def local(z):return z if z.shape[1]==24 else z[:,24:]


def design(z,a,h_mean=None,j_mean=None):
    z=inputs(z);a=np.asarray(a,float)
    if a.shape!=(25,) or not np.isfinite(a).all():raise ValueError('Finite 25-parameter gate')
    h=np.tanh(z[:,:24]@a[:24]+a[24]);hm=float(h.mean()) if h_mean is None else float(h_mean)
    w=local(z);j=(h-hm)[:,None]*w; jm=j.mean(0) if j_mean is None else np.asarray(j_mean,float)
    if jm.shape!=(w.shape[1],) or not np.isfinite(jm).all() or not np.isfinite(hm):raise ValueError('Training centering state')
    return np.c_[z,j-jm],h,hm,jm.copy()


def profile(a,z,y):
    z,y=inputs(z,y);x,h,hm,jm=design(z,a);p=z.shape[1];q=local(z).shape[1]
    penalty=np.r_[np.full(p,ALPHA),np.full(q,ALPHA*LOCAL_MULTIPLIER)]
    center=float(y.mean());yc=y-center;scale=float(yc@yc);gamma=GATE_FRACTION*scale
    gram=x.T@x+np.diag(penalty);rhs=x.T@yc;beta=cho_solve(cho_factor(gram,lower=True),rhs)
    e=x@beta-yc;objective=float(.5*(e@e+np.dot(penalty,beta*beta)+gamma*np.dot(a,a))/scale)
    e0=e-e.mean();w=e0*(local(z)@beta[p:]);w-=w.mean();dh=w*(1-h*h)
    gradient=(np.r_[z[:,:24].T@dh,dh.sum()]+gamma*a)/scale
    return objective,gradient,dict(beta=beta,intercept=center,h_mean=hm,j_mean=jm,
        gamma=gamma,loss_scale=scale,normal_error=float(abs(gram@beta-rhs).max()),
        normal_limit=float(1e-9*(1+abs(rhs).max())),train_MSE=float(np.mean(e*e)))


def fit(z,y,save_initial=None,deadline=None):
    z,y=inputs(z,y);a0=initial_gate();calls=0;accepted=[]
    if save_initial is not None:save_initial(a0.copy())
    def evaluate(a):
        nonlocal calls
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('CPU stage deadline')
        obj,grad,model=profile(a,z,y);calls+=1;return obj,grad,model
    initial_objective,_,_=evaluate(a0)
    def objective(a):
        obj,grad,_=evaluate(a);return obj,grad
    def callback(a):
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('CPU stage deadline')
        accepted.append(np.asarray(a,float).copy())
    result=minimize(objective,a0,method='L-BFGS-B',jac=True,callback=callback,
        options=dict(maxiter=MAXITER,maxfun=MAXFUN,maxls=MAXLS,ftol=FTOL,gtol=GTOL))
    obj,grad,model=evaluate(result.x)
    valid=bool(result.success and np.max(abs(grad))<=AUDIT_GTOL and model['normal_error']<=model['normal_limit'])
    model.update(gate=result.x.copy(),initial_gate=a0,initial_objective=float(initial_objective),objective=float(obj),
        gradient=grad,gradient_max=float(abs(grad).max()),numerically_qualified=valid,
        success=bool(result.success),message=str(result.message),iterations=int(result.nit),function_evaluations=int(result.nfev),
        new_linear_solves=calls,accepted_gates=np.asarray(accepted).reshape(-1,25),
        shared_alpha=ALPHA,local_multiplier=LOCAL_MULTIPLIER,gate_fraction=GATE_FRACTION)
    return model


def predict(z,model):
    x,h,_,_=design(z,model['gate'],model['h_mean'],model['j_mean'])
    return x@model['beta']+model['intercept'],(1+h)/2


def branch_predict(z,model):
    z=inputs(z);p=z.shape[1];b=np.asarray(model['beta']);a=np.asarray(model['gate'])
    g=(1+np.tanh(z[:,:24]@a[:24]+a[24]))/2;w=local(z)
    # This is algebra for a jointly fitted head, not separately fitted model predictions.
    base=z@b[:p]+model['intercept']-np.asarray(model['j_mean'])@b[p:]
    left=base+(-1-model['h_mean'])*(w@b[p:])
    right=base+(1-model['h_mean'])*(w@b[p:])
    return (1-g)*left+g*right,g


def qualify(records,numeric):
    core=[r for r in records if r['source']=='core_real']
    if len(core)!=3:raise ValueError('Three primary pools')
    performance=(sum(r['delta_PCC_vs_R4']>=.005 and r['MSE_reduction_vs_R4_pct']>=0 for r in core)>=2
                 and all(r['delta_PCC_vs_R4']>=-.005 for r in core))
    paired=[]
    for r in core:
        td=next(t for t in records if t['setting']==r['setting'] and t['source']=='TD_only')
        paired.append(dict(setting=r['setting'],delta_PCC=r['metrics']['PCC']-td['metrics']['PCC'],
            MSE_reduction_pct=100*(1-r['metrics']['MSE']/td['metrics']['MSE'])))
    incremental=sum(r['delta_PCC']>=0 for r in paired)>=2 and np.mean([r['MSE_reduction_pct'] for r in paired])>=0
    return dict(qualified=bool(performance and incremental and numeric),performance_gate=bool(performance),
        source_increment_gate=bool(incremental),all_numerically_qualified=bool(numeric),paired_gated_TD=paired)
