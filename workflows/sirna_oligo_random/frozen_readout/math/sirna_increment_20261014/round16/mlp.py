"""One fixed small FP64 regression head; no source-network training."""
import time
import numpy as np
from scipy.optimize import minimize
HIDDEN=16; ALPHA=10.
OPTIONS=dict(maxiter=1000,maxfun=20000,maxls=50,ftol=1e-12,gtol=1e-7)


def unpack(theta,d):
    theta=np.asarray(theta,float)
    if theta.shape!=(d*HIDDEN+2*HIDDEN+1,): raise ValueError('Parameter contract')
    k=d*HIDDEN
    return theta[:k].reshape(d,HIDDEN),theta[k:k+HIDDEN],theta[k+HIDDEN:k+2*HIDDEN],theta[-1]


def initialize(d):
    if d not in (24,40): raise ValueError('Input dimension')
    rng=np.random.default_rng(42)
    w=rng.uniform(-np.sqrt(6/56),np.sqrt(6/56),(40,HIDDEN))
    a=rng.uniform(-np.sqrt(6/17),np.sqrt(6/17),HIDDEN)
    return np.r_[w[:d].ravel(),np.zeros(HIDDEN),a,0.]


def forward(z,theta):
    w,b,a,c=unpack(theta,z.shape[1]); return np.tanh(z@w+b)@a+c


def loss_grad(theta,z,y):
    w,b,a,c=unpack(theta,z.shape[1]); h=np.tanh(z@w+b); residual=h@a+c-y; n=len(y)
    loss=.5*np.mean(residual*residual)+ALPHA/(2*n)*(np.sum(w*w)+a@a)
    output=residual/n; hidden=output[:,None]*a*(1-h*h)
    gradient=np.r_[(z.T@hidden+(ALPHA/n)*w).ravel(),hidden.sum(0),h.T@output+(ALPHA/n)*a,output.sum()]
    return float(loss),gradient


def fit(z,y,save_initial=None,save_checkpoint=None,deadline=None):
    z=np.asarray(z,dtype=np.float64); y=np.asarray(y,dtype=np.float64)
    if z.ndim!=2 or z.shape[1] not in (24,40) or y.shape!=(len(z),) or len(z)<2: raise ValueError('Input contract')
    if not np.isfinite(z).all() or not np.isfinite(y).all(): raise ValueError('Finite training values required')
    mean=float(y.mean()); sd=float(y.std()); scale=sd if sd>=1e-12 else 1.; target=(y-mean)/scale
    initial=initialize(z.shape[1]); initial_loss,_=loss_grad(initial,z,target)
    if save_initial is not None: save_initial(initial,mean,scale)
    trace=[]; updates=0
    def callback(theta):
        nonlocal updates
        updates+=1; value,gradient=loss_grad(theta,z,target)
        trace.append([updates,value,float(np.max(abs(gradient)))])
        if save_checkpoint is not None and updates%500==0: save_checkpoint(updates,theta)
        if deadline is not None and time.monotonic()>deadline: raise TimeoutError('Fixed CPU stage deadline')
    result=minimize(loss_grad,initial,args=(z,target),method='L-BFGS-B',jac=True,callback=callback,options=OPTIONS)
    value,gradient=loss_grad(result.x,z,target); maximum=float(np.max(abs(gradient)))
    if not np.isfinite(result.x).all() or not np.isfinite(gradient).all(): raise RuntimeError('Nonfinite optimization')
    assert value<=initial_loss+1e-10 and updates==result.nit
    return dict(theta=result.x,initial=initial,y_mean=mean,y_scale=scale,d=z.shape[1],parameter_count=len(initial),
        optimizer_success=bool(result.success),optimizer_status=int(result.status),optimizer_message=str(result.message),
        n_iter=int(result.nit),n_function=int(result.nfev),n_gradient=int(result.njev),
        objective=value,initial_objective=initial_loss,max_abs_gradient=maximum,
        qualified_numerically=bool(result.success and maximum<=1e-4),trace=np.asarray(trace,float).reshape(-1,3))


def predict(z,state):
    return state['y_mean']+state['y_scale']*forward(np.asarray(z,float),state['theta'])
