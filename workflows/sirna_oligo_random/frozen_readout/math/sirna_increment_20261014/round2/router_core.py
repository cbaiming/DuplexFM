"""Convex soft-label linear gate, deterministic hard inference."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

def targets(y,oof):
    e=(np.asarray(oof)-np.asarray(y)[:,None])**2
    tau=max(float(np.median(e.max(1)-e.min(1))),1e-8)
    logits=-(e-e.min(1,keepdims=True))/tau
    q=np.exp(logits-logsumexp(logits,axis=1,keepdims=True))
    return q,tau

def objective(theta,z,q):
    d=z.shape[1];k=q.shape[1];w=theta[:d*k].reshape(d,k);b=theta[d*k:]
    logits=z@w+b;logp=logits-logsumexp(logits,axis=1,keepdims=True);p=np.exp(logp)
    loss=float(-np.mean(np.sum(q*logp,axis=1))+.5*np.sum(w*w))
    diff=(p-q)/len(z);grad=np.r_[(z.T@diff+w).ravel(),diff.sum(0)]
    return loss,grad

def fit_gate(z,q):
    z=np.asarray(z,float);q=np.asarray(q,float)
    if len(z)!=len(q) or z.shape[1]!=20 or q.shape[1]!=7 or not np.isfinite(z).all() or not np.isfinite(q).all():raise ValueError('Gate shape/finite')
    if np.any(q<0) or not np.allclose(q.sum(1),1,rtol=0,atol=1e-12):raise ValueError('Targets')
    mean=q.mean(0);bias=np.log(np.maximum(mean,1e-15));bias-=bias.mean()
    initial=np.r_[np.zeros(20*7),bias]
    result=minimize(objective,initial,args=(z,q),jac=True,method='L-BFGS-B',
        options=dict(maxiter=1000,gtol=1e-9,ftol=1e-14))
    loss,grad=objective(result.x,z,q);gn=float(np.max(abs(grad)))
    if not result.success or gn>=1e-7:raise RuntimeError(f'Gate convergence: {result.message}, grad={gn}')
    return result.x,dict(success=True,iterations=int(result.nit),function_evaluations=int(result.nfev),
        objective=loss,max_gradient=gn,regularization=1.,initial_bias=bias.tolist(),constant_route=int(np.argmax(mean)))

def route(z,theta):
    return np.argmax(np.asarray(z)@theta[:140].reshape(20,7)+theta[140:],axis=1)
