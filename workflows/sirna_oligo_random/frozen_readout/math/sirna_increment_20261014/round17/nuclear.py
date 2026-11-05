"""Convex trace-norm bilinear regression with certified profiled solves."""
import time
import numpy as np
from scipy.linalg import cho_factor,cho_solve
ALPHA=100.; FRACTION=.25; MAXITER=5000; TOL=1e-9

def raw(z,kind):
    if kind=='source' and z.shape[1]==40: a,b=z[:,:24],z[:,24:]
    elif kind=='TD24' and z.shape[1]==24: a=b=z
    else: raise ValueError('Map contract')
    return np.einsum('ni,nj->nij',a,b).reshape(len(z),-1),(a.shape[1],b.shape[1])

def fit_map(z,kind):
    z=np.asarray(z,float); j,shape=raw(z,kind); m=j.mean(0); variance=(j-m).var(0)
    scale=float(np.sqrt(variance.sum()/16)); scale=scale if scale>=1e-12 else 1.
    state=dict(kind=kind,matrix_shape=shape,main_mean=z.mean(0),product_mean=m,product_scale=scale)
    return transform(z,state),state

def transform(z,t):
    j,shape=raw(np.asarray(z,float),t['kind']); assert tuple(t['matrix_shape'])==shape
    return z-t['main_mean'],(j-t['product_mean'])/t['product_scale']

def profile(x,j,y):
    y=np.asarray(y,float); center=float(y.mean()); target=y-center
    a=x.T@x+ALPHA*np.eye(x.shape[1]); af=cho_factor(a,lower=True); cross=x.T@j; rhs=x.T@target
    q=j.T@target-cross.T@cho_solve(af,rhs)
    Q=j.T@j+ALPHA*np.eye(j.shape[1])-cross.T@cho_solve(af,cross); Q=(Q+Q.T)/2
    return Q,q,af,cross,rhs,center

def shrink(matrix,tau):
    u,s,v=np.linalg.svd(matrix,full_matrices=False); return (u*np.maximum(s-tau,0))@v

def certificate(Q,q,c,lam,shape,factor):
    singular=np.linalg.svd(c.reshape(shape),compute_uv=False); nuclear=float(singular.sum())
    z=q-Q@c; norm=float(np.linalg.norm(z.reshape(shape),2))
    if norm>lam: z*=lam/norm
    dual=-.5*float((q-z)@cho_solve(factor,q-z)); primal=.5*float(c@Q@c)-float(q@c)+lam*nuclear
    gap=primal-dual
    if gap < -1e-8: raise ArithmeticError('Negative dual gap')
    return dict(primal_profile=primal,dual_profile=dual,gap=max(0.,gap),dual=z,
        rank=int(np.sum(singular>1e-8)),singular_values=singular)

def solve(Q,q,shape,lam,callback=None,deadline=None):
    factor=cho_factor(Q,lower=True); ridge=cho_solve(factor,q); scale=max(1.,.5*float(q@ridge))
    if lam==0:
        info=certificate(Q,q,ridge,0.,shape,factor); info.update(iterations=0,certified=info['gap']<=TOL*scale,gap_scale=scale)
        return ridge,info
    L=float(np.linalg.eigvalsh(Q)[-1]); c=np.zeros_like(q); look=c.copy(); momentum=1.; history=[]
    for step in range(1,MAXITER+1):
        new=shrink((look-(Q@look-q)/L).reshape(shape),lam/L).ravel()
        next_momentum=(1+np.sqrt(1+4*momentum*momentum))/2
        look=new+((momentum-1)/next_momentum)*(new-c); c=new; momentum=next_momentum
        if callback is not None and step%500==0: callback(step,c)
        if deadline is not None and time.monotonic()>deadline: raise TimeoutError('CPU deadline')
        if step%10==0 or step==MAXITER:
            info=certificate(Q,q,c,lam,shape,factor); history.append([step,info['gap']])
            if info['gap']<=TOL*scale: break
    info.update(iterations=step,certified=bool(info['gap']<=TOL*scale),gap_scale=scale,
        L=L,history=np.asarray(history))
    return c,info

def fit_pair(x,j,y,shape,save_initial=None,checkpoint=None,deadline=None):
    Q,q,af,cross,rhs,center=profile(x,j,y); critical=float(np.linalg.norm(q.reshape(shape),2)); lam=FRACTION*critical
    b0=cho_solve(af,rhs)
    if save_initial is not None: save_initial(b0,np.zeros(j.shape[1]),center)
    output={}
    for name,penalty in (('Ridge',0.),('Nuclear',lam)):
        c,info=solve(Q,q,shape,penalty,callback=checkpoint if name=='Nuclear' else None,deadline=deadline)
        b=cho_solve(af,rhs-cross@c)
        output[name]=dict(b=b,c=c,intercept=center,lambda_value=penalty,lambda_critical=critical,**info)
    return output
