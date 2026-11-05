"""Single-source local representation with a fixed two-block ridge family."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
ALPHAS=(.1,1.,10.,100.,1000.);WEIGHTS=(0.,.01,.1,1.,10.)
def configs():return [dict(weight=w,alpha=a) for w in WEIGHTS for a in ALPHAS]
def scaler(x,floor=False):
    mean=x.mean(0);sd=x.std(0)
    if floor:
        constant=np.ptp(x,axis=0)==0;positive=sd[(~constant)&(sd>0)]
        lo=.01*(float(np.median(positive)) if len(positive) else 1.)
        sd=np.where(constant,1.,np.maximum(sd,lo))
    else:sd[sd<1e-12]=1.
    return mean,sd
def transform(x,t):
    td=(x[:,:24]-t['td_mean'])/t['td_scale']
    z=(x[:,24:]-t['src_mean'])/t['src_scale']
    pc=(z-t['pca_mean'])@np.asarray(t['components']).T
    return np.c_[td,(pc-t['pc_mean'])/t['pc_scale']]
def fit_transform(x):
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=216 or len(x)<20 or not np.isfinite(x).all():raise ValueError('Input contract')
    tm,ts=scaler(x[:,:24]);sm,ss=scaler(x[:,24:],True)
    z=(x[:,24:]-sm)/ss;pm=z.mean(0);z=z-pm
    e,v=np.linalg.eigh(z.T@z/len(z));c=v[:,-16:][:,::-1].T;ev=np.maximum(e[-16:][::-1],0)
    for row in c:
        if row[np.argmax(abs(row))]<0:row*=-1
    pcm,pcs=scaler(z@c.T,True)
    t=dict(td_mean=tm,td_scale=ts,src_mean=sm,src_scale=ss,pca_mean=pm,
        components=c,eigenvalues=ev,pc_mean=pcm,pc_scale=pcs)
    return transform(x,t),t
def penalty(cfg):
    return np.r_[np.full(24,cfg['alpha']),np.full(16,cfg['alpha']/cfg['weight'] if cfg['weight'] else 0)]
def fit_heads(z,y,cfg):
    center=float(y.mean());g=z.T@z;q=z.T@(y-center);b=np.zeros((len(cfg),40))
    for k,c in enumerate(cfg):
        n=40 if c['weight']>0 else 24
        b[k,:n]=cho_solve(cho_factor(g[:n,:n]+np.diag(penalty(c)[:n]),lower=True),q[:n])
    return b,center
def score(y,p):
    yc=y-y.mean();pc=p-p.mean(0);den=np.sqrt((yc@yc)*np.sum(pc*pc,axis=0))
    if yc@yc<=0:raise ValueError('Constant response')
    return np.divide(yc@pc,den,out=np.full(p.shape[1],-2.),where=den>1e-14),np.mean((p-y[:,None])**2,axis=0)
def choose(pc,ms,cfg,indices):
    return min(indices,key=lambda k:(-float(pc[k]),float(ms[k]),cfg[k]['weight'],-cfg[k]['alpha'],k))
