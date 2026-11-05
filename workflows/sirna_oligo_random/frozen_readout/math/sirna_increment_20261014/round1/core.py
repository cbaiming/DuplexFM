"""Fixed group-PCA and linear hard routes; no source model or test access."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

ALPHAS=(.1,1.,10.,100.,1000.)
MASKS=(31,30,29,27,23,15,0)

def configs():
    return [dict(mask=m,td=True,alpha=a) for m in MASKS for a in ALPHAS]+[
        dict(mask=31,td=False,alpha=a) for a in ALPHAS]

def columns(cfg):
    return np.array((list(range(24)) if cfg['td'] else [])+
        [i for j in range(5) if cfg['mask']&(1<<j) for i in range(24+4*j,24+4*j+4)],int)

def fit_transform(x):
    x=np.asarray(x,np.float64)
    if x.ndim!=2 or x.shape[1]!=184 or len(x)<5 or not np.isfinite(x).all():raise ValueError('Training shape')
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-12]=1
    z=(x-mean)/scale;components=[];eigenvalues=[];pc_scale=[]
    for j in range(5):
        s=z[:,24+32*j:24+32*(j+1)];e,v=np.linalg.eigh(s.T@s/len(s))
        v=v[:,-4:][:,::-1].T;e=np.maximum(e[-4:][::-1],0)
        for row in v:
            if row[np.argmax(abs(row))]<0:row*=-1
        sd=np.sqrt(e);sd[sd<1e-12]=1
        components.append(v);eigenvalues.append(e);pc_scale.append(sd)
    t=dict(mean=mean,scale=scale,components=np.array(components),eigenvalues=np.array(eigenvalues),pc_scale=np.array(pc_scale))
    return transform(x,t),t

def transform(x,t):
    z=(np.asarray(x)-np.asarray(t['mean']))/np.asarray(t['scale'])
    return np.c_[z[:,:24],*[z[:,24+32*j:24+32*(j+1)]@np.asarray(t['components'])[j].T/np.asarray(t['pc_scale'])[j] for j in range(5)]]

def fit_heads(z,y,cfg):
    y=np.asarray(y);intercept=float(y.mean());g=z.T@z;q=z.T@(y-intercept)
    coef=np.zeros((len(cfg),44))
    for k,c in enumerate(cfg):
        keep=columns(c)
        coef[k,keep]=cho_solve(cho_factor(g[np.ix_(keep,keep)]+c['alpha']*np.eye(len(keep)),lower=True),q[keep])
    return coef,intercept

def score(y,p):
    yc=y-y.mean();pc=p-p.mean(0);den=np.sqrt((yc@yc)*np.sum(pc*pc,axis=0))
    if yc@yc<=0:raise ValueError('Constant response')
    return np.divide(yc@pc,den,out=np.full(p.shape[1],-2.),where=den>1e-14),np.mean((p-y[:,None])**2,axis=0)

def choose(pcc,mse,cfg,indices):
    return min(indices,key=lambda k:(-float(pcc[k]),float(mse[k]),cfg[k]['mask'].bit_count(),-cfg[k]['alpha'],k))
