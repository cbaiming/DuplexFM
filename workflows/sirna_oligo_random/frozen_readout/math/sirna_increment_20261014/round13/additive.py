"""One natural-cubic curvature term per feature; fixed train-only geometry."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
path=Path(__file__).resolve().parents[1]/'round4/core.py'
spec=importlib.util.spec_from_file_location('_r13_base',path);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

def hinge(z,knots,active):
    z=np.asarray(z,float);knots=np.asarray(knots,float);active=np.asarray(active,bool)
    out=np.zeros_like(z)
    if not active.any():return out
    x=z[:,active];k=knots[:,active];u=(x-k[0])/(k[2]-k[0]);a=(k[1]-k[0])/(k[2]-k[0])
    mid=np.clip(u,0.,1.);h=mid**3-np.maximum(mid-a,0)**3/(1-a)
    h=np.where(u>=1,3*a*u-a*(1+a),h)
    out[:,active]=h;return out

def curvature_fit(z):
    z=np.asarray(z,float)
    if z.ndim!=2 or len(z)<20 or not np.isfinite(z).all():raise ValueError('Curve input contract')
    knots=np.quantile(z,[.1,.5,.9],axis=0,method='linear');mean=z.mean(0);zc=z-mean;var=np.mean(zc**2,0)
    active=(np.min(np.diff(knots,axis=0),axis=0)>1e-8)&(var>1e-12)
    h=hinge(z,knots,active);hm=h.mean(0)
    slope=np.divide(np.mean(zc*(h-hm),0),var,out=np.zeros(z.shape[1]),where=var>1e-12)
    q=h-hm-zc*slope;sd=q.std(0);active=active&(sd>1e-10);scale=np.where(active,sd,1.)
    state=dict(knots=knots,mean=mean,h_mean=hm,slope=slope,scale=scale,active=active)
    return curvature_transform(z,state),state

def curvature_transform(z,t):
    z=np.asarray(z,float);active=np.asarray(t['active'],bool)
    if z.ndim!=2 or z.shape[1]!=len(active) or not np.isfinite(z).all():raise ValueError('Curve input contract')
    q=(hinge(z,t['knots'],active)-t['h_mean']-(z-t['mean'])*t['slope'])/t['scale']
    q[:,~active]=0.;return q

def fit_transform(x,projection,view):
    original=base.transform(x,projection)
    z=original[:,:24] if view=='TD24' else original
    variables=z if view=='TD24' else z[:,24:]
    q,curve=curvature_fit(variables)
    state=dict(projection=projection,curve=curve,view=view,new_PCA_fit=False)
    return np.c_[z,q],state

def transform(x,t):
    original=base.transform(x,t['projection']);z=original[:,:24] if t['view']=='TD24' else original
    return np.c_[z,curvature_transform(z if t['view']=='TD24' else z[:,24:],t['curve'])]

def fit_heads(z,y,configs):
    z=np.asarray(z,float);y=np.asarray(y,float)
    if z.ndim!=2 or z.shape[0]!=len(y) or z.shape[1] not in (48,56):raise ValueError('Head input contract')
    mean=float(y.mean());g=z.T@z;rhs=z.T@(y-mean);out=np.zeros((len(configs),z.shape[1]))
    for i,c in enumerate(configs):
        n=z.shape[1] if c['weight'] else 24
        penalty=np.r_[np.full(24,c['alpha']),np.full(n-24,c['alpha']/c['weight'] if c['weight'] else 0)]
        out[i,:n]=cho_solve(cho_factor(g[:n,:n]+np.diag(penalty),lower=True),rhs[:n])
    return out,mean
