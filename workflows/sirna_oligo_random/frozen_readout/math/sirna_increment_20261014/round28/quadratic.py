"""Frozen-source second-order response surface; one ridge regression."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
PAIRS=tuple((i,j) for i in range(16) for j in range(i,16))
FACTOR=np.sqrt(16./136.)
ALPHA=100.


def check(z):
    z=np.asarray(z,float)
    if z.ndim!=2 or z.shape[1]!=40 or not np.isfinite(z).all():
        raise ValueError('Finite frozen TD24+PC16 required')
    return z


def raw_terms(z):
    z=check(z);s=z[:,24:]
    i,j=np.triu_indices(16)
    return s[:,i]*s[:,j]


def fit_map(z):
    z=check(z)
    if len(z)<20:raise ValueError('At least 20 training rows')
    q=raw_terms(z);mean=q.mean(0);sd=q.std(0);sd[sd<1e-12]=1.
    state=dict(mean=mean,scale=sd,source_dimension=16,pairs=PAIRS,factor=float(FACTOR))
    return transform(z,state),state


def transform(z,state):
    z=check(z);mean=np.asarray(state['mean'],float);scale=np.asarray(state['scale'],float)
    if mean.shape!=(136,) or scale.shape!=(136,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<=0):
        raise ValueError('Training quadratic statistics contract')
    if state['source_dimension']!=16 or tuple(tuple(p) for p in state['pairs'])!=PAIRS or state['factor']!=float(FACTOR):
        raise ValueError('Locked quadratic layout/scale')
    return np.c_[z,(raw_terms(z)-mean)/scale*FACTOR]


def fit(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim!=2 or x.shape[1]!=176 or y.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Finite 176D regression contract')
    if np.ptp(y)==0 or abs(x.mean(0)).max()>1e-8:raise ValueError('Centered features/nonconstant response')
    center=float(y.mean());h=x.T@x+ALPHA*np.eye(176);target=x.T@(y-center)
    b=cho_solve(cho_factor(h,lower=True),target)
    residual=y-center-x@b;normal=float(abs(h@b-target).max());limit=1e-9*(1+float(abs(target).max()))
    return b,center,dict(alpha=ALPHA,normal_error=normal,normal_limit=limit,numerically_qualified=normal<=limit,
                         objective=float(residual@residual+ALPHA*(b@b)),features=176)


def direct_predict(z,state,b,center):
    """Deploy as a symmetric quadratic in the existing 16 coordinates."""
    z=check(z);b=np.asarray(b,float)
    if b.shape!=(176,):raise ValueError('Coefficient width')
    weights=b[40:]*FACTOR/np.asarray(state['scale'])
    matrix=np.zeros((16,16))
    for coefficient,(i,j) in zip(weights,PAIRS):
        if i==j:matrix[i,j]=coefficient
        else:matrix[i,j]=matrix[j,i]=coefficient*.5
    source=z[:,24:];offset=float(center-np.asarray(state['mean'])@weights)
    return z@b[:40]+np.einsum('ni,ij,nj->n',source,matrix,source)+offset


def qualify(records,numeric):
    core=[r for r in records if r['source']=='core_real']
    if len(core)!=3:raise ValueError('Three fixed primary pools')
    performance=(sum(r['delta_PCC_vs_R4']>=.005 and r['MSE_reduction_vs_R4_pct']>=0 for r in core)>=2
                 and all(r['delta_PCC_vs_R4']>=-.005 for r in core))
    paired=[dict(setting=r['setting'],delta_PCC=r['metrics']['PCC']-r['TD_quadratic_metrics']['PCC'],
        MSE_reduction_pct=100*(1-r['metrics']['MSE']/r['TD_quadratic_metrics']['MSE'])) for r in core]
    incremental=sum(r['delta_PCC']>=0 for r in paired)>=2 and np.mean([r['MSE_reduction_pct'] for r in paired])>=0
    return dict(qualified=bool(performance and incremental and numeric),performance_gate=bool(performance),
                source_increment_gate=bool(incremental),all_numerically_qualified=bool(numeric),paired_TD_quadratic=paired)
