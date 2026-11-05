"""Training-only partial-correlation screening followed by PCA16."""
import numpy as np
from scipy.linalg import cho_factor,cho_solve

SELECTED=32
COMPONENTS=16


def base_coordinates(x,reference):
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=216 or not np.isfinite(x).all():raise ValueError('Expected finite TD24+source192')
    return ((x[:,:24]-reference['td_mean'])/reference['td_scale'],
            (x[:,24:]-reference['src_mean'])/reference['src_scale'])


def screening(td,source,y):
    y=np.asarray(y,float)
    if y.shape!=(len(td),) or not np.isfinite(y).all():raise ValueError('Response contract')
    t=td-td.mean(0);z=source-source.mean(0);yc=y-y.mean()
    a,_,rank,_=np.linalg.lstsq(t,np.c_[z,yc],rcond=1e-10)
    residual=np.c_[z,yc]-t@a;zs=residual[:,:192];yr=residual[:,192]
    yn=float(yr@yr);den=np.sqrt(np.sum(zs*zs,axis=0)*yn)
    score=np.divide(zs.T@yr,den,out=np.zeros(192),where=den>1e-12)
    score=np.clip(score,-1.,1.)
    return dict(partial_correlation=score,td_rank=int(rank),residual_y_norm_squared=yn,
                residual_source_norm_squared=np.sum(zs*zs,axis=0),
                response_residual_defined=bool(yn>1e-12))


def select_indices(scores,mode):
    if mode=='conditional':
        return np.sort(np.lexsort((np.arange(192),-np.abs(scores)))[:SELECTED])
    if mode=='random':
        return np.sort(np.random.default_rng(42).choice(192,SELECTED,replace=False))
    raise ValueError('Unknown fixed selector')


def fit_transform(x,y,reference,mode):
    td,source=base_coordinates(x,reference);s=screening(td,source,y)
    selected=select_indices(s['partial_correlation'],mode);raw=source[:,selected];mean=raw.mean(0);z=raw-mean
    e,v=np.linalg.eigh(z.T@z/len(z));components=v[:,-COMPONENTS:][:,::-1].T;values=np.maximum(e[-COMPONENTS:][::-1],0)
    for row in components:
        if row[np.argmax(np.abs(row))]<0:row*=-1
    pc=z@components.T;pm=pc.mean(0);sd=pc.std(0);constant=np.ptp(pc,axis=0)==0
    positive=sd[(~constant)&(sd>0)];floor=.01*(float(np.median(positive)) if len(positive) else 1.)
    sd=np.where(constant,1.,np.maximum(sd,floor))
    state={k:np.asarray(reference[k]) for k in ('td_mean','td_scale','src_mean','src_scale')}
    state.update(s,mode=mode,selected=selected,pca_mean=mean,components=components,eigenvalues=values,
                 pc_mean=pm,pc_scale=sd,rank16_valid=bool(values[0]>0 and values[-1]>1e-10*values[0]))
    return transform(x,state),state


def transform(x,state):
    td,source=base_coordinates(x,state);selected=np.asarray(state['selected'],int)
    pc=(source[:,selected]-state['pca_mean'])@np.asarray(state['components']).T
    return np.c_[td,(pc-state['pc_mean'])/state['pc_scale']]


def fit_head(z,y):
    center=float(np.mean(y));beta=cho_solve(cho_factor(z.T@z+100*np.eye(40),lower=True),z.T@(y-center))
    return beta,center


def deployment(state,beta,center):
    selected=np.asarray(state['selected'],int);components=np.asarray(state['components'])
    pc_coef=beta[24:]/np.asarray(state['pc_scale']);selected_coef=components.T@pc_coef
    coef=np.zeros(216);coef[:24]=beta[:24]/np.asarray(state['td_scale'])
    coef[24+selected]=selected_coef/np.asarray(state['src_scale'])[selected]
    intercept=float(center-np.asarray(state['td_mean'])@coef[:24]
        -np.asarray(state['src_mean'])[selected]@coef[24+selected]
        -np.asarray(state['pca_mean'])@selected_coef-np.asarray(state['pc_mean'])@pc_coef)
    return coef,intercept
