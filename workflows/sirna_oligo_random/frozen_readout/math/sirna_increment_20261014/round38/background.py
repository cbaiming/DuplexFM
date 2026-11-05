"""Label-free cross-prediction between matched frozen representations."""
import hashlib
import numpy as np


def fit_map(b,f):
    b,f=np.asarray(b,float),np.asarray(f,float)
    if b.ndim!=2 or f.ndim!=2 or len(b)!=len(f) or not len(b) or not b.shape[1] or not f.shape[1]:raise ValueError('Matrix contract')
    if not np.isfinite(b).all() or not np.isfinite(f).all():raise ValueError('Finite matrices required')
    if max(abs(b.mean(0)).max(),abs(f.mean(0)).max())>1e-8:raise ValueError('Training-centered matrices required')
    g=b.T@b+100*np.eye(b.shape[1]);rhs=b.T@f;a=np.linalg.solve(g,rhs);prediction=b@a
    gram=prediction.T@prediction
    return a,gram,dict(normal_error=float(abs(g@a-rhs).max()),train_SSE=float(np.sum((prediction-f)**2)),alpha=100.)


def permutation(ids,setting,inner):
    if len(ids)!=len(set(ids)) or not ids:raise ValueError('Unique nonempty identity list')
    return np.array(sorted(range(len(ids)),key=lambda j:hashlib.sha256(f'r38:42:{setting}:{inner}:{ids[j]}'.encode()).hexdigest()),dtype=np.int64)


def statistics(target,prediction):
    target,prediction=np.asarray(target,float),np.asarray(prediction,float)
    if target.shape!=prediction.shape or target.ndim!=2 or target.shape[1]!=192 or not np.isfinite(target).all() or not np.isfinite(prediction).all():raise ValueError('Finite native192 matrices required')
    blocks=[]
    for j in range(6):
        sl=slice(32*j,32*(j+1));null=float(np.sum(target[:,sl]**2));error=float(np.sum((target[:,sl]-prediction[:,sl])**2))
        blocks.append(dict(SSE=error,null_SSE=null,R2=1-error/null if null>0 else None))
    null=sum(x['null_SSE'] for x in blocks);error=sum(x['SSE'] for x in blocks)
    if null<=0:raise ValueError('Zero null variance')
    return dict(SSE=error,null_SSE=null,R2=1-error/null,blocks=blocks)


def aggregate(parts):
    null=sum(p['null_SSE'] for p in parts);error=sum(p['SSE'] for p in parts)
    if null<=0:raise ValueError('Zero null variance')
    blocks=[]
    for j in range(6):
        n=sum(p['blocks'][j]['null_SSE'] for p in parts);e=sum(p['blocks'][j]['SSE'] for p in parts)
        blocks.append(dict(SSE=e,null_SSE=n,R2=1-e/n if n>0 else None))
    return dict(SSE=error,null_SSE=null,R2=1-error/null,blocks=blocks)


def qualify(records):
    if len(records)!=3:raise ValueError('Three input pools required')
    pools=[]
    for r in records:
        a=r['conditions']['paired_initial']['R2'];b=r['conditions']['TD24']['R2'];c=r['conditions']['shuffled_initial']['R2']
        pools.append(dict(setting=r['setting'],R2=a,delta_vs_TD24=a-b,delta_vs_shuffled=a-c,qualified=bool(a>=.1 and a-b>=.05 and a-c>=.05)))
    return dict(qualified=bool(sum(p['qualified'] for p in pools)>=2 and all(p['R2']>=0 for p in pools)),pool_qualification=pools)
