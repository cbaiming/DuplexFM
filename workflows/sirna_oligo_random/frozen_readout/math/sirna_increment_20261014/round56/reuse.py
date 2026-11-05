"""Strict immutable sampling and two-reference statistics transport contract."""
import hashlib,json
from pathlib import Path
import numpy as np


def load_fixed_sampling(root,ids,expected,draws=2000):
    root=Path(root)
    for name in ('groups_fixed.json','statistics.npz'):
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected[name]:raise ValueError('Prior sampling hash mismatch')
    g=json.loads((root/'groups_fixed.json').read_text(encoding='utf-8-sig'))
    if g['ids']!=list(ids) or len(set(ids))!=len(ids):raise ValueError('Exact ordered input identities required')
    if not g['input_only'] or not g['before_numeric_labels']:raise ValueError('Prior input-only qualification required')
    labels=np.asarray(g['labels']);k=g['global_components']
    if labels.shape!=(len(ids),) or not np.issubdtype(labels.dtype,np.integer) or set(labels.tolist())!=set(range(k)):raise ValueError('Invalid component labels')
    shapes={'observed_PCC':(4,6),'observed_MSE':(4,6),'leave_PCC':(k,4,6),'leave_MSE':(k,4,6),
            'bootstrap_PCC':(draws,4,6),'bootstrap_MSE':(draws,4,6)}
    with np.load(root/'statistics.npz',allow_pickle=False) as z:
        if not np.array_equal(z['labels'],labels):raise ValueError('Array/identity components differ')
        counts=z['counts'].copy();refs={}
        if counts.shape!=(draws,k) or not np.issubdtype(counts.dtype,np.integer) or np.any(counts<0) or not np.all(counts.sum(1)==k):raise ValueError('Invalid fixed draws')
        for name,shape in shapes.items():
            value=z[name]
            if value.shape!=shape or np.isinf(value).any():raise ValueError('Invalid prior reference shape/values')
            refs[name]=value[...,[0,1]].copy()
    return labels,counts,refs


def stitch(current,reference):
    current,reference=np.asarray(current),np.asarray(reference)
    if current.ndim<1 or current.shape[-1]!=4 or reference.shape!=current.shape[:-1]+(2,):raise ValueError('Four new conditions and two ordered references required')
    return np.concatenate((reference[...,:1],current,reference[...,1:]),axis=-1)
