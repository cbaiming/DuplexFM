"""Label-free coordinate tail concentration, not a predictor or significance test."""
import math
import numpy as np
TOP_FRACTION=.01
TAIL_SHARE=.20
MAX_Z=6.
MIN_COLUMNS=16
MIN_FRACTION=.10


def inspect(x):
    x=np.asarray(x,dtype=float)
    if x.ndim!=2 or len(x)<100 or x.shape[1] not in (24,192) or not np.isfinite(x).all():
        raise ValueError('Finite training TD24/source192, at least 100 rows')
    n,p=x.shape;k=int(math.ceil(TOP_FRACTION*n));mean=x.mean(0);center=x-mean
    variance=np.mean(center*center,axis=0);sd=np.sqrt(variance);nonconstant=np.ptp(x,axis=0)>0
    positive=sd[nonconstant & (sd>0)]
    floor=.01*(float(np.median(positive)) if len(positive) else 1.)
    eligible=nonconstant & (sd>=floor)
    sq=center*center;total=sq.sum(0);top=np.sort(sq,axis=0)[-k:].sum(0)
    share=np.divide(top,total,out=np.zeros(p),where=total>0)
    peak=np.divide(abs(center).max(0),sd,out=np.zeros(p),where=sd>0)
    fourth=np.divide(np.mean(sq*sq,axis=0),variance*variance,out=np.zeros(p),where=variance>0)
    flagged=eligible & (share>=TAIL_SHARE) & (peak>=MAX_Z)
    needed=max(MIN_COLUMNS,int(math.ceil(MIN_FRACTION*int(eligible.sum()))))
    return dict(n=n,columns=p,top_rows=k,mean=mean,sd=sd,scale_floor=floor,eligible=eligible,
        top_fraction_squared_deviation=share,max_abs_z=peak,raw_kurtosis=fourth,flagged=flagged,
        eligible_columns=int(eligible.sum()),flagged_columns=int(flagged.sum()),required_columns=needed,
        pool_qualified=bool(flagged.sum()>=needed),
        median_tail_share=float(np.median(share[eligible])) if eligible.any() else 0.,
        maximum_tail_share=float(share[eligible].max()) if eligible.any() else 0.,
        maximum_abs_z=float(peak[eligible].max()) if eligible.any() else 0.)


def qualify(records):
    core=[r for r in records if r['source']=='core_real']
    if len(core)!=3 or len({r['setting'] for r in core})!=3:raise ValueError('Three original primary pools')
    return sum(r['statistics']['pool_qualified'] for r in core)>=2
