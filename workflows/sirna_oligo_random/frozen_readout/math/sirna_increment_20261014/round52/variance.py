"""Split-half reproducibility of OOF residual variance; no efficiency fitting."""
import hashlib
import numpy as np
from scipy.stats import spearmanr
PRIOR_DF=5.

def halves(ids,labels,layout):
    if layout not in (0,1) or len(ids)!=len(labels) or len(set(ids))!=len(ids):raise ValueError('Two fixed layouts and unique aligned IDs required')
    g=np.asarray(labels,int);blocks=[]
    for group in sorted(set(g.tolist())):
        members=np.flatnonzero(g==group).tolist()
        if len(members)<8:continue
        order=sorted(members,key=lambda i:(hashlib.sha256(f'R52|20261015|layout{layout}|{ids[i]}'.encode()).hexdigest(),ids[i]))
        blocks.append(dict(group=int(group),A=order[::2],B=order[1::2]))
    return blocks

def sample_stats(r,indices):
    x=np.asarray(r,float)[indices]
    if len(x)<4 or not np.isfinite(x).all():raise ValueError('At least four finite residuals per half')
    sse=float(np.sum((x-x.mean())**2));return dict(n=len(x),df=len(x)-1,SSE=sse,variance=sse/(len(x)-1))

def direction(calibration,validation):
    ss=np.array([r['SSE'] for r in calibration]);df=np.array([r['df'] for r in calibration],float)
    vs=np.array([r['variance'] for r in validation]);vd=np.array([r['df'] for r in validation],float)
    if not len(ss):return dict(defined=False,reason='no_eligible_groups',predicted_variances=[],global_variance=None,score_improvement=None)
    v0=float(ss.sum()/df.sum());vp=(ss+PRIOR_DF*v0)/(df+PRIOR_DF)
    if not np.isfinite(np.r_[vp,vs]).all() or v0<=0 or np.any(vp<=0) or np.any(ss<=0) or np.any(vs<=0):
        return dict(defined=False,reason='zero_or_nonfinite_variance',predicted_variances=[],global_variance=v0,score_improvement=None)
    base=.5*(np.log(v0)+vs/v0);pred=.5*(np.log(vp)+vs/vp);weights=vd/vd.sum()
    return dict(defined=True,predicted_variances=vp.tolist(),global_variance=v0,
        baseline_score=float(weights@base),group_score=float(weights@pred),score_improvement=float(weights@(base-pred)),
        group_score_improvements=(base-pred).tolist(),validation_degrees_of_freedom=vd.tolist())

def diagnose(r,blocks,n):
    groups=[]
    for b in blocks:groups.append(dict(group=b['group'],A=sample_stats(r,b['A']),B=sample_stats(r,b['B'])))
    va=np.array([g['A']['variance'] for g in groups]);vb=np.array([g['B']['variance'] for g in groups])
    defined=len(va)>=2 and np.all(va>0) and np.all(vb>0) and np.ptp(va)>0 and np.ptp(vb)>0
    correlation=float(spearmanr(va,vb).statistic) if defined else None
    ab=direction([g['A'] for g in groups],[g['B'] for g in groups]);ba=direction([g['B'] for g in groups],[g['A'] for g in groups])
    avg=(ab['score_improvement']+ba['score_improvement'])/2 if ab['defined'] and ba['defined'] else None
    covered=sum(g['A']['n']+g['B']['n'] for g in groups)
    coverage=covered/n;available=len(groups)>=8 and coverage>=.5
    qualifies=available and correlation is not None and correlation>=.3 and avg is not None and avg>=.01
    return dict(eligible_groups=len(groups),covered_samples=covered,coverage=coverage,structurally_eligible=bool(available),
        variance_spearman=correlation,A_to_B=ab,B_to_A=ba,mean_score_improvement=avg,qualified=bool(qualifies),groups=groups)

def gate(records):
    pools={}
    for name in ('Huesken','Mixset','Takayuki'):
        p=[r for r in records if r['setting']==name and r['source']=='core_real']
        if len(p)!=2 or {r['layout'] for r in p}!={0,1}:raise ValueError('Both predeclared layouts required')
        pools[name]=all(r['qualified'] for r in p)
    return dict(qualified=sum(pools.values())>=2,primary_pool_qualifications=pools,required_pools=2)
