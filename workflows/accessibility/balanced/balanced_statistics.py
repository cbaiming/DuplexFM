"""Per-length sufficient statistics, with shared gene-group bootstrap draws."""
import numpy as np
def grouped_length_stats(p,y,lengths,groups):
 mask=np.arange(60)[None,:]<lengths[:,None];x=np.where(mask,p,0).astype('float64');v=np.where(mask,y,0).astype('float64')
 rows=np.stack([mask.sum(1),x.sum(1),v.sum(1),(x*x).sum(1),(v*v).sum(1),(x*v).sum(1)],axis=1)
 _,g=np.unique(groups,return_inverse=True);s=np.zeros((g.max()+1,11,6),dtype='float64');np.add.at(s,(g,lengths-50),rows);return s
def correlation(s):
 n=s[...,0];sx=s[...,1];sy=s[...,2]
 with np.errstate(divide='ignore',invalid='ignore'):
  cov=s[...,5]-sx*sy/n;den=(s[...,3]-sx*sx/n)*(s[...,4]-sy*sy/n)
  return np.where(den>1e-15,cov/np.sqrt(np.maximum(den,0)),np.nan)
def bootstrap(s,draws=2000):
 n=len(s);rng=np.random.default_rng(20261104);w=rng.multinomial(n,np.ones(n)/n,size=draws)
 totals=(w@s.reshape(n,-1)).reshape(draws,11,6)
 per_length=correlation(totals);assert np.isfinite(per_length).all()
 return dict(macro=per_length.mean(1),pooled=correlation(totals.sum(1)),per_length=per_length)
