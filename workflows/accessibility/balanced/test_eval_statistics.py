import unittest,numpy as np
from scipy.stats import pearsonr
from balanced_statistics import grouped_length_stats,correlation,bootstrap
class Tests(unittest.TestCase):
 def test_pooled_and_macro(self):
  rng=np.random.default_rng(41);lens=np.tile(np.arange(50,61),3);g=np.repeat(np.arange(3),11);p=rng.random((33,60));y=rng.random((33,60));mask=np.arange(60)[None,:]<lens[:,None];y[~mask]=np.nan
  st=grouped_length_stats(p,y,lens,g)
  self.assertAlmostEqual(float(correlation(st.sum((0,1)))),pearsonr(p[mask],y[mask]).statistic,places=12)
  rr=[pearsonr(p[lens==l,:l].ravel(),y[lens==l,:l].ravel()).statistic for l in range(50,61)]
  np.testing.assert_allclose(correlation(st.sum(0)),rr,rtol=0,atol=1e-12)
  # Shared resampling of a group repeats it in every length, preserving covariance.
  ids=np.r_[np.flatnonzero(g==0),np.flatnonzero(g==0),np.flatnonzero(g==2)]
  expected=[pearsonr(p[ids][lens[ids]==l,:l].ravel(),y[ids][lens[ids]==l,:l].ravel()).statistic for l in range(50,61)]
  actual=correlation(np.einsum('g,gls->ls',np.array([2,0,1]),st))
  np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
 def test_bootstrap_deterministic(self):
  rng=np.random.default_rng(42);lens=np.tile(np.arange(50,61),4);g=np.repeat(np.arange(4),11);p=rng.random((44,60));y=rng.random((44,60))
  st=grouped_length_stats(p,y,lens,g);a=bootstrap(st,20);b=bootstrap(st,20)
  np.testing.assert_array_equal(a['macro'],b['macro']);np.testing.assert_allclose(a['macro'],a['per_length'].mean(1))
if __name__=='__main__':unittest.main()
