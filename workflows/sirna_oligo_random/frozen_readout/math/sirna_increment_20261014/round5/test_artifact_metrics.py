import numpy as np
from scipy.stats import spearmanr
import pytest

def test_rank_discontinuity_and_artifact_metric():
    y=np.array([0.,1.,2.,3.]);p=np.array([0.,.5,.5,1.]);q=p.copy();q[1]=np.nextafter(q[1],1.)
    np.testing.assert_allclose(p,q,atol=1e-10,rtol=1e-10)
    assert spearmanr(y,p).statistic != spearmanr(y,q).statistic
    restored=np.array([float(repr(float(v))) for v in p])
    assert spearmanr(y,p).statistic==spearmanr(y,restored).statistic

def test_material_prediction_change_rejected():
    with pytest.raises(AssertionError):np.testing.assert_allclose([.1,.2],[.1,.3],atol=1e-10,rtol=1e-10)
