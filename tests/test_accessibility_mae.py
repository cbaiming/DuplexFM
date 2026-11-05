import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('mae_reproduction',Path(__file__).resolve().parents[1]/'scripts/reproduce_accessibility_mae.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_measured_zero_is_retained_missing_and_padding_are_excluded():
    y=np.array([[0.,1.,np.nan,99.],[.5,np.nan,99.,99.]])
    p=np.array([[.25,.5,np.nan,np.inf],[.5,np.nan,np.inf,np.inf]])
    r=m.observed_mae(y,p,np.array([3,2]))
    assert r['observed_nucleotides']==3
    assert r['MAE']==pytest.approx(.75/3)

def test_mae_is_observation_pooled_not_mean_of_window_means():
    y=np.array([[0.,0.,0.],[0.,np.nan,np.nan]])
    p=np.array([[1.,1.,1.],[0.,0.,0.]])
    assert m.observed_mae(y,p,np.array([3,1]))['MAE']==.75

def test_nonfinite_observed_prediction_and_empty_window_rejected():
    with pytest.raises(ValueError):m.observed_mae([[0.]],[[np.nan]],np.array([1]))
    with pytest.raises(ValueError):m.observed_mae([[np.nan]],[[0.]],np.array([1]))

def test_impossible_error_counts_and_negative_sums_rejected():
    with pytest.raises(ValueError):m.error_sum_mae([1.],np.array([51]),np.array([50]))
    with pytest.raises(ValueError):m.error_sum_mae([-1.],np.array([50]),np.array([50]))
