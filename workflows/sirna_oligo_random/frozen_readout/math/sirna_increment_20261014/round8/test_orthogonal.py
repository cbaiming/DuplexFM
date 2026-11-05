import os,sys,copy
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pytest
from orthogonal import configs,fit_transform,transform,fit_heads,collapse,base
from audit_projection import expand

def sample():
    rng=np.random.default_rng(42);x=rng.normal(size=(125,216));x[:,24:]+=x[:,:24]@rng.normal(size=(24,192))*.3
    return x,rng.normal(size=125),rng.normal(size=(23,216))

def test_orthogonal_and_td_coefficients_at_every_alpha():
    x,y,_=sample();z,t=fit_transform(x);b,c=fit_heads(z,y,configs())
    np.testing.assert_allclose(z[:,:24].T@z[:,24:],0,atol=1e-10)
    for k in range(25):np.testing.assert_allclose(b[k,:24],b[k%5,:24],atol=1e-10)
    oldz,_=base.fit_transform(x);oldb,oldc=base.fit_heads(oldz,y,configs())
    np.testing.assert_allclose(z@b[:5].T+c,oldz@oldb[:5].T+oldc,atol=1e-10)

def test_collapsed_deployment_and_independent_reconstruction():
    x,y,other=sample();z,t=fit_transform(x);b,c=fit_heads(z,y,configs());coef,offset=collapse(t,b,c)
    for value in (x,other):
        pred=transform(value,t)@b.T+c
        np.testing.assert_allclose(value@coef.T+offset,pred,atol=1e-10)
        np.testing.assert_allclose(expand(x,value,t)@b.T+c,pred,atol=1e-10)

def test_unpenalized_span_is_unchanged():
    x,y,_=sample();z,t=fit_transform(x);original=base.transform(x,t['projection'])
    p=z@np.linalg.lstsq(z,y-y.mean(),rcond=1e-10)[0]
    q=original@np.linalg.lstsq(original,y-y.mean(),rcond=1e-10)[0]
    np.testing.assert_allclose(p,q,atol=1e-10)

def test_rank_deficient_td_and_constant_source_are_finite():
    x,y,other=sample();x[:,1]=x[:,0];x[:,2]=3.;x[:,24:28]=7.
    z,t=fit_transform(x);b,c=fit_heads(z,y,configs());assert t['TD_rank']==22
    assert np.isfinite(b).all();np.testing.assert_allclose(expand(x,x,t),z,atol=1e-10)

def test_unseen_data_cannot_change_map_and_corruption_is_rejected():
    x,y,other=sample();_,t=fit_transform(x);a=t['td_to_source'].copy()
    assert np.isfinite(transform(other+100,t)).all();np.testing.assert_array_equal(a,t['td_to_source'])
    bad=copy.deepcopy(t);bad['td_to_source'][0,0]+=.1
    with pytest.raises(AssertionError):expand(x,other,bad)
