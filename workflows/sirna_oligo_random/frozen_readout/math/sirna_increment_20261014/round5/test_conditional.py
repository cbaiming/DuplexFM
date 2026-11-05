import os,sys
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pytest
from conditional import configs,fit_transform,transform,fit_heads,second_order,choose

def sample(n=90):
    r=np.random.default_rng(42);return r.normal(size=(n,216)),r.normal(size=n)

@pytest.mark.parametrize('kind,width',[('bilinear',424),('td_quadratic',324)])
def test_regularized_equations_and_td_equivalence(kind,width):
    x,y=sample();x=x if kind=='bilinear' else x[:,:24]
    z,t=fit_transform(x,kind);assert z.shape==(90,width)
    b,c=fit_heads(z,y,configs());np.testing.assert_allclose(z.mean(0),0,atol=1e-12)
    for k,cf in enumerate(configs()):
        n=width if cf['weight'] else 24
        pen=np.r_[np.full(24,cf['alpha']),np.full(n-24,cf['alpha']/cf['weight'] if cf['weight'] else 0)]
        np.testing.assert_allclose(z[:,:n].T@(z[:,:n]@b[k,:n]+c-y)+pen*b[k,:n],0,atol=1e-9)
        if not cf['weight']:
            np.testing.assert_allclose(b[k,:24],np.linalg.solve(z[:,:24].T@z[:,:24]+cf['alpha']*np.eye(24),z[:,:24].T@(y-y.mean())),atol=1e-10)
            assert np.count_nonzero(b[k,24:])==0

def test_conditional_synthetic_signal():
    x,_=sample(120);z,t=fit_transform(x,'bilinear');y=.7+2*z[:,40+7*16+3]
    b,c=fit_heads(z,y,[{'alpha':1e-8,'weight':1.}]);np.testing.assert_allclose(z@b.T+c,y[:,None],atol=1e-6)
    raw=np.arange(80,dtype=float).reshape(2,40)
    q=second_order(raw,'bilinear');assert q[1,7*16+3]==raw[1,7]*raw[1,24+3]

def test_train_only_state_and_normalization():
    x,_=sample();z,t=fit_transform(x,'bilinear');q=second_order(np.c_[z[:,:24],z[:,24:40]*np.sqrt(2.)],'bilinear')
    np.testing.assert_allclose(z[:,40:],((q-t['second_mean'])/t['second_scale'])/np.sqrt(48.),atol=1e-12)
    before=np.array(t['second_mean']).copy();transform(x+50,t);np.testing.assert_array_equal(before,t['second_mean'])
    np.testing.assert_allclose(np.sum(z[:,24:].var(0)),16.,atol=1e-10)

def test_constant_features_and_shapes():
    x,y=sample();x[:,0]=2.;x[:,30]=3.;z,t=fit_transform(x,'bilinear')
    assert np.isfinite(z).all();b,c=fit_heads(z,y,[{'alpha':10.,'weight':1.}]);assert np.isfinite(b).all()
    with pytest.raises(ValueError):fit_transform(x,'td_quadratic')
    with pytest.raises(ValueError):second_order(np.ones((3,39)),'bilinear')

def test_same_fixed_tie_break():
    cfg=configs();assert len(cfg)==25
    assert choose(np.ones(25),np.ones(25),cfg,range(25))==4
    assert choose(np.ones(25),np.ones(25),cfg,range(15,20))==19
