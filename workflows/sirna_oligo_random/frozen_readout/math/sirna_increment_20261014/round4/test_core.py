import os,sys
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from core import configs,fit_transform,transform,fit_heads,penalty,score,choose
def sample():
    rng=np.random.default_rng(42);return rng.normal(size=(80,216)),rng.normal(size=80)
def test_zero_and_uniform_equivalence():
    x,y=sample();z,_=fit_transform(x);cfg=configs();b,c=fit_heads(z,y,cfg)
    for k,d in enumerate(cfg):
        n=24 if d['weight']==0 else 40
        ref=np.linalg.solve(z[:,:n].T@z[:,:n]+np.diag(penalty(d)[:n]),z[:,:n].T@(y-y.mean()))
        np.testing.assert_allclose(b[k,:n],ref,atol=1e-10,rtol=1e-10)
        assert np.count_nonzero(b[k,n:])==0
def test_pca_reference_and_train_stats():
    x,_=sample();z,t=fit_transform(x);a=(x[:,24:]-t['src_mean'])/t['src_scale'];a-=a.mean(0)
    _,_,v=np.linalg.svd(a,full_matrices=False)
    np.testing.assert_allclose(t['components'].T@t['components'],v[:16].T@v[:16],atol=1e-9)
    np.testing.assert_allclose(z.mean(0),0,atol=1e-12)
    before={k:np.array(v).copy() for k,v in t.items()};transform(x+100,t)
    for k,v in t.items():np.testing.assert_array_equal(v,before[k])
def test_single_model_equivariance():
    x,y=sample();z,t=fit_transform(x);b,c=fit_heads(z,y,configs())
    assert z.shape==(80,40) and b.shape==(25,40)
    expected=z@b.T+c;rot=z.copy();rot[:,24:]*=-1
    bb,cc=fit_heads(rot,y,configs());np.testing.assert_allclose(expected,rot@bb.T+cc,atol=1e-10)
def test_fixed_selection_and_no_extra_energy():
    cfg=configs();assert len(cfg)==25 and [d['weight'] for d in cfg[15:20]]==[1]*5
    pc=np.ones(25);ms=np.ones(25);assert choose(pc,ms,cfg,range(25))==4
    x,y=sample()
    try:fit_transform(x[:,:215])
    except ValueError:pass
    else:raise AssertionError('Wrong dimension accepted')
def test_constant_features_are_finite():
    x,y=sample();x[:,0]=2;x[:,30]=3;z,t=fit_transform(x);b,c=fit_heads(z,y,configs())
    assert np.isfinite(z).all() and np.isfinite(b).all() and t['td_scale'][0]==1 and t['src_scale'][6]==1
