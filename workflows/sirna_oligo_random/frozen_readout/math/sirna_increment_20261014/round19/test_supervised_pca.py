import numpy as np
import pytest
from supervised_pca import screening,select_indices,fit_transform,transform,fit_head,deployment


def fixture():
    rng=np.random.default_rng(43);td=rng.normal(size=(200,24));source=rng.normal(size=(200,192))
    source[:,0]=td[:,0]*10+rng.normal(size=200)*.01
    y=td[:,0]+source[:,1]+.1*rng.normal(size=200)
    x=np.c_[td,source]
    ref=dict(td_mean=td.mean(0),td_scale=td.std(0),src_mean=source.mean(0),src_scale=source.std(0))
    return x,y,ref


def test_partial_scores_match_independent_svd():
    x,y,ref=fixture();td=x[:,:24];source=x[:,24:];s=screening(td,source,y)
    u,e,_=np.linalg.svd(td-td.mean(0),full_matrices=False);q=u[:,e>e[0]*1e-10]
    zr=source-source.mean(0);zr-=q@(q.T@zr);yr=y-y.mean();yr-=q@(q.T@yr)
    expected=zr.T@yr/np.sqrt((yr@yr)*np.sum(zr*zr,axis=0))
    np.testing.assert_allclose(s['partial_correlation'],expected,atol=1e-10,rtol=1e-10)
    assert abs(s['partial_correlation'][1])>.98
    assert abs(s['partial_correlation'][0])<.2


def test_ties_and_fixed_random_control():
    np.testing.assert_array_equal(select_indices(np.zeros(192),'conditional'),np.arange(32))
    a=select_indices(np.zeros(192),'random');b=select_indices(np.ones(192),'random')
    np.testing.assert_array_equal(a,b);assert len(set(a))==32


def test_pca_spectrum_independent_svd():
    x,y,ref=fixture();z,s=fit_transform(x,y,ref,'conditional')
    raw=((x[:,24:]-ref['src_mean'])/ref['src_scale'])[:,s['selected']];raw-=raw.mean(0)
    _,ev,v=np.linalg.svd(raw,full_matrices=False)
    np.testing.assert_allclose(s['eigenvalues'],ev[:16]**2/len(x),rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(s['components'].T@s['components'],v[:16].T@v[:16],atol=1e-9)
    assert z.shape==(200,40) and s['rank16_valid']


def test_head_equations_and_single_vector_deployment():
    x,y,ref=fixture();z,s=fit_transform(x,y,ref,'conditional');b,c=fit_head(z,y);coef,offset=deployment(s,b,c)
    np.testing.assert_allclose((z.T@z+100*np.eye(40))@b,z.T@(y-y.mean()),atol=1e-10)
    np.testing.assert_allclose(z@b+c,x@coef+offset,atol=1e-10)
    assert np.count_nonzero(coef[24:])<=32


def test_affine_label_screening_invariance():
    x,y,ref=fixture();z,a=fit_transform(x,y,ref,'conditional');zz,b=fit_transform(x,-3*y+10,ref,'conditional')
    np.testing.assert_array_equal(a['selected'],b['selected']);np.testing.assert_allclose(z,zz,atol=1e-10)


def test_heldout_does_not_mutate_fitted_transform():
    x,y,ref=fixture();z,s=fit_transform(x,y,ref,'conditional');before=s['selected'].copy()
    transform(np.full((4,216),1e4),s)
    np.testing.assert_array_equal(before,s['selected']);np.testing.assert_array_equal(z,transform(x,s))


def test_constant_feature_and_singular_td():
    x,y,ref=fixture();x[:,1]=x[:,0];x[:,26]=0
    ref['td_mean']=x[:,:24].mean(0);ref['td_scale']=x[:,:24].std(0)
    ref['src_mean']=x[:,24:].mean(0);ref['src_scale']=x[:,24:].std(0);ref['src_scale'][2]=1
    z,s=fit_transform(x,y,ref,'conditional')
    assert s['td_rank']==23 and s['partial_correlation'][2]==0 and np.isfinite(z).all()


def test_invalid_inputs_rejected():
    x,y,ref=fixture()
    with pytest.raises(ValueError):fit_transform(x[:,:100],y,ref,'conditional')
    with pytest.raises(ValueError):fit_transform(x,y[:-1],ref,'conditional')
    with pytest.raises(ValueError):fit_transform(x,y,ref,'not_registered')
