import time
import numpy as np
import pytest
from scipy.optimize import minimize
from native_kernel import transform,embedding,fit,predict,arc,base,qualify


def sample():
    g=np.random.default_rng(20261015);x=g.normal(size=(72,216));x-=x.mean(0)
    y=.3*x[:,0]+.2*np.maximum(x[:,-1],0)+g.normal(size=len(x))*.04
    return x,y


def test_penalized_correlation_direction_is_ridge():
    rng=np.random.default_rng(9);z=rng.normal(size=(50,7));z-=z.mean(0);y=rng.normal(size=50);y-=y.mean()
    h=z.T@z+3*np.eye(7);q=z.T@y;b=np.linalg.solve(h,q);direction=b/np.sqrt(b@h@b)
    result=minimize(lambda a:-q@a,np.ones(7)*.01,jac=lambda a:-q,method='SLSQP',
        constraints=[dict(type='ineq',fun=lambda a:1-a@h@a,jac=lambda a:-2*h@a)],options={'ftol':1e-12,'maxiter':500})
    assert result.success
    np.testing.assert_allclose(result.x,direction,atol=2e-7)
    assert np.corrcoef(z@b,z@direction)[0,1]==pytest.approx(1.,abs=1e-12)


def test_ols_correlation_direction_and_response_affine():
    rng=np.random.default_rng(7);z=rng.normal(size=(60,6));z-=z.mean(0);y=rng.normal(size=60);y-=y.mean()
    b=np.linalg.lstsq(z,y,rcond=None)[0];q=z.T@y;g=z.T@z
    np.testing.assert_allclose(g@b,q,atol=1e-12)
    # Cauchy-Schwarz in G metric bounds correlation; OLS attains the bound.
    upper=np.sqrt(q@np.linalg.solve(g,q)/(y@y))
    assert np.corrcoef(z@b,y)[0,1]==pytest.approx(upper,abs=1e-12)
    for _ in range(20):
        a=rng.normal(size=6)
        assert q@a/np.sqrt((a@g@a)*(y@y))<=upper+1e-12


def test_native_embedding_budgets_and_saved_scalers_only():
    z,y=sample();s=dict(td_mean=np.zeros(24),td_scale=np.ones(24),src_mean=np.zeros(192),src_scale=np.ones(192))
    np.testing.assert_array_equal(transform(z,s),z)
    u=embedding(z);np.testing.assert_allclose(u[:,:24],z[:,:24]/np.sqrt(24))
    np.testing.assert_allclose(u[:,24:],z[:,24:]/np.sqrt(192))
    assert u.shape==(72,216)
    before={k:v.copy() for k,v in s.items()};transform(np.r_[z[:1],100*z[1:]],s)
    for k in s:np.testing.assert_array_equal(s[k],before[k])


def test_psd_center_trace_and_independent_augmented_features():
    z,y=sample();m=fit(z,y);u=embedding(z);raw=arc(u,u);raw=(raw+raw.T)/2
    c=np.eye(len(z))-np.ones((len(z),len(z)))/len(z);kc=c@raw@c
    gram=z[:,:24]@z[:,:24].T+m['aux_scale']*kc
    eig,vec=np.linalg.eigh((gram+gram.T)/2);assert eig.min()>-1e-9
    feature=vec*np.sqrt(np.maximum(eig,0))
    beta=np.linalg.lstsq(np.r_[feature,10*np.eye(len(z))],np.r_[y-y.mean(),np.zeros(len(z))],rcond=None)[0]
    np.testing.assert_allclose(predict(z,m),feature@beta+y.mean(),atol=1e-10)
    assert m['mean_aux_diagonal']==pytest.approx(16) and m['numerically_qualified']


def test_training_order_response_and_deployment_batches():
    z,y=sample();m=fit(z,y);p=predict(z,m)
    q=np.random.default_rng(42).permutation(len(z));m2=fit(z[q],3*y[q]+2)
    np.testing.assert_allclose(predict(z,m2),3*p+2,atol=1e-10)
    np.testing.assert_allclose(predict(z[:1],m),predict(np.r_[z[:1],z[1:]*100],m)[:1],atol=1e-12)


def test_native_source_zero_reduces_exactly_to_unchanged_td_control():
    z,y=sample();z[:,24:]=0.;native=fit(z,y);td=base.fit(z[:,:24],y)
    np.testing.assert_allclose(predict(z,native),base.predict(z[:,:24],td),atol=1e-12)


def test_zero_and_contracts():
    z,y=sample();m=fit(np.zeros_like(z),y);assert m['aux_scale']==0
    np.testing.assert_allclose(predict(np.zeros_like(z),m),y.mean(),atol=1e-14)
    for x,response in ((z[:,:40],y),(z+1,y),(z,np.ones(len(y))),(np.zeros((2001,216)),np.arange(2001))):
        with pytest.raises(ValueError):fit(x,response)
    with pytest.raises(TimeoutError):fit(z,y,deadline=time.monotonic()-1)
    with pytest.raises(ValueError):embedding(np.full((20,216),np.nan))


def test_gate_requires_compression_and_strong_native_comparisons():
    rows=[dict(source='core_real',setting=s,delta_PCC_vs_R4=.006,MSE_reduction_vs_R4_pct=.1,
        delta_PCC_vs_R33=.001,MSE_reduction_vs_R33_pct=.1,delta_PCC_vs_R25=.001,MSE_reduction_vs_R25_pct=.1,
        metrics=dict(PCC=.6,MSE=.09)) for s in ('Huesken','Mixset','Takayuki')]
    td=[dict(setting=r['setting'],metrics=dict(PCC=.59,MSE=.1)) for r in rows]
    assert qualify(rows,True,td)['qualified']
    for r in rows:r['MSE_reduction_vs_R25_pct']=-.1
    assert not qualify(rows,True,td)['qualified']
