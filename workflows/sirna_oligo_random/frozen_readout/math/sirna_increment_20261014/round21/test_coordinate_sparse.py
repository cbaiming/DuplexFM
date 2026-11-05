import numpy as np
import pytest
from coordinate_sparse import transform,profile,solve_profile,minimum_subgradient,fit_head,deployment,ALPHA

def data():
    rng=np.random.default_rng(42);z=rng.normal(size=(95,216));z-=z.mean(0)
    return z,.2*z[:,0]+.3*z[:,30]+rng.normal(size=95)*.1+.5

def test_dense_equivalent_to_joint_ridge():
    z,y=data();b,c,s=fit_head(z,y,'dense')
    expected=np.linalg.solve(z.T@z+ALPHA*np.eye(216),z.T@(y-y.mean()))
    np.testing.assert_allclose(b,expected,atol=1e-12);assert s['converged'] and c==y.mean()

def test_lambda_max_closes_source():
    z,y=data();h,q,t,coupling,c=profile(z,y);b,steps,ok=solve_profile(h,q,max(abs(q)))
    assert ok and steps==0 and np.count_nonzero(b)==0
    np.testing.assert_allclose(t,np.linalg.solve(z[:,:24].T@z[:,:24]+ALPHA*np.eye(24),z[:,:24].T@(y-c)))

def test_sparse_joint_kkt():
    z,y=data();b,c,s=fit_head(z,y,'sparse');g=z.T@(z@b+c-y)+ALPHA*b
    err=np.r_[g[:24],np.where(b[24:]!=0,g[24:]+s['lambda_value']*np.sign(b[24:]),np.sign(g[24:])*np.maximum(abs(g[24:])-s['lambda_value'],0))]
    assert s['converged'] and max(abs(err))<1e-6 and 0<s['source_nonzero']<192
    h,*_=profile(z,y);assert np.linalg.eigvalsh(h)[0]>=ALPHA-1e-10

def test_deployment_and_trace():
    z,y=data();st=dict(td_mean=z[:,:24].mean(0),td_scale=z[:,:24].std(0),src_mean=z[:,24:].mean(0),src_scale=z[:,24:].std(0))
    x=transform(z,st);b,c,_=fit_head(x,y,'sparse');coef,offset=deployment(st,b,c)
    np.testing.assert_allclose(z@coef+offset,x@b+c,atol=1e-12)
    np.testing.assert_allclose(np.var(x[:,24:],axis=0).sum(),16.,atol=1e-12)

def test_heldout_does_not_refit():
    z,_=data();st=dict(td_mean=np.zeros(24),td_scale=np.ones(24),src_mean=np.zeros(192),src_scale=np.ones(192))
    np.testing.assert_array_equal(transform(z[:1],st),transform(np.r_[z[:1],np.ones((2,216))*10000],st)[:1])

def test_deterministic():
    z,y=data();a,c,sa=fit_head(z,y,'sparse');b,d,sb=fit_head(z,y,'sparse')
    np.testing.assert_array_equal(a,b);assert c==d and sa==sb

def test_invalid_inputs():
    z,y=data()
    with pytest.raises(ValueError):fit_head(z,np.ones(len(y)),'sparse')
    with pytest.raises(ValueError):fit_head(z+1,y,'sparse')
    with pytest.raises(ValueError):fit_head(z,y,'unknown')
    with pytest.raises(ValueError):solve_profile(np.eye(2),np.ones(2),-1)

def test_independent_coordinate_descent_certificate():
    z,y=data();h,q,_,_,_=profile(z,y);lam=.1*max(abs(q));b,_,ok=solve_profile(h,q,lam);cd=np.zeros(192)
    for _ in range(1000):
        for j in range(192):
            r=q[j]-h[j]@cd+h[j,j]*cd[j];cd[j]=np.sign(r)*max(abs(r)-lam,0)/h[j,j]
        if max(abs(minimum_subgradient(h,q,cd,lam)))<1e-10:break
    objective=lambda v:.5*v@h@v-q@v+lam*abs(v).sum();sg=minimum_subgradient(h,q,b,lam)
    assert ok and abs(objective(b)-objective(cd))<=sg@sg/(2*ALPHA)+1e-12
    np.testing.assert_allclose(b,cd,atol=1e-8)
