import time
import numpy as np
import pytest
from scipy.integrate import quad
from joint_kernel import fit,predict,even_kernel,embed,kernel_state,native,qualify


def sample(dim=216):
    rng=np.random.default_rng(20261015);x=rng.normal(size=(65,dim));x-=x.mean(0)
    if dim==216:x[:,24:]/=np.sqrt(12)
    y=.4+x[:,0]*.3+x[:,-1]*.2+np.abs(x[:,1])*.1+rng.normal(size=len(x))*.03
    return x,y


def test_even_kernel_integral_psd_and_sign_symmetry():
    for theta in (.3,1.1,2.4):
        points=sorted({np.pi/2,3*np.pi/2,(theta+np.pi/2)%(2*np.pi),(theta+3*np.pi/2)%(2*np.pi)})
        expected=quad(lambda p:abs(np.cos(p)*np.cos(p-theta)),0,2*np.pi,points=points,epsabs=1e-12)[0]/(2*np.pi)
        actual=even_kernel(np.array([[1.,0.]]),np.array([[np.cos(theta),np.sin(theta)]]))[0,0]
        assert actual==pytest.approx(expected,abs=1e-11)
    x=np.random.default_rng(42).normal(size=(30,7));k=even_kernel(x,x)
    np.testing.assert_allclose(k,even_kernel(-x,x),atol=1e-12)
    assert np.linalg.eigvalsh((k+k.T)/2).min()>-1e-10
    np.testing.assert_allclose(np.diag(k),np.sum(x*x,axis=1)/2,atol=1e-12)


def test_no_kernel_recovers_existing_native_head():
    x,y=sample();m=fit(x,y,aux_trace=0);b,c,info=native.fit(x,y)
    np.testing.assert_allclose(m['beta'],b,atol=1e-9)
    np.testing.assert_allclose(predict(x,m),x@b+c,atol=1e-9)
    assert m['certificate']['numerically_qualified']


def test_no_nuclear_penalty_matches_joint_feature_ridge():
    x,y=sample();m=fit(x,y,ratio=0);k,_=kernel_state(x)
    ev,vec=np.linalg.eigh((k+k.T)/2);f=vec*np.sqrt(np.maximum(ev,0));design=np.c_[x,f]
    b=np.linalg.lstsq(np.r_[design,10*np.eye(design.shape[1])],np.r_[y-y.mean(),np.zeros(design.shape[1])],rcond=None)[0]
    np.testing.assert_allclose(predict(x,m),design@b+y.mean(),atol=1e-10)
    np.testing.assert_allclose(m['beta'],b[:x.shape[1]],atol=1e-10)


def test_independent_full_feature_convex_certificate():
    x,y=sample();m=fit(x,y);cert=m['certificate'];k,_=kernel_state(x);ev,vec=np.linalg.eigh((k+k.T)/2)
    f=vec*np.sqrt(np.maximum(ev,0));design=np.c_[x,f];b=np.r_[m['beta'],f.T@m['dual']];q=design.T@(y-y.mean());h=design.T@design+100*np.eye(design.shape[1])
    sg=q-h@b;dual=np.zeros_like(sg);source=sg[24:216];lam=cert['lambda_value'];norm=np.linalg.norm(source.reshape(6,32),2)
    if norm>lam:source*=lam/norm
    dual[24:216]=source
    penalty=lam*np.linalg.svd(b[24:216].reshape(6,32),compute_uv=False).sum()
    primal=.5*b@h@b-q@b+penalty;lower=-.5*(q-dual)@np.linalg.solve(h,q-dual)
    assert -1e-8<=primal-lower<=1e-9*cert['gap_scale']+1e-10 and cert['numerically_qualified']
    np.testing.assert_allclose(sg[:24],0,atol=1e-9);np.testing.assert_allclose(sg[216:],0,atol=1e-9)


def test_TD_only_independent_combined_kernel_solution():
    x,y=sample(24);m=fit(x,y);k,_=kernel_state(x);full=x@x.T+k
    c=np.linalg.solve(full+100*np.eye(len(x)),y-y.mean())
    np.testing.assert_allclose(predict(x,m),full@c+y.mean(),atol=1e-11)
    assert m['certificate']['lambda_value']==0 and m['certificate']['numerically_qualified']


def test_batch_response_and_training_order():
    x,y=sample();m=fit(x,y);p=predict(x,m)
    np.testing.assert_allclose(predict(x[:1],m),predict(np.r_[x[:1],x[1:]*100],m)[:1],atol=1e-12)
    permutation=np.random.default_rng(42).permutation(len(x));other=fit(x[permutation],3*y[permutation]+2)
    np.testing.assert_allclose(predict(x,other),3*p+2,atol=3e-6)


def test_six_pool_shared_basis_invariance():
    x,y=sample();rng=np.random.default_rng(3);q=np.linalg.qr(rng.normal(size=(32,32)))[0]
    other=np.c_[x[:,:24],(x[:,24:].reshape(-1,6,32)@q).reshape(-1,192)]
    np.testing.assert_allclose(predict(x,fit(x,y)),predict(other,fit(other,y)),atol=3e-7)


def test_contracts_deadline_and_zero_inputs():
    x,y=sample();m=fit(np.zeros_like(x),y);np.testing.assert_allclose(predict(np.zeros_like(x),m),y.mean(),atol=1e-12)
    for z,v in ((x[:,:40],y),(x+1,y),(x,np.ones(len(y)))):
        with pytest.raises(ValueError):fit(z,v)
    with pytest.raises(ValueError):fit(x,y,aux_trace=-1)
    with pytest.raises(TimeoutError):fit(x,y,deadline=time.monotonic()-1)


def test_gate_preserves_strong_native_and_kernel_comparisons():
    rows=[]
    for name in ('Huesken','Mixset','Takayuki'):
        rows += [dict(setting=name,source='core_real',delta_PCC_vs_R25=.006,MSE_reduction_vs_R25_pct=.1,
            delta_PCC_vs_R40=.001,MSE_reduction_vs_R40_pct=.1,metrics=dict(PCC=.6,MSE=.09)),
            dict(setting=name,source='TD_only',metrics=dict(PCC=.59,MSE=.1))]
    assert qualify(rows,True)['qualified']
    rows[4]['delta_PCC_vs_R25']=-.006
    assert not qualify(rows,True)['qualified']
