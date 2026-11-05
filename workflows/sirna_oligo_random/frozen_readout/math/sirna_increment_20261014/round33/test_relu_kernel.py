import numpy as np
import pytest
from scipy.integrate import quad
from relu_kernel import arc,embedding,fit,predict,ridge_from_gram


def sample(dim=40):
    rng=np.random.default_rng(42);z=rng.normal(size=(80,dim));z-=z.mean(0)
    y=.3*z[:,0]+np.maximum(z[:,-1],0)*.2+rng.normal(size=len(z))*.03
    return z,y


def test_exact_geometry():
    x=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,0.]])
    k=arc(x,x)
    np.testing.assert_allclose(np.diag(k),[1,1,1,0],atol=1e-14)
    assert k[0,1]==pytest.approx(1/np.pi) and abs(k[0,2])<1e-14
    assert not np.count_nonzero(k[3])


def test_gaussian_relu_integral_independent_quadrature():
    # Polar Gaussian integral: 2 E[(w.x)+ (w.y)+] = 2/pi integral cos+ cos+.
    for theta in (.2,.9,2.2):
        points=sorted({np.pi/2,3*np.pi/2,(theta+np.pi/2)%(2*np.pi),(theta+3*np.pi/2)%(2*np.pi)})
        expected=2/np.pi*quad(lambda phi:max(np.cos(phi),0)*max(np.cos(phi-theta),0),0,2*np.pi,points=points,epsabs=1e-12)[0]
        actual=arc(np.array([[1.,0.]]),np.array([[np.cos(theta),np.sin(theta)]]))[0,0]
        assert actual==pytest.approx(expected,abs=1e-11)


def test_psd_homogeneity_and_rotation():
    rng=np.random.default_rng(9);x=rng.normal(size=(50,7));x[1]=x[0];x[2]=0
    q,_=np.linalg.qr(rng.normal(size=(7,7)));k=arc(x,x)
    assert np.linalg.eigvalsh((k+k.T)/2).min()>-1e-10
    np.testing.assert_allclose(arc(2*x,3*x),6*k,atol=1e-12)
    np.testing.assert_allclose(arc(x@q,x@q),k,atol=1e-12)


def test_centering_and_training_norm():
    z,y=sample();m=fit(z,y)
    assert m['numerically_qualified'] and m['mean_aux_diagonal']==pytest.approx(16)
    assert abs(m['sum_dual'])<1e-12 and m['centered_kernel_row_mean_max']<1e-12
    p=predict(z,m);assert np.mean(p)==pytest.approx(np.mean(y))
    assert np.mean((p-y)**2)==pytest.approx(m['train_MSE'])


def test_linear_kernel_dual_matches_primal():
    z,y=sample(24);c,b,err,limit=ridge_from_gram(z@z.T,y)
    beta=np.linalg.lstsq(np.r_[z,10*np.eye(24)],np.r_[y-y.mean(),np.zeros(24)],rcond=None)[0]
    np.testing.assert_allclose(z.T@c,beta,atol=1e-12)
    assert b==y.mean() and err<=limit


def test_frozen_deployment_independent_batch():
    z,y=sample();m=fit(z,y)
    old={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in m.items()}
    one=predict(z[:1],m);batch=predict(np.r_[z[:1],100*z[1:]],m)
    np.testing.assert_allclose(one,batch[:1],atol=1e-12)
    for k in m:np.testing.assert_array_equal(m[k],old[k])


def test_response_affine_and_training_permutation():
    z,y=sample();m=fit(z,y);m2=fit(z,3*y+2)
    np.testing.assert_allclose(predict(z,m2),3*predict(z,m)+2,atol=1e-12)
    order=np.random.default_rng(3).permutation(len(z));m3=fit(z[order],y[order])
    np.testing.assert_allclose(predict(z,m3),predict(z,m),atol=1e-12)


def test_zero_features_td_and_source_interfaces():
    y=np.arange(80)/80
    for dim in (24,40):
        z=np.zeros((80,dim));m=fit(z,y)
        assert m['aux_scale']==0
        np.testing.assert_allclose(predict(z,m),y.mean(),atol=1e-14)
    z,y=sample(24);assert embedding(z).shape==(80,24)


def test_contract():
    z,y=sample()
    for zz,yy in ((z+1,y),(z,np.ones(len(y))),(z[:,:32],y),(np.zeros((2001,40)),np.arange(2001))):
        with pytest.raises(ValueError):fit(zz,yy)
    with pytest.raises(ValueError):arc(np.ones((2,3)),np.ones((2,4)))
    with pytest.raises(ValueError):embedding(np.full((20,40),np.nan))
