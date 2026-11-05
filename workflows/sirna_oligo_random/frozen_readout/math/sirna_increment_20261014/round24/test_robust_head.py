import numpy as np
import pytest
from scipy.optimize import minimize
from robust_head import fit, qualify, ALPHA


def sample():
    r = np.random.default_rng(42)
    z = r.normal(size=(90, 5)); g = np.r_[np.zeros(55, int), np.ones(35, int)]
    y = .2 + z @ np.array([.1, -.2, .3, .05, -.1]) + r.normal(size=90) * np.where(g, .6, .1)
    return z, y, g


def independent(z, y, g, q):
    w = np.where(g == 0, len(y) * q / sum(g == 0), len(y) * (1-q) / sum(g == 1))
    x = np.c_[np.ones(len(y)), z]
    augment = np.diag(np.r_[0., np.full(z.shape[1], np.sqrt(ALPHA))])
    return np.linalg.lstsq(np.r_[np.sqrt(w)[:, None]*x, augment],
                           np.r_[np.sqrt(w)*y, np.zeros(x.shape[1])], rcond=None)[0]


def test_balanced_weighted_least_squares():
    z,y,g=sample(); b,c,s=fit(z,y,g,'balanced')
    np.testing.assert_allclose(np.r_[c,b],independent(z,y,g,.5),atol=1e-12)
    assert s['numerically_qualified'] and s['weighted_ridge_solves']==1


def test_equal_group_sizes_recover_original_ridge():
    z,y,_=sample();g=np.tile([0,1],len(y)//2)
    b,c,_=fit(z,y,g,'balanced');mu=z.mean(0);ym=y.mean()
    expected=np.linalg.solve((z-mu).T@(z-mu)+ALPHA*np.eye(z.shape[1]),(z-mu).T@(y-ym))
    np.testing.assert_allclose(b,expected,atol=1e-12)
    assert abs(c-(ym-mu@expected))<1e-12


def test_robust_global_certificate_and_epigraph():
    z,y,g=sample(); b,c,s=fit(z,y,g,'robust'); x=np.c_[np.ones(len(y)),z]; n=len(y)
    p=np.r_[0.,np.full(z.shape[1],ALPHA)]
    def fun(v): return v[-1]+np.dot(p,v[:-1]**2)
    def jac(v): return np.r_[2*p*v[:-1],1.]
    constraints=[]
    for k in (0,1):
        xx,yy=x[g==k],y[g==k]
        constraints.append(dict(type='ineq',fun=lambda v,xx=xx,yy=yy:v[-1]-n*np.mean((xx@v[:-1]-yy)**2),
             jac=lambda v,xx=xx,yy=yy:np.r_[-2*n/len(yy)*xx.T@(xx@v[:-1]-yy),1.]))
    initial=np.r_[np.zeros(x.shape[1]),n*np.mean(y*y)*3]
    other=minimize(fun,initial,jac=jac,constraints=constraints,method='SLSQP',options=dict(maxiter=400,ftol=1e-10))
    assert other.success,other.message
    np.testing.assert_allclose(other.x[:-1],np.r_[c,b],atol=2e-6)
    np.testing.assert_allclose(np.r_[c,b],independent(z,y,g,s['group0_weight']),atol=1e-12)
    assert s['numerically_qualified'] and s['duality_gap']<=s['gap_limit']


def test_equal_intercept_only_groups():
    z=np.zeros((20,0));g=np.repeat([0,1],10);y=2*g
    b,c,s=fit(z,y,g,'robust')
    np.testing.assert_allclose([c,s['group0_weight']],[1,.5],atol=1e-11)
    assert len(b)==0 and s['location']=='interior'


def test_boundary_high_irreducible_variance():
    z=np.zeros((20,0));g=np.repeat([0,1],10);y=np.r_[np.zeros(10),np.tile([-10.,10.],5)]
    _,c,s=fit(z,y,g,'robust')
    assert c==0 and s['group0_weight']==0 and s['location']=='lower_boundary'
    assert s['duality_gap']==0


def test_swap_groups_and_row_order():
    z,y,g=sample();b,c,s=fit(z,y,g,'robust');order=np.random.default_rng(43).permutation(len(y))
    bb,cc,ss=fit(z[order],y[order],1-g[order],'robust')
    np.testing.assert_allclose(np.r_[c,b],np.r_[cc,bb],atol=1e-11)
    assert abs(s['group0_weight']+ss['group0_weight']-1)<1e-9


def test_label_shift():
    z,y,g=sample();b,c,_=fit(z,y,g,'robust');bb,cc,_=fit(z,y+2,g,'robust')
    np.testing.assert_allclose(b,bb,atol=1e-11);assert abs(cc-c-2)<1e-10


def test_constant_and_collinear_features():
    z,y,g=sample();z=np.c_[z,np.ones(len(z)),z[:,0]]
    b,c,s=fit(z,y,g,'robust')
    assert s['numerically_qualified'] and abs(b[-2])<1e-10
    np.testing.assert_allclose(np.r_[c,b],independent(z,y,g,s['group0_weight']),atol=1e-11)


def test_invalid_contract():
    z,y,g=sample()
    with pytest.raises(ValueError):fit(z,y,np.zeros(len(y)),'robust')
    with pytest.raises(ValueError):fit(z,y,g,'invalid')
    bad=z.copy();bad[0,0]=np.nan
    with pytest.raises(ValueError):fit(bad,y,g,'robust')


def test_fixed_qualification_does_not_promote_secondary():
    records=[]
    for setting in ('a','b','c'):
        for mode in ('robust','balanced'):
            records.append(dict(setting=setting,source='core_real',mode=mode,
                metrics=dict(PCC=.5,MSE=1.),delta_PCC_vs_R4=.006,MSE_reduction_vs_R4_pct=.1))
    assert qualify(records,True)['qualified']
    records[0]['delta_PCC_vs_R4']=-.006
    assert not qualify(records,True)['qualified']
    assert not qualify(records,False)['qualified']
