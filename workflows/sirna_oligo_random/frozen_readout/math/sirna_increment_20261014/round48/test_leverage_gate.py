import numpy as np
import pytest
from leverage_gate import fit_gate,apply_gate,apply_design,fit_head,leverage,weights,qualify


def fixture(seed=1,n=90,p=12):
    rng=np.random.default_rng(seed);x=rng.normal(size=(n,p));return x-x.mean(0)


def test_precision_and_effective_degrees():
    x=fixture();g=fit_gate(x);h=leverage(x,g['precision'])
    np.testing.assert_allclose(g['precision'],np.linalg.inv(x.T@x+100*np.eye(x.shape[1])),atol=1e-12)
    singular=np.linalg.svd(x,compute_uv=False)
    assert h.sum()==pytest.approx(np.sum(singular**2/(singular**2+100)))
    from verify import gate_audit
    xv=fixture(8,n=17)*3
    _,th,tw=apply_gate(x,g);_,vh,vw=apply_gate(xv,g)
    model=dict(gate_precision=g['precision'],gate_mean=g['mean'],train_leverage=th,train_weight=tw,valid_leverage=vh,valid_weight=vw)
    result=gate_audit(x,xv,model,{k:v for k,v in g.items() if k not in ('precision','mean')})
    np.testing.assert_allclose(result[0],apply_gate(x,g)[0],atol=1e-10)


def test_quantile_and_radius_cap():
    x=fixture();g=fit_gate(x);h=leverage(x,g['precision']);w=weights(h,g['threshold'])
    assert g['threshold']==sorted(h)[int(np.ceil(.9*(len(x)-1)))]
    assert np.max(w*w*h)<=g['threshold']+1e-12
    np.testing.assert_allclose(w[h<=g['threshold']],1)


def test_centering_and_inlier_identity():
    x=fixture();g=fit_gate(x);z,h,w=apply_gate(x,g)
    np.testing.assert_allclose(z.mean(0),0,atol=1e-12)
    np.testing.assert_allclose((z+g['mean'])[h<=g['threshold']],x[h<=g['threshold']],atol=1e-12)


def test_zero_training_geometry():
    x=np.zeros((20,3));g=fit_gate(x);z,h,w=apply_gate(np.array([[0,0,0],[10,1,2.]]),g)
    assert g['threshold']==0;np.testing.assert_allclose(w,[1,0]);np.testing.assert_allclose(z,0)


def test_batch_independence_and_state_immutable():
    g=fit_gate(fixture());p=g['precision'].copy();m=g['mean'].copy();x=fixture(8,n=17)
    batch=apply_gate(x,g)[0];one=np.vstack([apply_gate(row[None,:],g)[0] for row in x])
    np.testing.assert_allclose(batch,one,atol=1e-12);np.testing.assert_array_equal(g['precision'],p);np.testing.assert_array_equal(g['mean'],m)


def test_orthogonal_geometry_invariance():
    x=fixture();q=np.linalg.qr(np.random.default_rng(42).normal(size=(12,12)))[0]
    a=fit_gate(x);b=fit_gate(x@q);z,h,w=apply_gate(x,a);zz,hh,ww=apply_gate(x@q,b)
    np.testing.assert_allclose(h,hh,atol=1e-12);np.testing.assert_allclose(w,ww,atol=1e-12);np.testing.assert_allclose(z@q,zz,atol=1e-12)


def test_anisotropy_not_unit_norm():
    x=np.array([[10.,0],[-10.,0],[0,1],[0,-1]])
    g=fit_gate(x);h=leverage(np.eye(2)*3,g['precision'])
    assert h[0]<h[1] and np.linalg.norm([3,0])==np.linalg.norm([0,3])


def test_TD_head_matches_augmented_lstsq():
    z=fixture(n=90,p=216);y=np.random.default_rng(7).normal(size=90);gate=fit_gate(z[:,:24])
    beta,c,cert=fit_head(z,y,gate,False);design,_,_=apply_design(z,gate,False)
    expected=np.linalg.lstsq(np.r_[design,10*np.eye(24)],np.r_[y-y.mean(),np.zeros(24)],rcond=None)[0]
    np.testing.assert_allclose(beta,expected,atol=1e-12);assert c==pytest.approx(y.mean()) and cert['certified']
    from verify import audit_TD
    assert audit_TD(design,y,beta,c,cert)[0]


def test_source_head_and_native_deployment():
    z=fixture(n=90,p=216);y=np.random.default_rng(8).normal(size=90);gate=fit_gate(z[:,24:])
    beta,c,cert=fit_head(z,y,gate,True);design,h,w=apply_design(z,gate,True)
    explicit=z[:,:24]@beta[:24]+w*(z[:,24:]@beta[24:])-gate['mean']@beta[24:]+c
    np.testing.assert_allclose(design@beta+c,explicit,atol=1e-12);np.testing.assert_array_equal(design[:,:24],z[:,:24])
    assert cert['certified'] and cert['alpha']==100 and cert['lambda_ratio']==.1
    from verify import certify
    disk_certificate=dict(cert,matrix_shape=list(cert['matrix_shape']))
    assert certify(design,y,beta,c,disk_certificate,False)[0]


def test_invalid_geometry_rejected():
    with pytest.raises(ValueError):fit_gate(np.full((5,3),np.nan))
    with pytest.raises(ValueError):fit_gate(np.empty((0,2)))
    with pytest.raises(ValueError):weights(np.array([-1.]),1.)
    with pytest.raises(ValueError):leverage(np.ones((2,2)),np.eye(3))
    with pytest.raises(ValueError):apply_gate(np.ones((2,4)),fit_gate(fixture(p=3)))


def test_gate_and_control_qualification():
    records=[]
    for s in ('Hu','Mix','Tak'):
        records.append(dict(setting=s,source='core_real',delta_PCC_vs_reference=.006,MSE_reduction_vs_reference_pct=.2,metrics=dict(PCC=.6,MSE=.02)))
        records.append(dict(setting=s,source='TD_only',delta_PCC_vs_reference=.001,MSE_reduction_vs_reference_pct=.1,metrics=dict(PCC=.55,MSE=.03)))
    assert qualify(records,True)['qualified']
    records[0]['delta_PCC_vs_reference']=-.006;assert not qualify(records,True)['qualified']
    records[0]['delta_PCC_vs_reference']=.006
    for r in records:
        if r['source']=='TD_only':r['metrics']=dict(PCC=.7,MSE=.01)
    assert not qualify(records,True)['qualified']
