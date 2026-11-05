import numpy as np
import pytest
from group_nuisance import keys,groups,project,fit,qualify


def sequences():
    rng=np.random.default_rng(42)
    s=''.join(rng.choice(list('ACGT'),120));t=''.join(rng.choice(list('ACGT'),57))
    return [s[:57],s[25:82],s[50:107],t]


def design(n=80,p=216):
    z=np.random.default_rng(6).normal(size=(n,p));return z-z.mean(0)


def test_canonical_keys_and_transitive_components():
    seq=sequences();rc=seq[0].translate(str.maketrans('ACGT','TGCA'))[::-1]
    assert keys(seq[0])==keys(rc) and keys(seq[0])==keys(seq[0].replace('T','U'))
    assert keys(seq[0]).isdisjoint(keys(seq[2]))
    g,m=groups(seq);np.testing.assert_array_equal(g,[0,0,0,1]);assert m['sizes']==[3,1]


def test_partition_order_equivariance():
    seq=sequences();g,_=groups(seq);order=np.array([3,2,0,1]);gg,_=groups([seq[i] for i in order])
    np.testing.assert_array_equal(gg[:,None]==gg[None,:],g[order,None]==g[None,order])


def test_invalid_inputs():
    with pytest.raises(ValueError):groups(['A'*57])
    with pytest.raises(ValueError):keys('A'*56+'N')
    with pytest.raises(ValueError):keys('A'*58)
    with pytest.raises(ValueError):project(np.ones((5,4)),np.zeros(5,int))
    with pytest.raises(ValueError):project(design(8,4),np.ones(8,int))


def test_projection_gram_and_trace():
    z=design(80,12);g=np.arange(80)%7;k=7
    G=np.eye(k)[g];P=G@np.diag(1/np.bincount(g))@G.T;J=np.ones((80,80))/80;c=79/(79-.5*6)
    M=c*(np.eye(80)-.5*(P-J));out=project(z,g)
    np.testing.assert_allclose(out.T@out,z.T@M@z,atol=1e-10)
    assert np.trace((np.eye(80)-J)@M)==pytest.approx(79)
    np.testing.assert_allclose(out.mean(0),0,atol=1e-12)


def test_single_group_and_singletons_identity():
    z=design(80,12)
    np.testing.assert_array_equal(project(z,np.zeros(80,int)),z)
    np.testing.assert_array_equal(project(z,np.arange(80)),z)


def test_group_label_permutation():
    z=design(80,12);g=np.arange(80)%5
    np.testing.assert_allclose(project(z,g),project(z,4-g),atol=1e-12)


def test_independent_graph_and_projection():
    from verify import independent_groups,weighted_design
    seq=sequences();g,m=groups(seq);gg,mm=independent_groups(seq)
    np.testing.assert_array_equal(g,gg);assert m==mm
    z=design(4,12);np.testing.assert_allclose(project(z,g),weighted_design(z,g),atol=1e-12)


def test_TD_matches_independent_lstsq():
    z=design();g=np.arange(80)%9;y=np.random.default_rng(5).normal(size=80)
    b,c,cert=fit(z,y,g,False);zz=project(z[:,:24],g);yc=project(y-y.mean(),g)
    expected=np.linalg.lstsq(np.r_[zz,10*np.eye(24)],np.r_[yc,np.zeros(24)],rcond=None)[0]
    np.testing.assert_allclose(b,expected,atol=1e-11);assert c==pytest.approx(y.mean()) and cert['certified']
    from verify import audit_TD
    assert audit_TD(zz,yc,b,cert)[0]


def test_native_certificate_and_group_free_prediction():
    from verify import certify
    z=design();g=np.arange(80)%9;y=np.random.default_rng(9).normal(size=80)
    b,c,cert=fit(z,y,g,True);zz=project(z,g);yy=project(y-y.mean(),g)+y.mean()
    assert certify(zz,yy,b,c,dict(cert,matrix_shape=list(cert['matrix_shape'])),False)[0]
    xv=design(17);batch=xv@b+c;one=np.array([row@b+c for row in xv]);np.testing.assert_allclose(batch,one,atol=1e-12)


def test_residual_decomposition():
    from group_nuisance import residual_stats
    z=design();g=np.arange(80)%8;y=np.random.default_rng(15).normal(size=80);b=np.zeros(216);c=y.mean()
    s=residual_stats(z,y,b,c,g,True);factor=79/(79-.5*7)
    assert s['centered_between_SSE']+s['centered_within_SSE']==pytest.approx(s['centered_residual_SSE'])
    assert s['transformed_residual_SSE']==pytest.approx(factor*(s['centered_within_SSE']+.5*s['centered_between_SSE']))


def test_qualification_fixed():
    records=[]
    for name in ('Hu','Mix','Tak'):
        records.extend([dict(setting=name,source='core_real',delta_PCC_vs_reference=.006,MSE_reduction_vs_reference_pct=.1,metrics=dict(PCC=.6,MSE=.02)),
                        dict(setting=name,source='TD_only',delta_PCC_vs_reference=.001,MSE_reduction_vs_reference_pct=.1,metrics=dict(PCC=.55,MSE=.03))])
    assert qualify(records,True)['qualified']
    records[0]['delta_PCC_vs_reference']=-.006;assert not qualify(records,True)['qualified']
    records[0]['delta_PCC_vs_reference']=.006;assert not qualify(records,False)['qualified']
