import importlib.util
from pathlib import Path
import numpy as np
import pytest
s=importlib.util.spec_from_file_location('_background_test',Path(__file__).with_name('background.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


def arrays():
    rng=np.random.default_rng(38);b=rng.normal(size=(40,7));f=rng.normal(size=(40,192));return b-b.mean(0),f-f.mean(0)


def test_independent_augmented_lstsq():
    b,f=arrays();a,_,_=m.fit_map(b,f);expected=np.linalg.lstsq(np.r_[b,10*np.eye(7)],np.r_[f,np.zeros((7,192))],rcond=None)[0]
    np.testing.assert_allclose(a,expected,atol=1e-13)


def test_normal_equations_and_gram_psd():
    b,f=arrays();a,g,c=m.fit_map(b,f)
    np.testing.assert_allclose((b.T@b+100*np.eye(7))@a,b.T@f,atol=1e-12);np.testing.assert_allclose(g,a.T@b.T@b@a,atol=1e-12)
    assert c['normal_error']<1e-10 and np.linalg.eigvalsh(g)[0]>-1e-10


def test_orthogonal_background_coordinates():
    b,f=arrays();q=np.linalg.qr(np.random.default_rng(39).normal(size=(7,7)))[0]
    a,g,_=m.fit_map(b,f);aa,gg,_=m.fit_map(b@q,f)
    np.testing.assert_allclose(b@a,b@q@aa,atol=1e-13);np.testing.assert_allclose(g,gg,atol=1e-12)


def test_validation_cannot_change_mapping():
    b,f=arrays();a,g,_=m.fit_map(b,f);before=a.copy();validation=np.ones((4,7))*1e6;validation@a
    np.testing.assert_array_equal(a,before);np.testing.assert_array_equal(g,(b@a).T@(b@a))


def test_zero_background_prediction():
    b,f=arrays();a,g,c=m.fit_map(np.zeros_like(b),f);assert not a.any() and not g.any();assert c['alpha']==100


def test_invalid_inputs():
    b,f=arrays()
    for bb,ff in ((b,f[:-1]),(np.full_like(b,np.nan),f),(b+1,f),(b,np.ones_like(f)),(b[:0],f[:0])):
        with pytest.raises(ValueError):m.fit_map(bb,ff)


def test_fixed_permutation_identity():
    ids=[f's{i}' for i in range(80)];p=m.permutation(ids,'Hu',0)
    np.testing.assert_array_equal(p,m.permutation(ids,'Hu',0));assert sorted(p)==list(range(80))
    with pytest.raises(ValueError):m.permutation(['a','a'],'Hu',0)


def test_pooled_r2_is_ratio_of_sums():
    b,f=arrays();a,_,_=m.fit_map(b,f);pred=b@a;parts=[m.statistics(f[:10],pred[:10]),m.statistics(f[10:],pred[10:])]
    g=m.aggregate(parts);full=m.statistics(f,pred);np.testing.assert_allclose(g['R2'],full['R2'])
    with pytest.raises(ValueError):m.statistics(np.zeros((3,192)),np.zeros((3,192)))


def test_all_qualification_conditions_required():
    def record(name,a,b,c):return dict(setting=name,conditions={k:dict(R2=v) for k,v in zip(('paired_initial','TD24','shuffled_initial'),(a,b,c))})
    yes=[record('a',.3,.1,-.1),record('b',.4,.2,-.2),record('c',0.,.1,-.2)]
    assert m.qualify(yes)['qualified'];yes[2]=record('c',-.01,.1,-.2);assert not m.qualify(yes)['qualified']
    yes[2]=record('c',.05,.1,-.2);yes[1]=record('b',.3,.28,-.2);assert not m.qualify(yes)['qualified']
