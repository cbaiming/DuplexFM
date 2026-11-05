import hashlib,json
import numpy as np
import pytest
from reuse import load_fixed_sampling,stitch


def fixture(tmp_path):
    ids=['a','b','c','d'];g=np.array([0,0,1,2]);k=3;draws=2000
    (tmp_path/'groups_fixed.json').write_text(json.dumps(dict(ids=ids,labels=g.tolist(),global_components=k,input_only=True,before_numeric_labels=True)),encoding='utf-8')
    arrays=dict(labels=g,counts=np.tile([1,1,1],(draws,1)))
    for key,shape in {'observed_PCC':(4,6),'observed_MSE':(4,6),'leave_PCC':(k,4,6),'leave_MSE':(k,4,6),'bootstrap_PCC':(draws,4,6),'bootstrap_MSE':(draws,4,6)}.items():arrays[key]=np.broadcast_to(np.arange(6),shape).astype(float).copy()
    np.savez(tmp_path/'statistics.npz',**arrays)
    hashes={name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() for name in ('groups_fixed.json','statistics.npz')}
    return ids,hashes,arrays


def test_exact_reference_columns_and_stitch(tmp_path):
    ids,h,a=fixture(tmp_path);g,c,r=load_fixed_sampling(tmp_path,ids,h)
    np.testing.assert_array_equal(g,a['labels']);np.testing.assert_array_equal(c,a['counts'])
    for key,v in r.items():
        np.testing.assert_array_equal(v,a[key][...,[0,1]])
        new=np.broadcast_to([10,11,12,13],v.shape[:-1]+(4,));z=stitch(new,v)
        np.testing.assert_array_equal(z,np.broadcast_to([0,10,11,12,13,1],z.shape))


def test_reject_reordered_identity_and_changed_hash(tmp_path):
    ids,h,a=fixture(tmp_path)
    with pytest.raises(ValueError,match='identities'):load_fixed_sampling(tmp_path,ids[::-1],h)
    with pytest.raises(ValueError,match='hash'):load_fixed_sampling(tmp_path,ids,{**h,'statistics.npz':'0'*64})


def test_reject_invalid_saved_draw_contract(tmp_path):
    ids,h,a=fixture(tmp_path);a['counts'][0,0]=2;np.savez(tmp_path/'statistics.npz',**a)
    h['statistics.npz']=hashlib.sha256((tmp_path/'statistics.npz').read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='draws'):load_fixed_sampling(tmp_path,ids,h)


def test_nan_ineligible_reference_preserved_and_shape_guard():
    z=stitch(np.full((2,4),np.nan),np.full((2,2),np.nan));assert z.shape==(2,6) and np.isnan(z).all()
    with pytest.raises(ValueError):stitch(np.ones((2,4)),np.ones((3,2)))
    with pytest.raises(ValueError):stitch(np.ones((2,3)),np.ones((2,2)))
