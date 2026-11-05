import copy
import numpy as np
import pytest
from assets import align

def fixture():
    meta=[dict(sample_id=f's{i}',cohort='a',guide='AUG',target57='UUC') for i in range(3)]
    x=np.arange(12).reshape(3,4);return meta,x

def test_align_by_identity_not_order():
    m,x=fixture();np.testing.assert_array_equal(align([m[2],m[0]],m,x),x[[2,0]])

def test_reject_changed_sequence():
    m,x=fixture();r=copy.deepcopy(m[:1]);r[0]['guide']='CGG'
    with pytest.raises(ValueError):align(r,m,x)

def test_reject_duplicate_identity():
    m,x=fixture()
    with pytest.raises(ValueError):align([m[0],m[0]],m,x)
    with pytest.raises(ValueError):align(m,[m[0],m[0],m[2]],x)

def test_reject_nonfinite_or_missing():
    m,x=fixture();x=x.astype(float);x[0,0]=np.nan
    with pytest.raises(ValueError):align(m,m,x)
    with pytest.raises(KeyError):align([dict(m[0],sample_id='missing')],m,x)
