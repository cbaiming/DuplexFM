import os,sys,copy
from pathlib import Path
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pytest
from kernel import fit_transform,transform
from audit_kernel import expand


@pytest.mark.parametrize('view,kind,width',[('core_real','joint',216),('TDKernel','td',24)])
def test_independent_train_and_unseen_reconstruction(view,kind,width):
    rng=np.random.default_rng(4321);x=rng.normal(size=(83,width));other=rng.normal(size=(17,width))
    z,state=fit_transform(x,kind)
    np.testing.assert_allclose(expand(x,x,state,view),z,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(expand(x,other,state,view),transform(other,state),atol=1e-10,rtol=1e-10)


def test_audit_rejects_bandwidth_and_foreign_landmarks():
    x=np.random.default_rng(42).normal(size=(83,24));_,state=fit_transform(x,'td')
    bad=copy.deepcopy(state);bad['kernel']['bandwidth']*=1.1
    with pytest.raises(AssertionError):expand(x,x,bad,'TDKernel')
    bad=copy.deepcopy(state);bad['kernel']['indices'][0]=82
    with pytest.raises(AssertionError):expand(x,x,bad,'TDKernel')
