import sys
from pathlib import Path
import numpy as np
from scipy.optimize._numdiff import approx_derivative
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_router_numeric_compat import hessian,polish
from router_core import objective,fit_gate

def fixture():
    r=np.random.default_rng(211);z=r.normal(size=(55,20));q=r.dirichlet(np.ones(7),size=55)
    return z,q

def test_exact_hessian_matches_numerical_gradient_jacobian():
    z,q=fixture();theta=np.arange(147)*.0001
    expected=approx_derivative(lambda t:objective(t,z,q)[1],theta,method='3-point')
    np.testing.assert_allclose(hessian(theta,z,q),expected,atol=1e-9,rtol=1e-6)

def test_polish_reduces_gradient_without_changing_objective_or_gauge():
    z,q=fixture();theta,rec=fit_gate(z,q);theta+=np.linspace(-1,1,147)*1e-5
    start=objective(theta,z,q)[0];bmean=theta[140:].mean();fixed,info=polish(theta,z,q)
    assert info['gradient_after']<1e-7 and info['newton_steps']>=1
    assert objective(fixed,z,q)[0]<=start+1e-14
    np.testing.assert_allclose(fixed[140:].mean(),bmean,atol=1e-12)
