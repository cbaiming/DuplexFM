import sys
from pathlib import Path
import numpy as np
from scipy.optimize._numdiff import approx_derivative
sys.path.insert(0,str(Path(__file__).resolve().parent))
from router_core import targets,objective,fit_gate,route

def test_analytic_gradient_matches_finite_difference():
    r=np.random.default_rng(42);z=r.normal(size=(30,20));q=r.dirichlet(np.ones(7),size=30);theta=r.normal(size=147)*.1
    loss,g=objective(theta,z,q);num=approx_derivative(lambda t:objective(t,z,q)[0],theta,method='3-point').ravel()
    np.testing.assert_allclose(g,num,atol=2e-9,rtol=2e-7)

def test_uniform_cost_targets_and_deterministic_hard_choice():
    q,tau=targets(np.arange(20.),np.repeat(np.arange(20.)[:,None],7,axis=1))
    np.testing.assert_allclose(q,1/7);assert tau==1e-8
    theta,receipt=fit_gate(np.zeros((20,20)),q)
    assert receipt['success'];np.testing.assert_array_equal(route(np.zeros((10,20)),theta),np.zeros(10,int))

def test_convex_gate_improves_its_own_training_objective():
    r=np.random.default_rng(42);z=r.normal(size=(80,20));logits=z[:,:7];q=np.exp(logits-logits.max(1,keepdims=True));q/=q.sum(1,keepdims=True)
    theta,rec=fit_gate(z,q);b=np.array(rec['initial_bias']);initial=np.r_[np.zeros(140),b]
    assert rec['objective']<=objective(initial,z,q)[0]+1e-12 and rec['max_gradient']<1e-7
    assert len(set(route(z,theta)))>1

def test_cost_labels_use_relative_error_and_are_normalized():
    y=np.arange(8.)/10;p=np.c_[y,y+.1,y-.2,y+.3,y+.4,y-.5,y+.6]
    q,t=targets(y,p);np.testing.assert_allclose(q.sum(1),1)
    assert np.all(np.argmax(q,axis=1)==0) and t>0
