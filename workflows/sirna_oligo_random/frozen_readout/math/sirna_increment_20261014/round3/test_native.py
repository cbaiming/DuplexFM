import os,sys
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from native import model,loss,predict,parameters,P12,P13,normalize_fit,normalize
from train import fit,state_hash,select,ENDPOINTS
torch.set_num_threads(1)

def test_architecture_and_anchor_parity():
    a=model(P12);b=model(P13)
    assert parameters(a)==881 and parameters(b)==6387
    for k,v in a.anchor.state_dict().items():assert torch.equal(v,b.anchor.state_dict()[k])
    assert all(p.device.type=='cpu' for p in b.parameters())

def test_independent_forward_and_loss():
    torch.manual_seed(91);x=torch.randn(9,184);y=torch.rand(9)
    for cond in (P12,P13):
        m=model(cond);m.eval();out=m(x)
        a=m.anchor(x[:,160:]).squeeze(-1)
        r=torch.zeros(9) if cond==P12 else .5*torch.tanh(m.rho_raw)*m.concat_residual(x[:,:160]).squeeze(-1)
        torch.testing.assert_close(out['final_logit'],a+r)
        expected=((torch.sigmoid(a+r)-y)**2).mean()+(.03*(r*r).mean() if cond==P13 else 0)
        torch.testing.assert_close(loss(cond,out,y),expected)

def test_source_invariance_td_and_gradient():
    x=torch.randn(10,184);m=model(P12);m.eval();p=predict(m,x.numpy())
    x[:,:160]*=100;np.testing.assert_array_equal(p,predict(m,x.numpy()))
    m=model(P13);loss(P13,m(x),torch.linspace(.1,.9,10)).backward()
    assert m.concat_residual[1].weight.grad.abs().sum()>0 and m.anchor[1].weight.grad.abs().sum()>0

def test_normalizer_train_only_and_replay():
    x=np.random.default_rng(42).normal(size=(35,184)).astype('float32');y=np.linspace(.05,.95,35)
    t=normalize_fit(x[:25]);np.testing.assert_allclose(t['mean'],x[:25].mean(0,dtype=np.float64).astype('float32'))
    a=fit(x[:25],y[:25],P13,8,probe=x[25:]);b=fit(x[:25],y[:25],P13,8,probe=x[25:]+100)
    assert state_hash(a['state'])==state_hash(b['state']) and a['initial_sha256']!=a['final_sha256']

def test_fixed_selection_ties():
    y=np.arange(10,dtype=float);p=np.tile(y[:,None],(1,5));k,_,_=select(y,p)
    assert k==0 and ENDPOINTS==(8,16,32,64,100)
