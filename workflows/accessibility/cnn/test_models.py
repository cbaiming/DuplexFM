import unittest,torch,numpy as np
from models import Baseline,loss
class Tests(unittest.TestCase):
 def test_padding(self):
  torch.manual_seed(42)
  for kind in ['sequence','structure']:
   m=Baseline(kind);c=4+(kind=='structure');x=torch.randn(2,c,60);l=torch.tensor([50,60]);y=m(x,l)
   x[0,:,50:]=999;z=m(x,l);torch.testing.assert_close(y,z);self.assertTrue((y[0,50:]==0).all())
   torch.testing.assert_close(y[0,:50],m(x[:1,:,:50],l[:1])[0])
 def test_loss_mask_gradient(self):
  p=torch.full((2,60),.5,requires_grad=True);y=torch.full((2,60),.1);y[0,2]=float('nan');l=torch.tensor([50,60])
  z=loss(p,y,l);z.backward();self.assertAlmostEqual(z.item(),.4,places=6);self.assertEqual(p.grad[0,2],0);self.assertTrue((p.grad[0,50:]==0).all())
 def test_parameter_updates(self):
  for kind in ['sequence','structure']:
   m=Baseline(kind);x=torch.rand(4,4+(kind=='structure'),60);l=torch.tensor([50,53,60,59]);y=torch.rand(4,60)
   loss(m(x,l),y,l).backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.parameters()))
if __name__=='__main__':unittest.main()
