import unittest,tempfile
from pathlib import Path
import torch,numpy as np
from models import Baseline,loss
from train import save
import accessibility_metrics as am
class Tests(unittest.TestCase):
 def test_original_loss_parity(self):
  torch.manual_seed(42);p=torch.rand(3,60);y=torch.rand(3,60);y[1,2:20]=float('nan');l=torch.tensor([50,56,60])
  torch.testing.assert_close(loss(p,y,l),am.observed_loss(p,y,l,0,.5)[0])
 def test_resume(self):
  torch.manual_seed(42);m=Baseline('sequence');o=torch.optim.AdamW(m.parameters(),lr=.0003);x=torch.rand(8,4,60);y=torch.rand(8,60);l=torch.full((8,),60)
  def step(model,opt,i):
   opt.zero_grad();loss(model(x[i:i+2],l[i:i+2]),y[i:i+2],l[i:i+2]).backward();opt.step()
  for i in [0,2]:step(m,o,i)
  with tempfile.TemporaryDirectory() as tmp:
   f=Path(tmp)/'last.pt';save(f,dict(model=m.state_dict(),optimizer=o.state_dict(),cursor=4,rng=torch.get_rng_state()))
   for i in [4,6]:step(m,o,i)
   ck=torch.load(f,weights_only=False);n=Baseline('sequence');no=torch.optim.AdamW(n.parameters(),lr=.0003);n.load_state_dict(ck['model']);no.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng'])
   for i in [ck['cursor'],6]:step(n,no,i)
   for a,b in zip(m.parameters(),n.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
 def test_train_validation_membership(self):
  split=np.array([0,1,2,0,2,1]);ids=np.flatnonzero(split==0);val=np.flatnonzero(split==1)
  self.assertFalse(set(ids)&set(val));self.assertEqual(ids.tolist(),[0,3])
if __name__=='__main__':unittest.main()
