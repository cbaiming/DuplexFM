"""Synthetic checks of the exact production functions, without public models."""
import ast,io,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
tree=ast.parse((HERE/'export_cpu.py').read_text())
subset=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('manual_design_pool','array','resource_guard')],type_ignores=[])
scope=dict(np=np,Path=Path,io=io,os=os)
exec(compile(subset,str(HERE/'export_cpu.py'),'exec'),scope)

class Contract(unittest.TestCase):
    def test_design_orientation_and_padding(self):
        # Unique coordinate signal distinguishes reversed guide, wrong endpoint,
        # PAD offsets, channel/region order, and accidentally averaged PAD cells.
        b,g,t,c=2,21,60,32
        h=torch.full((b,c,g,t),99999.)
        valid=torch.zeros((b,g,t),dtype=torch.bool)
        gp=torch.tensor([[-1]+list(range(19))+[-1]]*b)
        tp=torch.tensor([[-1]+list(range(57))+[-1,-1]]*b)
        for i in range(b):
            for x in range(19):
                for y in range(57):
                    h[i,:,x+1,y+1]=torch.arange(c)*1000+i*100+x*10+y
                    valid[i,x+1,y+1]=True
        class Fusion:
            def grid_features(self,batch):return h,valid
        class Model:
            fusion=Fusion()
            def interaction_inputs(self,batch):return batch
        got=scope['manual_design_pool'](Model(),dict(guide_pos=gp,target_pos=tp))
        for i in range(b):
            regions=[list(range(1,8)),list(range(8,12)),[0]+list(range(12,19))]
            expected=[]
            for region in regions:
                v=torch.tensor([37+9*x for x in region],dtype=torch.float)
                expected.extend([torch.arange(c)*1000+i*100+v.mean(),torch.arange(c)*1000+i*100+v.max()])
            torch.testing.assert_close(got[i],torch.cat(expected),rtol=0,atol=0)

    def test_invalid_design_coordinate_fails(self):
        class Model:
            def interaction_inputs(self,b):return b
            def grid_features(self,b):return torch.zeros(1,32,19,60),torch.zeros(1,19,60,dtype=torch.bool)
        model=Model();model.fusion=model
        with self.assertRaises(AssertionError):scope['manual_design_pool'](model,dict(guide_pos=torch.arange(19)[None],target_pos=torch.arange(60)[None]))

    def test_committed_array_not_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'array.npy';a=np.arange(20).reshape(4,5)
            scope['array'](p,a);original=p.read_bytes();scope['array'](p,a)
            with self.assertRaises(AssertionError):scope['array'](p,a+1)
            self.assertEqual(p.read_bytes(),original)
            np.testing.assert_array_equal(np.load(p,allow_pickle=False),a)

    def test_cpu_memory_guards(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}),patch.object(os,'sched_getaffinity',return_value=set(range(128)),create=True):
            p=Path(tmp);(p/'cpu.max').write_text('1400000 100000');(p/'memory.max').write_text(str(60<<30));(p/'memory.current').write_text(str(3<<30))
            self.assertEqual(scope['resource_guard'](p)['quota_cores'],14)
            (p/'cpu.max').write_text('200000 100000')
            with self.assertRaises(AssertionError):scope['resource_guard'](p)
            (p/'cpu.max').write_text('1400000 100000');(p/'memory.current').write_text(str(45<<30))
            with self.assertRaises(AssertionError):scope['resource_guard'](p)

if __name__=='__main__':unittest.main()
