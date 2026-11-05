import torch
from torch import nn
class Baseline(nn.Module):
 """Same local CNN readout; structural model adds label-free u1 probability."""
 def __init__(self,kind):
  super().__init__();self.kind=kind
  channels=[4+(kind=='structure'),32,32,32]
  self.blocks=nn.ModuleList(nn.Conv1d(a,b,5,padding=2) for a,b in zip(channels,channels[1:]))
  self.output=nn.Conv1d(32,1,1)
 def forward(self,x,lengths):
  mask=(torch.arange(x.shape[-1],device=x.device)[None,:]<lengths[:,None])[:,None,:]
  z=x.masked_fill(~mask,0)
  for block in self.blocks:z=torch.relu(block(z)).masked_fill(~mask,0)
  return self.output(z)[:,0].sigmoid().masked_fill(~mask[:,0],0)
def loss(pred,y,lengths):
 mask=torch.isfinite(y)&(torch.arange(y.shape[1],device=y.device)[None,:]<lengths[:,None])
 if (mask.sum(1)==0).any():raise ValueError('no observed labels')
 return ((pred.masked_fill(~mask,0)-y.masked_fill(~mask,0)).abs().sum(1)/mask.sum(1)).mean()

