"""SHA-locked corrected/BERT/native-spline fits; no test input in run_fit.
Caller binds one visible GPU. Importing this module does not import torch.
"""
from __future__ import annotations
import ast, contextlib, copy, hashlib, importlib.util, json, math, os, random, tempfile, time, types
from pathlib import Path
MODELS=('OligoFormer-corrected','BERT-siRNA','RNAiSpline')
POLICY={
 'OligoFormer-corrected':dict(epochs=120,batch=128,lr=3e-4,wd=1e-4,clip=1.,amp=True,patience=15,min_delta=1e-5,dropout=.02,eval_batch=128,ssl_epochs=0),
 'BERT-siRNA':dict(epochs=30,batch=100,lr=5e-5,wd=.01,clip=None,amp=False,patience=100,min_delta=0.,dropout=.1,eval_batch=128,ssl_epochs=0),
 'RNAiSpline':dict(epochs=50,batch=16,lr=1e-4,wd=1e-5,clip=1.,amp=False,patience=20,min_delta=0.,eval_batch=256,ssl_epochs=60,ssl_lr=1e-4,ssl_T0=50,ssl_Tmult=1,supervised_T0=10,supervised_Tmult=2)}
SOURCE_HASHES={
 'corrected_model':'13fac6ac99d2d5145b5e5fc87f8b7804f91fd5d6f2c8eef421e112c5189b6177',
 'bert_source':'4dd3b0f8c2eaa22bf06e5e22442d1816b72764de76709b81636e1bc17e092069',
 'spline_model':'f833a72eee03852275775efdae885c89b8e15421a9af9821a985d16c95528b3f',
 'spline_preprocess':'6750650efd611ca8ff5a93ddb424ee49706d5e81551669aca489f1ab16389eb1'}
DNABERT_HASHES={
 'config.json':'4263c6ae1df85a5d119bad7636a0a7701c17ad2bdd110a3bbbc801a5c5e6bd88',
 'vocab.txt':'d8f61c247fc10f9b8c7f6baf4d31876f30076d4aaf9f31ff47b9e68badd62e61',
 'pytorch_model.bin':'e1688c3a3f881daff02a42b487e04612c4ec9ef33f1f02fa5f4e67d7b9865ca9'}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8<<20),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def _atomic_bytes(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
 try:
  with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def write(path,value):_atomic_bytes(path,(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
def _save(path,value):
 import torch
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
 try:
  with os.fdopen(fd,'wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
  os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def _load(path):
 import torch
 return torch.load(path,map_location='cpu',weights_only=True)
def _checked(path,expected):
 path=Path(path).resolve()
 if sha(path)!=expected:raise ValueError('Source/asset SHA mismatch: '+str(path))
 return path
def _asset_path(assets,key):
 if key in assets:value=assets[key]
 elif key+'_path' in assets:value=assets[key+'_path']
 elif key.startswith('spline_'):value=Path(assets['spline_source_dir'])/(key.removeprefix('spline_')+'.py')
 else:raise KeyError('Missing asset '+key)
 if isinstance(value,dict):
  if value.get('sha256',SOURCE_HASHES[key])!=SOURCE_HASHES[key]:raise ValueError('Unregistered model source')
  value=value['path']
 return _checked(value,SOURCE_HASHES[key])
def source_manifest(model,assets):
 if model not in MODELS:raise ValueError('Unknown model '+str(model))
 keys={'OligoFormer-corrected':('corrected_model',),'BERT-siRNA':('bert_source',),'RNAiSpline':('spline_model','spline_preprocess')}[model]
 result={key:dict(path=str(_asset_path(assets,key)),sha256=SOURCE_HASHES[key]) for key in keys}
 if model=='BERT-siRNA':
  folder=Path(assets['dnabert_dir']).resolve()
  for name,expected in DNABERT_HASHES.items():result['dnabert/'+name]=dict(path=str(_checked(folder/name,expected)),sha256=expected)
 return result
def _module(path,prefix):
 spec=importlib.util.spec_from_file_location(prefix+'_'+sha(path)[:16],path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def _seed(seed):
 import numpy as np
 import torch
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def _rng():
 import numpy as np
 import torch
 n=np.random.get_state()
 return dict(python=random.getstate(),numpy=(n[0],n[1].tolist(),n[2],n[3],n[4]),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def _set_rng(state):
 import numpy as np
 import torch
 random.setstate(state['python']);n=state['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),*n[2:]));torch.set_rng_state(state['cpu'])
 if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])
def _state_digest(state):
 h=hashlib.sha256()
 for key,value in state.items():
  value=value.detach().cpu().contiguous();h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
 return h.hexdigest()
def _cpu_state(model):return {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
def _cpu_tree(value):
 import torch
 if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
 if isinstance(value,dict):return {key:_cpu_tree(item) for key,item in value.items()}
 if isinstance(value,list):return [_cpu_tree(item) for item in value]
 if isinstance(value,tuple):return tuple(_cpu_tree(item) for item in value)
 return copy.deepcopy(value)
def _basis(self,x):
 x=x.clamp(*self.grid_range).unsqueeze(-1);t=self.knots
 basis=((x>=t[:-1])&(x<t[1:])).to(x.dtype);basis[...,-1]+=(x[...,0]==t[-1]).to(x.dtype)
 for k in range(1,self.spline_order+1):
  n=len(t)-k-1;dl=t[k:k+n]-t[:n];dr=t[k+1:k+1+n]-t[1:1+n]
  left=(x-t[:n])/dl.clamp_min(1e-6)*basis[...,:n]*(dl>1e-6)
  right=(t[k+1:k+1+n]-x)/dr.clamp_min(1e-6)*basis[...,1:n+1]*(dr>1e-6);basis=left+right
 return basis
def _verify_basis(module):
 import torch
 torch.manual_seed(13);result=[];original=module.CoxDeBoorBSpline.compute_basis_functions
 for grid_range in ((-1,1),(-2,2)):
  layer=module.CoxDeBoorBSpline(12,7,grid_range=grid_range)
  values=torch.cat([torch.randn(24,12)*3,layer.knots[:,None].expand(-1,12)])
  x=values.clone().requires_grad_();y=values.clone().requires_grad_();a,b=original(layer,x),_basis(layer,y);weight=torch.randn_like(a)
  (a*weight).sum().backward();(b*weight).sum().backward()
  if not torch.allclose(a,b,atol=2e-6,rtol=1e-5) or not torch.allclose(x.grad,y.grad,atol=2e-6,rtol=1e-5):raise ValueError('Native spline basis parity failed')
  result.append(dict(range=grid_range,forward_max_delta=float((a-b).abs().max()),gradient_max_delta=float((x.grad-y.grad).abs().max())))
 module.CoxDeBoorBSpline.compute_basis_functions=_basis;return result
def _bert(source,folder):
 import torch
 from transformers import BertConfig,BertModel
 cfg=BertConfig(**read(folder/'config.json'));cfg.output_attentions=True;cfg._attn_implementation='eager'
 def backbone(*args,**kwargs):
  model=BertModel(cfg);weights=_load(folder/'pytorch_model.bin');weights={k.removeprefix('bert.'):v for k,v in weights.items() if k.startswith('bert.')}
  result=model.load_state_dict(weights,strict=False)
  if not set(result.missing_keys)<={'pooler.dense.weight','pooler.dense.bias'} or not set(result.unexpected_keys)<={'embeddings.position_ids'}:raise ValueError('DNABERT unexpected state keys')
  return model
 namespace=dict(torch=torch,nn=torch.nn,BertConfig=types.SimpleNamespace(from_pretrained=lambda *a,**k:cfg),BertModel=types.SimpleNamespace(from_pretrained=backbone))
 tree=ast.parse(source.read_text(encoding='utf-8-sig'));nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='siRNABertRegressor']
 if len(nodes)!=1:raise ValueError('Native BERT class not unique')
 exec(compile(ast.Module(nodes,type_ignores=[]),str(source),'exec'),namespace);return namespace['siRNABertRegressor']()
def configure(device='cuda:0'):
 import torch
 if str(device).startswith('cuda'):
  if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count()!=1:raise RuntimeError('Bind exactly one explicit GPU per worker')
  if str(device) not in ('cuda','cuda:0'):raise ValueError('Worker must use its visible cuda:0')
 os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');torch.set_num_threads(2)
 torch.backends.cudnn.enabled=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
def _indices(arrays,values,role):
 import numpy as np
 if values is None:return np.empty(0,dtype=np.int64)
 result=np.asarray(values,dtype=np.int64)
 if result.ndim!=1 or len(set(result.tolist()))!=len(result):raise ValueError(role+' indices not unique 1D')
 if len(result) and (result.min()<0 or result.max()>=len(arrays['ids'])):raise ValueError(role+' index outside data')
 return result
def prepare_arrays(rows,oligo_features=None):
 import numpy as np
 result=dict(ids=[str(row['sample_id']) for row in rows],labels=np.asarray([float(row['label']) for row in rows],dtype=np.float32),guide=[row['guide'] for row in rows],target19=[row['target19'] for row in rows],target57=[row['target57'] for row in rows])
 alphabet={base:i for i,base in enumerate('AUCG')}
 result['guide_tokens']=np.asarray([[alphabet[base] for base in row['guide']] for row in rows],dtype=np.int64)
 result['target_tokens']=np.asarray([[alphabet[base] for base in row['target57']] for row in rows],dtype=np.int64)
 if oligo_features is not None:
  if isinstance(oligo_features,np.ndarray):result['features']=np.asarray(oligo_features,dtype=np.float32)
  else:
   for key in ('guide_tokens','target_tokens','features'):
    if key in oligo_features:result[key]=oligo_features[key]
   if 'features' not in result and all(key in oligo_features for key in ('guide_fm','target_fm','td24')):result['features']=np.c_[oligo_features['guide_fm'],oligo_features['target_fm'],oligo_features['td24']].astype(np.float32)
  if result['features'].shape!=(len(rows),280):raise ValueError('Frozen Oligo features must have shape (n,280)')
 return result

POOL_COMPATIBILITY_REVISION='spline_AdaptiveMaxPool1d_output1_first_argmax_v1'

def _first_max_pool_class():
 """Construct a parameter-free local replacement; never patch torch.nn."""
 import torch
 class FirstMaximumPool1d(torch.nn.Module):
  def forward(self,x):
   if x.ndim not in (2,3) or x.shape[-1]<1:raise ValueError('Pool expects nonempty CxL or NxCxL input')
   return torch.max(x,dim=-1,keepdim=True).values
 return FirstMaximumPool1d

def _install_spline_pool_compatibility(model):
 """Only native Spline's two global 1D max pools; state and RNG unchanged."""
 import torch
 expected={'siRNA_cnn.inc.pool_layer','mRNA_cnn.inc.pool_layer'}
 found={name:module for name,module in model.named_modules() if isinstance(module,torch.nn.AdaptiveMaxPool1d)}
 if set(found)!=expected:raise ValueError('Unexpected native Spline adaptive max-pool topology')
 before=_state_digest(model.state_dict());pool_class=_first_max_pool_class();replaced=[]
 for name,module in found.items():
  output=module.output_size
  if output not in (1,(1,)) or module.return_indices:raise ValueError('Unregistered adaptive max-pool configuration')
  parent_name,child_name=name.rsplit('.',1);parent=model.get_submodule(parent_name)
  replacement=pool_class();replacement.train(module.training);parent._modules[child_name]=replacement;replaced.append(name)
 if _state_digest(model.state_dict())!=before:raise ValueError('Pool compatibility changed model state')
 return dict(revision=POOL_COMPATIBILITY_REVISION,replaced_modules=sorted(replaced),
             original='torch.nn.AdaptiveMaxPool1d(1,return_indices=False)',
             replacement='torch.max(x,dim=-1,keepdim=True).values',
             source_bytes_unchanged=True,model_state_unchanged=True,parameter_free=True,
             first_argmax_ties=True,qualification_required=True)

def verify_spline_pool_compatibility(device='cuda:0'):
 """Synthetic native/replacement forward, gradient and first-tie parity.

 Only the native reference operations run in a temporary deterministic=False
 context; strict flags and all RNG states are restored even after failure.
 The replacement forward/backward must execute under deterministic=True.
 """
 import torch
 rng=_rng();enabled=torch.are_deterministic_algorithms_enabled()
 warn_only=torch.is_deterministic_algorithms_warn_only_enabled();cases=[]
 generator=torch.Generator().manual_seed(17421)
 try:
  fixtures=[]
  for length in (1,2,19,33):
   fixtures.append(('random_L'+str(length),torch.randn(2,3,length,generator=generator)))
   fixtures.append(('zero_ties_L'+str(length),torch.zeros(2,3,length)))
   fixtures.append(('negative_constant_ties_L'+str(length),torch.full((2,3,length),-2.)))
  ties=torch.full((2,3,19),-3.);ties[...,0]=4.;ties[...,8]=4.;ties[...,-1]=4.
  fixtures.append(('first_middle_last_ties',ties))
  ties2=ties.clone();ties2[...,0]=-3.;fixtures.append(('middle_last_ties',ties2))
  fixtures.append(('unbatched_ties',ties[0]))
  fixtures.append(('noncontiguous',torch.randn(2,3,38,generator=generator)[...,::2]))
  pool_class=_first_max_pool_class()
  for name,values in fixtures:
   a=values.to(device).detach().requires_grad_();b=values.to(device).detach().requires_grad_()
   weights=torch.linspace(-1.25,2.5,steps=a.numel()//a.shape[-1],device=device).reshape(a.shape[:-1]+(1,))
   torch.use_deterministic_algorithms(False)
   native,indices=torch.nn.functional.adaptive_max_pool1d(a,1,return_indices=True)
   native_gradient=torch.autograd.grad((native*weights).sum(),a)[0]
   torch.use_deterministic_algorithms(True,warn_only=False)
   replacement=pool_class()(b);replacement_gradient=torch.autograd.grad((replacement*weights).sum(),b)[0]
   expected_indices=torch.max(b.detach(),dim=-1,keepdim=True).indices
   if not torch.equal(native,replacement) or not torch.equal(native_gradient,replacement_gradient) or not torch.equal(indices,expected_indices):
    raise ValueError('Native first-argmax pool parity failed: '+name)
   cases.append(dict(case=name,shape=list(values.shape),forward_exact=True,gradient_exact=True,first_argmax_indices_exact=True,
                     forward_max_abs_delta=float((native-replacement).abs().max()),gradient_max_abs_delta=float((native_gradient-replacement_gradient).abs().max())))
  return dict(passed=True,revision=POOL_COMPATIBILITY_REVISION,device=str(device),dtype='float32',cases=cases,
              replacement_strict_determinism=True,native_reference_only_temporary_deterministic_false=True,
              production_determinism_relaxed=False,RNG_restored=True)
 finally:
  torch.use_deterministic_algorithms(enabled,warn_only=warn_only);_set_rng(rng)

class Runner:
 """Per-fit update machine, including native SSL reconstruction heads."""
 def __init__(self,model,arrays,fit_indices,seed,assets,device='cuda:0',normalization=None):
  import numpy as np
  import torch
  self.name,self.policy,self.arrays=model,POLICY[model],arrays;self.fit=_indices(arrays,fit_indices,'fit');self.seed=int(seed)
  if not len(self.fit):raise ValueError('Empty fit')
  self.device=torch.device(device);self.normalization=normalization;self.sources=source_manifest(model,assets);self.basis_equivalence=None;self.pool_compatibility=None
  if model=='OligoFormer-corrected':
   raw=torch.as_tensor(np.asarray(arrays['features'],dtype=np.float32))
   if raw.shape!=(len(arrays['ids']),280):raise ValueError('Corrected Oligo needs 280 frozen RNAFM+TD24 features')
   if normalization is None:
    mean=raw[self.fit].mean(0);scale=raw[self.fit].std(0,unbiased=False).clamp_min(1e-6)
    self.normalization=dict(mean=mean.tolist(),scale=scale.tolist(),fit_ids=[arrays['ids'][int(i)] for i in self.fit])
   mean=torch.tensor(self.normalization['mean']);scale=torch.tensor(self.normalization['scale'])
   self.data=[torch.as_tensor(np.asarray(arrays['guide_tokens'],dtype=np.int64)),torch.as_tensor(np.asarray(arrays['target_tokens'],dtype=np.int64)),(raw-mean)/scale]
   mod=_module(Path(self.sources['corrected_model']['path']),'corrected_oligo');_seed(seed);self.model=mod.CorrectedOligoFormerSiRNA(.02).to(self.device)
  elif model=='BERT-siRNA':
   folder=Path(assets['dnabert_dir']);vocab={word:i for i,word in enumerate((folder/'vocab.txt').read_text().splitlines())};tokens=[]
   for guide in arrays['guide']:
    guide=guide.replace('U','T')
    if len(guide)!=19 or not set(guide)<=set('ACGT'):raise ValueError('BERT guide must be canonical 19 nt')
    tokens.append([vocab[word] for word in ['[CLS]']+[guide[i:i+6] for i in range(14)]+['[SEP]']])
   self.data=[torch.tensor(tokens,dtype=torch.long),torch.ones((len(tokens),16),dtype=torch.long)]
   _seed(seed);self.model=_bert(Path(self.sources['bert_source']['path']),folder).to(self.device)
  else:
   self.mod=_module(Path(self.sources['spline_model']['path']),'native_spline');self.prep=_module(Path(self.sources['spline_preprocess']['path']),'native_spline_prep');self.basis_equivalence=_verify_basis(self.mod)
   self.data=[torch.stack([self.prep.one_hot_encode_rna_sequence(s,19) for s in arrays[col]]) for col in ('guide','target19')]
   self.data.append(torch.tensor(np.asarray([self.prep.calculate_thermodynamic_feature_vector(s) for s in arrays['guide']]),dtype=torch.float32))
   _seed(seed);self.model=self.mod.SiRNAEfficacyClassificationModel().to(self.device);self.pool_compatibility=_install_spline_pool_compatibility(self.model)
  self.labels=torch.as_tensor(np.asarray(arrays['labels'],dtype=np.float32));self.sampler=torch.Generator().manual_seed(seed)
  self.state=dict(phase='ssl' if self.policy['ssl_epochs'] else 'supervised',epoch=0,cursor=0,order=None,updates=0,epoch_loss_sum=0.,epoch_batches=0,best=None,best_epoch=None,bad=0,history=[],ssl_history=[])
  self.pre=None;self.optimizer=None;self.scheduler=None;self._phase_optimizer(self.state['phase']);self.initial_digest=_state_digest(self.model.state_dict())
 def _phase_optimizer(self,phase):
  import torch
  self.pre=None;self.scheduler=None
  if self.name=='RNAiSpline':
   if phase=='ssl':
    self.pre=self.mod.PretrainingModel(self.model).to(self.device);self.optimizer=torch.optim.Adam(self.pre.parameters(),lr=1e-4)
    self.scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=50,T_mult=1)
   else:
    self.optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-4,weight_decay=1e-5);self.scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=10,T_mult=2)
  else:
   cls=torch.optim.AdamW if self.name=='OligoFormer-corrected' else torch.optim.Adam
   self.optimizer=cls(self.model.parameters(),lr=self.policy['lr'],weight_decay=self.policy['wd'])
 def state_dict(self):
  return dict(model=_cpu_state(self.model),pretraining=_cpu_state(self.pre) if self.pre is not None else None,optimizer=_cpu_tree(self.optimizer.state_dict()),scheduler=_cpu_tree(self.scheduler.state_dict()) if self.scheduler else None,sampler=self.sampler.get_state(),rng=_rng(),training=copy.deepcopy(self.state),normalization=copy.deepcopy(self.normalization),initial_digest=self.initial_digest)
 def load_state_dict(self,checkpoint):
  self._phase_optimizer(checkpoint['training']['phase']);self.model.load_state_dict(checkpoint['model'],strict=True)
  if self.pre is not None:
   if checkpoint['pretraining'] is None:raise ValueError('SSL checkpoint lacks reconstruction heads')
   self.pre.load_state_dict(checkpoint['pretraining'],strict=True)
  self.optimizer.load_state_dict(checkpoint['optimizer'])
  if self.scheduler is not None:self.scheduler.load_state_dict(checkpoint['scheduler'])
  self.sampler.set_state(checkpoint['sampler']);self.state=checkpoint['training'];self.initial_digest=checkpoint['initial_digest']
  if checkpoint['normalization']!=self.normalization:raise ValueError('Checkpoint normalization differs')
  _set_rng(checkpoint['rng'])
 def _new_order(self):
  import torch
  if self.name=='RNAiSpline':
   # Native DataLoader's base_seed, then RandomSampler generator seed.
   torch.empty((),dtype=torch.int64).random_().item();seed=int(torch.empty((),dtype=torch.int64).random_().item());generator=torch.Generator().manual_seed(seed)
  else:generator=self.sampler
  self.state['order']=torch.randperm(len(self.fit),generator=generator).tolist();self.state['cursor']=0
 def _forward(self,values):
  if self.name=='OligoFormer-corrected':return self.model(guide_tokens=values[0],target_tokens=values[1],features=values[2])
  if self.name=='BERT-siRNA':return self.model(*values)[0].reshape(-1)
  return self.model(*values)[:,1]
 def step(self):
  """One optimizer update; True means the epoch is consumed."""
  import torch
  state=self.state
  if state['order'] is None:self._new_order()
  selected=state['order'][state['cursor']:state['cursor']+self.policy['batch']]
  if not selected:raise RuntimeError('No remaining update in epoch')
  indices=self.fit[selected];values=[tensor[indices].to(self.device) for tensor in self.data];self.optimizer.zero_grad(set_to_none=True)
  if state['phase']=='ssl':
   self.pre.train();left,right=self.pre(self.prep.mask_sequence(values[0]),self.prep.mask_sequence(values[1]),values[2])
   loss=torch.nn.functional.mse_loss(left,values[0])+torch.nn.functional.mse_loss(right,values[1])
  else:
   self.model.train()
   with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.policy['amp']):prediction=self._forward(values)
   loss=torch.nn.functional.mse_loss(prediction.float(),self.labels[indices].to(self.device))
  if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
  loss.backward()
  if state['phase']!='ssl' and self.policy['clip'] is not None:torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.policy['clip'])
  self.optimizer.step()
  if self.scheduler is not None:self.scheduler.step(state['epoch']+state['epoch_batches']/math.ceil(len(self.fit)/self.policy['batch']))
  state['cursor']+=len(selected);state['updates']+=1;state['epoch_loss_sum']+=float(loss.detach());state['epoch_batches']+=1
  return state['cursor']==len(state['order'])
 def predict(self,indices):
  import numpy as np
  import torch
  indices=_indices(self.arrays,indices,'predict')
  if not len(indices):return np.empty(0,dtype=np.float32)
  self.model.eval();result=[]
  # Native non-shuffling evaluation DataLoader also draws a base seed.
  if self.name=='RNAiSpline':torch.empty((),dtype=torch.int64).random_().item()
  with torch.no_grad():
   for start in range(0,len(indices),self.policy['eval_batch']):
    ii=indices[start:start+self.policy['eval_batch']];values=[tensor[ii].to(self.device) for tensor in self.data]
    with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.policy['amp']):prediction=self._forward(values)
    result.append(prediction.float().cpu().numpy())
  value=np.concatenate(result)
  if not np.isfinite(value).all():raise ValueError('Nonfinite prediction')
  return value

@contextlib.contextmanager
def _object_lock(path):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a+b') as handle:
  if os.name!='nt':
   import fcntl
   fcntl.flock(handle,fcntl.LOCK_EX)
  try:yield
  finally:
   if os.name!='nt':fcntl.flock(handle,fcntl.LOCK_UN)
def _initial_object(runner,assets,out):
 initial=runner.state_dict();digest=_state_digest(initial['pretraining'] if initial['pretraining'] is not None else initial['model'])
 folder=Path(assets.get('initial_objects_dir',out.parent/'initial_objects')).resolve();path=folder/(runner.name.replace('/','_')+'_'+digest+'.pt')
 with _object_lock(path.with_suffix('.lock')):
  if not path.exists():_save(path,dict(model=initial['model'],pretraining=initial['pretraining'],initial_digest=runner.initial_digest,combined_digest=digest,seed=runner.seed,source_manifest=runner.sources))
  existing=_load(path)
  if existing['combined_digest']!=digest or existing['initial_digest']!=runner.initial_digest or _state_digest(existing['pretraining'] if existing['pretraining'] is not None else existing['model'])!=digest:raise ValueError('Initialization object differs')
 return dict(path=str(path),sha256=sha(path),model_digest=runner.initial_digest,combined_digest=digest,seed=runner.seed)
def _identity(model,arrays,fit,validation,seed,assets,fixed_epochs,execution_identity,device):
 import numpy as np
 import torch, platform
 def labels_digest(indices):return hashlib.sha256(np.asarray(arrays['labels'],dtype=np.float32)[indices].tobytes()).hexdigest()
 return dict(schema=1,pool_compatibility_revision=POOL_COMPATIBILITY_REVISION if model=='RNAiSpline' else None,runtime=dict(torch=str(torch.__version__),numpy=np.__version__,python=platform.python_version()),model=model,seed=int(seed),policy=POLICY[model],fit_sample_ids=[str(arrays['ids'][int(i)]) for i in fit],validation_sample_ids=[str(arrays['ids'][int(i)]) for i in validation],fit_label_sha256=labels_digest(fit),validation_label_sha256=labels_digest(validation),fixed_epochs=fixed_epochs,execution_identity=execution_identity,sources=source_manifest(model,assets),adapter_sha256=sha(__file__),device=str(device),precision='BF16' if POLICY[model]['amp'] else 'FP32',cudnn_enabled=False,deterministic=True,tf32=False,SSL_sample_ids=[str(arrays['ids'][int(i)]) for i in fit] if model=='RNAiSpline' else [])
def verify_fit(out,expected_identity=None):
 out=Path(out);identity=read(out/'identity.json')
 if expected_identity is not None and identity['execution_identity']!=expected_identity:raise ValueError('Fit execution identity differs')
 receipt=read(out/'complete.json')
 if receipt.get('status')!='complete' or receipt['identity_sha256']!=sha(out/'identity.json'):raise ValueError('Fit not complete with this identity')
 for relative,expected in receipt['files'].items():
  path=(out/relative).resolve()
  if not path.is_relative_to(out.resolve()) or sha(path)!=expected:raise ValueError('Fit artifact mismatch '+str(relative))
 reference=receipt['initial_reference']
 if sha(reference['path'])!=reference['sha256']:raise ValueError('Initial reference SHA mismatch')
 history=read(out/'history.json');supervised=history['supervised'];ssl=history['ssl']
 if [row['epoch'] for row in supervised]!=list(range(1,len(supervised)+1)):raise ValueError('Invalid supervised epoch sequence')
 required_ssl=POLICY[identity['model']]['ssl_epochs']
 if [row['epoch'] for row in ssl]!=list(range(1,required_ssl+1)):raise ValueError('SSL epochs do not equal registered fit-only budget')
 if identity['fixed_epochs'] is None:
  best=None;selected=None
  for row in supervised:
   value=float(row['val_mse'])
   if not math.isfinite(value):raise ValueError('Invalid validation MSE')
   if best is None or value<best-POLICY[identity['model']]['min_delta']:best=value;selected=row['epoch']
  if receipt['selected_epochs']!=selected or receipt['best_validation_mse']!=best:raise ValueError('Selected checkpoint is not the registered validation-MSE choice')
 else:
  if len(supervised)!=identity['fixed_epochs'] or receipt['selected_epochs']!=identity['fixed_epochs'] or any('val_mse' in row for row in supervised):raise ValueError('Final fit differs from fixed training-only epoch budget')
  if receipt['best_validation_mse'] is not None:raise ValueError('Final fit unexpectedly has validation selection')
 return receipt

def run_fit(model,arrays,fit_indices,seed,out,assets,validation_indices=None,fixed_epochs=None,execution_identity=None,stop_requested=None,checkpoint_callback=None,device='cuda:0'):
 """Fit inner fit/validation or fresh fixed-epoch outer refit; accepts no test.

 checkpoint_callback(out:Path,reason:str) runs after atomic last and SHA writes.
 Reasons: initial,epoch,update500,stopped,complete. Stop raises InterruptedError.
 """
 import numpy as np
 function_started=time.monotonic()
 if model not in MODELS:raise ValueError('Unknown model')
 device=assets.get('device',device) if device=='cuda:0' else device
 device='cuda:0' if str(device)=='cuda' else device
 out=Path(out).resolve();fit=_indices(arrays,fit_indices,'fit');validation=_indices(arrays,validation_indices,'validation')
 if not len(fit) or set(fit.tolist())&set(validation.tolist()):raise ValueError('Nonempty fit and disjoint validation required')
 if fixed_epochs is None:
  if not len(validation):raise ValueError('Inner selection requires independent validation')
 else:
  if isinstance(fixed_epochs,bool) or int(fixed_epochs)!=fixed_epochs or not 1<=int(fixed_epochs)<=POLICY[model]['epochs']:raise ValueError('Fixed epoch budget outside policy')
  fixed_epochs=int(fixed_epochs)
  if len(validation):raise ValueError('Final fixed-epoch fit cannot receive validation')
 if len(set(map(str,arrays['ids'])))!=len(arrays['ids']):raise ValueError('Duplicate sample IDs')
 expected=_identity(model,arrays,fit,validation,seed,assets,fixed_epochs,execution_identity,device)
 if (out/'complete.json').exists():
  if read(out/'identity.json')!=expected:raise ValueError('Completed fit identity differs')
  return verify_fit(out,execution_identity)
 existed=out.exists() and any(out.iterdir())
 if existed:
  if not (out/'identity.json').exists() or read(out/'identity.json')!=expected:raise ValueError('Incomplete fit identity differs')
  if not (out/'last.pt').exists() or not (out/'last.sha256.json').exists():raise RuntimeError('Partial fit lacks verified last; retain for recovery audit')
 else:out.mkdir(parents=True,exist_ok=True);write(out/'identity.json',expected)
 configure(device);runner=Runner(model,arrays,fit,seed,assets,device=device)
 if runner.normalization is not None:
  if (out/'normalization.json').exists() and read(out/'normalization.json')!=runner.normalization:raise ValueError('Saved normalization differs')
  write(out/'normalization.json',runner.normalization)
 if runner.basis_equivalence is not None:write(out/'basis_equivalence.json',runner.basis_equivalence)
 if runner.pool_compatibility is not None:write(out/'pool_compatibility.json',runner.pool_compatibility)
 identity_sha=sha(out/'identity.json');started=function_started;previous_seconds=0.
 if existed:
  pointer=read(out/'last.sha256.json')
  if pointer['identity_sha256']!=identity_sha or sha(out/'last.pt')!=pointer['sha256']:raise ValueError('Resume checkpoint SHA differs')
  checkpoint=_load(out/'last.pt')
  if checkpoint['identity_sha256']!=identity_sha:raise ValueError('Resume checkpoint identity differs')
  runner.load_state_dict(checkpoint['runner']);previous_seconds=checkpoint['seconds'];initial_reference=read(out/'initial_reference.json')
  if sha(initial_reference['path'])!=initial_reference['sha256']:raise ValueError('Resume initial reference differs')
 else:initial_reference=_initial_object(runner,assets,out);write(out/'initial_reference.json',initial_reference)
 def save_checkpoint(reason):
  _save(out/'last.pt',dict(schema=1,identity_sha256=identity_sha,runner=runner.state_dict(),seconds=previous_seconds+time.monotonic()-started))
  write(out/'last.sha256.json',dict(identity_sha256=identity_sha,sha256=sha(out/'last.pt'),phase=runner.state['phase'],epoch=runner.state['epoch'],updates=runner.state['updates']))
  if checkpoint_callback:checkpoint_callback(out,reason)
 def save_best(epoch,mse):_save(out/'best.pt',dict(identity_sha256=identity_sha,model=_cpu_state(runner.model),selected_epoch=epoch,validation_mse=mse,normalization=runner.normalization,initial_digest=runner.initial_digest))
 if not existed:save_checkpoint('initial')
 target=fixed_epochs if fixed_epochs is not None else runner.policy['epochs']
 while True:
  state=runner.state
  if stop_requested and stop_requested():save_checkpoint('stopped');raise InterruptedError('Controlled stop after checkpoint')
  if state['phase']=='ssl' and state['epoch']>=runner.policy['ssl_epochs']:
   runner._phase_optimizer('supervised');state.update(phase='supervised',epoch=0,cursor=0,order=None,epoch_loss_sum=0.,epoch_batches=0);save_checkpoint('epoch')
  if state['phase']=='supervised' and (state['epoch']>=target or (fixed_epochs is None and state['bad']>=runner.policy['patience'])):break
  ended=runner.step()
  if not ended:
   if state['updates']%500==0:save_checkpoint('update500')
   continue
  epoch=state['epoch']+1;record=dict(epoch=epoch,train_loss=state['epoch_loss_sum']/state['epoch_batches'],updates=state['updates'])
  if state['phase']=='ssl':state['ssl_history'].append(record)
  else:
   if fixed_epochs is None:
    prediction=runner.predict(validation);mse=float(np.mean((prediction-np.asarray(arrays['labels'],dtype=np.float32)[validation])**2))
    if not math.isfinite(mse):raise ValueError('Nonfinite validation MSE')
    if state['best'] is None or mse<state['best']-runner.policy['min_delta']:
     state['best']=mse;state['best_epoch']=epoch;state['bad']=0;save_best(epoch,mse)
    else:state['bad']+=1
    record.update(val_mse=mse,best_epoch=state['best_epoch'],best_validation_mse=state['best'])
   elif epoch==target:state['best_epoch']=epoch;save_best(epoch,None)
   state['history'].append(record)
  state.update(epoch=epoch,cursor=0,order=None,epoch_loss_sum=0.,epoch_batches=0)
  write(out/'history.json',dict(ssl=state['ssl_history'],supervised=state['history']));save_checkpoint('epoch')
 if not (out/'best.pt').exists() or runner.state['best_epoch'] is None:raise RuntimeError('No valid selected checkpoint')
 best=_load(out/'best.pt')
 if best['identity_sha256']!=identity_sha or best['selected_epoch']!=runner.state['best_epoch']:raise ValueError('Best checkpoint selection differs')
 names=['identity.json','initial_reference.json','history.json','best.pt']+[name for name in ('normalization.json','basis_equivalence.json','pool_compatibility.json') if (out/name).exists()]
 receipt=dict(schema=1,status='complete',model=model,seed=int(seed),identity_sha256=identity_sha,selected_epochs=best['selected_epoch'],best_validation_mse=best['validation_mse'],fixed_epochs=fixed_epochs,checkpoint_file='best.pt',checkpoint_sha256=sha(out/'best.pt'),last_checkpoint_sha256=sha(out/'last.pt'),initial_reference=initial_reference,initial_digest=runner.initial_digest,files={name:sha(out/name) for name in names},test_used_for_selection=False,SSL_fit_only=True,fit_n=len(fit),validation_n=len(validation),ssl_epochs=len(runner.state['ssl_history']),supervised_epochs=len(runner.state['history']),updates=runner.state['updates'],seconds=previous_seconds+time.monotonic()-started)
 write(out/'complete.json',receipt);verify_fit(out,execution_identity)
 if checkpoint_callback:checkpoint_callback(out,'complete')
 del runner
 return receipt

def predict_fit(model,arrays,indices,out,assets,expected_identity=None,device='cuda:0'):
 """Load a completed selected checkpoint; only caller supplies final test now."""
 import numpy as np
 import torch
 device=assets.get('device',device) if device=='cuda:0' else device
 device='cuda:0' if str(device)=='cuda' else device
 out=Path(out);receipt=verify_fit(out,expected_identity);identity=read(out/'identity.json')
 if identity['model']!=model or source_manifest(model,assets)!=identity['sources']:raise ValueError('Prediction model/source differs')
 lookup={str(value):i for i,value in enumerate(arrays['ids'])};fit=np.asarray([lookup[value] for value in identity['fit_sample_ids']],dtype=np.int64);indices=_indices(arrays,indices,'test')
 if set(fit.tolist())&set(indices.tolist()):raise ValueError('Test prediction overlaps fit IDs')
 configure(device);checkpoint=_load(out/receipt['checkpoint_file'])
 if checkpoint['identity_sha256']!=receipt['identity_sha256']:raise ValueError('Prediction checkpoint identity differs')
 runner=Runner(model,arrays,fit,identity['seed'],assets,device=device,normalization=checkpoint['normalization']);runner.model.load_state_dict(checkpoint['model'],strict=True);prediction=runner.predict(indices);del runner
 if torch.cuda.is_available():torch.cuda.empty_cache()
 return prediction
