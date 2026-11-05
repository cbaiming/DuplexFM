"""NEW registered HPO adapter, forked without changing the old experiment.

Architecture/public sources/preprocessing match the old SHA-pinned backend;
only registered LR/weight decay/supervised batch vary. Prefixes remain partial,
promotions retain optimizer/RNG/cursor, and Spline SSL is separately shared by
exact fit inputs and seed. No test input enters run_fit. Import is torch-free.
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

BASELINE_BACKEND_SHA256='05927ac83872372f77fed5efb25df2493f269ec820e230f8199eed69f8c45b28'
HPO_REVISION='registered_eight_policy_successive_halving_v1'
GRID_IDS=('base','lr_half','lr_double','wd_zero','wd_ten','batch_half','batch_double','lr_double_batch_double')

def candidate_grid(model):
 if model not in MODELS:raise ValueError('Unknown model')
 base={k:POLICY[model][k] for k in ('lr','wd','batch')};items=[]
 for name in GRID_IDS:
  value=dict(base)
  if name=='lr_half':value['lr']*=.5
  if name in ('lr_double','lr_double_batch_double'):value['lr']*=2
  if name=='wd_zero':value['wd']=0.
  if name=='wd_ten':value['wd']*=10
  if name=='batch_half':value['batch']//=2
  if name in ('batch_double','lr_double_batch_double'):value['batch']*=2
  items.append(dict(id=name,hyperparameters=value))
 return items

def resolve_policy(model,hyperparameters=None,candidate_id='base'):
 matches={x['id']:x['hyperparameters'] for x in candidate_grid(model)}
 if candidate_id not in matches:raise ValueError('Unregistered HPO candidate ID')
 chosen=matches[candidate_id]
 if hyperparameters is not None and hyperparameters!=chosen:raise ValueError('Candidate hyperparameters differ from the fixed eight-grid registration')
 value=dict(POLICY[model]);value.update(chosen);value['ssl_batch']=16 if model=='RNAiSpline' else None
 return value

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
 def __init__(self,model,arrays,fit_indices,seed,assets,device='cuda:0',normalization=None,policy=None):
  import numpy as np
  import torch
  self.name,self.policy,self.arrays=model,dict(POLICY[model] if policy is None else policy),arrays;self.fit=_indices(arrays,fit_indices,'fit');self.seed=int(seed)
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
    self.pre=self.mod.PretrainingModel(self.model).to(self.device);self.optimizer=torch.optim.Adam(self.pre.parameters(),lr=self.policy['ssl_lr'])
    self.scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=self.policy['ssl_T0'],T_mult=self.policy['ssl_Tmult'])
   else:
    self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.policy['lr'],weight_decay=self.policy['wd']);self.scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=self.policy['supervised_T0'],T_mult=self.policy['supervised_Tmult'])
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
  batch=self.policy.get('ssl_batch',16) if state['phase']=='ssl' else self.policy['batch']
  selected=state['order'][state['cursor']:state['cursor']+batch]
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
  if self.scheduler is not None:self.scheduler.step(state['epoch']+state['epoch_batches']/math.ceil(len(self.fit)/batch))
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
def _digest_json(value):
 return hashlib.sha256((json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()).hexdigest()

def _runtime():
 import numpy as np
 import torch,platform,transformers
 return dict(torch=str(torch.__version__),numpy=np.__version__,python=platform.python_version(),
             transformers=transformers.__version__,torch_CUDA_runtime=torch.version.cuda,
             cuDNN_version=torch.backends.cudnn.version())

def _ssl_identity(arrays,fit,seed,assets,execution_identity):
 # Labels are deliberately absent: source reconstruction uses fit inputs only.
 ids=[str(arrays['ids'][int(i)]) for i in fit]
 inputs=[[str(arrays[col][int(i)]) for col in ('guide','target19')] for i in fit]
 return dict(schema=1,kind='fit_only_spline_SSL_cache',adapter_sha256=sha(__file__),runtime=_runtime(),
             fit_sample_ids=ids,fit_input_sha256=_digest_json(inputs),seed=int(seed),
             sources=source_manifest('RNAiSpline',assets),execution_identity=execution_identity,
             ssl_policy=dict(epochs=60,batch=16,lr=1e-4,T0=50,Tmult=1),labels_used=False,
             pool_compatibility_revision=POOL_COMPATIBILITY_REVISION)

def verify_ssl_cache(cache,expected_identity=None):
 path=Path(cache['path']).resolve()
 if path.name!='complete.json' or sha(path)!=cache['sha256']:raise ValueError('SSL cache completion SHA differs')
 out=path.parent;receipt=read(path);identity=read(out/'identity.json')
 if (receipt.get('status')!='complete' or receipt.get('kind')!='fit_only_spline_SSL_cache'
     or receipt['identity_sha256']!=sha(out/'identity.json') or receipt['identity_sha256']!=cache['identity_sha256']):raise ValueError('SSL cache identity/completion differs')
 if expected_identity is not None and identity!=expected_identity:raise ValueError('SSL cache fit inputs, source, seed, or execution context differs')
 for name,digest in receipt['files'].items():
  target=(out/name).resolve()
  if not target.is_relative_to(out) or sha(target)!=digest:raise ValueError('SSL cache artifact differs')
 initial=receipt['initial_reference']
 if sha(initial['path'])!=initial['sha256']:raise ValueError('SSL cache initial source object differs')
 if receipt.get('ssl_epochs')!=60 or receipt.get('labels_used') is not False:raise ValueError('SSL cache is not the registered label-free 60-epoch stage')
 if (identity.get('adapter_sha256')!=sha(__file__) or identity.get('labels_used') is not False
     or identity.get('ssl_policy')!=dict(epochs=60,batch=16,lr=1e-4,T0=50,Tmult=1)):
  raise ValueError('SSL cache policy/backend differs from the fixed registration')
 history=read(out/'history.json')
 if history.get('supervised') or [r['epoch'] for r in history['ssl']]!=list(range(1,61)):
  raise ValueError('SSL cache history is not exactly 60 reconstruction epochs')
 if any(not math.isfinite(float(r['train_loss'])) for r in history['ssl']) or history['ssl'][-1]['updates']!=receipt['ssl_updates']:
  raise ValueError('SSL cache loss/update history differs')
 return receipt

def ensure_ssl_cache(model,arrays,fit_indices,seed,cache_root,assets,execution_identity=None,stop_requested=None,checkpoint_callback=None,device='cuda:0'):
 """Build once per exact fit-ID/order/input/seed/source; no candidate or label use.

 Returns None for non-Spline. For Spline, path points to complete.json; back up
 its entire parent directory and the initial_reference target independently.
 Stop raises InterruptedError after atomic last/SHA; restart verifies both.
 """
 if model not in MODELS:raise ValueError('Unknown model')
 if model!='RNAiSpline':return None
 began=time.monotonic();fit=_indices(arrays,fit_indices,'SSL fit')
 if not len(fit):raise ValueError('Empty SSL fit')
 device=assets.get('device',device) if device=='cuda:0' else device
 expected=_ssl_identity(arrays,fit,seed,assets,execution_identity);key=_digest_json(expected)
 out=Path(cache_root).resolve()/key
 def value(created):
  receipt=read(out/'complete.json')
  result=dict(path=str(out/'complete.json'),sha256=sha(out/'complete.json'),identity_sha256=sha(out/'identity.json'),
              cache_directory=str(out),initial_reference=receipt['initial_reference'],ssl_epochs=60,ssl_updates=receipt['ssl_updates'],created_now=created)
  verify_ssl_cache(result,expected);return result
 with _object_lock(Path(cache_root).resolve()/(key+'.lease')):
  if (out/'complete.json').exists():return value(False)
  existed=out.exists() and any(out.iterdir())
  if existed:
   if not (out/'identity.json').exists() or read(out/'identity.json')!=expected:raise ValueError('Incomplete SSL cache identity differs')
   if not (out/'last.pt').exists() or not (out/'last.sha256.json').exists():raise ValueError('Partial SSL cache lacks a verified checkpoint pair')
  else:out.mkdir(parents=True,exist_ok=True);write(out/'identity.json',expected)
  identity_sha=sha(out/'identity.json');configure(device)
  runner=Runner(model,arrays,fit,seed,assets,device=device,policy=resolve_policy(model))
  if runner.basis_equivalence is not None:write(out/'basis_equivalence.json',runner.basis_equivalence)
  if runner.pool_compatibility is not None:write(out/'pool_compatibility.json',runner.pool_compatibility)
  previous=0.
  if existed:
   pointer=read(out/'last.sha256.json')
   if pointer['identity_sha256']!=identity_sha or sha(out/'last.pt')!=pointer['sha256']:raise ValueError('SSL cache checkpoint pair differs')
   checkpoint=_load(out/'last.pt')
   if checkpoint['identity_sha256']!=identity_sha:raise ValueError('SSL cache checkpoint identity differs')
   runner.load_state_dict(checkpoint['runner']);previous=checkpoint['seconds'];initial=read(out/'initial_reference.json')
   if sha(initial['path'])!=initial['sha256']:raise ValueError('SSL cache initial reference differs')
  else:initial=_initial_object(runner,assets,out);write(out/'initial_reference.json',initial)
  def save(reason):
   _save(out/'last.pt',dict(schema=1,identity_sha256=identity_sha,runner=runner.state_dict(),seconds=previous+time.monotonic()-began))
   write(out/'last.sha256.json',dict(identity_sha256=identity_sha,sha256=sha(out/'last.pt'),phase=runner.state['phase'],epoch=runner.state['epoch'],updates=runner.state['updates']))
   if checkpoint_callback:checkpoint_callback(out,reason)
  if not existed:save('ssl_initial')
  while runner.state['phase']=='ssl' and runner.state['epoch']<60:
   if stop_requested and stop_requested():save('stopped');raise InterruptedError('Controlled SSL cache stop')
   ended=runner.step();state=runner.state
   if ended:
    epoch=state['epoch']+1
    state['ssl_history'].append(dict(epoch=epoch,train_loss=state['epoch_loss_sum']/state['epoch_batches'],updates=state['updates']))
    state.update(epoch=epoch,cursor=0,order=None,epoch_loss_sum=0.,epoch_batches=0)
    write(out/'history.json',dict(ssl=state['ssl_history'],supervised=[]));save('ssl_epoch')
   elif state['updates']%500==0:save('update500')
  if len(runner.state['ssl_history'])!=60:raise ValueError('SSL cache did not execute exactly 60 native epochs')
  # The transition is deterministic and consumes no training minibatch. Its RNG
  # and sampler state are what every fresh supervised candidate receives.
  runner._phase_optimizer('supervised');runner.state.update(phase='supervised',epoch=0,cursor=0,order=None,epoch_loss_sum=0.,epoch_batches=0)
  _save(out/'checkpoint.pt',dict(schema=1,identity_sha256=identity_sha,runner=runner.state_dict()))
  names=['identity.json','initial_reference.json','history.json','checkpoint.pt']+[n for n in ('basis_equivalence.json','pool_compatibility.json') if (out/n).exists()]
  receipt=dict(schema=1,status='complete',kind='fit_only_spline_SSL_cache',identity_sha256=identity_sha,
               files={n:sha(out/n) for n in names},initial_reference=initial,ssl_epochs=60,ssl_updates=runner.state['updates'],
               initial_digest=runner.initial_digest,labels_used=False,fit_only=True,seconds=previous+time.monotonic()-began)
  write(out/'complete.json',receipt)
  if checkpoint_callback:checkpoint_callback(out,'ssl_complete')
  del runner
  return value(True)

def _identity(model,arrays,fit,validation,seed,assets,fixed_epochs,execution_identity,device,policy,candidate_id,ssl_cache):
 import numpy as np
 def labels_digest(indices):return hashlib.sha256(np.asarray(arrays['labels'],dtype=np.float32)[indices].tobytes()).hexdigest()
 cache=None if ssl_cache is None else {k:ssl_cache[k] for k in ('path','sha256','identity_sha256')}
 return dict(schema=2,HPO_revision=HPO_REVISION,baseline_backend_sha256=BASELINE_BACKEND_SHA256,
             pool_compatibility_revision=POOL_COMPATIBILITY_REVISION if model=='RNAiSpline' else None,runtime=_runtime(),
             model=model,seed=int(seed),candidate_id=candidate_id,policy=policy,
             fit_sample_ids=[str(arrays['ids'][int(i)]) for i in fit],validation_sample_ids=[str(arrays['ids'][int(i)]) for i in validation],
             fit_label_sha256=labels_digest(fit),validation_label_sha256=labels_digest(validation),fixed_epochs=fixed_epochs,
             execution_identity=execution_identity,sources=source_manifest(model,assets),adapter_sha256=sha(__file__),device=str(device),
             precision='BF16' if policy['amp'] else 'FP32',cudnn_enabled=False,deterministic=True,tf32=False,
             SSL_sample_ids=[str(arrays['ids'][int(i)]) for i in fit] if model=='RNAiSpline' else [],ssl_cache=cache,
             operational_rung_cap_excluded_from_identity=True)

def _verified_receipt(out,receipt,expected_identity=None,allow_rung=False):
 out=Path(out).resolve();identity=read(out/'identity.json');status=receipt.get('status')
 if expected_identity is not None and identity['execution_identity']!=expected_identity:raise ValueError('Fit execution identity differs')
 policy=resolve_policy(identity['model'],{k:identity['policy'][k] for k in ('lr','wd','batch')},identity['candidate_id'])
 if identity['policy']!=policy or identity.get('HPO_revision')!=HPO_REVISION or identity.get('adapter_sha256')!=sha(__file__):raise ValueError('HPO policy/backend identity differs')
 if status not in (('complete','rung_complete') if allow_rung else ('complete',)) or receipt['identity_sha256']!=sha(out/'identity.json'):raise ValueError('Fit receipt status/identity differs')
 for relative,expected in receipt['files'].items():
  path=(out/relative).resolve()
  if not path.is_relative_to(out) or sha(path)!=expected:raise ValueError('Fit artifact mismatch '+str(relative))
 if sha(receipt['initial_reference']['path'])!=receipt['initial_reference']['sha256']:raise ValueError('Initial reference SHA mismatch')
 history=read(out/receipt['history_file']);supervised=history['supervised'];ssl=history['ssl']
 if [row['epoch'] for row in supervised]!=list(range(1,len(supervised)+1)):raise ValueError('Invalid supervised epoch sequence')
 if len(supervised)>policy['epochs']:raise ValueError('Supervised fit exceeded the registered epoch budget')
 if [row['epoch'] for row in ssl]!=list(range(1,policy['ssl_epochs']+1)):raise ValueError('SSL epochs do not equal fixed fit-only budget')
 if status=='rung_complete':
  if receipt.get('full_fit_complete') is not False or receipt['supervised_epochs']>receipt['rung_epoch_cap']:raise ValueError('Prefix incorrectly marked fully complete')
 else:
  if receipt.get('full_fit_complete') is not True:raise ValueError('Full fit lacks explicit completion')
 if identity['fixed_epochs'] is None:
  best=None;selected=None;bad=0
  for row in supervised:
   value=float(row['val_mse'])
   if not math.isfinite(value):raise ValueError('Invalid validation MSE')
   if bad>=policy['patience']:raise ValueError('Training continued beyond the registered patience endpoint')
   if best is None or value<best-policy['min_delta']:best=value;selected=row['epoch'];bad=0
   else:bad+=1
  if receipt['selected_epochs']!=selected or receipt['best_validation_mse']!=best:raise ValueError('Selected checkpoint is not the registered validation-MSE choice')
  if receipt.get('stopped_by_patience')!=(bad>=policy['patience']):raise ValueError('Patience receipt differs from validation history')
  if status=='complete' and len(supervised)<policy['epochs'] and bad<policy['patience']:raise ValueError('Full inner fit stopped before its registered endpoint')
 else:
  if status!='complete' or len(supervised)!=identity['fixed_epochs'] or receipt['selected_epochs']!=identity['fixed_epochs'] or any('val_mse' in row for row in supervised):raise ValueError('Final fit differs from training-only fixed epoch budget')
  if receipt['best_validation_mse'] is not None:raise ValueError('Final fit unexpectedly has validation selection')
 if identity['model']=='RNAiSpline':
  cache=identity['ssl_cache'];verified=verify_ssl_cache(cache)
  if receipt['ssl_reused_updates']!=verified['ssl_updates'] or receipt['initial_digest']!=verified['initial_digest']:raise ValueError('Reused SSL receipt differs')
 return receipt

def verify_fit(out,expected_identity=None,allow_rung=False):
 out=Path(out)
 if (out/'complete.json').exists():return _verified_receipt(out,read(out/'complete.json'),expected_identity,allow_rung=False)
 if allow_rung and (out/'rung_complete.json').exists():return _verified_receipt(out,read(out/'rung_complete.json'),expected_identity,allow_rung=True)
 raise ValueError('No full completed fit (prefixes cannot be used as final fits)')

def verify_rung(out,epoch_cap,expected_identity=None):
 out=Path(out);path=out/'rungs'/('epoch_'+str(int(epoch_cap)))/'receipt.json'
 return _verified_receipt(out,read(path),expected_identity,allow_rung=True)

def _copy_atomic_file(source,target):
 import shutil
 source,target=Path(source),Path(target);target.parent.mkdir(parents=True,exist_ok=True)
 fd,temp=tempfile.mkstemp(prefix=target.name+'.',suffix='.tmp',dir=target.parent)
 try:
  with source.open('rb') as inp,os.fdopen(fd,'wb') as out:shutil.copyfileobj(inp,out,length=8<<20);out.flush();os.fsync(out.fileno())
  if sha(temp)!=sha(source):raise ValueError('Rung immutable copy failed SHA readback')
  if target.exists():
   if sha(target)!=sha(temp):raise ValueError('Immutable rung endpoint already differs')
  else:os.replace(temp,target)
 finally:
  if os.path.exists(temp):os.unlink(temp)

def _start_supervised_from_cache(runner,ssl_cache,expected):
 receipt=verify_ssl_cache(ssl_cache,expected);checkpoint=_load(Path(ssl_cache['path']).parent/'checkpoint.pt');state=checkpoint['runner']
 if checkpoint['identity_sha256']!=ssl_cache['identity_sha256'] or state['training']['phase']!='supervised' or state['training']['epoch']!=0 or state['training']['history']:raise ValueError('SSL cache is not the exact pre-supervised boundary')
 if runner.initial_digest!=receipt['initial_digest']:raise ValueError('Fresh source initialization differs from SSL cache initial object')
 runner.model.load_state_dict(state['model'],strict=True);runner.sampler.set_state(state['sampler']);runner.state=copy.deepcopy(state['training'])
 runner.initial_digest=state['initial_digest'];runner._phase_optimizer('supervised')
 # Do not import the base supervised optimizer from the cache: each candidate
 # receives a fresh optimizer with its own registered LR/WD. Restore RNG last.
 _set_rng(state['rng'])
 return receipt['initial_reference'],receipt['ssl_updates']

def run_fit(model,arrays,fit_indices,seed,out,assets,validation_indices=None,fixed_epochs=None,execution_identity=None,stop_requested=None,checkpoint_callback=None,device='cuda:0',hyperparameters=None,candidate_id='base',max_supervised_epochs=None,finalize=True,ssl_cache=None,ssl_execution_identity=None):
 """No test accepted. Prefix caps are operational, not an identity/optimizer reset.

 R1: finalize=False,max_supervised_epochs=ceil(.25*registered epochs).
 Promotion: same out/identity, finalize=True, no cap. Uses last optimizer/RNG.
 Final outer fit: fixed_epochs, no validation, finalize=True.
 """
 import numpy as np
 began=time.monotonic();policy=resolve_policy(model,hyperparameters,candidate_id)
 device=assets.get('device',device) if device=='cuda:0' else device;device='cuda:0' if str(device)=='cuda' else device
 out=Path(out).resolve();fit=_indices(arrays,fit_indices,'fit');validation=_indices(arrays,validation_indices,'validation')
 if not len(fit) or set(fit.tolist())&set(validation.tolist()):raise ValueError('Nonempty fit and disjoint validation required')
 if fixed_epochs is None:
  if not len(validation):raise ValueError('Inner selection requires independent validation')
 else:
  if isinstance(fixed_epochs,bool) or int(fixed_epochs)!=fixed_epochs or not 1<=int(fixed_epochs)<=policy['epochs']:raise ValueError('Fixed epoch budget outside policy')
  fixed_epochs=int(fixed_epochs)
  if len(validation) or not finalize or max_supervised_epochs is not None:raise ValueError('Final fixed fit cannot receive validation or a partial rung')
 if not isinstance(finalize,bool):raise ValueError('finalize must be a boolean')
 if finalize and max_supervised_epochs is not None:raise ValueError('Full promotion must use the registered full budget without an operational cap')
 if not finalize:
  if isinstance(max_supervised_epochs,bool) or not isinstance(max_supervised_epochs,int) or not 1<=max_supervised_epochs<policy['epochs']:raise ValueError('Prefix cap must be an integer below the full registered epoch budget')
 if len(set(map(str,arrays['ids'])))!=len(arrays['ids']):raise ValueError('Duplicate sample IDs')
 if model=='RNAiSpline':
  if ssl_cache is None:raise ValueError('Spline requires an explicit completed, fit-only SSL cache')
  ssl_expected=_ssl_identity(arrays,fit,seed,assets,execution_identity if ssl_execution_identity is None else ssl_execution_identity)
  verify_ssl_cache(ssl_cache,ssl_expected)
 elif ssl_cache is not None:raise ValueError('Only Spline accepts an SSL cache')
 expected=_identity(model,arrays,fit,validation,seed,assets,fixed_epochs,execution_identity,device,policy,candidate_id,ssl_cache)
 if (out/'identity.json').exists() and read(out/'identity.json')!=expected:raise ValueError('HPO fit identity differs; old fits/other candidates cannot be reused')
 if not finalize and (out/'rungs'/('epoch_'+str(max_supervised_epochs))/'receipt.json').exists():return verify_rung(out,max_supervised_epochs,execution_identity)
 if (out/'complete.json').exists():return verify_fit(out,execution_identity)
 existed=out.exists() and any(out.iterdir())
 if existed:
  if not (out/'identity.json').exists() or not (out/'last.pt').exists() or not (out/'last.sha256.json').exists():raise RuntimeError('Partial HPO fit lacks a verified identity/checkpoint pair; preserve for explicit recovery')
 else:out.mkdir(parents=True,exist_ok=True);write(out/'identity.json',expected)
 configure(device);runner=Runner(model,arrays,fit,seed,assets,device=device,policy=policy)
 if runner.normalization is not None:
  if (out/'normalization.json').exists() and read(out/'normalization.json')!=runner.normalization:raise ValueError('Saved normalization differs')
  write(out/'normalization.json',runner.normalization)
 if runner.basis_equivalence is not None:write(out/'basis_equivalence.json',runner.basis_equivalence)
 if runner.pool_compatibility is not None:write(out/'pool_compatibility.json',runner.pool_compatibility)
 identity_sha=sha(out/'identity.json');previous_seconds=0.;ssl_reused_updates=0
 if existed:
  pointer=read(out/'last.sha256.json')
  if pointer['identity_sha256']!=identity_sha or sha(out/'last.pt')!=pointer['sha256']:raise ValueError('Resume HPO checkpoint SHA pair differs')
  checkpoint=_load(out/'last.pt')
  if checkpoint['identity_sha256']!=identity_sha:raise ValueError('Resume HPO checkpoint identity differs')
  runner.load_state_dict(checkpoint['runner']);previous_seconds=checkpoint['seconds'];ssl_reused_updates=checkpoint.get('ssl_reused_updates',0)
  initial_reference=read(out/'initial_reference.json')
  if sha(initial_reference['path'])!=initial_reference['sha256']:raise ValueError('Resume source initial reference differs')
 elif model=='RNAiSpline':
  initial_reference,ssl_reused_updates=_start_supervised_from_cache(runner,ssl_cache,ssl_expected)
  write(out/'initial_reference.json',initial_reference)
  write(out/'ssl_cache_reference.json',{k:ssl_cache[k] for k in ('path','sha256','identity_sha256')})
 else:initial_reference=_initial_object(runner,assets,out);write(out/'initial_reference.json',initial_reference)
 def save(reason):
  _save(out/'last.pt',dict(schema=2,identity_sha256=identity_sha,runner=runner.state_dict(),seconds=previous_seconds+time.monotonic()-began,ssl_reused_updates=ssl_reused_updates))
  write(out/'last.sha256.json',dict(identity_sha256=identity_sha,sha256=sha(out/'last.pt'),phase=runner.state['phase'],epoch=runner.state['epoch'],updates=runner.state['updates']))
  if checkpoint_callback:checkpoint_callback(out,reason)
 def best_save(epoch,mse):_save(out/'best.pt',dict(identity_sha256=identity_sha,model=_cpu_state(runner.model),selected_epoch=epoch,validation_mse=mse,normalization=runner.normalization,initial_digest=runner.initial_digest))
 if not existed:save('initial')
 target=fixed_epochs if fixed_epochs is not None else (policy['epochs'] if finalize else max_supervised_epochs)
 while True:
  state=runner.state
  if state['phase']!='supervised':raise ValueError('Candidate may not run or repeat SSL; use the fixed shared cache')
  if stop_requested and stop_requested():save('stopped');raise InterruptedError('Controlled HPO stop after checkpoint')
  if state['epoch']>=target or (fixed_epochs is None and state['bad']>=policy['patience']):break
  ended=runner.step()
  if not ended:
   if state['updates']%500==0:save('update500')
   continue
  epoch=state['epoch']+1;record=dict(epoch=epoch,train_loss=state['epoch_loss_sum']/state['epoch_batches'],updates=state['updates'])
  if fixed_epochs is None:
   prediction=runner.predict(validation);mse=float(np.mean((prediction-np.asarray(arrays['labels'],dtype=np.float32)[validation])**2))
   if not math.isfinite(mse):raise ValueError('Nonfinite validation MSE')
   if state['best'] is None or mse<state['best']-policy['min_delta']:
    state['best']=mse;state['best_epoch']=epoch;state['bad']=0;best_save(epoch,mse)
   else:state['bad']+=1
   record.update(val_mse=mse,best_epoch=state['best_epoch'],best_validation_mse=state['best'])
  elif epoch==target:state['best_epoch']=epoch;best_save(epoch,None)
  state['history'].append(record);state.update(epoch=epoch,cursor=0,order=None,epoch_loss_sum=0.,epoch_batches=0)
  write(out/'history.json',dict(ssl=state['ssl_history'],supervised=state['history']));save('epoch')
 if not (out/'best.pt').exists() or runner.state['best_epoch'] is None:raise RuntimeError('No valid selected checkpoint')
 best=_load(out/'best.pt')
 if best['identity_sha256']!=identity_sha or best['selected_epoch']!=runner.state['best_epoch']:raise ValueError('Best selected checkpoint differs')
 history_file,checkpoint_file='history.json','best.pt'
 if not finalize:
  rung='rungs/epoch_'+str(max_supervised_epochs)
  _copy_atomic_file(out/'history.json',out/rung/'history.json');_copy_atomic_file(out/'best.pt',out/rung/'best.pt')
  history_file,checkpoint_file=rung+'/history.json',rung+'/best.pt'
 names=['identity.json','initial_reference.json',history_file,checkpoint_file]+[n for n in ('normalization.json','basis_equivalence.json','pool_compatibility.json','ssl_cache_reference.json') if (out/n).exists()]
 receipt=dict(schema=2,status='complete' if finalize else 'rung_complete',full_fit_complete=finalize,model=model,seed=int(seed),candidate_id=candidate_id,
              identity_sha256=identity_sha,selected_epochs=best['selected_epoch'],best_validation_mse=best['validation_mse'],fixed_epochs=fixed_epochs,
              checkpoint_file=checkpoint_file,checkpoint_sha256=sha(out/checkpoint_file),history_file=history_file,
              resume_checkpoint_sha256_at_endpoint=sha(out/'last.pt'),initial_reference=initial_reference,initial_digest=runner.initial_digest,
              files={name:sha(out/name) for name in names},test_used_for_selection=False,SSL_fit_only=True,
              fit_n=len(fit),validation_n=len(validation),ssl_epochs=len(runner.state['ssl_history']),ssl_reused_updates=ssl_reused_updates,
              new_SSL_updates=0,supervised_epochs=len(runner.state['history']),updates=runner.state['updates'],
              new_supervised_updates=runner.state['updates']-ssl_reused_updates,stopped_by_patience=runner.state['bad']>=policy['patience'],
              rung_epoch_cap=None if finalize else max_supervised_epochs,seconds=previous_seconds+time.monotonic()-began)
 if finalize:write(out/'complete.json',receipt);verify_fit(out,execution_identity)
 else:
  write(out/rung/'receipt.json',receipt);write(out/'rung_complete.json',receipt);verify_rung(out,max_supervised_epochs,execution_identity)
 if checkpoint_callback:checkpoint_callback(out,'complete' if finalize else 'rung_complete')
 del runner
 return receipt

def _predict_selected(model,arrays,indices,out,assets,receipt,device):
 import numpy as np
 import torch
 out=Path(out);identity=read(out/'identity.json')
 if identity['model']!=model or source_manifest(model,assets)!=identity['sources']:raise ValueError('Prediction model/source differs')
 lookup={str(value):i for i,value in enumerate(arrays['ids'])};fit=np.asarray([lookup[value] for value in identity['fit_sample_ids']],dtype=np.int64)
 configure(device);checkpoint=_load(out/receipt['checkpoint_file'])
 if checkpoint['identity_sha256']!=receipt['identity_sha256'] or checkpoint['selected_epoch']!=receipt['selected_epochs']:raise ValueError('Prediction checkpoint identity differs')
 runner=Runner(model,arrays,fit,identity['seed'],assets,device=device,normalization=checkpoint['normalization'],policy=identity['policy'])
 runner.model.load_state_dict(checkpoint['model'],strict=True);prediction=runner.predict(indices);del runner
 if torch.cuda.is_available():torch.cuda.empty_cache()
 return prediction

def predict_validation(model,arrays,indices,out,assets,expected_identity=None,device='cuda:0',rung_epoch_cap=None):
 """Rank a best-validation checkpoint, never a final-test prediction."""
 out=Path(out);identity=read(out/'identity.json');indices=_indices(arrays,indices,'validation prediction')
 if [str(arrays['ids'][int(i)]) for i in indices]!=identity['validation_sample_ids'] or not len(indices):raise ValueError('Validation prediction must use exactly the registered validation IDs/order')
 receipt=verify_fit(out,expected_identity,allow_rung=True) if rung_epoch_cap is None else verify_rung(out,rung_epoch_cap,expected_identity)
 device=assets.get('device',device) if device=='cuda:0' else device
 return _predict_selected(model,arrays,indices,out,assets,receipt,device)

def predict_fit(model,arrays,indices,out,assets,expected_identity=None,device='cuda:0'):
 """Full final-fit only. Inner/rung checkpoints may not be tested."""
 out=Path(out);receipt=verify_fit(out,expected_identity);identity=read(out/'identity.json')
 if identity['fixed_epochs'] is None or identity['validation_sample_ids']:raise ValueError('Final test requires a fresh fixed-epoch fit without validation')
 indices=_indices(arrays,indices,'test');used=set(identity['fit_sample_ids'])
 if used&{str(arrays['ids'][int(i)]) for i in indices}:raise ValueError('Test prediction overlaps training IDs')
 device=assets.get('device',device) if device=='cuda:0' else device
 return _predict_selected(model,arrays,indices,out,assets,receipt,device)
