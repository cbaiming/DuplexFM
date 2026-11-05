"""CPU synthetic validation: original fusion layers, mocked foundation models.

No source data, foundation weights, network or GPU work are used here.
"""
from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn

from fastpath_patch import BASE_MODEL_SHA256, SINGLE_CONDITIONS, apply_to, sha256

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=HERE/'code' if (HERE/'code').exists() else HERE.parent/'release/code'
BASE_MODEL_FILE=BASE/'fastpath_originals/model.py' if (BASE/'fastpath_originals/model.py').exists() else BASE/'duplexfm_mechanism/model.py'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='session')
def modules(tmp_path_factory):
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    original=dict(sys.modules)
    root=tmp_path_factory.mktemp('new_fast_release')
    (root/'duplexfm_mechanism').mkdir()
    for name in ('model.py','protocol.py'):
        source=BASE_MODEL_FILE if name=='model.py' else BASE/'duplexfm_mechanism'/name
        (root/'duplexfm_mechanism'/name).write_bytes(source.read_bytes())
    receipt=apply_to(root)
    for name,path in [('duplexfm_joint',BASE/'duplexfm_joint'),
                      ('duplexfm_mechanism',root/'duplexfm_mechanism')]:
        module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
    load('duplexfm_joint.model',BASE/'duplexfm_joint/model.py')
    backbones=types.ModuleType('duplexfm_joint.backbones')

    class MockTokenizer:
        unk_token_id=99
        def __call__(self,texts,**kwargs):
            seqs=[s.replace(' ','') for s in texts]
            width=max(map(len,seqs))+2
            ids=torch.zeros(len(seqs),width,dtype=torch.long)
            attn=torch.zeros_like(ids);special=torch.ones_like(ids)
            for i,s in enumerate(seqs):
                n=len(s);ids[i,1:n+1]=torch.tensor([{'A':1,'C':2,'G':3,'U':4,'T':4}[x] for x in s])
                attn[i,:n+2]=1;special[i,1:n+1]=0
            return {'input_ids':ids,'attention_mask':attn,'special_tokens_mask':special}

    class MockEncoder(nn.Module):
        def __init__(self):
            super().__init__();self.embedding=nn.Embedding(6,4);self.drop=nn.Dropout(.25)
            self.lora_B=nn.Parameter(torch.zeros(2,2));self.calls=0
        def forward(self,input_ids,**kwargs):
            self.calls+=1
            return types.SimpleNamespace(last_hidden_state=self.drop(self.embedding(input_ids)))

    class MockJointModel(nn.Module):
        def __init__(self,config):
            super().__init__()
            self.target_encoder=MockEncoder();self.guide_encoder=MockEncoder()
            self.target_projection=nn.Linear(4,8);self.guide_projection=nn.Linear(4,8)
            self.q_projection=nn.Linear(8,256);self.k_projection=nn.Linear(8,256)
            self.target_tokenizer=MockTokenizer();self.guide_tokenizer=MockTokenizer()
            self.fusion=config['_test_fusion_class'](config)
    backbones.JointModel=MockJointModel;sys.modules[backbones.__name__]=backbones
    protocol=load('duplexfm_mechanism.protocol',root/'duplexfm_mechanism/protocol.py')
    fast=load('duplexfm_mechanism.model',root/'duplexfm_mechanism/model.py')
    old=load('legacy_pairac_model',BASE_MODEL_FILE)
    old.__package__='duplexfm_mechanism'
    yield fast,old,protocol,root,receipt
    for name in tuple(sys.modules):
        if name.startswith(('duplexfm_joint','duplexfm_mechanism','legacy_pairac_model')):
            if name in original:sys.modules[name]=original[name]
            else:sys.modules.pop(name,None)
    assert not torch.cuda.is_initialized()


def batch(glen=(22,19),tlen=(50,38),left=(5,11)):
    b=2;g=max(glen)+3;t=60
    gm=torch.arange(g)[None]<torch.tensor(glen)[:,None]
    tp=torch.arange(t)[None]-torch.tensor(left)[:,None]
    tm=(tp>=0)&(tp<torch.tensor(tlen)[:,None])
    guide=torch.randint(0,4,(b,g)).masked_fill(~gm,99)
    target=torch.randint(0,4,(b,t)).masked_fill(~tm,99)
    return {'guide':guide,'target':target,'guide_mask':gm,'target_mask':tm,
        'guide_pos':torch.arange(g)[None].expand(b,-1).masked_fill(~gm,-1),
        'target_pos':tp.masked_fill(~tm,-1),'target_lengths':torch.tensor(tlen),
        'guide_lengths':torch.tensor(glen),
        'guide_sequences':[''.join('ACGU'[v] for v in guide[i,:n]) for i,n in enumerate(glen)],
        'target_sequences':[''.join('ACGU'[v] for v in target[i,tm[i]]) for i in range(b)],
        'accessibility':torch.rand(b,t).masked_fill(~tm,123).requires_grad_(),
        'q':torch.randn(b,t,256,requires_grad=True),'k':torch.randn(b,g,256,requires_grad=True)}


def clone_batch(b):
    return {k:(v.detach().clone().requires_grad_(v.requires_grad) if torch.is_tensor(v) else copy.deepcopy(v)) for k,v in b.items()}


def model_pair(modules,condition,seed=42):
    fast,old,p,_,_=modules
    cfg=p.source_config('/unused')
    torch.manual_seed(seed);a=old.PairACModel(dict(cfg,_test_fusion_class=old.PairACFusion),condition+'_seed42')
    torch.manual_seed(seed);b=fast.PairACModel(dict(cfg,_test_fusion_class=fast.PairACFusion),condition+'_seed42')
    return a,b


class GuardDict(dict):
    def __init__(self,b,forbidden):super().__init__(b);self.forbidden=set(forbidden)
    def __getitem__(self,k):
        if k in self.forbidden:raise AssertionError('Disabled input read: '+k)
        return super().__getitem__(k)


@pytest.mark.parametrize('condition',SINGLE_CONDITIONS)
def test_map_parity_masks_pad_and_no_disabled_reads(modules,condition):
    old,fast=model_pair(modules,condition);b=batch()
    expected,valid=old.fusion.make_maps(b)
    forbidden={'only_pair':{'q','k','accessibility'},'only_cross':{'guide','target','accessibility'},
               'only_access':{'q','k','guide','target'}}[condition]
    actual,mask=fast.fusion.make_maps(GuardDict(b,forbidden))
    assert actual.shape==(2,16,25,60) and torch.equal(actual,expected) and torch.equal(valid,mask)
    assert torch.count_nonzero(actual.masked_select(~mask[:,None].expand_as(actual)))==0
    assert torch.equal(actual[:,14:],expected[:,14:])


@pytest.mark.parametrize('condition',SINGLE_CONDITIONS)
def test_state_keys_shapes_trainable_capacity_unchanged(modules,condition):
    old,fast=model_pair(modules,condition)
    assert list(old.state_dict())==list(fast.state_dict())
    assert all(torch.equal(v,fast.state_dict()[k]) for k,v in old.state_dict().items())
    assert sum(p.numel() for p in old.parameters() if p.requires_grad)==sum(p.numel() for p in fast.parameters() if p.requires_grad)
    assert old.masked_parameter_names()==fast.masked_parameter_names()
    assert list(old.named_buffers())==list(fast.named_buffers())


@pytest.mark.parametrize('condition',SINGLE_CONDITIONS)
@pytest.mark.parametrize('training',[False,True])
def test_fusion_logits_enabled_gradients_parity_at_aligned_rng(modules,condition,training):
    old,fast=model_pair(modules,condition);old.train(training);fast.train(training)
    a=batch();b=clone_batch(a)
    torch.manual_seed(923);expected=old.fusion(a);expected.square().sum().backward();rng=torch.get_rng_state()
    torch.manual_seed(923);actual=fast.fusion(b);actual.square().sum().backward()
    assert torch.equal(actual,expected) and torch.equal(rng,torch.get_rng_state())
    masked=old.masked_parameter_names()
    for (name,p),(other,q) in zip(old.named_parameters(),fast.named_parameters()):
        assert name==other
        if name in masked:
            assert q.grad is None and (p.grad is None or torch.count_nonzero(p.grad)==0)
        elif p.grad is None:
            assert q.grad is None
        else:assert torch.equal(p.grad,q.grad),name
    if condition=='only_cross':assert torch.equal(a['q'].grad,b['q'].grad)
    else:assert b['q'].grad is None
    if condition=='only_access':assert torch.equal(a['accessibility'].grad,b['accessibility'].grad)
    else:assert b['accessibility'].grad is None


@pytest.mark.parametrize('condition',SINGLE_CONDITIONS)
def test_actual_encoder_projection_and_pair_call_counts(modules,condition):
    old,fast=model_pair(modules,condition);old.eval();fast.eval();b=batch()
    calls=[];handles=[]
    for name in ['target_encoder','guide_encoder','target_projection','guide_projection','q_projection','k_projection']:
        handles.append(getattr(fast,name).register_forward_hook(lambda m,a,o,n=name:calls.append(n)))
    handles.append(fast.fusion.pair_embedding.register_forward_hook(lambda *args:calls.append('pair')))
    a=old.interaction_inputs(b);x=fast.interaction_inputs(b)
    expected=old.fusion(a);actual=fast.fusion(x)
    for h in handles:h.remove()
    assert torch.equal(expected,actual)
    if condition=='only_cross':
        assert calls==['target_encoder','target_projection','q_projection','guide_encoder','guide_projection','k_projection']
    elif condition=='only_pair':assert calls==['pair']
    else:assert calls==[]
    if condition!='only_cross':assert 'q' not in x and 'k' not in x


@pytest.mark.parametrize('condition',SINGLE_CONDITIONS)
def test_source192_design_site_and_mixed_padding_eval_parity(modules,condition):
    old,fast=model_pair(modules,condition);old.eval();fast.eval()
    b=batch(glen=(19,19),tlen=(57,57),left=(1,2))
    assert torch.equal(old.source192(b),fast.source192(b))
    result=fast.source192(b)
    assert result.shape==(2,192) and torch.isfinite(result).all()
    changed=clone_batch(b)
    changed['guide'][~b['guide_mask']]=783
    changed['target'][~b['target_mask']]=671
    changed['accessibility'].data[~b['target_mask']]=-998
    assert torch.equal(result,fast.source192(changed))


@pytest.mark.parametrize('condition',SINGLE_CONDITIONS)
def test_legacy_alias_switch_same_instance_eval_parity(modules,condition):
    _,fast=model_pair(modules,condition);fast.eval();b=batch()
    expected=fast.fusion(fast.interaction_inputs(b))
    inputs,maps=fast.interaction_inputs,fast.fusion.make_maps
    try:
        fast.interaction_inputs=fast._legacy_interaction_inputs
        fast.fusion.make_maps=fast.fusion._legacy_make_maps
        assert torch.equal(expected,fast.fusion(fast.interaction_inputs(b)))
    finally:fast.interaction_inputs=inputs;fast.fusion.make_maps=maps


@pytest.mark.parametrize('condition',['only_pair','only_access'])
def test_skipped_encoder_dropout_changes_rng_not_claiming_bitwise_training(modules,condition):
    old,fast=model_pair(modules,condition);old.train();fast.train();b=batch()
    torch.manual_seed(42);old.interaction_inputs(b);a=torch.get_rng_state()
    torch.manual_seed(42);fast.interaction_inputs(b);z=torch.get_rng_state()
    assert not torch.equal(a,z)


@pytest.mark.parametrize('condition',SINGLE_CONDITIONS)
def test_optimizer_disabled_owned_parameters_not_updated(modules,condition):
    _,fast=model_pair(modules,condition);fast.train();b=batch()
    params=dict(fast.named_parameters());before={k:v.detach().clone() for k,v in params.items()}
    optimizer=torch.optim.AdamW(fast.parameters(),lr=.005,weight_decay=.1)
    fast.fusion(fast.interaction_inputs(b)).square().sum().backward()
    for n in fast.masked_parameter_names():params[n].grad=None
    optimizer.step()
    assert all(torch.equal(before[n],params[n]) for n in fast.masked_parameter_names())
    assert any(not torch.equal(before[n],p) for n,p in params.items() if n.startswith('fusion.stem.'))


def test_patch_is_sha_gated_idempotent_and_original_retained(modules):
    _,_,_,root,receipt=modules
    assert apply_to(root)==receipt
    assert sha256((root/'fastpath_originals/model.py').read_bytes())==BASE_MODEL_SHA256
    assert sha256(BASE_MODEL_FILE.read_bytes())==BASE_MODEL_SHA256
    with pytest.raises(ValueError,match='historical|base model SHA mismatch'):apply_to(BASE)


@pytest.mark.parametrize('condition',['full','no_pair','no_cross','no_access'])
def test_legacy_full_and_leave_one_out_path_unchanged(modules,condition):
    fast,old,p,_,_=modules
    torch.manual_seed(42);a=old.PairACFusion(p.source_config('/unused')).train()
    torch.manual_seed(42);b=fast.PairACFusion(p.source_config('/unused')).train()
    a.ablation=b.ablation=condition;data=batch()
    torch.manual_seed(923);expected=a(data);expected.sum().backward();rng=torch.get_rng_state()
    torch.manual_seed(923);actual=b(data);actual.sum().backward()
    assert torch.equal(expected,actual) and torch.equal(rng,torch.get_rng_state())
    for (n,x),(m,y) in zip(a.named_parameters(),b.named_parameters()):
        assert n==m and torch.equal(x.grad,y.grad)
