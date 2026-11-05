import copy, json
from pathlib import Path
import numpy as np
import pytest
import torch
from test_pairac import batch
from duplexfm_mechanism.model import PairACFusion, PairACModel
from duplexfm_mechanism.protocol import (source_config,contract,TRAIN_VARIANTS,CONDITIONS,
                                       SOURCE_SEEDS,OFFICIAL_TESTS,variant_parts)
from duplexfm_joint.io_utils import seed_all


@pytest.mark.parametrize('condition,lo,hi',[('no_pair',0,4),('no_cross',4,12),('no_access',12,14)])
def test_exact_channel_mask_no_shape_change(condition,lo,hi):
    seed_all(42);m=PairACFusion(source_config('/unused'));b=batch()
    x,v=m.make_maps(b);state={n:p.clone() for n,p in m.state_dict().items()}
    m.ablation=condition;y,w=m.make_maps(b)
    expected=x.clone();expected[:,lo:hi]=0
    assert torch.equal(y,expected) and torch.equal(v,w)
    assert all(torch.equal(p,m.state_dict()[n]) for n,p in state.items())
    assert y.shape[1]==16 and torch.equal(y[:,14:],x[:,14:])


@pytest.mark.parametrize('condition',['full',*CONDITIONS])
def test_input_dependency_and_zero_gradients(condition):
    seed_all(42);m=PairACFusion(source_config('/unused'));m.ablation=condition;m.eval()
    b=batch();b['accessibility'].requires_grad_()
    output=m(b);output.sum().backward()
    groups={'pair':m.pair_embedding.weight.grad,'cross':b['q'].grad,'access':b['accessibility'].grad}
    for kind,grad in groups.items():
        assert grad is not None
        assert (torch.count_nonzero(grad)==0) if condition=='no_'+kind else (grad.abs().sum()>0)
    if condition=='full':return
    altered={k:v.detach().clone() for k,v in b.items()}
    if condition=='no_pair':
        altered['guide'][b['guide_mask']]=(altered['guide'][b['guide_mask']]+1)%4
        altered['target'][b['target_mask']]=(altered['target'][b['target_mask']]+2)%4
    elif condition=='no_cross':
        altered['q']*=93;altered['k']+=27
    else:
        altered['accessibility'][b['target_mask']]=torch.rand_like(altered['accessibility'][b['target_mask']])*90-45
    assert torch.equal(output,m(altered))


@pytest.mark.parametrize('seed',[42,43,44])
def test_unmasked_forward_backward_bitwise_original(seed):
    from reference_fusion import PairACFusion as Original
    seed_all(seed);old=Original(source_config('/unused'))
    seed_all(seed);new=PairACFusion(source_config('/unused'))
    assert list(old.state_dict())==list(new.state_dict())
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    b=batch();a={k:v.detach().clone().requires_grad_(v.requires_grad) for k,v in b.items()}
    c={k:v.detach().clone().requires_grad_(v.requires_grad) for k,v in b.items()}
    seed_all(123);p=old(a);p.sum().backward();rng=torch.get_rng_state()
    seed_all(123);q=new(c);q.sum().backward()
    assert torch.equal(p,q) and torch.equal(rng,torch.get_rng_state())
    for (n,x),(m,y) in zip(old.named_parameters(),new.named_parameters()):
        assert n==m and torch.equal(x.grad,y.grad)


@pytest.mark.parametrize('condition',CONDITIONS)
def test_fusion_mask_preserves_rng_consumption(condition):
    m=PairACFusion(source_config('/unused'));b=batch()
    seed_all(111);m(b);expected=torch.get_rng_state()
    m.ablation=condition;seed_all(111);m(b)
    assert torch.equal(expected,torch.get_rng_state())


@pytest.mark.parametrize('condition',CONDITIONS)
def test_source192_uses_identical_mask(condition):
    m=PairACModel.__new__(PairACModel);torch.nn.Module.__init__(m)
    m.fusion=PairACFusion(source_config('/unused'));m.fusion.ablation=condition;m.eval()
    m.interaction_inputs=lambda b:b
    b=batch(19,38);b['guide_sequences']=['A'*19]
    source=m.source192(b,core_start=5)
    h,v=m.fusion.grid_features(b);sites,allowed=m.fusion.site_pool(h,v,b)
    assert source.shape==(1,192) and torch.equal(source,sites[:,39])


@pytest.mark.parametrize('key,value',[('heads',4),('hidden_width',64),('pair_embedding_dim',8),
    ('projection_dim',128),('local_accessibility',False),('accessibility_window',3),
    ('explicit_products',True),('site_energy',True)])
def test_misleading_config_rejected(key,value):
    cfg=source_config('/unused');cfg[key]=value
    with pytest.raises(ValueError):PairACFusion(cfg)


def test_variant_seed_contract_and_reject_unknown():
    assert len(TRAIN_VARIANTS)==3
    assert {variant_parts(v) for v in TRAIN_VARIANTS}=={(c,s) for c in CONDITIONS for s in SOURCE_SEEDS}
    for v in ['no_pair_seed45','full_seed42','no_access','seed1','no_cross_seed42_extra']:
        with pytest.raises(ValueError):variant_parts(v)
    assert variant_parts('seed42')==('full',42)
    assert contract()['micro_batch']==512 and not contract()['effective_capacity_equal']


def test_summary_keeps_conditions_and_seed_pairing():
    from duplexfm_mechanism.ablation_summary import summarize
    full={'records':[{'variant':f'seed{s}','test':t,'rows':10,'metrics':{'aps':.7+i*.01}}
                     for i,s in enumerate(SOURCE_SEEDS) for t in OFFICIAL_TESTS]}
    records=[{'variant':f'{c}_seed{s}','test':t,'rows':10,'metrics':{'aps':.7+i*.01+.01*(j-1)}}
             for j,c in enumerate(CONDITIONS) for i,s in enumerate(SOURCE_SEEDS) for t in OFFICIAL_TESTS]
    result=summarize(records,full)
    assert set(result['conditions'])==set(CONDITIONS)
    for j,c in enumerate(CONDITIONS):
        assert result['conditions'][c]['mean_delta_macro_AP']==pytest.approx(.01*(j-1))
    with pytest.raises(ValueError):summarize(records[:-1],full)
    with pytest.raises(ValueError):summarize(records+[records[0]],full)
    bad=copy.deepcopy(records);bad[0]['rows']=11
    with pytest.raises(ValueError):summarize(bad,full)


def test_masked_parameter_declaration_excludes_shared_fusion():
    from duplexfm_mechanism.training_hourly import optimizer_for
    m=PairACModel.__new__(PairACModel);torch.nn.Module.__init__(m)
    m.fusion=PairACFusion(source_config('/unused'));m.q_projection=torch.nn.Linear(2,2)
    m.guide_encoder=torch.nn.Module();m.guide_encoder.lora_B=torch.nn.Parameter(torch.zeros(2,2))
    names={n for n,p in m.named_parameters() if p.requires_grad}
    m.condition='no_pair';assert m.masked_parameter_names()=={'fusion.pair_embedding.weight'}
    m.condition='no_access';assert m.masked_parameter_names()==set()
    m.condition='no_cross';masked=m.masked_parameter_names()
    assert masked=={n for n in names if not n.startswith('fusion.')}|{'fusion.log_cross_scale'}
    opt=optimizer_for(m,{})
    assert sum(len(g['params']) for g in opt.param_groups)==len(names)


def test_efficiency_search_cannot_change_reference_micro_batch():
    from duplexfm_mechanism.efficiency import MICRO_BATCHES,CPU_THREADS
    from duplexfm_mechanism.parallel import EXECUTION
    assert MICRO_BATCHES==(512,) and CPU_THREADS==(1,2,4)
    assert EXECUTION['efficiency_search']['micro_batches']==[512]
    assert EXECUTION['experiment_allocation_gpu_hours']==128


def test_no_deferred_seed_can_be_dispatched():
    from duplexfm_mechanism.protocol import ENGINEERING_VARIANTS
    from duplexfm_mechanism.parallel import EXECUTION, JOBS
    assert SOURCE_SEEDS==(42,)
    assert TRAIN_VARIANTS==('no_pair_seed42','no_cross_seed42','no_access_seed42')
    assert ENGINEERING_VARIANTS==('seed42',*TRAIN_VARIANTS)
    assert contract()['source_seeds']==[42]
    assert EXECUTION['dispatch_order']==list(TRAIN_VARIANTS)
    assert JOBS==(*TRAIN_VARIANTS,'evaluation')
    for v in ('seed43','seed44','no_pair_seed43','no_access_seed44','no_cross_seed43'):
        with pytest.raises(ValueError):variant_parts(v)


def test_single_seed_sd_missing_and_reference_other_seeds_ignored():
    from duplexfm_mechanism.ablation_summary import summarize
    full={'records':[{'variant':f'seed{s}','test':t,'rows':10,'metrics':{'aps':.7 if s==42 else .99}}
                     for s in (42,43,44) for t in OFFICIAL_TESTS]}
    records=[{'variant':v,'test':t,'rows':10,'metrics':{'aps':.72}}
             for v in TRAIN_VARIANTS for t in OFFICIAL_TESTS]
    result=summarize(records,full)
    assert result['source_seed_count']==1 and result['source_seeds']==[42]
    assert not result['between_source_seed_SD_available']
    for item in result['conditions'].values():
        assert item['mean_delta_macro_AP']==pytest.approx(.02)
        assert item['sample_SD_macro_AP'] is None and item['sample_SD_delta_macro_AP'] is None
        assert set(item['per_seed_macro_AP'])=={'42'}
    json.dumps(result,allow_nan=False)
    full['records'].append(full['records'][0])
    with pytest.raises(ValueError):summarize(records,full)


def test_single_seed_evaluation_completion_contract(tmp_path):
    from duplexfm_mechanism.evaluation import check_complete
    from duplexfm_mechanism.common import sha
    from duplexfm_mechanism.durable_io import durable_json as write
    endpoints={}
    for v in TRAIN_VARIANTS:
        p=tmp_path/'runs'/v/'terminal.pt';p.parent.mkdir(parents=True);p.write_bytes(v.encode())
        endpoints[v]=sha(p)
    p=tmp_path/'evaluation'/'summary.json';write(p,{'synthetic':True})
    record={'status':'complete','source_endpoints':endpoints,'seed_count':1,'condition_count':3,
            'test_count_per_seed':4,'files_sha256':{'evaluation/summary.json':sha(p)}}
    write(tmp_path/'evaluation/complete.json',record)
    assert check_complete(tmp_path)
    record['seed_count']=3;write(tmp_path/'evaluation/complete.json',record)
    assert not check_complete(tmp_path)
