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
    assert len(TRAIN_VARIANTS)==6
    assert {variant_parts(v) for v in TRAIN_VARIANTS}=={(c,s) for c in CONDITIONS for s in SOURCE_SEEDS}
    for v in ['no_pair_seed45','full_seed42','no_access','seed1','no_cross_seed42_extra']:
        with pytest.raises(ValueError):variant_parts(v)
    assert variant_parts('seed43')==('full',43)
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


def test_remaining_scope_and_fixed_scientific_budget():
    from duplexfm_mechanism.protocol import ENGINEERING_VARIANTS
    from duplexfm_mechanism.parallel import JOBS
    assert SOURCE_SEEDS == (43, 44)
    assert len(TRAIN_VARIANTS) == 6 and len(ENGINEERING_VARIANTS) == 8
    assert JOBS == TRAIN_VARIANTS + ('evaluation',)
    assert all('_seed42' not in v for v in TRAIN_VARIANTS)
    for v in ['seed42'] + [f'{c}_seed42' for c in CONDITIONS]:
        with pytest.raises(ValueError): variant_parts(v)
    c=contract()
    assert c['source_seeds'] == [43,44] and c['updates']==32728
    assert c['backup_seconds']==3600 and c['checkpoint_steps']==500
    assert c['batch']==c['micro_batch']==512 and c['max_gpu_hours']==128


def test_full_reference_can_include_completed_seed42_without_averaging_it():
    from duplexfm_mechanism.ablation_summary import summarize
    full={'records':[{'variant':f'seed{s}','test':t,'rows':10,'metrics':{'aps':.99 if s==42 else .7}}
        for s in (42,43,44) for t in OFFICIAL_TESTS]}
    records=[{'variant':f'{c}_seed{s}','test':t,'rows':10,'metrics':{'aps':.71 if s==43 else .73}}
        for c in CONDITIONS for s in SOURCE_SEEDS for t in OFFICIAL_TESTS]
    v=summarize(records,full)
    assert v['source_seed_count']==2 and v['source_seeds']==[43,44]
    for c in CONDITIONS:
        assert v['conditions'][c]['mean_macro_AP']==pytest.approx(.72)
        assert v['conditions'][c]['mean_delta_macro_AP']==pytest.approx(.02)
        assert v['conditions'][c]['sample_SD_macro_AP']==pytest.approx(np.std([.71,.73],ddof=1))
    with pytest.raises(ValueError):
        summarize(records,{'records':[r for r in full['records'] if r['variant']!='seed44']})
    with pytest.raises(ValueError):
        summarize(records,{'records':full['records']+[full['records'][4]]})


def test_engineering_and_efficiency_use_registered_seed_not_deferred_seed42():
    code=Path(__file__).resolve().parents[1]/'duplexfm_mechanism'
    native=(code/'engineering_three.py').read_text()
    efficiency=(code/'efficiency.py').read_text()
    assert 'for variant in TRAIN_VARIANTS:' in native
    assert "variant=condition+'_seed42'" not in native
    assert "seed=SOURCE_SEEDS[0]" in efficiency
    assert "PairACModel(source_config(assets),'seed42')" not in efficiency


def test_evaluation_completion_requires_two_new_seeds(monkeypatch,tmp_path):
    import duplexfm_mechanism.evaluation as e
    dest=tmp_path/'evaluation';dest.mkdir()
    v={'status':'complete','condition_count':3,'seed_count':2,'test_count_per_seed':4,
       'files_sha256':{},'source_endpoints':{n:'locked' for n in TRAIN_VARIANTS}}
    monkeypatch.setattr(e,'verify_files',lambda *args:None)
    monkeypatch.setattr(e,'sha',lambda p:'locked')
    (dest/'complete.json').write_text(json.dumps(v))
    assert e.check_complete(tmp_path)
    for count in (1,3):
        v['seed_count']=count;(dest/'complete.json').write_text(json.dumps(v))
        assert not e.check_complete(tmp_path)
