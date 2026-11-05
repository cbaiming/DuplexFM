import ast,copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from duplexfm_joint.data import PairStore,sha256
from duplexfm_joint.io_utils import seed_all
from duplexfm_mechanism.model import PairACFusion
from duplexfm_mechanism.protocol import source_config,contract,TRAIN_VARIANTS
from duplexfm_mechanism.efficiency import choose

def batch(g=21,t=27):
    gm=torch.arange(g+3)[None]<g;tm=(torch.arange(60)[None]>=16)&(torch.arange(60)[None]<16+t)
    gp=torch.arange(g+3)[None].masked_fill(~gm,-1)
    tp=(torch.arange(60)[None]-16).masked_fill(~tm,-1)
    return {'guide':torch.randint(0,4,(1,g+3)).masked_fill(~gm,4),'target':torch.randint(0,4,(1,60)).masked_fill(~tm,4),
            'guide_mask':gm,'target_mask':tm,'guide_pos':gp,'target_pos':tp,
            'target_lengths':torch.tensor([t]),'guide_lengths':torch.tensor([g]),
            'accessibility':torch.rand(1,60).masked_fill(~tm,0),
            'q':torch.randn(1,60,256,requires_grad=True),'k':torch.randn(1,g+3,256,requires_grad=True)}

def fusion():return PairACFusion(source_config('/synthetic'))

def test_ordered_pairs_and_no_products():
    m=fusion();b=batch();x,valid=m.make_maps(b)
    assert x.shape==(1,16,24,60)
    ids=b['guide'][0,:21,None]*4+b['target'][0,None,16:43]
    assert torch.equal(x[0,:4,:21,16:43].permute(1,2,0),m.pair_embedding(ids))
    assert m.pair_embedding.weight.shape==(16,4)
    assert not torch.equal(m.pair_embedding.weight[3],m.pair_embedding.weight[12])
    assert len(m.channel_names())==16
    assert all('energy' not in n and 'bio' not in n for n,_ in m.named_parameters())

@pytest.mark.parametrize('g',[1,6,19,22,27])
def test_variable_guide_and_all_window_pool(g):
    m=fusion().eval();b=batch(g)
    h,valid=m.grid_features(b);sites,allowed=m.site_pool(h,valid,b)
    assert sites.shape==(1,g+3+60-1,192) # no extra physical candidate
    assert allowed.any() and torch.isfinite(m(b)).all()
    assert torch.count_nonzero(h.masked_select(~valid[:,None].expand_as(h)))==0

def test_padding_nan_and_coordinate_shift_invariance():
    seed_all(42);m=fusion().eval();b=batch();expected=m(b)
    other={k:v.detach().clone() for k,v in b.items()}
    other['q'][~b['target_mask']]=float('nan');other['k'][~b['guide_mask']]=float('nan')
    other['accessibility'][~b['target_mask']]=float('nan')
    other['target'][~b['target_mask']]=999;other['guide'][~b['guide_mask']]=-999
    assert torch.equal(expected,m(other))
    shifted=dict(b)
    for k in ['target','target_mask','target_pos','accessibility','q']:
        shifted[k]=torch.roll(b[k],3,1)
    assert torch.allclose(expected,m(shifted),atol=1e-6,rtol=1e-5)

def test_unknown_real_base_rejected():
    b=batch();b['guide'][0,0]=4
    with pytest.raises(ValueError):fusion()(b)

def test_pair_cross_access_gradient_and_no_pad_gradient():
    m=fusion();b=batch();b['accessibility'].requires_grad_()
    m(b).sum().backward()
    assert m.pair_embedding.weight.grad.abs().sum()>0
    for k,mask in [('q','target_mask'),('k','guide_mask')]:
        assert b[k].grad[b[mask]].abs().sum()>0
        assert b[k].grad[~b[mask]].abs().sum()==0
    assert b['accessibility'].grad[b['target_mask']].abs().sum()>0
    assert b['accessibility'].grad[~b['target_mask']].abs().sum()==0

def test_unused_legacy_inputs_cannot_influence_output():
    m=fusion().eval();b=batch();a=m(b)
    b.update(energy=torch.full((1,3),float('nan')),partner=torch.full((1,60),999),
             bio=torch.full((1,4,24,60),float('nan')),energy_valid=torch.ones(1,dtype=torch.bool))
    assert torch.equal(a,m(b))

def test_direct_pair_embedding_is_position_shared():
    m=fusion();b=batch();b['guide'][0,:21]=0;b['target'][0,16:43]=3
    x,_=m.make_maps(b)
    assert torch.equal(x[0,:4,1,20],x[0,:4,18,40])
    assert not torch.equal(x[0,14:,1,20],x[0,14:,18,40])

def tiny_store(root):
    arrays={'data/target_tokens.npy':np.array([[0,1,2,3,0,1,2]],np.uint8),
            'data/guide_tokens.npy':np.array([[3,2,1,0],[0,1,4,4]],np.uint8),
            'data/guide_lengths.npy':np.array([4,2],np.int64),
            'data/target_ids.npy':np.array([0,0],np.int64),'data/guide_ids.npy':np.array([0,1],np.int64),
            'data/labels.npy':np.array([0,1],np.float32),'accessibility.npy':np.array([[.1,.2,.3,.4,.5,.6,.7]],np.float32),
            'data/indices_train.npy':np.array([0],np.int64),'data/indices_val.npy':np.array([1],np.int64)}
    for name,a in arrays.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);np.save(p,a,allow_pickle=False)
    manifest={'files_sha256':{name:sha256(root/name) for name in arrays},
              'target_padding':'collate_only_center'}
    # Legacy manifest may still list unavailable physics. PairAC deliberately
    # verifies only actually consumed arrays, not fictitious replacement data.
    manifest['files_sha256']['physics/energies.npy']='0'*64
    (root/'manifest.json').write_text(json.dumps(manifest))

def test_data_without_physics_or_pair_ids(tmp_path):
    tiny_store(tmp_path)
    with PairStore(tmp_path) as s:
        v=s.verify_integrity();assert not v['unused_legacy_physics_arrays_opened']
        b=s.collate([0,1]);assert b['target'].shape==(2,60)
        assert b['guide_mask'].sum(1).tolist()==[4,2]
        assert b['target_pos'][0,26:33].tolist()==list(range(7))
        assert b['accessibility'][0,26].item()==pytest.approx(.1)
        assert b['accessibility'][0,32].item()==pytest.approx(.7)
        assert b['target_sequences']==['ACGUACG']*2
        assert not {'energy','partner','bio','energy_valid'}&set(b)

def test_tampered_actual_input_rejected(tmp_path):
    tiny_store(tmp_path)
    np.save(tmp_path/'accessibility.npy',np.zeros((1,7),np.float32))
    with PairStore(tmp_path) as s:
        with pytest.raises(ValueError):s.verify_integrity()

def test_three_seed_dispatch_and_fixed_budget():
    from duplexfm_mechanism.parallel import EXECUTION,JOBS
    assert len(TRAIN_VARIANTS)==6
    assert JOBS==TRAIN_VARIANTS+('evaluation',)
    assert EXECUTION['cards']==2 and EXECUTION['experiment_allocation_gpu_hours']==128
    assert contract()['updates']==32728 and contract()['backup_seconds']==3600
    path=Path(__file__).resolve().parents[1]/'duplexfm_mechanism/parallel.py'
    src=path.read_text()
    assert "condition,seed=variant_parts(variant)" in src and "'source_seed': seed" in src

def test_efficiency_eligibility_and_tie():
    def r(m,t,s,ratio=.5):return {'status':'complete','micro_batch':m,'cpu_threads':t,'examples_per_second':s,
                                'peak_reserved_bytes':ratio*100,'total_memory_bytes':100}
    assert choose([r(64,2,99),r(128,1,100),r(512,4,120,.9)])['micro_batch']==64

def test_no_idle_calls_or_old_fusion_implementation():
    root=Path(__file__).resolve().parents[1]
    for p in root.rglob('*.py'):
        tree=ast.parse(p.read_text(encoding='utf-8'))
        for n in ast.walk(tree):
            if isinstance(n,ast.Call):assert getattr(n.func,'attr',None)!='sleep' and getattr(n.func,'id',None)!='sleep'
    s=(root/'duplexfm_joint/model.py').read_text()
    assert 'pairing_maps' not in s and 'FourSourceFusion' not in s

def test_design_site_is_antiparallel_and_not_a_center_gate():
    b=batch(19,38);h=torch.zeros(1,32,22,60)
    # At design site t_real=5..23, guide position0 pairs to t_real23.
    for i in range(19):h[:,:,i,16+23-i]=1
    valid=b['guide_mask'][:,:,None]&b['target_mask'][:,None,:]
    sites,allowed=PairACFusion.site_pool(h,valid,b)
    assert allowed[0,39] and torch.allclose(sites[0,39],torch.ones(192))
    assert allowed[0].sum()>1
