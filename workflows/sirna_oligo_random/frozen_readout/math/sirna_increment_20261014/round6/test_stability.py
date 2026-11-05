import os,sys
from pathlib import Path
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pytest
from scipy.stats import pearsonr
from stability import select
CFG=[{'weight':w,'alpha':a} for w in (0.,.01,.1,1.,10.) for a in (.1,1.,10.,100.,1000.)]
def example():
    rng=np.random.default_rng(42);y=rng.normal(size=60);p=y[:,None]+rng.normal(size=(60,25))
    blocks=[list(range(i*12,(i+1)*12)) for i in range(5)];return y,p,blocks

def test_independent_selection_definition():
    y,p,blocks=example();d=select(y,p,CFG,blocks,range(25));winners=[]
    def key(k,ii):return (-pearsonr(y[ii],p[ii,k]).statistic,np.mean((y[ii]-p[ii,k])**2),CFG[k]['weight'],-CFG[k]['alpha'],k)
    for block in blocks:
        ii=np.array([i for i in range(60) if i not in block]);winners.append(min(range(25),key=lambda k:key(k,ii)))
    expected=min(range(25),key=lambda k:(-winners.count(k),key(k,np.arange(60))))
    assert d['winners']==winners and d['selected']==expected

def test_tied_scores_retain_fixed_rule():
    y,p,blocks=example();p=np.repeat(y[:,None],25,axis=1)
    assert select(y,p,CFG,blocks,range(25))['selected']==4
    assert select(y,p,CFG,blocks,range(15,20))['selected']==19

def test_partition_is_mandatory():
    y,p,blocks=example();blocks[0].append(blocks[1][0])
    with pytest.raises(ValueError):select(y,p,CFG,blocks,range(25))

def test_constant_response_rejected():
    y,p,blocks=example()
    with pytest.raises(ValueError):select(np.ones_like(y),p,CFG,blocks,range(25))

def test_block_order_not_a_tuning_parameter():
    y,p,blocks=example();a=select(y,p,CFG,blocks,range(25));b=select(y,p,CFG,blocks[::-1],range(25))
    assert a['selected']==b['selected'] and a['votes']==b['votes'] and a['winners']==b['winners'][::-1]
