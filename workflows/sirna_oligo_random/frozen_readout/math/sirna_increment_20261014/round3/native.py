"""Load only named, hash-pinned model functions; never execute legacy CLI/imports."""
import ast, math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Sequence
import numpy as np
import torch
from torch import nn

WORK=Path(__file__).resolve().parents[3]
RUNTIME=WORK/'DuplexFM-main/runtime'
FILES=(RUNTIME/'benchmark_sirna_ar_bridge_moe_expert32_td24.py',
       RUNTIME/'tools/run_sirna_td24_ungated_residual_control.py',
       RUNTIME/'tools/run_sirna_td24_regression_only_controls.py')
EXPERTS=('accessibility','utr_cross_attention','score_matrix_attention','biological_priors','intarna_energy_geometry')
P12='p12_td24_anchor_regression_only'
P13='p13_td24_ungated_source_residual_regression_only'
UNGATED='p10_td24_ungated_source_residual_binary'

def functions(path,names,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

ns=dict(torch=torch,nn=nn,np=np,math=math,Dict=Dict,Sequence=Sequence,Any=Any,
        EXPERT_GROUPS={'structure':[],'interaction':[],'bioprior':[]})
functions(FILES[0],('inverse_bounded_rho','make_model_class','fit_normalizer','apply_normalizer'),ns)
bridge=SimpleNamespace(inverse_bounded_rho=ns['inverse_bounded_rho'],
    EXPERT_GROUPS=ns['EXPERT_GROUPS'],sp=SimpleNamespace(torch=torch,nn=nn))
un=dict(bridge=bridge,Dict=Dict,Sequence=Sequence,EXPERT_NAMES=EXPERTS,
    CONDITION=UNGATED,_ORIGINAL_MAKE_MODEL_CLASS=ns['make_model_class'])
functions(FILES[1],('make_model_class',),un)
reg=dict(bridge=bridge,P0_REGRESSION=P12,P1_REGRESSION=P13)
functions(FILES[2],('compute_loss',),reg)

def model(condition,seed=42):
    torch.manual_seed(seed)
    args=dict(expert_names=EXPERTS,expert_dims={n:32 for n in EXPERTS},td24_dim=24,
        anchor_hidden_dim=32,branch_hidden_dim=32,branch_output_dim=16,router_hidden_dim=16,
        router_temperature=1.,dropout=.1,rho_init=.1)
    if condition==P12:return ns['make_model_class'](torch,nn)(condition='p0_td24_anchor',**args).cpu()
    if condition==P13:return un['make_model_class'](torch,nn)(condition=UNGATED,**args).cpu()
    raise ValueError(condition)

def normalize_fit(x):return ns['fit_normalizer'](x,'train_zscore',1e-6)
def normalize(x,t):return ns['apply_normalizer'](x,t)
def loss(condition,output,y):
    return reg['compute_loss'](condition=condition,output=output,target_continuous=y,
        target_binary=None,args=SimpleNamespace(residual_l2_weight=.03))[0]

def predict(m,x):
    m.eval()
    with torch.no_grad():return torch.sigmoid(m(torch.as_tensor(x,dtype=torch.float32))['final_logit']).numpy().astype(float)

def parameters(m):return sum(p.numel() for p in m.parameters() if p.requires_grad)
