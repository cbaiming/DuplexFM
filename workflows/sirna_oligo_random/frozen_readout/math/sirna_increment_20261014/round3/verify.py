"""Independent forward/selection/coverage audit plus exact final training replay."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score
R=Path(__file__).resolve().parent;B=R.parents[1];S=('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix')
M=('P12_native_head','P13_native_head');EP=(8,16,32,64,100)
P=B/'expert_selection_20261014/results_download/verified/features';BENCH=B/'core_six_settings_20261011/benchmark'
sys.path.insert(0,str(R));from train import fit,state_hash
from native import P12,P13
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b,tol=1e-9):np.testing.assert_allclose(a,b,atol=tol,rtol=tol)
def check(root,files):
    for name,h in files.items():
        p=(root/name).resolve();assert p.is_relative_to(root.resolve()) and sha(p)==h,name
def normalizer(train,other,t):
    mean=train.mean(0,dtype=np.float64).astype('float32');std=train.std(0,dtype=np.float64).astype('float32');std[std<1e-6]=1
    np.testing.assert_array_equal(mean,t['mean']);np.testing.assert_array_equal(std,t['std'])
    assert t['eps']==1e-6 and t['method']=='train_zscore'
    return ((other-mean)/std).astype('float32')
def forward(x,state,method):
    a=torch.from_numpy(x)
    def branch(z,name):
        z=F.layer_norm(z,(z.shape[1],),state[name+'.0.weight'],state[name+'.0.bias'],1e-5)
        z=F.gelu(F.linear(z,state[name+'.1.weight'],state[name+'.1.bias']))
        return F.linear(z,state[name+'.4.weight'],state[name+'.4.bias']).squeeze(-1)
    with torch.no_grad():
        logit=branch(a[:,160:],'anchor')
        if method==M[1]:logit=logit+.5*torch.tanh(state['rho_raw'])*branch(a[:,:160],'concat_residual')
        return torch.sigmoid(logit).numpy().astype(float)
def metric(y,p,b):
    return dict(n=len(y),PCC=float(pearsonr(y,p).statistic),Spearman=float(spearmanr(y,p).statistic),
        MSE=float(np.mean((y-p)**2)),MAE=float(np.mean(abs(y-p))),AP=float(average_precision_score(b,p)),
        AUROC=float(roc_auc_score(b,p)),F1=float(f1_score(b,p>=.5,zero_division=0)))

def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for name,h in lock['dependencies'].items():assert sha(Path(name))==h,name
    rec=read(R/'results/complete.json');check(R/'results',rec['files_sha256']);assert rec['lock_sha256']==sha(R/'lock.json')
    executions=[read(p) for p in sorted(R.glob('execution_*.json'))];assert executions[-1]['returncode']==0 and all(x['GPU_hours']==0 for x in executions)
    plan=read(R/'plan.json');assert plan['folds']==read(R.parent/'round1/plan.json')['folds'] and plan['endpoints']==list(EP)
    rows=sum([table(BENCH/'data'/f'{s}.csv') for s in S[:3]],[]);assert len(rows)==3329
    index={r['sample_id']:i for i,r in enumerate(rows)};identity=table(P/'row_identity.csv');ii={r['sample_id']:i for i,r in enumerate(identity)}
    for row in rows:assert all(row[k]==identity[ii[row['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
    slots=np.load(P/'slots.npy',allow_pickle=False)[[ii[r['sample_id']] for r in rows]]
    x=np.c_[slots,np.array([[float(v) for v in r['td'].split(',')] for r in rows])].astype('float32')
    y=np.array([float(r['label']) for r in rows]);binary=np.array([int(r['y']) for r in rows])
    predictions=table(R/'results/oof_predictions.csv');summary=table(R/'results/summary.csv')
    assert len(predictions)==7596 and len(summary)==8
    count=0;replays=0;maxerr=0.;anchors={};canonical={}
    for item in plan['folds']:
        s,fold=item['setting'],item['fold'];fi=np.array([index[t] for t in item['fit_ids']]);ti=np.array([index[t] for t in item['test_ids']])
        assert not set(fi)&set(ti)
        for method,condition in zip(M,(P12,P13)):
            root=R/'results/folds'/s/f'fold{fold}'/method;oof=np.full((len(fi),5),np.nan)
            for j,sp in enumerate(item['inner']):
                tr=np.array(sp['train']);va=np.array(sp['valid']);out=root/f'inner{j}'
                data=torch.load(out/'result.pt',map_location='cpu',weights_only=False)
                initial=torch.load(out/'initial.pt',map_location='cpu',weights_only=False)
                assert initial['updates']==0 and initial['epoch']==0 and state_hash(initial['model'])==data['initial_sha256']
                assert data['epochs']==100 and data['updates']==100*((len(tr)+127)//128)
                assert data['trainable_parameters']==(881 if method==M[0] else 6387)
                assert list(data['states'])==list(EP) and len(data['history'])==100
                key=(s,fold,j)
                ah=state_hash({k:v for k,v in initial['model'].items() if k.startswith('anchor.')})
                if method==M[0]:anchors[key]=ah
                else:assert anchors[key]==ah
                xx=normalizer(x[fi[tr]],x[fi[va]],data['normalizer'])
                for k,ep in enumerate(EP):
                    pp=forward(xx,data['states'][ep],method);err=float(np.max(abs(pp-data['predictions'][:,k])));maxerr=max(maxerr,err)
                    close(pp,data['predictions'][:,k],1e-7);count+=1
                oof[va]=data['predictions']
                check(out,read(out/'complete.json')['files_sha256'])
            assert np.isfinite(oof).all()
            yc=y[fi]-y[fi].mean();pc=oof-oof.mean(0);den=np.sqrt((yc@yc)*np.sum(pc*pc,axis=0))
            corr=np.divide(yc@pc,den,out=np.full(5,-2.),where=den>1e-14);mse=np.mean((oof-y[fi,None])**2,axis=0)
            k=min(range(5),key=lambda z:(-corr[z],mse[z],EP[z]));sel=read(root/'selection.json')
            close(corr,sel['inner_PCC']);close(mse,sel['inner_MSE']);assert sel['selected']==k and sel['epoch']==EP[k]
            assert sel['fit_ids']==item['fit_ids'] and sel['test_ids']==item['test_ids'] and sel['outer_validation_unused']
            data=torch.load(root/'final/result.pt',map_location='cpu',weights_only=False)
            initial=torch.load(root/'final/initial.pt',map_location='cpu',weights_only=False)
            assert initial['updates']==0 and state_hash(initial['model'])==data['initial_sha256']
            ah=state_hash({k:v for k,v in initial['model'].items() if k.startswith('anchor.')})
            if method==M[0]:anchors[s,fold,'final']=ah
            else:assert anchors[s,fold,'final']==ah
            # Same fixed training repeated, using training inputs only; no test enters fit.
            replay=fit(x[fi],y[fi],condition,EP[k]);assert replay['initial_sha256']==data['initial_sha256']
            assert state_hash(replay['state'])==data['final_sha256']==state_hash(data['state']);replays+=1
            pp=forward(normalizer(x[fi],x[ti],data['normalizer']),data['state'],method)
            part=[r for r in predictions if r['setting']==s and int(r['fold'])==fold and r['condition']==method]
            assert [r['sample_id'] for r in part]==item['test_ids']
            canonical_p=np.array([float(r['prediction']) for r in part]);close(pp,canonical_p,1e-7)
            for row,i in zip(part,ti):assert float(row['label'])==y[i] and int(row['binary'])==binary[i]
            canonical.setdefault((s,method),[]).extend(zip(ti,canonical_p))
        print('VERIFIED',s,fold,flush=True)
    assert count==800 and replays==32
    baseline={(r['setting'],r['condition']):r for r in table(B/'expert_banded_20261014/results/summary.csv')}
    for r in summary:
        pairs=canonical[r['setting'],r['condition']];ii=np.array([i for i,_ in pairs]);p=np.array([v for _,v in pairs])
        cohort='Mixset' if r['setting']=='Inter_Hu_to_Mix' else r['setting'];expected={i for i,row in enumerate(rows) if row['cohort']==cohort}
        assert len(ii)==len(expected) and set(ii)==expected
        vals=metric(y[ii],p,binary[ii])
        for k,v in vals.items():close(float(r[k]),v)
        base=baseline[r['setting'],'TD24_Ridge_CV'];bp=float(base['PCC'])
        close(float(r['TD24_PCC']),bp);close(float(r['PCC_delta']),vals['PCC']-bp)
        close(float(r['PCC_relative_gain_pct']),100*(vals['PCC']/bp-1))
        close(float(r['MSE_relative_reduction_pct']),100*(1-vals['MSE']/float(base['MSE'])))
    for goal in read(R/'results/goal_summary.json'):
        part=[r for r in summary if r['condition']==goal['condition']];g=float(np.mean([float(r['PCC_relative_gain_pct']) for r in part]))
        close(goal['mean_relative_PCC_gain_pct'],g);assert goal['reaches_development_5pct']==(g>=5)
    result=dict(passed=True,outer_folds=16,inner_endpoint_forwards=count,final_exact_training_replays=replays,
        rows=7596,max_inner_forward_error=maxerr,fit_only_statistics_and_selection=True,paired_anchor_initial_equal=True,
        lock_sha256=sha(R/'lock.json'),complete_sha256=sha(R/'results/complete.json'),verifier_sha256=sha(Path(__file__)),
        CPU_wall_seconds=time.monotonic()-started,GPU_hours=0,source_updates=0)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
