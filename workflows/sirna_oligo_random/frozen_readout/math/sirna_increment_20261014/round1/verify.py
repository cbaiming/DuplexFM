"""Independent CPU verification, without importing the fitted model code."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score
R=Path(__file__).resolve().parent;BASE=R.parents[1];OLD=BASE/'expert_banded_20261014'
P=BASE/'expert_selection_20261014/results_download/verified';B=BASE/'core_six_settings_20261011/benchmark'
M=('PCA4_Full','PCA4_GlobalHard','PCA4_SourceOnly');S=('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b,tol=1e-9):np.testing.assert_allclose(a,b,atol=tol,rtol=tol)
def check(root,items):
    for name,h in items.items():
        p=(root/name).resolve();assert p.is_relative_to(root.resolve()) and sha(p)==h,name
def active(c):return np.array((list(range(24)) if c['td'] else [])+[i for j in range(5) if c['mask']&(1<<j) for i in range(24+4*j,28+4*j)])

def transformation(train,other,t):
    mu=train.mean(0);sd=train.std(0);sd[sd<1e-12]=1
    close(mu,t['mean'],1e-12);close(sd,t['scale'],1e-12)
    z=(train-mu)/sd;v=(other-mu)/sd;parts=[v[:,:24]]
    for j in range(5):
        a=z[:,24+32*j:56+32*j];g=a.T@a/len(a);comp=np.array(t['components'][j]);e=np.array(t['eigenvalues'][j])
        close(comp@comp.T,np.eye(4),1e-10);close(g@comp.T,comp.T*e,1e-9)
        close(e,np.maximum(np.linalg.eigvalsh(g)[-4:][::-1],0),1e-9)
        scale=np.sqrt(e);scale[scale<1e-12]=1;close(scale,t['pc_scale'][j],1e-12)
        for row in comp:assert row[np.argmax(abs(row))]>=0
        parts.append(v[:,24+32*j:56+32*j]@comp.T/scale)
    return np.c_[*parts]

def metric(y,p,b):
    return dict(n=len(y),PCC=float(pearsonr(y,p).statistic),Spearman=float(spearmanr(y,p).statistic),
        MSE=float(np.mean((y-p)**2)),MAE=float(np.mean(abs(y-p))),AP=float(average_precision_score(b,p)),
        AUROC=float(roc_auc_score(b,p)),F1=float(f1_score(b,p>=.5,zero_division=0)))

def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for name,h in lock['dependencies'].items():assert sha(Path(name))==h
    complete=read(R/'results/complete.json');check(R/'results',complete['files_sha256']);assert complete['lock_sha256']==sha(R/'lock.json')
    executions=[read(p) for p in R.glob('execution_*.json')];assert executions and executions[-1]['returncode']==0 and all(x['GPU_hours']==0 for x in executions)
    rows=sum([table(B/'data'/f'{s}.csv') for s in S[:3]],[]);assert len(rows)==3329
    ids={r['sample_id']:i for i,r in enumerate(rows)};identity=table(P/'features/row_identity.csv');idx={r['sample_id']:i for i,r in enumerate(identity)}
    for r in rows:assert all(r[k]==identity[idx[r['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
    src=np.load(P/'features/slots.npy',allow_pickle=False)[[idx[r['sample_id']] for r in rows]].astype(float)
    x=np.c_[np.array([[float(v) for v in r['td'].split(',')] for r in rows]),src]
    y=np.array([float(r['label']) for r in rows]);binary=np.array([int(r['y']) for r in rows])
    plan=read(R/'plan.json');oldplan=read(OLD/'plan.json');assert plan['folds']==[f for f in oldplan['folds'] if f['setting'] in S]
    cfg=[dict(mask=m,td=True,alpha=a) for m in (31,30,29,27,23,15,0) for a in (.1,1.,10.,100.,1000.)]+[
        dict(mask=31,td=False,alpha=a) for a in (.1,1.,10.,100.,1000.)]
    assert cfg==plan['configs'] and plan['excluded']==['wang2009','alas1_patent']
    oldaudit=read(OLD/'local_verification.json');assert oldaudit['passed'] and oldaudit['all_selection_and_predictions_checked']
    predictions=table(R/'results/oof_predictions.csv');summary=table(R/'results/summary.csv');selections=table(R/'results/selection.csv')
    assert len(predictions)==11394 and len(summary)==12 and len(selections)==48
    count=0;maxres=0.;canonical={}
    for item in plan['folds']:
        s,f=item['setting'],item['fold'];out=R/'results/folds'/s/f'fold{f}'
        rec=read(out/'complete.json');check(out,rec['files_sha256']);assert rec['lock_sha256']==sha(R/'lock.json')
        a=np.load(out/'arrays.npz',allow_pickle=False);t=read(out/'transforms.json');sel=read(out/'selection.json')
        assert sel['outer_validation_unused'] and sel['fit_ids']==item['fit_ids'] and sel['test_ids']==item['test_ids']
        fi=np.array([ids[z] for z in item['fit_ids']]);ti=np.array([ids[z] for z in item['test_ids']])
        assert not set(fi)&set(ti);oof=np.full((len(fi),40),np.nan);coverage=[]
        for j,sp in enumerate(item['inner']):
            tr=np.array(sp['train']);va=np.array(sp['valid']);coverage+=va.tolist()
            z=transformation(x[fi[tr]],x[fi[tr]],t['inner'][j]);zv=transformation(x[fi[tr]],x[fi[va]],t['inner'][j])
            c=y[fi[tr]].mean();close(c,a['inner_intercept'][j],1e-12)
            gram=z.T@z;q=z.T@(y[fi[tr]]-c)
            for k,conf in enumerate(cfg):
                b=a['inner_coef'][j,k];keep=active(conf);assert np.count_nonzero(b[np.setdiff1d(np.arange(44),keep)])==0
                residual=((gram+conf['alpha']*np.eye(44))@b-q)[keep]/len(tr)
                err=float(np.max(abs(residual)));assert err<1e-8;maxres=max(maxres,err);count+=1
            oof[va]=zv@a['inner_coef'][j].T+c
        assert sorted(coverage)==list(range(len(fi)));close(oof,a['inner_oof'],1e-10)
        pc=oof-oof.mean(0);yc=y[fi]-y[fi].mean();den=np.sqrt(np.sum(pc*pc,axis=0)*(yc@yc))
        corr=np.divide(yc@pc,den,out=np.full(40,-2.),where=den>1e-14);mse=np.mean((oof-y[fi,None])**2,axis=0)
        close(corr,sel['PCC'],1e-10);close(mse,sel['MSE'],1e-10)
        key=lambda k:(-corr[k],mse[k],cfg[k]['mask'].bit_count(),-cfg[k]['alpha'],k)
        chosen=dict(zip(M,[min(range(5),key=key),min(range(35),key=key),min(range(35,40),key=key)]))
        assert chosen==sel['selected']
        z=transformation(x[fi],x[fi],t['final']);zt=transformation(x[fi],x[ti],t['final']);c=y[fi].mean()
        close(c,a['final_intercept'],1e-12);g=z.T@z;q=z.T@(y[fi]-c)
        for k,m in enumerate(M):
            conf=cfg[chosen[m]];keep=active(conf);b=a['final_coef'][k]
            assert np.count_nonzero(b[np.setdiff1d(np.arange(44),keep)])==0
            err=float(np.max(abs(((g+conf['alpha']*np.eye(44))@b-q)[keep]/len(fi))));assert err<1e-8;maxres=max(maxres,err)
            close(zt@b+c,a['test_pred'][:,k],1e-10)
            part=[r for r in predictions if r['setting']==s and int(r['fold'])==f and r['condition']==m]
            assert [r['sample_id'] for r in part]==item['test_ids']
            close([float(r['prediction']) for r in part],a['test_pred'][:,k],1e-12)
            for r,i in zip(part,ti):assert float(r['label'])==y[i] and int(r['binary'])==binary[i]
            canonical.setdefault((s,m),[]).extend(zip(ti,a['test_pred'][:,k]))
        a.close()
    assert count==3200
    baseline={(r['setting'],r['condition']):r for r in table(OLD/'results/summary.csv')}
    for r in summary:
        pairs=canonical[r['setting'],r['condition']];ii=np.array([i for i,p in pairs]);p=np.array([p for i,p in pairs])
        cohort='Mixset' if r['setting']=='Inter_Hu_to_Mix' else r['setting'];expected={i for i,r0 in enumerate(rows) if r0['cohort']==cohort}
        assert len(ii)==len(expected) and set(ii)==expected
        vals=metric(y[ii],p,binary[ii])
        for k,v in vals.items():close(float(r[k]),v,1e-10)
        base=baseline[r['setting'],'TD24_Ridge_CV'];bp=float(base['PCC'])
        close(float(r['TD24_PCC']),bp,1e-12);close(float(r['PCC_delta']),vals['PCC']-bp,1e-12)
        close(float(r['PCC_relative_gain_pct']),100*(vals['PCC']/bp-1),1e-10)
        close(float(r['MSE_relative_reduction_pct']),100*(1-vals['MSE']/float(base['MSE'])),1e-10)
    goals=read(R/'results/goal_summary.json')
    for r in goals:
        part=[x0 for x0 in summary if x0['condition']==r['condition']];gain=float(np.mean([float(x0['PCC_relative_gain_pct']) for x0 in part]))
        close(r['mean_relative_PCC_gain_pct'],gain,1e-10);assert r['reaches_development_5pct']==(gain>=5)
        assert r['negative_settings']==[x0['setting'] for x0 in part if float(x0['PCC_delta'])<0]
    result=dict(passed=True,outer_folds=16,inner_folds=80,inner_normal_equations=3200,final_heads=48,
        OOF_rows=11394,excluded=['wang2009','alas1_patent'],max_normal_residual=maxres,
        split_purging_reused_from_verified_locked_prior=True,all_pca_and_selection_and_predictions_checked=True,
        lock_sha256=sha(R/'lock.json'),complete_sha256=sha(R/'results/complete.json'),verifier_sha256=sha(Path(__file__)),
        GPU_hours=0,wall_seconds=time.monotonic()-started)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
