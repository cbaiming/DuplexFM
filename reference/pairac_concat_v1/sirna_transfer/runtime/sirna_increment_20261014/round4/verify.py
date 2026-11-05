"""Independent readback of all PCA, linear equations and selected predictions."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score
R=Path(__file__).resolve().parent;B=R.parents[1];OLD=B/'expert_banded_20261014'
BENCH=B/'core_six_settings_20261011/benchmark';EXPORT=B/'core_six_settings_20261011/results/exports'
S=('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix');SOURCES=('core_real','full_real','initial_core');MODES=('Uniform40','Adaptive40')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b,tol=1e-9):np.testing.assert_allclose(a,b,atol=tol,rtol=tol)
def check(root,files):
    for name,h in files.items():
        p=(root/name).resolve();assert p.is_relative_to(root.resolve()) and sha(p)==h,name
def scale(x,floor):
    mu=x.mean(0);sd=x.std(0)
    if floor:
        fixed=np.ptp(x,axis=0)==0;v=sd[(~fixed)&(sd>0)];lo=.01*(np.median(v) if len(v) else 1.)
        sd=np.where(fixed,1.,np.maximum(sd,lo))
    else:sd[sd<1e-12]=1.
    return mu,sd
def project(train,other,t):
    tm,ts=scale(train[:,:24],False);sm,ss=scale(train[:,24:],True)
    for a,b in ((tm,t['td_mean']),(ts,t['td_scale']),(sm,t['src_mean']),(ss,t['src_scale'])):close(a,b,1e-11)
    z=(train[:,24:]-sm)/ss;pm=z.mean(0);close(pm,t['pca_mean'],1e-11);z-=pm
    g=z.T@z/len(z);c=np.array(t['components']);e=np.array(t['eigenvalues'])
    close(c@c.T,np.eye(16),1e-10);close(g@c.T,c.T*e,1e-8)
    close(e,np.maximum(np.linalg.eigvalsh(g)[-16:][::-1],0),1e-9)
    for row in c:assert row[np.argmax(abs(row))]>=0
    pcm,pcs=scale(z@c.T,True);close(pcm,t['pc_mean'],1e-10);close(pcs,t['pc_scale'],1e-10)
    return np.c_[(other[:,:24]-tm)/ts,(((other[:,24:]-sm)/ss-pm)@c.T-pcm)/pcs]
def metric(y,p,b):
    return dict(n=len(y),PCC=float(pearsonr(y,p).statistic),Spearman=float(spearmanr(y,p).statistic),
        MSE=float(np.mean((y-p)**2)),MAE=float(np.mean(abs(y-p))),AP=float(average_precision_score(b,p)),
        AUROC=float(roc_auc_score(b,p)),F1=float(f1_score(b,p>=.5,zero_division=0)))
def normal(z,y,b,cfg):
    n=40 if cfg['weight']>0 else 24;pen=np.r_[np.full(24,cfg['alpha']),np.full(16,cfg['alpha']/cfg['weight'] if cfg['weight'] else 0)]
    assert np.count_nonzero(b[n:])==0
    err=float(np.max(abs((z[:,:n].T@z[:,:n]+np.diag(pen[:n]))@b[:n]-z[:,:n].T@(y-y.mean())))/len(y))
    assert err<1e-8,err
    return err

def main():
    start=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'verification.json').exists()
    lock=read(R/'lock.json');check(R,lock['files'])
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    complete=read(R/'results/complete.json');assert complete['lock_sha256']==sha(R/'lock.json');check(R/'results',complete['files_sha256'])
    executions=[read(p) for p in sorted(R.glob('execution_*.json'))];assert executions[-1]['returncode']==0 and all(e['GPU_hours']==0 for e in executions)
    plan=read(R/'plan.json');assert plan['folds']==read(R.parent/'round1/plan.json')['folds']
    cfg=[dict(weight=w,alpha=a) for w in (0.,.01,.1,1.,10.) for a in (.1,1.,10.,100.,1000.)];assert cfg==plan['configs']
    rows=sum([table(BENCH/'data'/f'{s}.csv') for s in S[:3]],[]);assert len(rows)==3329
    ids={r['sample_id']:i for i,r in enumerate(rows)};td=np.array([[float(v) for v in r['td'].split(',')] for r in rows]);xx={}
    for name in SOURCES:
        identity=table(EXPORT/name/'row_identity.csv');idx={r['sample_id']:i for i,r in enumerate(identity)}
        for r in rows:assert all(r[k]==identity[idx[r['sample_id']]][k] for k in ('sample_id','cohort','guide','target57'))
        source=np.load(EXPORT/name/'source.npy',allow_pickle=False)[[idx[r['sample_id']] for r in rows]].astype(float)
        xx[name]=np.c_[td,source]
    y=np.array([float(r['label']) for r in rows]);binary=np.array([int(r['y']) for r in rows])
    pred=table(R/'results/oof_predictions.csv');summary=table(R/'results/summary.csv');assert len(pred)==22788 and len(summary)==24
    canonical={};count=0;maxres=0.;maxTDerror=0.;final=0
    for item in plan['folds']:
        s,fold=item['setting'],item['fold'];fi=np.array([ids[a] for a in item['fit_ids']]);ti=np.array([ids[a] for a in item['test_ids']])
        assert not set(fi)&set(ti)
        for source in SOURCES:
            x=xx[source];out=R/'results/folds'/s/f'fold{fold}'/source
            c=read(out/'complete.json');assert c['lock_sha256']==sha(R/'lock.json');check(out,c['files_sha256'])
            a=np.load(out/'arrays.npz',allow_pickle=False);t=read(out/'transforms.json');sel=read(out/'selection.json');oof=np.full((len(fi),25),np.nan);coverage=[]
            for j,sp in enumerate(item['inner']):
                tr=np.array(sp['train']);va=np.array(sp['valid']);coverage+=va.tolist()
                z=project(x[fi[tr]],x[fi[tr]],t['inner'][j]);zv=project(x[fi[tr]],x[fi[va]],t['inner'][j])
                close(a['inner_intercept'][j],y[fi[tr]].mean(),1e-12)
                for k,cf in enumerate(cfg):maxres=max(maxres,normal(z,y[fi[tr]],a['inner_coef'][j,k],cf));count+=1
                oof[va]=zv@a['inner_coef'][j].T+a['inner_intercept'][j]
            assert sorted(coverage)==list(range(len(fi)));close(oof,a['inner_oof'],1e-10)
            with np.load(OLD/'results/folds'/s/f'fold{fold}'/'arrays.npz',allow_pickle=False) as old:
                err=float(np.max(abs(oof[:,:5]-old['inner_oof'][:,5:10])));maxTDerror=max(maxTDerror,err);assert err<1e-10
            yc=y[fi]-y[fi].mean();pc=oof-oof.mean(0);den=np.sqrt((yc@yc)*np.sum(pc*pc,axis=0))
            corr=np.divide(yc@pc,den,out=np.full(25,-2.),where=den>1e-14);mse=np.mean((oof-y[fi,None])**2,axis=0)
            close(corr,sel['PCC']);close(mse,sel['MSE'])
            key=lambda k:(-corr[k],mse[k],cfg[k]['weight'],-cfg[k]['alpha'],k)
            chosen=dict(zip(MODES,(min(range(15,20),key=key),min(range(25),key=key))));assert chosen==sel['selected']
            assert sel['fit_ids']==item['fit_ids'] and sel['test_ids']==item['test_ids'] and sel['outer_validation_unused']
            z=project(x[fi],x[fi],t['final']);zt=project(x[fi],x[ti],t['final']);close(a['final_intercept'],y[fi].mean(),1e-12)
            for k,mode in enumerate(MODES):
                maxres=max(maxres,normal(z,y[fi],a['final_coef'][k],cfg[chosen[mode]]));final+=1
                pp=zt@a['final_coef'][k]+a['final_intercept'];close(pp,a['test_pred'][:,k],1e-10)
                condition=source+'_'+mode;part=[r for r in pred if r['setting']==s and int(r['fold'])==fold and r['condition']==condition]
                assert [r['sample_id'] for r in part]==item['test_ids']
                canonical_p=np.array([float(r['prediction']) for r in part]);close(canonical_p,pp,1e-10)
                for r,i in zip(part,ti):assert float(r['label'])==y[i] and int(r['binary'])==binary[i]
                canonical.setdefault((s,condition),[]).extend(zip(ti,canonical_p))
            a.close()
        print('VERIFIED',s,fold,flush=True)
    assert count==6000 and final==96
    baseline={(r['setting'],r['condition']):r for r in table(OLD/'results/summary.csv')}
    for row in summary:
        pairs=canonical[row['setting'],row['condition']];ii=np.array([i for i,p in pairs]);p=np.array([p for i,p in pairs])
        cohort='Mixset' if row['setting']=='Inter_Hu_to_Mix' else row['setting'];expected={i for i,r in enumerate(rows) if r['cohort']==cohort}
        assert len(ii)==len(expected) and set(ii)==expected
        m=metric(y[ii],p,binary[ii])
        for k,v in m.items():close(float(row[k]),v)
        b=baseline[row['setting'],'TD24_Ridge_CV'];bp=float(b['PCC']);close(float(row['TD24_PCC']),bp)
        close(float(row['PCC_delta']),m['PCC']-bp);close(float(row['PCC_relative_gain_pct']),100*(m['PCC']/bp-1))
        close(float(row['MSE_relative_reduction_pct']),100*(1-m['MSE']/float(b['MSE'])))
    for r in read(R/'results/goal_summary.json'):
        part=[x for x in summary if x['condition']==r['condition']];g=float(np.mean([float(x['PCC_relative_gain_pct']) for x in part]))
        close(r['mean_relative_PCC_gain_pct'],g);assert r['reaches_development_5pct']==(g>=5)
        assert r['primary']==(r['condition']=='core_real_Adaptive40')
    result=dict(passed=True,inner_normal_equations=count,final_heads=final,OOF_rows=22788,source_models=3,
        max_normal_residual=maxres,max_TD24_candidate_prediction_difference=maxTDerror,
        all_PCA_selection_predictions_metrics_checked=True,source_checkpoint_hashes_verified=True,
        lock_sha256=sha(R/'lock.json'),complete_sha256=sha(R/'results/complete.json'),verifier_sha256=sha(Path(__file__)),
        GPU_hours=0,new_source_forward=0,CPU_wall_seconds=time.monotonic()-start)
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
