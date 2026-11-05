"""Independent audit of cost targets, convex classifier, branches and hard paths."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import csv,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score
R=Path(__file__).resolve().parent;R1=R.parent/'round1';BASE=R.parents[1];OLD=BASE/'expert_banded_20261014'
B=BASE/'core_six_settings_20261011/benchmark';P=BASE/'expert_selection_20261014/results_download/verified'
S=('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix');M=('InterceptGate','LinearGate')
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b,tol=1e-10):np.testing.assert_allclose(a,b,atol=tol,rtol=tol)
def check(root,d):
    for n,h in d.items():
        p=(root/n).resolve();assert p.is_relative_to(root.resolve()) and sha(p)==h
def transform(x,t):
    z=(x-np.array(t['mean']))/np.array(t['scale'])
    return np.c_[z[:,:24],*[z[:,24+32*j:56+32*j]@np.array(t['components'][j]).T/np.array(t['pc_scale'][j]) for j in range(5)]]
def columns(c):return np.array(list(range(24))+[i for j in range(5) if c['mask']&(1<<j) for i in range(24+4*j,28+4*j)])

def main():
    started=time.monotonic();assert not (R/'verification.json').exists() and not (R/'execution.mutex').exists()
    lock=read(R/'lock.json');check(R,lock['files']);assert sha(R1/'lock.json')==lock['prior_lock_sha256']
    l1=read(R1/'lock.json');check(R1,l1['files'])
    for name,h in l1['dependencies'].items():assert sha(Path(name))==h
    assert sha(R1/'results/complete.json')==lock['prior_complete_sha256'] and sha(R1/'verification.json')==lock['prior_verification_sha256']
    assert read(R1/'verification.json')['passed'];check(R1/'results',read(R1/'results/complete.json')['files_sha256'])
    comp=read(R/'results/complete.json');check(R/'results',comp['files_sha256']);assert comp['lock_sha256']==sha(R/'lock.json')
    costs=[read(p) for p in sorted(R.glob('execution_*.json'))];assert costs and costs[-1]['returncode']==0 and all(x['GPU_hours']==0 for x in costs)
    rows=sum([table(B/'data'/f'{s}.csv') for s in S[:3]],[]);assert len(rows)==3329
    ids={r['sample_id']:i for i,r in enumerate(rows)};identity=table(P/'features/row_identity.csv');ix={r['sample_id']:i for i,r in enumerate(identity)}
    src=np.load(P/'features/slots.npy',allow_pickle=False)[[ix[r['sample_id']] for r in rows]].astype(float)
    x=np.c_[np.array([[float(v) for v in r['td'].split(',')] for r in rows]),src];y=np.array([float(r['label']) for r in rows]);binary=np.array([int(r['y']) for r in rows])
    plan=read(R1/'plan.json');oldcfg=read(OLD/'plan.json')['configs'];allpred=table(R/'results/oof_predictions.csv');summary=table(R/'results/summary.csv')
    assert len(allpred)==7596 and len(summary)==8;rebuild={};maxgrad=0.;maxnormal=0.
    for item in plan['folds']:
        s,f=item['setting'],item['fold'];out=R/'results/folds'/s/f'fold{f}';pr=R1/'results/folds'/s/f'fold{f}'
        training=read(out/'training.json');assert training['outer_validation_unused'] and training['test_labels_used'] is False
        assert training['fit_ids']==item['fit_ids'];fi=np.array([ids[i] for i in item['fit_ids']]);ti=np.array([ids[i] for i in item['test_ids']])
        assert not set(fi)&set(ti)
        sel=read(pr/'selection.json');alpha=plan['configs'][sel['selected']['PCA4_Full']]['alpha']
        oldsel=read(OLD/'results/folds'/s/f'fold{f}'/'selection.json');ki=oldsel['selected']['TD24_Ridge_CV'];tdalpha=oldcfg[ki]['alpha']
        cfg=[dict(mask=m,td=True,alpha=alpha) for m in (31,30,29,27,23,15)]+[dict(mask=0,td=True,alpha=tdalpha)]
        assert cfg==training['branch_configs'] and training['original_TD24_candidate']==ki
        with np.load(pr/'arrays.npz',allow_pickle=False) as p:
            cols=[next(k for k,c in enumerate(plan['configs']) if c==cf) for cf in cfg[:6]];cross=p['inner_oof'][:,cols]
        with np.load(OLD/'results/folds'/s/f'fold{f}'/'arrays.npz',allow_pickle=False) as p:cross=np.c_[cross,p['inner_oof'][:,ki]]
        a=np.load(out/'arrays.npz',allow_pickle=False);close(cross,a['crossfit_branches'],1e-12)
        err=(cross-y[fi,None])**2;tau=max(float(np.median(err.max(1)-err.min(1))),1e-8);close(tau,training['tau'],1e-12)
        logits=-(err-err.min(1,keepdims=True))/tau;q=np.exp(logits-logsumexp(logits,axis=1,keepdims=True));close(q,a['soft_targets'],1e-12)
        t=read(pr/'transforms.json')['final'];z=transform(x[fi],t);zt=transform(x[ti],t)
        theta=a['theta'];close(theta,training['theta'],1e-12);w=theta[:140].reshape(20,7);b=theta[140:]
        logits=z[:,24:]@w+b;lp=logits-logsumexp(logits,axis=1,keepdims=True);prob=np.exp(lp);diff=(prob-q)/len(fi)
        gradient=np.r_[(z[:,24:].T@diff+w).ravel(),diff.sum(0)];gn=float(np.max(abs(gradient)))
        assert gn<1e-7;maxgrad=max(maxgrad,gn);close(gn,training['gate']['max_gradient'],1e-12)
        loss=-np.mean(np.sum(q*lp,axis=1))+.5*np.sum(w*w);close(loss,training['gate']['objective'],1e-12)
        constant=int(np.argmax(q.mean(0)));assert constant==training['gate']['constant_route']
        coef=a['coef'];c=float(y[fi].mean());close(c,a['intercept'],1e-12);gram=z.T@z;rhs=z.T@(y[fi]-c)
        for k,cf in enumerate(cfg):
            keep=columns(cf);assert np.count_nonzero(coef[k,np.setdiff1d(np.arange(44),keep)])==0
            residual=((gram+cf['alpha']*np.eye(44))@coef[k]-rhs)[keep]/len(fi);nr=float(np.max(abs(residual)))
            assert nr<1e-8;maxnormal=max(maxnormal,nr)
        close(zt@coef.T+c,a['branch_pred'],1e-10)
        routes=np.c_[np.full(len(ti),constant),np.argmax(zt[:,24:]@w+b,axis=1)]
        np.testing.assert_array_equal(routes,a['routes']);pred=np.take_along_axis(a['branch_pred'],routes,axis=1);close(pred,a['test_pred'],1e-12)
        for j,m in enumerate(M):
            part=[r for r in allpred if r['setting']==s and int(r['fold'])==f and r['condition']==m]
            assert [r['sample_id'] for r in part]==item['test_ids']
            close([float(r['prediction']) for r in part],pred[:,j],1e-12)
            for r,i,k in zip(part,ti,routes[:,j]):assert int(r['route'])==k and int(r['mask'])==cfg[k]['mask'] and float(r['label'])==y[i] and int(r['binary'])==binary[i]
            rebuild.setdefault((s,m),[]).extend(zip(ti,pred[:,j]))
        a.close()
    baseline={(r['setting'],r['condition']):r for r in table(OLD/'results/summary.csv')}
    for r in summary:
        pairs=rebuild[r['setting'],r['condition']];ii=np.array([i for i,p in pairs]);p=np.array([p for i,p in pairs]);yy=y[ii];bb=binary[ii]
        expected={i for i,r0 in enumerate(rows) if r0['cohort']==('Mixset' if r['setting']=='Inter_Hu_to_Mix' else r['setting'])}
        assert len(ii)==len(expected) and set(ii)==expected
        vals=dict(n=len(ii),PCC=float(pearsonr(yy,p).statistic),Spearman=float(spearmanr(yy,p).statistic),MSE=float(np.mean((yy-p)**2)),
            MAE=float(np.mean(abs(yy-p))),AP=float(average_precision_score(bb,p)),AUROC=float(roc_auc_score(bb,p)),F1=float(f1_score(bb,p>=.5,zero_division=0)))
        for k,v in vals.items():close(float(r[k]),v,1e-10)
        base=baseline[r['setting'],'TD24_Ridge_CV'];bp=float(base['PCC']);close(float(r['TD24_PCC']),bp)
        close(float(r['PCC_delta']),vals['PCC']-bp);close(float(r['PCC_relative_gain_pct']),100*(vals['PCC']/bp-1))
        close(float(r['MSE_relative_reduction_pct']),100*(1-vals['MSE']/float(base['MSE'])))
    for r in read(R/'results/goal_summary.json'):
        part=[z for z in summary if z['condition']==r['condition']];g=float(np.mean([float(z['PCC_relative_gain_pct']) for z in part]))
        close(g,r['mean_relative_PCC_gain_pct']);assert r['reaches_development_5pct']==(g>=5)
    for u in table(R/'results/usage.csv'):
        count=sum(r['setting']==u['setting'] and r['fold']==u['fold'] and r['condition']==u['condition'] and r['route']==u['route'] for r in allpred)
        assert count==int(u['count'])
    result=dict(passed=True,folds=16,branches=112,linear_gates=16,OOF_rows=7596,max_gate_gradient=maxgrad,max_normal_residual=maxnormal,
        full_hard_routes_recomputed=True,training_only_costs=True,GPU_hours=0,source_updates=0,wall_seconds=time.monotonic()-started,
        lock_sha256=sha(R/'lock.json'),complete_sha256=sha(R/'results/complete.json'),verifier_sha256=sha(Path(__file__)))
    with (R/'verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
if __name__=='__main__':main()
