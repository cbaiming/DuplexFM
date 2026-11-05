"""Independent numerical audit. Does not import two_stage or run_experiment."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score

HERE=Path(__file__).resolve().parent
COHORTS=('Huesken','Mixset','Takayuki','wang2009','alas1_patent')
SETTINGS=('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix','wang2009','alas1_patent')
NAMES=('ExpertsOnly_Ridge','ExpertsOnly_GroupSparse','SelectedExperts_Ridge','SelectedExperts_TD24_Ridge')
OLD=('TD24_Ridge','Expert5_Ridge','Expert5_GroupSparse')
EXPERTS=('accessibility','utr_cross_attention','score_matrix_attention','biological_priors','intarna_energy_geometry')


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b,tol=1e-9):np.testing.assert_allclose(a,b,atol=tol,rtol=tol)
def files(root,manifest):
    for name,digest in manifest.items():
        p=(root/name).resolve();assert p.is_relative_to(root.resolve())
        assert sha(p)==digest,name


def measures(part):
    y=np.array([float(r['label']) for r in part]);p=np.array([float(r['prediction']) for r in part]);b=np.array([int(r['binary']) for r in part])
    return dict(n=len(y),PCC=float(pearsonr(y,p).statistic) if y.std()>0 and p.std()>0 else None,
        Spearman=float(spearmanr(y,p).statistic) if y.std()>0 and p.std()>0 else None,
        MSE=float(np.mean((y-p)**2)),MAE=float(np.mean(abs(y-p))),
        AP=float(average_precision_score(b,p)) if len(set(b))==2 else None,
        AUROC=float(roc_auc_score(b,p)) if len(set(b))==2 else None,F1=float(f1_score(b,p>=.5,zero_division=0)))


def check_metrics(row,part):
    for k,v in measures(part).items():
        if v is None:assert row[k]==''
        else:close(float(row[k]),v,1e-10)


def verify(root,prior,benchmark):
    t=time.monotonic()
    if (root/'local_verification.json').exists():raise FileExistsError('Do not repeat completed audit')
    assert not (root/'execution.mutex').exists(),'Execution must have exited'
    lock=read(root/'experiment_lock.json');locksha=sha(root/'experiment_lock.json')
    files(root,lock['code']);files(benchmark,lock['benchmark'])
    assert sha(root/'pytest.xml')==lock['pytest_sha256']
    for name,k in [('features/complete.json','feature_complete_sha256'),('results/complete.json','prior_complete_sha256'),
                   ('local_verification.json','prior_verification_sha256'),('readout_lock.json','prior_readout_lock_sha256')]:
        assert sha(prior/name)==lock[k],name
    files(prior/'features',read(prior/'features/complete.json')['files_sha256'])
    files(prior/'results',read(prior/'results/complete.json')['files_sha256'])
    complete=read(root/'results/complete.json');assert complete['experiment_lock_sha256']==locksha
    files(root/'results',complete['files_sha256'])
    costs=[read(p) for p in root.glob('execution_*.json')]
    assert costs and costs[-1]['returncode']==0
    assert all(c['experiment_lock_sha256']==locksha and c['new_GPU_hours']==0 and c['device']=='CPU' for c in costs)
    rows=[]
    for co in COHORTS:rows+=table(benchmark/'data'/f'{co}.csv')
    identities=table(prior/'features/row_identity.csv')
    assert len(rows)==len(identities)==3688
    assert all(all(a[k]==b[k] for k in ('sample_id','cohort','guide','target57')) for a,b in zip(rows,identities))
    ids={r['sample_id']:i for i,r in enumerate(rows)};assert len(ids)==3688
    s=np.load(prior/'features/slots.npy',allow_pickle=False).astype(np.float64)
    td=np.array([[float(v) for v in r['td'].split(',')] for r in rows]);x=np.c_[td,s]
    y=np.array([float(r['label']) for r in rows])
    fullpred=table(root/'results/combined_oof_predictions.csv');newpred=table(root/'results/new_oof_predictions.csv')
    oldpred=table(prior/'results/oof_predictions.csv')
    assert fullpred[:len(oldpred)]==oldpred and fullpred[len(oldpred):]==newpred
    assert len(newpred)==16628 and len(fullpred)==29099
    selections=table(root/'results/selection.csv');foldmetrics=table(root/'results/fold_metrics.csv')
    assert len(selections)==len(foldmetrics)==104
    maxkkt=0.;maxridge=0.;nf=0;nc=0;stitched=[]
    for setting in SETTINGS:
        paths=sorted((benchmark/'splits'/setting).glob('fold*.json'))
        assert len(paths)==(1 if setting=='Inter_Hu_to_Mix' else 5)
        for path in paths:
            sp=read(path);fold=sp['fold'];d=root/'results/folds'/setting/f'fold{fold}'
            receipt=read(d/'complete.json');assert receipt['experiment_lock_sha256']==locksha
            files(d,receipt['files_sha256'])
            decision=read(d/'selection_lock.json');assert decision['stage1_sha256']==sha(d/'stage1.json')
            assert decision['split_sha256']==sha(path) and decision['experiment_lock_sha256']==locksha
            assert decision['TD24_used'] is False
            for a,b in [('fit_ids','fit_sample_ids'),('validation_ids','validation_sample_ids'),('test_ids','test_sample_ids')]:assert decision[a]==sp[b]
            fi,vi,ti=[[ids[k] for k in sp[name]] for name in ('fit_sample_ids','validation_sample_ids','test_sample_ids')]
            assert not(set(fi)&set(vi) or set(fi)&set(ti) or set(vi)&set(ti))
            first=read(d/'stage1.json');second=read(d/'stage2.json')
            mu=s[fi].mean(0);sd=s[fi].std(0);sd[sd<1e-12]=1
            close(first['mean'],mu,1e-12);close(first['scale'],sd,1e-12)
            z=(s[fi]-mu)/sd;zv=(s[vi]-mu)/sd;zt=(s[ti]-mu)/sd;n=len(fi)
            assert len(first['trials'])==30
            assert {(r['alpha'],r['ratio']) for r in first['trials']}=={(a,b) for a in (.1,1,10,100,1000) for b in (1,.3,.1,.03,.01,0)}
            for r in first['trials']:
                coef=np.array(r['coef']);inter=r['intercept'];a=r['alpha'];lam=r['lambda_']
                close(inter,y[fi].mean(),1e-12)
                q=z.T@(y[fi]-inter)/n
                lm=max(np.linalg.norm(q[j*32:(j+1)*32])/np.sqrt(32) for j in range(5))
                close(lm,r['lambda_max'],1e-12);close(lam,lm*r['ratio'],1e-12)
                grad=z.T@(z@coef+inter-y[fi])/n+a/n*coef
                norms=[];errors=[]
                for j in range(5):
                    b=coef[j*32:(j+1)*32];g=grad[j*32:(j+1)*32];bn=np.linalg.norm(b);norms.append(float(bn))
                    errors.append(float(np.linalg.norm(g+lam*np.sqrt(32)*b/bn)) if bn>0 else float(max(0,np.linalg.norm(g)-lam*np.sqrt(32))))
                err=max(errors);maxkkt=max(maxkkt,err)
                assert err<=1e-7*(1+max(abs(q)))+1e-12,(setting,fold,err)
                close(norms,r['group_norms'],1e-12);assert r['active_groups']==sum(v>0 for v in norms)
                close(r['val_mse'],np.mean((zv@coef+inter-y[vi])**2),1e-11)
                if r['ratio']==1:assert np.count_nonzero(coef)==0
                if r['ratio']==0:
                    ref=Ridge(alpha=a,solver='svd').fit(z,y[fi])
                    close(zt@coef+inter,ref.predict(zt),1e-8)
            key=lambda r:(r['val_mse'],r['active_groups'],-r['alpha'],-r['ratio'])
            assert first['selected']==min(first['trials'],key=key)
            assert first['dense']==min((r for r in first['trials'] if r['ratio']==0),key=key)
            mask=[v>0 for v in first['selected']['group_norms']]
            assert mask==first['mask']==decision['mask']
            expected={NAMES[0]:zt@np.array(first['dense']['coef'])+first['dense']['intercept'],
                      NAMES[1]:zt@np.array(first['selected']['coef'])+first['selected']['intercept']}
            chosen={NAMES[0]:first['dense'],NAMES[1]:first['selected']}
            for name,with_td in zip(NAMES[2:],(False,True)):
                st=second[name];assert st['mask']==mask and st['with_td']==with_td
                cols=(list(range(24)) if with_td else [])+[k for j,v in enumerate(mask) if v for k in range(24+32*j,24+32*(j+1))]
                assert st['columns']==cols
                mu2=x[fi][:,cols].mean(0);sd2=x[fi][:,cols].std(0);sd2[sd2<1e-12]=1
                close(st['mean'],mu2,1e-12);close(st['scale'],sd2,1e-12)
                z2=(x[fi][:,cols]-mu2)/sd2;v2=(x[vi][:,cols]-mu2)/sd2;t2=(x[ti][:,cols]-mu2)/sd2
                assert len(st['trials'])==5 and {r['alpha'] for r in st['trials']}=={.1,1,10,100,1000}
                for r in st['trials']:
                    b=np.array(r['coef']);c=r['intercept'];res=z2@b+c-y[fi]
                    close(res.mean(),0.,1e-10)
                    if cols:
                        err=float(np.max(np.abs((z2.T@res+r['alpha']*b)/n)));maxridge=max(maxridge,err)
                        assert err<1e-9,(setting,fold,name,err)
                    else:assert len(b)==0
                    close(r['val_mse'],np.mean((v2@b+c-y[vi])**2),1e-10)
                assert st['selected']==min(st['trials'],key=lambda r:(r['val_mse'],-r['alpha']))
                chosen[name]=st['selected'];expected[name]=t2@np.array(st['selected']['coef'])+st['selected']['intercept']
            part=table(d/'predictions.csv');stitched+=part
            for name in NAMES:
                pred=[r for r in part if r['condition']==name]
                assert [r['sample_id'] for r in pred]==sp['test_sample_ids']
                close([float(r['prediction']) for r in pred],expected[name],1e-10)
                for r,i in zip(pred,ti):
                    assert float(r['label'])==y[i] and int(r['binary'])==int(rows[i]['y']) and r['cohort']==rows[i]['cohort']
                sel=[r for r in selections if r['condition']==name and r['setting']==setting and int(r['fold'])==fold]
                assert len(sel)==1;sel=sel[0]
                expected_mask=[v>0 for v in chosen[name]['group_norms']] if name in NAMES[:2] else mask
                assert [int(sel[k]) for k in EXPERTS]==[int(v) for v in expected_mask]
                assert int(sel['active_groups'])==sum(expected_mask)
                close(float(sel['alpha']),chosen[name]['alpha']);close(float(sel['validation_mse']),chosen[name]['val_mse'])
                fm=[r for r in foldmetrics if r['condition']==name and r['setting']==setting and int(r['fold'])==fold]
                assert len(fm)==1;check_metrics(fm[0],pred)
            nf+=1;nc+=40
    assert stitched==newpred and nf==26 and nc==1040
    summary=table(root/'results/summary.csv');assert len(summary)==42
    for row in summary:
        assert row['origin']==('reused' if row['condition'] in OLD else 'new')
        part=[r for r in fullpred if r['condition']==row['condition'] and r['setting']==row['setting']]
        co='Mixset' if row['setting']=='Inter_Hu_to_Mix' else row['setting']
        wanted={r['sample_id'] for r in rows if r['cohort']==co}
        assert len(part)==len(wanted) and {r['sample_id'] for r in part}==wanted
        check_metrics(row,part)
    result=dict(passed=True,folds=nf,new_candidate_checks=nc,new_selected_heads=104,new_OOF_rows=16628,
        reused_OOF_rows=12471,all_stage1_choices_recomputed=True,all_stage2_masks_unchanged=True,
        all_predictions_reconstructed=True,all_metrics_recomputed=True,prior_bytes_unchanged=True,
        max_group_KKT=maxkkt,max_Ridge_normal_residual=maxridge,source_updates=0,GPU_hours=0,
        wall_seconds=time.monotonic()-t,experiment_lock_sha256=locksha,
        results_complete_sha256=sha(root/'results/complete.json'),verifier_sha256=sha(Path(__file__)),
        execution_wall_seconds_including_failures=sum(c['wall_seconds'] for c in costs))
    with (root/'local_verification.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=HERE)
    p.add_argument('--prior',type=Path,default=HERE.parent/'expert_selection_20261014/results_download/verified')
    p.add_argument('--benchmark',type=Path,default=HERE.parent/'core_six_settings_20261011/benchmark')
    a=p.parse_args();verify(a.root.resolve(),a.prior.resolve(),a.benchmark.resolve())
