"""Three frozen PairAC masked sources with the unchanged, certified R55 readout."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,hashlib,importlib.util,io,json,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score

ROOT=Path(__file__).resolve().parent;RESEARCH=ROOT.parent/'runtime';INC=RESEARCH/'sirna_increment_20261014';R55=INC/'round55'
BENCH=RESEARCH/'core_six_settings_20261011/benchmark';BASE=RESEARCH/'expert_banded_20261014/results'
VARIANTS=('no_pair_seed43', 'no_cross_seed43', 'no_access_seed43', 'no_pair_seed44', 'no_cross_seed44', 'no_access_seed44');SETTINGS=('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix')
sys.path[:0]=[str(INC/'round4'),str(RESEARCH/'expert_first_20261014')]
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
head=load('_ablation_pairac_readout55',R55/'head.py')
scale=load('_ablation_pairac_scale4',INC/'round4/core.py')
math_audit=load('_ablation_pairac_audit55',R55/'audit_math.py')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def table(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def plain(v):
    if isinstance(v,np.ndarray):return plain(v.tolist())
    if isinstance(v,np.generic):return plain(v.item())
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(plain(v),f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def csv_write(p,rows):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(plain(rows))
def npz(p,**arrays):
    with p.open('xb') as f:np.savez_compressed(f,**arrays)
def check(root,files):
    for n,h in files.items():assert sha(root/n)==h,n
def rows():
    result=[]
    for name,n in zip(SETTINGS[:3],(2196,469,664)):
        part=table(BENCH/'data'/f'{name}.csv');assert len(part)==n;result+=part
    assert len({r['sample_id'] for r in result})==3329;return result
def fit_scaler(x):
    x=np.asarray(x,float);assert x.ndim==2 and x.shape[1]==216 and np.isfinite(x).all()
    tm,ts=scale.scaler(x[:,:24]);sm,ss=scale.scaler(x[:,24:],True)
    return dict(td_mean=tm,td_scale=ts,src_mean=sm,src_scale=ss)
def metric(part):
    y=np.array([float(r['label']) for r in part]);p=np.array([float(r['prediction']) for r in part]);b=np.array([int(r['binary']) for r in part])
    assert np.isfinite(p).all()
    return dict(n=len(y),PCC=float(pearsonr(y,p).statistic),Spearman=float(spearmanr(y,p).statistic),MSE=float(np.mean((y-p)**2)),
      MAE=float(np.mean(abs(y-p))),AP=float(average_precision_score(b,p)) if len(set(b))==2 else None,
      AUROC=float(roc_auc_score(b,p)) if len(set(b))==2 else None,F1=float(f1_score(b,p>=.5,zero_division=0)) if len(set(b))==2 else None)

def register():
    assert not (ROOT/'readout_lock.json').exists()
    port=read(ROOT/'port_manifest.json');deps={}
    for name,h in port['files_sha256'].items():
        p=ROOT/name;assert sha(p)==h,name;deps[str(p)]=h
    for name in SETTINGS[:3]:
        p=BENCH/'data'/f'{name}.csv';h=port['benchmark_sha256'][name];assert sha(p)==h;deps[str(p)]=h
    assert read(ROOT/'synthetic_receipt.json')['passed']
    deps[str(ROOT/'synthetic_receipt.json')]=sha(ROOT/'synthetic_receipt.json')
    deps[str(ROOT/'port_manifest.json')]=sha(ROOT/'port_manifest.json')
    write(ROOT/'readout_lock.json',dict(dependencies=deps,variants=VARIANTS,settings=SETTINGS,
        alphas=head.ALPHAS,new_inner_heads=2400,new_final_heads=96,rows=22788,
        source_seeds=[43,44],new_PCA=0,CPU_only=True,maximum_seconds=1800,
        mathematical_head_modified=False,original_readout_sha256=port['original_readout_sha256'],
        baseline='original_TD24_Ridge_CV',same_treatment_TD_reported=True))
    print('REGISTERED',sha(ROOT/'readout_lock.json'),flush=True)

def validate():
    lock=read(ROOT/'readout_lock.json')
    for n,h in lock['dependencies'].items():assert sha(n)==h,n
    return lock

def input_features(rs):
    imported=ROOT/'exported';receipt=read(imported/'complete.json');check(imported,receipt['files_sha256'])
    assert receipt['status']=='complete' and receipt['rows_per_ablation']==3329 and receipt['new_source_rows']==19974
    identity=read(imported/'export_lock.json')['rows'][:3329]
    assert identity==[{k:r[k] for k in ('sample_id','cohort','guide','target57')} for r in rs]
    req=read(ROOT/'request.json');td=np.array([[float(x) for x in r['td'].split(',')] for r in rs],float)
    result={}
    for name in VARIANTS:
        folder=imported/'features'/name;c=read(folder/'complete.json');check(folder,c['files_sha256'])
        assert c['identity']['checkpoint_sha256']==req['checkpoint_sha256'][name] and c['identity']['variant']==name
        assert c['identity']['source_seed']==int(name[-2:])
        assert c['identity']['checkpoint']==('initial.pt' if name.startswith('initial') else 'terminal.pt')
        assert c['export_lock_sha256']==sha(imported/'export_lock.json')
        assert c['identity']['source_registration_sha256']==req['source_registration_sha256']
        a=np.load(folder/'source.npy',allow_pickle=False).astype(float);assert a.shape==(3329,192) and np.isfinite(a).all()
        result[name]=np.c_[td,a[:3329]]
    return result

def groups(item,rs,fi,stage):
    label='final' if stage==5 else f'inner{stage}'
    path=R55/'results/groups'/item['setting']/f"fold{item['fold']}"/(label+'.json');saved=read(path)
    tr=np.arange(len(fi)) if stage==5 else np.array(item['inner'][stage]['train'])
    assert saved['train_indices']==tr.tolist() and saved['train_ids']==[item['fit_ids'][i] for i in tr]
    text='\n'.join(rs[fi[i]]['target57'] for i in tr);assert hashlib.sha256(text.encode()).hexdigest()==saved['target_strings_sha256']
    return np.array(saved['labels'],int),sha(path)

def run(start):
    validate();out=ROOT/'readouts';assert not out.exists();out.mkdir()
    rs=rows();xx=input_features(rs);lookup={r['sample_id']:i for i,r in enumerate(rs)}
    plan=read(R55/'plan.json');allpred=[];selections=[];count=0
    write(out/'feature_binding.json',dict(complete_sha256=sha(ROOT/'exported/complete.json'),
        request_sha256=sha(ROOT/'request.json'),readout_lock_sha256=sha(ROOT/'readout_lock.json')))
    for item in plan['folds']:
        s,f=item['setting'],item['fold'];fi=np.array([lookup[x] for x in item['fit_ids']]);ti=np.array([lookup[x] for x in item['test_ids']])
        assert set(fi).isdisjoint(ti);yf=np.array([float(rs[i]['label']) for i in fi])
        for source in VARIANTS:
            x=xx[source];folder=out/'folds'/s/f'fold{f}'/source;folder.mkdir(parents=True)
            oof=np.full((len(fi),5),np.nan);coefs=np.empty((5,5,216));intercepts=np.empty((5,5));transforms=[];group_shas=[]
            coverage=np.zeros(len(fi),int)
            for j,split in enumerate(item['inner']):
                tr=np.array(split['train']);va=np.array(split['valid']);assert set(tr).isdisjoint(va)
                labels,gh=groups(item,rs,fi,j);group_shas.append(gh)
                t=fit_scaler(x[fi[tr]]);transforms.append(t);z=head.transform(x[fi[tr]],t);zv=head.transform(x[fi[va]],t)
                for k,a in enumerate(head.ALPHAS):
                    b,c,cert=head.fit(z,yf[tr],labels,a,True,deadline=start+1800);assert cert['certified']
                    coefs[j,k]=b;intercepts[j,k]=c;oof[va,k]=zv@b+c;count+=1
                    write(folder/f'inner{j}_alpha{k}_certificate.json',cert)
                coverage[va]+=1
            assert np.all(coverage==1) and np.isfinite(oof).all()
            pc,ms=scale.score(yf,oof);chosen=head.choose(pc,ms);a=head.ALPHAS[chosen]
            write(folder/'selection.json',dict(alpha=a,candidate=chosen,inner_PCC=pc,inner_MSE=ms,
                fit_ids=item['fit_ids'],test_ids=item['test_ids'],outer_validation_unused=True))
            labels,gh=groups(item,rs,fi,5);group_shas.append(gh);t=fit_scaler(x[fi]);z=head.transform(x[fi],t);zt=head.transform(x[ti],t)
            b,c,cert=head.fit(z,yf,labels,a,True,deadline=start+1800);assert cert['certified'];count+=1
            pred=zt@b+c;dc,di=head.deployment(t,b,c);np.testing.assert_allclose(pred,x[ti]@dc+di,atol=1e-10,rtol=1e-10)
            write(folder/'transforms.json',dict(inner=transforms,final=t));write(folder/'final_certificate.json',cert)
            write(folder/'group_references.json',dict(training_group_sha256=group_shas))
            npz(folder/'arrays.npz',inner_coef=coefs,inner_intercept=intercepts,inner_oof=oof,final_coef=b,final_intercept=c,
                deployment_coef=dc,deployment_intercept=di,test_pred=pred)
            part=[dict(setting=s,fold=f,source=source,sample_id=rs[i]['sample_id'],label=rs[i]['label'],binary=rs[i]['y'],prediction=float(pred[j])) for j,i in enumerate(ti)]
            csv_write(folder/'predictions.csv',part);allpred+=part
            selections.append(dict(setting=s,fold=f,source=source,alpha=a,candidate=chosen,inner_PCC=float(pc[chosen])))
            write(folder/'complete.json',dict(readout_lock_sha256=sha(ROOT/'readout_lock.json'),
                files_sha256={p.name:sha(p) for p in folder.iterdir() if p.is_file()}))
        print('CLOSED',s,f,flush=True)
    assert count==2496 and len(allpred)==22788
    write(out/'models_fixed.json',dict(heads=count,new_inner=2400,new_final=96,rows=22788,all_models_fixed=True,
        files_sha256={p.relative_to(out).as_posix():sha(p) for p in (out/'folds').rglob('complete.json')}))
    base={r['setting']:r for r in table(BASE/'summary.csv') if r['condition']=='TD24_Ridge_CV'}
    refs={(r['setting'],r['source']):r for r in table(R55/'results/summary.csv')}
    pairac_full={(r['setting'],r['source']):r for r in table(RESEARCH/'pairac_transfer_20261017/readouts/summary.csv') if r['source'] in ('trained43','trained44')}
    summary=[];effects=[]
    for source in VARIANTS:
        for setting in SETTINGS:
            part=[r for r in allpred if r['source']==source and r['setting']==setting]
            expected={r['sample_id'] for r in rs if r['cohort']==('Mixset' if setting==SETTINGS[3] else setting)}
            assert len(part)==len(expected) and {r['sample_id'] for r in part}==expected
            m=metric(part);bp=float(base[setting]['PCC']);td=float(refs[setting,'TD_only']['PCC']);full=float(pairac_full[setting,'trained'+source[-2:]]['PCC'])
            summary.append(dict(setting=setting,source=source,**m,original_TD24_PCC=bp,PCC_relative_gain_pct=100*(m['PCC']/bp-1),
                same_treatment_TD_PCC=td,PCC_gain_vs_same_TD_pct=100*(m['PCC']/td-1),Full_PCC=full,
                Full_minus_source_PCC=full-m['PCC'],Full_reference='PairAC_trained'+source[-2:],
                MSE_reduction_vs_TD24_pct=100*(1-m['MSE']/float(base[setting]['MSE'])),
                MSE_reduction_vs_same_TD_pct=100*(1-m['MSE']/float(refs[setting,'TD_only']['MSE']))))
    means=[]
    for name in VARIANTS:
        part=[r for r in summary if r['source']==name]
        means.append(dict(source=name,mean_relative_PCC_gain_pct=float(np.mean([r['PCC_relative_gain_pct'] for r in part])),
             mean_relative_gain_vs_same_TD_pct=float(np.mean([r['PCC_gain_vs_same_TD_pct'] for r in part])),
             negative_settings=[r['setting'] for r in part if r['PCC_relative_gain_pct']<0]))
    csv_write(out/'summary.csv',summary);csv_write(out/'oof_predictions.csv',allpred);csv_write(out/'selection.csv',selections)
    write(out/'mean_gains.json',means);validate()
    write(out/'complete.json',dict(status='complete',heads=count,new_inner=2400,new_final=96,rows=22788,outer_folds=16,
        readout_lock_sha256=sha(ROOT/'readout_lock.json'),CPU_wall_seconds=time.monotonic()-start,
        files_sha256={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}))
    print('COMPLETE',means,flush=True)

def verify(start):
    validate();out=ROOT/'readouts';comp=read(out/'complete.json');check(out,comp['files_sha256'])
    rs=rows();xx=input_features(rs);lookup={r['sample_id']:i for i,r in enumerate(rs)}
    errors=[];count=0;max_deploy=0.;joined=[]
    for item in read(R55/'plan.json')['folds']:
        s,f=item['setting'],item['fold'];fi=np.array([lookup[x] for x in item['fit_ids']]);ti=np.array([lookup[x] for x in item['test_ids']]);yf=np.array([float(rs[i]['label']) for i in fi])
        for source in VARIANTS:
            x=xx[source];folder=out/'folds'/s/f'fold{f}'/source;ts=read(folder/'transforms.json');sel=read(folder/'selection.json');arrays=np.load(folder/'arrays.npz',allow_pickle=False)
            replay=np.full((len(fi),5),np.nan)
            for j,split in enumerate(item['inner']):
                tr=np.array(split['train']);va=np.array(split['valid']);g,_=groups(item,rs,fi,j)
                expected=fit_scaler(x[fi[tr]])
                for k,v in expected.items():np.testing.assert_array_equal(v,ts['inner'][j][k])
                z=head.transform(x[fi[tr]],ts['inner'][j]);zv=head.transform(x[fi[va]],ts['inner'][j]);profile=math_audit.profile(z,yf[tr],g)
                for k,a in enumerate(head.ALPHAS):
                    b=arrays['inner_coef'][j,k];c=arrays['inner_intercept'][j,k];cert=read(folder/f'inner{j}_alpha{k}_certificate.json')
                    errors.append(math_audit.check_source(z,yf[tr],g,b,c,cert,a,cached=profile));count+=1;replay[va,k]=zv@b+c
            np.testing.assert_allclose(replay,arrays['inner_oof'],atol=1e-12,rtol=1e-12)
            pc,ms=scale.score(yf,replay);assert head.choose(pc,ms)==sel['candidate'] and head.ALPHAS[sel['candidate']]==sel['alpha']
            np.testing.assert_allclose(pc,sel['inner_PCC'],atol=1e-12,rtol=1e-12);np.testing.assert_allclose(ms,sel['inner_MSE'],atol=1e-12,rtol=1e-12)
            g,_=groups(item,rs,fi,5)
            for k,v in fit_scaler(x[fi]).items():np.testing.assert_array_equal(v,ts['final'][k])
            z=head.transform(x[fi],ts['final']);b=arrays['final_coef'];c=float(arrays['final_intercept'])
            errors.append(math_audit.check_source(z,yf,g,b,c,read(folder/'final_certificate.json'),sel['alpha']));count+=1
            p=head.transform(x[ti],ts['final'])@b+c;np.testing.assert_allclose(p,arrays['test_pred'],atol=1e-12,rtol=1e-12)
            raw=x[ti]@arrays['deployment_coef']+arrays['deployment_intercept'];max_deploy=max(max_deploy,float(np.max(abs(raw-p))))
            part=table(folder/'predictions.csv');assert [r['sample_id'] for r in part]==item['test_ids']
            np.testing.assert_allclose([float(r['prediction']) for r in part],p,atol=1e-12,rtol=1e-12)
            for r,i in zip(part,ti):assert float(r['label'])==float(rs[i]['label']) and int(r['binary'])==int(rs[i]['y'])
            joined+=part
        print('AUDITED',s,f,flush=True)
    assert count==2496 and len(joined)==22788 and max_deploy<1e-10
    summary={(r['source'],r['setting']):r for r in table(out/'summary.csv')}
    for source in VARIANTS:
        for setting in SETTINGS:
            part=[r for r in joined if r['source']==source and r['setting']==setting];actual=metric(part);row=summary[source,setting]
            assert len(part)==len({r['sample_id'] for r in part})==actual['n']
            for key,value in actual.items():
                if value is not None:np.testing.assert_allclose(value,float(row[key]),atol=1e-12,rtol=1e-12)
    write(ROOT/'readout_verification.json',dict(passed=True,heads=count,OOF_rows=len(joined),max_TD_normal_error=float(np.max(np.array(errors)[:,0])),
        max_relative_dual_gap=float(np.max(np.array(errors)[:,1])),max_deployment_error=max_deploy,
        complete_sha256=sha(out/'complete.json'),CPU_wall_seconds=time.monotonic()-start,new_fits=0))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('register','run','verify'));a=ap.parse_args();t=time.monotonic();rc=1;error=None
    try:
        {'register':lambda _:register(),'run':run,'verify':verify}[a.mode](t);rc=0
    except BaseException:error=traceback.format_exc();raise
    finally:write(ROOT/f'{a.mode}_cost_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-t,GPU_hours=0))
