"""Once-only metrics, equal-length macro and grouped uncertainty; no selection."""
import json,time,csv,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import numpy as np
from scipy.stats import pearsonr,spearmanr
from eval_settings import METHODS,DATA_SHA
BASE=Path(__file__).resolve().parent
ROOT=BASE/'strict_v2'
PARENT=BASE.parent
from purge_sequences_strict import check_lock
from common import atomic_json,sha
from balanced_statistics import grouped_length_stats,correlation,bootstrap
import accessibility_metrics as am
def main():
 out=ROOT/'evaluation';assert not out.exists()
 check_lock()
 frozen=json.loads((ROOT/'mask_frozen.json').read_text());assert frozen['passed']
 for n,h in frozen['files'].items():assert sha(ROOT/n)==h,n
 audit=json.loads((ROOT/'input_audit.json').read_text());assert audit['passed']
 contract=json.loads((BASE/'analysis_lock_strict.json').read_text())
 for n,h in contract['files'].items():assert sha(BASE/n)==h,n
 out.mkdir()
 complete=json.loads((PARENT/'evaluation/complete.json').read_text());assert sha(PARENT/'dataset_v1/dataset.npz')==DATA_SHA
 assert complete['methods']==contract['prediction_sha256']
 for name,h in complete['methods'].items():assert sha(PARENT/'evaluation'/(name+'.npy'))==h,name
 with np.load(PARENT/'dataset_v1/dataset.npz') as z:d={k:z[k] for k in z.files}
 with np.load(ROOT/'retained_masks.npz') as z:subsets={k:z[k] for k in z.files}
 summary=[];by_length=[];stats={};draws={};start=time.time()
 for subset,select in subsets.items():
  y=d['reactivity'][select];lens=d['lengths'][select].astype('int64');mask=d['observed_mask'][select];groups=d['evaluation_group'][select]
  assert set(lens)==set(range(50,61))
  for name in METHODS:
   p=np.load(PARENT/'evaluation'/(name+'.npy'),mmap_mode='r')[select];st=grouped_length_stats(p,y,lens,groups);stats[subset+'__'+name]=st
   boot=bootstrap(st);draws[subset,name]=boot['macro'];per=[]
   for length in range(50,61):
    pick=lens==length;pp=p[pick,:length].astype(float);yy=y[pick,:length];metrics=am.profile_metrics(p[pick],y[pick],lens[pick])
    v=dict(subset=subset,model=name,length=length,windows=int(pick.sum()),positions=int(pick.sum())*length,
      pcc=float(pearsonr(pp.ravel(),yy.ravel()).statistic),spearman=float(spearmanr(pp.ravel(),yy.ravel()).statistic),
      mae=float(np.abs(pp-yy).mean()),mse=float(np.square(pp-yy).mean()),within_pcc=metrics['within_sequence_pearson'],within_defined=metrics['defined_within_sequences'])
    assert abs(v['pcc']-float(correlation(st[:,length-50].sum(0))))<1e-9
    ci=np.quantile(boot['per_length'][:,length-50],[.025,.975]);v.update(pcc_low=float(ci[0]),pcc_high=float(ci[1]));by_length.append(v);per.append(v)
   pcc=float(pearsonr(p[mask].astype(float),y[mask]).statistic);check=am.profile_metrics(p,y,lens)
   assert abs(pcc-float(correlation(st.sum((0,1)))))<1e-9 and abs(pcc-check['micro_pearson'])<1e-10
   macro={k:float(np.mean([v[k] for v in per])) for k in ['pcc','spearman','mae','mse','within_pcc']}
   low,high=np.quantile(boot['macro'],[.025,.975]);pl,ph=np.quantile(boot['pooled'],[.025,.975])
   summary.append(dict(subset=subset,model=name,windows=len(lens),positions=int(mask.sum()),groups=len(st),macro_pcc=macro['pcc'],macro_pcc_low=float(low),macro_pcc_high=float(high),
      macro_spearman=macro['spearman'],macro_mae=macro['mae'],macro_mse=macro['mse'],macro_within_pcc=macro['within_pcc'],
      pooled_pcc=pcc,pooled_pcc_low=float(pl),pooled_pcc_high=float(ph),pooled_spearman=float(spearmanr(p[mask],y[mask]).statistic),
      pooled_mae=float(np.abs(p[mask]-y[mask]).mean()),pooled_mse=float(np.square(p[mask]-y[mask]).mean())))
 comparisons=[]
 for subset in subsets:
  for ref in ['PairAC_original','rebuilt_mRNABERT']:
   for base in ['sequence','structure']:
    low,high=np.quantile(draws[subset,ref]-draws[subset,base],[.025,.975]);comparisons.append(dict(subset=subset,reference=ref,baseline=base,macro_pcc_delta_ci95=[float(low),float(high)],exploratory_fixed_fit=True))
 for filename,rows in [('summary.csv',summary),('per_length.csv',by_length)]:
  with (out/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 atomic_json(out/'summary.json',summary);atomic_json(out/'per_length.json',by_length);atomic_json(out/'paired_intervals.json',comparisons)
 np.savez_compressed(out/'group_length_stats.npz',**stats)
 atomic_json(out/'analysis_complete.json',dict(passed=True,summary_groups=len(summary),length_groups=len(by_length),bootstrap_draws=2000,seconds=time.time()-start,
   main='arithmetic mean of PCC across 11 lengths; same connected-group draws reused across all lengths',
   inference_complete_sha256=sha(PARENT/'evaluation/complete.json'),mask_receipt_sha256=sha(ROOT/'mask_frozen.json'),input_audit_sha256=sha(ROOT/'input_audit.json'),no_training_or_inference=True,files={n:sha(out/n) for n in ['summary.csv','summary.json','per_length.csv','per_length.json','paired_intervals.json','group_length_stats.npz']}))
 print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
