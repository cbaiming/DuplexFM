"""Verify saved inputs, splits, predictions and metrics using Python's standard library."""
from __future__ import annotations
import argparse,csv,hashlib,itertools,json,math,statistics,time
from collections import Counter,defaultdict
from pathlib import Path
SETTINGS=('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix')
METRICS=('PCC','Spearman','MSE','RMSE','MAE','AP','AUROC','F1')
MODELS={'PairAC_TD24_R55':('42','43','44'),'TD24_R55':('deterministic',),
        'OligoFormer_corrected':('42','43','44'),'BERT-siRNA':('42','43','44'),'RNAiSpline':('42','43','44')}

def read_csv(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha256(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_csv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ranks(values):
 order=sorted(range(len(values)),key=values.__getitem__);out=[0.]*len(values);position=0
 for _,group in itertools.groupby(order,key=values.__getitem__):
  indices=list(group);midpoint=position+(len(indices)+1)/2.
  for i in indices:out[i]=midpoint
  position+=len(indices)
 return out

def corr(a,b):
 n=len(a);am,bm=math.fsum(a)/n,math.fsum(b)/n
 da,db=[x-am for x in a],[x-bm for x in b]
 denominator=math.sqrt(math.fsum(x*x for x in da)*math.fsum(x*x for x in db))
 return None if denominator==0 else math.fsum(x*y for x,y in zip(da,db))/denominator

def metrics(rows):
 a,b,y=[float(r['label']) for r in rows],[float(r['prediction']) for r in rows],[int(r['binary']) for r in rows]
 n=len(rows);mse=math.fsum((x-z)**2 for x,z in zip(a,b))/n;pos=sum(y);neg=n-pos;ap=auc=f1=None
 if pos and neg:
  tp=total=0;ap=0.
  for _,group in itertools.groupby(sorted(zip(b,y),reverse=True),key=lambda t:t[0]):
   block=list(group);new_tp=sum(z for _,z in block);tp+=new_tp;total+=len(block)
   ap+=(new_tp/pos)*(tp/total)
  rank=ranks(b);auc=(math.fsum(r for r,z in zip(rank,y) if z)-pos*(pos+1)/2)/(pos*neg)
  tp=sum(z==1 and x>=.5 for x,z in zip(b,y));fp=sum(z==0 and x>=.5 for x,z in zip(b,y));fn=pos-tp
  f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
 return dict(PCC=corr(a,b),Spearman=corr(ranks(a),ranks(b)),MSE=mse,RMSE=math.sqrt(mse),
             MAE=math.fsum(abs(x-z) for x,z in zip(a,b))/n,AP=ap,AUROC=auc,F1=f1)

def avg(rows,k):
 values=[r[k] for r in rows]
 return None if any(v is None for v in values) else statistics.mean(values)

def make_tables(predictions):
 folds,seeds=defaultdict(list),defaultdict(list)
 for r in predictions:
  folds[r['model'],r['seed'],r['setting'],int(r['fold'])].append(r)
  seeds[r['model'],r['seed'],r['setting']].append(r)
 fold_rows=[dict(model=m,seed=s,setting=d,fold=f,n=len(rs),**metrics(rs)) for (m,s,d,f),rs in sorted(folds.items())]
 pooled=[dict(model=m,seed=s,setting=d,n=len(rs),**metrics(rs)) for (m,s,d),rs in sorted(seeds.items())]
 means=[]
 for m,s,d in sorted(seeds):
  rs=[r for r in fold_rows if (r['model'],r['seed'],r['setting'])==(m,s,d)]
  means.append(dict(model=m,seed=s,setting=d,n=sum(r['n'] for r in rs),**{k:avg(rs,k) for k in METRICS}))
 output={'fold_metrics.csv':fold_rows}
 for method,rows in [('pooled_oof',pooled),('fold_mean',means)]:
  all_rows=list(rows)
  for model,ss in MODELS.items():
   for seed in ss:
    rs=[r for r in rows if r['model']==model and r['seed']==seed]
    assert {r['setting'] for r in rs}==set(SETTINGS)
    all_rows.append(dict(model=model,seed=seed,setting='Macro4',n='',**{k:avg(rs,k) for k in METRICS}))
  summaries=[]
  for model in MODELS:
   for setting in SETTINGS+('Macro4',):
    rs=[r for r in all_rows if r['model']==model and r['setting']==setting]
    row=dict(model=model,setting=setting,seeds=len(rs),n=rs[0]['n'])
    for k in METRICS:
     vs=[r[k] for r in rs];valid=all(v is not None for v in vs)
     row[k+'_mean']=statistics.mean(vs) if valid else None
     row[k+'_SD']=statistics.stdev(vs) if valid and len(vs)>1 else None
    summaries.append(row)
  output['seed_metrics_'+method+'.csv']=all_rows;output['setting_summary_'+method+'.csv']=summaries
 return output

def rc(s):return s.translate(str.maketrans('ACGU','UGCA'))[::-1]
def kmers(s):return {min(s[i:i+25],rc(s[i:i+25])) for i in range(len(s)-24)}
def neighbor_edges(data):
 km,gb=defaultdict(list),defaultdict(list)
 for sid,r in sorted(data.items()):
  for k in kmers(r['target57']):km[k].append(sid)
  # Any guide pair with <=2 mismatches shares one of these three disjoint blocks.
  for block,(a,b) in enumerate(((0,7),(7,13),(13,19))):gb[block,r['guide'][a:b]].append(sid)
 edges=set();examined=set()
 for ids in km.values():edges.update(tuple(sorted(x)) for x in itertools.combinations(ids,2))
 for ids in gb.values():
  for a,b in itertools.combinations(ids,2):
   pair=tuple(sorted((a,b)))
   if pair in edges or pair in examined:continue
   examined.add(pair)
   if sum(x!=z for x,z in zip(data[a]['guide'],data[b]['guide']))<=2:edges.add(pair)
 return edges

def compare_table(path,calculated):
 saved=read_csv(path);assert len(saved)==len(calculated),f'Table length: {path.name}'
 for old,new in zip(saved,calculated):
  assert set(old)==set(new),f'Table columns: {path.name}'
  for k,expected in new.items():
   value=old[k]
   if expected is None or expected=='':assert value=='',f'Unexpected value: {path.name}/{k}'
   elif isinstance(expected,(int,float)):
    assert math.isclose(float(value),expected,rel_tol=2e-11,abs_tol=2e-12),f'Metric mismatch: {path.name}/{k}'
   else:assert value==expected,f'Identity mismatch: {path.name}/{k}'

def verify(root):
 started=time.perf_counter();manifest=read_json(root/'audit/files_sha256.json')
 for relative,expected in manifest.items():assert sha256(root/relative)==expected,f'SHA256 mismatch: {relative}'
 exempt={'audit/files_sha256.json','audit/verification.json'}
 actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.relative_to(root).as_posix() not in exempt}
 assert actual==set(manifest),'Manifest inventory mismatch'
 data={}
 for cohort,count in [('Huesken',2196),('Mixset',469),('Takayuki',664)]:
  rows=read_csv(root/'data'/(cohort+'.csv'));assert len(rows)==count
  for r in rows:
   sid=r['sample_id'];assert sid not in data and r['cohort']==cohort
   assert len(r['guide'])==19 and len(r['target57'])==57 and set(r['guide']+r['target57'])<=set('ACGU')
   assert r['target19']==r['target57'][19:38]==rc(r['guide'])
   assert r['siRNA']==r['guide'] and r['mRNA']==r['target57']
   assert int(r['y']) in (0,1) and math.isfinite(float(r['label']))
   assert len(r['td'].split(','))==24 and all(math.isfinite(float(x)) for x in r['td'].split(','))
   data[sid]=r
 edges=neighbor_edges(data);splits={};audit=[]
 for setting in SETTINGS:
  coverage=Counter()
  for fold in range(1 if setting=='Inter_Hu_to_Mix' else 5):
   sp=read_json(root/'splits'/setting/f'fold{fold}.json')
   assert sp['protocol']=='purged_sequence_v1' and sp['trainable'] and sp['split_seed']==20261007
   assert not sp['gene_holdout'] and not sp['full_transitive_component_holdout'] and not sp['labels_used_to_construct_split']
   assert sp['setting']==setting and int(sp['fold'])==fold
   for cohort,h in sp['data_sha256'].items():assert sha256(root/'data'/(cohort+'.csv'))==h
   parts={n:sp[k] for n,k in [('fit','fit_sample_ids'),('val','validation_sample_ids'),('test','test_sample_ids'),('purged','purged_sample_ids')]}
   assert all(len(v)==len(set(v)) for v in parts.values())
   assert all(set(parts[a]).isdisjoint(parts[b]) for a,b in itertools.combinations(parts,2))
   universe={sid for sid,r in data.items() if r['cohort'] in {sp['train_cohort'],sp['test_cohort']}}
   assert set().union(*(set(v) for v in parts.values()))==universe
   assigned={sid:n for n,ids in parts.items() if n!='purged' for sid in ids}
   for a,b in edges:assert a not in assigned or b not in assigned or assigned[a]==assigned[b],f'Direct neighbor overlap: {setting}/{fold}'
   assert len(parts['fit'])>=20 and len(parts['val'])>=5 and len(parts['test'])>=10
   coverage.update(parts['test']);splits[setting,fold]=parts
   audit.append(dict(setting=setting,fold=fold,fit=len(parts['fit']),validation=len(parts['val']),test=len(parts['test']),purged=len(parts['purged']),fit_val_direct_edges=0,fit_test_direct_edges=0,val_test_direct_edges=0))
  cohort='Mixset' if setting=='Inter_Hu_to_Mix' else setting
  assert dict(coverage)=={sid:1 for sid,r in data.items() if r['cohort']==cohort}
 compare_table(root/'results/split_audit.csv',audit)
 predictions=read_csv(root/'predictions/all_models.csv');assert len(predictions)==49374
 counts=Counter();groups=defaultdict(set)
 for r in predictions:
  m,s,d,f,sid=r['model'],r['seed'],r['setting'],int(r['fold']),r['sample_id']
  assert m in MODELS and s in MODELS[m] and sid in data and math.isfinite(float(r['prediction']))
  assert float(r['label'])==float(data[sid]['label']) and int(r['binary'])==int(data[sid]['y'])
  assert math.isclose(float(r['source_label']),float(r['label']),rel_tol=0,abs_tol=1e-7)
  counts[m,s,d,f,sid]+=1;groups[m,s,d,f].add(sid)
 assert all(n==1 for n in counts.values()),'Duplicate prediction identity'
 for model,seeds in MODELS.items():
  for seed in seeds:
   for (setting,fold),parts in splits.items():
    assert groups.pop((model,seed,setting,fold))==set(parts['test']),f'Prediction coverage: {model}/{setting}/{fold}'
    if model in {'PairAC_TD24_R55','TD24_R55'}:
     receipt=read_json(root/'audit/selection'/model/setting/f'fold{fold}'/(seed+'.json'))
     assert set(receipt['fit_ids'])==set(parts['fit']) and set(receipt['test_ids'])==set(parts['test']) and receipt['outer_validation_unused']
     alphas=(.1,1.,10.,100.,1000.);pcc=receipt.get('inner_PCC',receipt.get('PCC'));mse=receipt.get('inner_MSE',receipt.get('MSE'))
     chosen=max(range(5),key=lambda i:(pcc[i],-mse[i],alphas[i]));assert receipt['alpha']==alphas[chosen]
 assert not groups
 baseline_receipts=0
 for model in ('BERT-siRNA','RNAiSpline'):
  for seed in MODELS[model]:
   for (setting,fold),parts in splits.items():
    rec=read_json(root/'audit/baseline_fits'/model/setting/f'fold{fold}'/(seed+'.json'))
    assert rec['model']==model and str(rec['seed'])==seed and rec['setting']==setting and int(rec['fold'])==fold
    assert rec['split_sha256']==sha256(root/'splits'/setting/f'fold{fold}.json')
    assert not rec['test_used_for_selection'] and rec['selection_metric']=='validation_mse'
    for a,b in [('fit_sample_ids','fit'),('validation_sample_ids','val'),('test_sample_ids','test')]:assert rec[a]==parts[b]
    if model=='RNAiSpline':
     assert rec['protocol']=='purged_sequence_v1' and rec['status']=='complete' and not rec['smoke']
     assert rec['SSL_sample_ids']==parts['fit'] and not rec['old_task_supervised_checkpoint_reused']
     assert rec['test_prediction_calls']==1 and rec['checkpoint_selection_exact_minimum']
    else:assert rec['complete'] and rec['fit_n']==len(parts['fit']) and rec['test_n']==len(parts['test']) and rec['val_n']==len(parts['val'])
    baseline_receipts+=1
 tables=make_tables(predictions)
 for name,rows in tables.items():compare_table(root/'results'/name,rows)
 return dict(status='passed',protocol='purged_sequence_v1',new_training=False,master_samples=len(data),split_manifests=len(splits),direct_neighbor_edges_within_master=len(edges),cross_fit_validation_test_direct_neighbor_edges=0,prediction_rows=len(predictions),prediction_runs=208,selection_receipts_checked=64,baseline_training_receipts_checked=baseline_receipts,public_files_sha256_checked=len(manifest),numeric_tables_checked=len(tables),elapsed_seconds=round(time.perf_counter()-started,3),scope='Saved inputs, partitions, truth, prediction identity, selection receipts, SHA256 and metrics; training is not rerun.')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--write-report',action='store_true');a=p.parse_args();r=verify(a.root.resolve())
 if a.write_report:(a.root/'audit/verification.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(r,indent=2))
if __name__=='__main__':main()
