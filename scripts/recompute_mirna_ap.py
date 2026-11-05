"""Recompute individual-prediction AP for the three Full-model miRNA prediction banks."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parents[1]
ROW_START={'AGO2_eCLIP_Klimentova2022_test':2495535,'AGO2_CLASH_Hejret2023_test':2496489,
           'AGO2_eCLIP_Manakov2022_test':2497507,'AGO2_eCLIP_Manakov2022_leftout':2821678}
COUNTS={'AGO2_CLASH_Hejret2023_test':1018,'AGO2_eCLIP_Klimentova2022_test':954,
        'AGO2_eCLIP_Manakov2022_test':324171,'AGO2_eCLIP_Manakov2022_leftout':20054}
CONDITIONS={'no_pair':'no_pair','no_cross':'no_cross','no_access':'no_access'}

def independent_ap(y,p):
    order=np.argsort(-p,kind='stable');y=y[order];p=p[order]
    end=np.r_[np.flatnonzero(np.diff(p)!=0),len(y)-1]
    tp=np.cumsum(y,dtype='float64')[end];precision=tp/(end+1)
    return float(np.sum(np.diff(np.r_[0.,tp])*precision)/y.sum())

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT);a=parser.parse_args()
    with (a.root/'results/miRNA/mirna_per_test.csv').open(encoding='utf-8-sig') as f:expected=list(csv.DictReader(f))
    records=[]
    banks=[('seed'+str(s),'Full',s) for s in [42,43,44]]

    for bank,condition,seed in banks:
        folder=a.root/'data/mirna_predictions'/bank
        if not folder.is_dir():raise FileNotFoundError('Missing retained bank: '+bank)
        for name,n in COUNTS.items():
            files=sorted((folder/name).glob('*.npz'))
            if not files:raise FileNotFoundError('Missing prediction shards: '+bank+'/'+name)
            ids=[];labels=[];scores=[]
            for p in files:
                with np.load(p,allow_pickle=False) as z:
                    ids.append(z['rows']);labels.append(z['labels']);scores.append(z['scores'])
            ids=np.concatenate(ids);y=np.concatenate(labels).astype('float64');p=np.concatenate(scores).astype('float64')
            if len(ids)!=n or len(np.unique(ids))!=n or len(y)!=n or len(p)!=n or not np.isfinite(p).all() or not np.isin(y,[0,1]).all():raise ValueError('Invalid row count/identity/labels/prediction')
            if not np.array_equal(ids,np.arange(ROW_START[name],ROW_START[name]+n)):
                raise ValueError('Saved prediction rows differ from the original official ordering')
            actual=float(average_precision_score(y,p));check=independent_ap(y,p)
            if abs(actual-check)>1e-12:raise ValueError('Independent AP implementation disagrees')
            match=[r for r in expected if r['condition']==condition and int(r['seed'])==seed and r['test']==name]
            if len(match)!=1 or abs(actual-float(match[0]['AP']))>1e-10:raise ValueError('Archived AP mismatch: '+bank+'/'+name)
            records.append(dict(condition=condition,seed=seed,test=name,rows=n,AP=actual))
    print(json.dumps(dict(passed=True,banks=len(banks),test_metrics=len(records),records=records,
       result_scope='Full model only; ablation and single-branch implementations are retained separately',
       limitation='Saved-prediction scoring; use reproduce_mirna.py with explicit weights for new model inference.'),indent=2))
if __name__=='__main__':main()
