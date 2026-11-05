"""Reproduce the MAE-focused reactivity table using retained error sums.

This reports observed-nucleotide absolute error only. The original checkpoint
selection remains correlation based and is not changed by this reporting entry.
Full metric records and original metric implementations remain supplementary.
"""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
EVIDENCE_SHA={
    'external_row_stats.npz':'37586284377b10a8d417cb7c64cb242921d80ddebabfeb9d632dfcbbcc3a2f64',
    'internal_test_row_stats.npz':'e1e4694dd43c5fb426160998761ecd260b0b11ac041d91cbf8c742bbb817b538'}

def observed_mae(labels,predictions,lengths):
    y=np.asarray(labels,float);p=np.asarray(predictions,float);n=np.asarray(lengths)
    if y.ndim!=2 or p.shape!=y.shape or n.shape!=(len(y),)or n.dtype.kind not in'iu':raise ValueError('Invalid profile dimensions/lengths')
    if np.any(n<1)or np.any(n>y.shape[1])or np.isinf(y).any():raise ValueError('Invalid lengths or labels')
    real=np.arange(y.shape[1])[None,:]<n[:,None];mask=real&np.isfinite(y)
    if np.any(mask.sum(1)==0)or not np.isfinite(p[mask]).all():raise ValueError('Unobserved row or nonfinite observed prediction')
    sums=np.where(mask,np.abs(np.where(mask,y,0)-np.where(mask,p,0)),0).sum(1)
    counts=mask.sum(1)
    return dict(MAE=float(sums.sum()/counts.sum()),observed_nucleotides=int(counts.sum()),windows=len(y))

def error_sum_mae(sums,counts,lengths):
    s=np.asarray(sums,float);c=np.asarray(counts);n=np.asarray(lengths)
    if s.ndim!=1 or c.shape!=s.shape or n.shape!=s.shape or c.dtype.kind not in'iu':raise ValueError('Invalid per-window error statistics')
    if not np.isfinite(s).all()or np.any(s<0)or np.any(c<=0)or np.any(c>n)or np.any(n<50)or np.any(n>60):raise ValueError('Invalid absolute error/mask/length')
    return float(s.sum()/c.sum())

def run(root,data_root=None):
    base=data_root or root/'data/accessibility/mae_evidence'
    with (root/'results/accessibility/MAE_comparison.csv').open(encoding='utf-8-sig',newline='')as f:table=list(csv.DictReader(f))
    expected={r['model']:r for r in table};verified=[]
    for filename,field,models,windows in [
        ('internal_test_row_stats.npz','internal_test_mae',('sequence','structure'),207979),
        ('external_row_stats.npz','external_mae',('sequence','structure','rebuilt_mRNABERT','PairAC_original'),7347)]:
        if hashlib.sha256((base/filename).read_bytes()).hexdigest()!=EVIDENCE_SHA[filename]:
            raise ValueError('Retained evidence SHA differs: '+filename)
        with np.load(base/filename,allow_pickle=False)as z:
            if len(z['row_ids'])!=windows or len(np.unique(z['row_ids']))!=windows:raise ValueError('Dataset row identity/count differs')
            for model in models:
                actual=error_sum_mae(z[model+'_absolute_error_sum'],z['observed_counts'],z['lengths'])
                target=float(expected[model][field])
                if abs(actual-target)>1e-12:raise ValueError('MAE differs: '+model+'/'+field)
                verified.append(dict(model=model,evaluation=field,MAE=actual,windows=windows,
                                     observed_nucleotides=int(z['observed_counts'].sum())))
            if filename.startswith('internal'):
                for model in models:
                    ids=z['sample_row_indices'];got=observed_mae(z['sample_y'],z[model+'_sample_pred'],z['lengths'][ids])
                    target=float(z[model+'_absolute_error_sum'][ids].sum()/z['observed_counts'][ids].sum())
                    if abs(got['MAE']-target)>1e-12:raise ValueError('Sampled raw profiles disagree with error sums')
    return dict(passed=True,metric='pooled observed-nucleotide MAE',verified=verified,
                aggregate_only_cells=['Validation MAE for all methods','Reconstructed mRNABERT internal-test MAE: retained aggregate; raw per-window predictions not included'],
                original_pairac_membership='unknown; external result descriptive only',
                checkpoint_selection='Original PCC composite with regional-MAE tie-break, unchanged',
                reporting='MAE-focused post-hoc presentation; does not establish superiority in correlation',
                model_retrained=False,new_inference=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--data-root',type=Path,help='Optional directory of retained per-window error NPZ files')
    p.add_argument('--predictions',type=Path,help='Instead score an NPZ with labels, predictions and true lengths')
    a=p.parse_args()
    if a.predictions:
        with np.load(a.predictions,allow_pickle=False)as z:r=observed_mae(z['labels'],z['predictions'],z['lengths'])
    else:r=run(a.root.resolve(),a.data_root)
    print(json.dumps(r,indent=2))
