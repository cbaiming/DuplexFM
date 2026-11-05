"""Unsupervised two-centroid routing and conditional OOF gain summaries."""
import numpy as np
from sklearn.cluster import KMeans

def predict(td,centers):
    td,centers=np.asarray(td,float),np.asarray(centers,float)
    if td.ndim!=2 or td.shape[1]!=24 or centers.shape!=(2,24):raise ValueError('TD24/two-centroid contract')
    if not np.isfinite(td).all() or not np.isfinite(centers).all():raise ValueError('Nonfinite input')
    distance=np.sum((td[:,None,:]-centers[None,:,:])**2,axis=2)
    return np.argmin(distance,axis=1)

def fit(td):
    td=np.asarray(td,float)
    if td.ndim!=2 or td.shape[1]!=24 or len(td)<20 or not np.isfinite(td).all():raise ValueError('Training TD contract')
    if len(np.unique(td,axis=0))<2:raise ValueError('Cannot fit two distinct routes')
    model=KMeans(n_clusters=2,init='k-means++',n_init=10,max_iter=300,tol=1e-4,random_state=42,algorithm='lloyd').fit(td)
    centers=model.cluster_centers_.copy()
    order=sorted(range(2),key=lambda k:tuple(centers[k].tolist()));centers=centers[order]
    gate=predict(td,centers)
    return dict(centers=centers,iterations=int(model.n_iter_),inertia=float(np.sum((td-centers[gate])**2)),
        training_counts=np.bincount(gate,minlength=2),distinct=bool(np.linalg.norm(centers[0]-centers[1])>1e-10))

def summary(y,td_prediction,source_prediction,gate):
    y,pt,ps=np.asarray(y,float),np.asarray(td_prediction,float),np.asarray(source_prediction,float)
    gate=np.asarray(gate,int)
    if y.ndim!=1 or y.shape!=pt.shape or y.shape!=ps.shape or y.shape!=gate.shape:raise ValueError('Summary shape')
    if not np.isfinite(np.r_[y,pt,ps]).all() or not np.isin(gate,[0,1]).all():raise ValueError('Summary values')
    base=(y-pt)**2;gain=base-(y-ps)**2;scale=float(base.mean())
    def corr(a,b):
        a=a-a.mean();b=b-b.mean();den=float(np.sqrt((a@a)*(b@b)))
        return float(a@b/den) if den>1e-14 else None
    groups=[]
    for g in (0,1):
        use=gate==g;n=int(use.sum())
        groups.append(dict(route=g,n=n,fraction=float(n/len(y)),mean_gain=float(gain[use].mean()) if n else None,
            TD_MSE=float(base[use].mean()) if n else None,source_MSE=float(np.mean((y[use]-ps[use])**2)) if n else None,
            TD_PCC=corr(y[use],pt[use]) if n>1 else None,source_PCC=corr(y[use],ps[use]) if n>1 else None))
    gap=groups[1]['mean_gain']-groups[0]['mean_gain'] if all(r['n'] for r in groups) else None
    return dict(n=len(y),TD_MSE=scale,source_MSE=float(np.mean((y-ps)**2)),groups=groups,
        gain_gap=gap,normalized_gain_gap=(gap/scale if gap is not None and scale>0 else None))

def qualify(record,route_record):
    gap=record['summary']['normalized_gain_gap']
    if gap is None:return False
    signs=[b['summary']['normalized_gain_gap'] for b in record['inner_blocks']]
    consistent=sum(g is not None and g*gap>0 for g in signs)
    return bool(route_record['balanced'] and route_record['median_ARI']>=.5 and route_record['all_converged']
        and abs(gap)>=.05 and consistent>=4)
