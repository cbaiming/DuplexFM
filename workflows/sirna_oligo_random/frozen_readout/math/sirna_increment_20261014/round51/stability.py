"""Fixed-prediction sequence-component sensitivity, without model fitting."""
import numpy as np

COMPARISONS=((1,0),(1,4),(1,3),(1,5),(2,0),(3,0),(4,0))
MODELS=('TD24','Core50','Full50','Initial50','TD50','Core26')

def normalized(s,n):
    s=str(s).upper().replace('U','T')
    if len(s)!=n or set(s)-set('ACGT'):raise ValueError('Canonical fixed-length RNA required')
    return s

def components(guides,targets):
    n=len(guides)
    if n!=len(targets) or not n:raise ValueError('Aligned nonempty inputs required')
    gs=[normalized(s,19) for s in guides];ts=[normalized(s,57) for s in targets]
    parent=list(range(n));seen={};buckets={};table=str.maketrans('ACGT','TGCA')
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def join(i,j):
        a,b=find(i),find(j);parent[max(a,b)]=min(a,b)
    for i,(g,t) in enumerate(zip(gs,ts)):
        candidates=set()
        for block,(a,b) in enumerate(((0,6),(6,12),(12,19))):
            key=(block,g[a:b]);candidates.update(buckets.get(key,()))
        for j in sorted(candidates):
            if sum(a!=b for a,b in zip(g,gs[j]))<=2:join(i,j)
        for block,(a,b) in enumerate(((0,6),(6,12),(12,19))):buckets.setdefault((block,g[a:b]),[]).append(i)
        for j in range(33):
            k=t[j:j+25];k=min(k,k.translate(table)[::-1])
            if k in seen:join(i,seen[k])
            else:seen[k]=i
    roots=[find(i) for i in range(n)];lookup={v:j for j,v in enumerate(sorted(set(roots)))}
    return np.array([lookup[v] for v in roots],dtype=int)

def qualification(labels):
    _,sizes=np.unique(labels,return_counts=True);n=int(sum(sizes));k=len(sizes)
    effective=float(n*n/np.sum(sizes.astype(float)**2)) if n else 0.
    largest=float(max(sizes)/n) if n else 1.
    return dict(n=n,components=k,effective_components=effective,largest_fraction=largest,
                eligible=bool(k>=10 and effective>=10 and largest<=.5))

def statistics(y,p,w=None):
    y=np.asarray(y,float);p=np.asarray(p,float)
    if p.ndim!=2 or y.shape!=(len(p),) or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Finite paired arrays required')
    w=np.ones(len(y)) if w is None else np.asarray(w,float)
    if w.shape!=y.shape or not np.isfinite(w).all() or np.any(w<0):raise ValueError('Finite nonnegative weights required')
    sw=float(w.sum())
    if sw<=1e-12:return np.full(p.shape[1],np.nan),np.full(p.shape[1],np.nan)
    my=float(w@y/sw);mp=w@p/sw
    vy=float(w@(y*y)/sw-my*my);vp=w@(p*p)/sw-mp*mp
    cov=(w*y)@p/sw-my*mp
    valid=(vy>1e-14)&(vp>1e-14)
    out=np.full(p.shape[1],np.nan);out[valid]=cov[valid]/np.sqrt(vy*vp[valid])
    return out,w@((p-y[:,None])**2)/sw

def paired(pc,ms):
    pc=np.asarray(pc,float);ms=np.asarray(ms,float);out=[]
    for a,b in COMPARISONS:
        delta=pc[a]-pc[b]
        rel=100*(pc[a]/pc[b]-1) if np.isfinite(pc[[a,b]]).all() and pc[b]>0 else np.nan
        mr=100*(1-ms[a]/ms[b]) if np.isfinite(ms[[a,b]]).all() and ms[b]>0 else np.nan
        out.append([delta,rel,mr])
    return np.array(out)

def interval(x,eligible):
    x=np.asarray(x,float);finite=x[np.isfinite(x)];enough=len(finite)>=.95*len(x)
    return dict(valid_draws=len(finite),invalid_draws=len(x)-len(finite),eligible=bool(eligible),
        interval_available=bool(eligible and enough),percentile95=np.quantile(finite,[.025,.975]).tolist() if eligible and enough else None)

def macro(relative):
    v=np.asarray(relative,float)
    if v.shape[0]!=4:raise ValueError('Exactly four settings required')
    return np.mean(v,axis=0)
