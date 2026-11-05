"""Partial downweighting of input-defined target-component intercept contrasts."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor,cho_solve
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_r49_native25',R.parent/'round25/native_trace.py')
native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
transform=native.transform;deployment=native.deployment
GAMMA=.5


def keys(seq):
    s=str(seq).upper().replace('U','T')
    if len(s)!=57 or set(s)-set('ACGT'):raise ValueError('Canonical unambiguous target57 required')
    table=str.maketrans('ACGT','TGCA')
    return {min(s[j:j+25],s[j:j+25].translate(table)[::-1]) for j in range(33)}


def groups(targets):
    if len(targets)<2:raise ValueError('At least two training inputs required')
    parent=list(range(len(targets)));seen={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,t in enumerate(targets):
        for key in sorted(keys(t)):
            if key in seen:
                a,b=find(i),find(seen[key]);parent[max(a,b)]=min(a,b)
            else:seen[key]=i
    roots=[find(i) for i in range(len(targets))];ordered={v:j for j,v in enumerate(sorted(set(roots)))}
    labels=np.array([ordered[v] for v in roots],dtype=int);sizes=np.bincount(labels);n=len(labels);k=len(sizes)
    c=(n-1)/(n-1-(1-GAMMA)*(k-1))
    return labels,dict(n=n,K=k,sizes=sizes.tolist(),unique_canonical_25mers=len(seen),gamma=GAMMA,normalizer=c,
        identity_on_centered_space=bool(k==1 or k==n),largest_fraction=float(max(sizes)/n),
        nonsingleton_groups=int(sum(sizes>1)),nonsingleton_rows=int(sum(sizes[sizes>1])),implicit_group_contrasts=k-1,
        confirmed_gene_or_assay=False,training_inputs_only=True)


def project(x,labels):
    x=np.asarray(x,float);g=np.asarray(labels,int)
    if x.ndim==1:x=x[:,None];squeeze=True
    else:squeeze=False
    if x.ndim!=2 or len(x)<2 or g.shape!=(len(x),) or not np.isfinite(x).all():raise ValueError('Finite aligned training matrix required')
    if g.min()!=0 or set(g.tolist())!=set(range(int(g.max())+1)):raise ValueError('Consecutive group ids required')
    if np.max(abs(x.mean(0)))>1e-8:raise ValueError('Global-centered training matrix required')
    n=len(x);k=int(g.max())+1
    if k in (1,n):out=x.copy()
    else:
        sizes=np.bincount(g);totals=np.zeros((k,x.shape[1]));np.add.at(totals,g,x)
        pg=totals[g]/sizes[g,None]-x.mean(0)
        c=(n-1)/(n-1-(1-GAMMA)*(k-1))
        out=np.sqrt(c)*(x-(1-np.sqrt(GAMMA))*pg)
    return out[:,0] if squeeze else out


def fit(z,y,labels,source,deadline=None):
    z=np.asarray(z,float);y=np.asarray(y,float)
    if z.ndim!=2 or z.shape[1]!=216 or y.shape!=(len(z),) or not np.isfinite(y).all() or np.ptp(y)==0:raise ValueError('Native216 finite training input/response required')
    center=float(y.mean());design=z if source else z[:,:24]
    zz=project(design,labels);yc=project(y-center,labels)
    if source:
        b,cc,cert=native.fit(zz,yc+center,deadline=deadline)
        return b,cc,cert
    gram=zz.T@zz+100*np.eye(24);rhs=zz.T@yc
    b=cho_solve(cho_factor(gram,lower=True),rhs)
    return b,center,td_certificate(zz,yc,b)


def td_certificate(zz,yc,b):
    rhs=zz.T@yc;normal=float(np.max(abs((zz.T@zz+100*np.eye(24))@b-rhs)))
    threshold=1e-8*(1+float(np.max(abs(rhs))))
    return dict(mode='TD_partial_target_group_Ridge',alpha=100.,normal_max=normal,normal_threshold=threshold,
        train_MSE=float(np.mean((zz@b-yc)**2)),certified=bool(normal<=threshold))


def residual_stats(z,y,b,c,labels,source):
    design=z if source else z[:,:24];r=y-(design@b+c);centered=r-r.mean();g=np.asarray(labels,int);sizes=np.bincount(g)
    total=np.bincount(g,weights=centered);between=float(np.sum(total**2/sizes));total_sse=float(centered@centered)
    return dict(raw_training_MSE=float(np.mean(r*r)),centered_residual_SSE=total_sse,
        centered_between_SSE=between,centered_within_SSE=total_sse-between,
        transformed_residual_SSE=float(np.sum(project(centered,g)**2)))


def qualify(records,numeric):
    core=[r for r in records if r['source']=='core_real']
    if len(core)!=3 or len({r['setting'] for r in core})!=3:raise ValueError('Three primary pools required')
    perf=sum(r['delta_PCC_vs_reference']>=.005 and r['MSE_reduction_vs_reference_pct']>=0 for r in core)>=2 and all(r['delta_PCC_vs_reference']>=-.005 for r in core)
    paired=[]
    for r in core:
        td=next(a for a in records if a['setting']==r['setting'] and a['source']=='TD_only')
        paired.append(dict(setting=r['setting'],delta_PCC=r['metrics']['PCC']-td['metrics']['PCC'],
            MSE_reduction_pct=100*(1-r['metrics']['MSE']/td['metrics']['MSE']),
            delta_gain_above_TD=r['delta_PCC_vs_reference']-td['delta_PCC_vs_reference']))
    increment=sum(r['delta_PCC']>=0 for r in paired)>=2 and np.mean([r['MSE_reduction_pct'] for r in paired])>=0
    return dict(qualified=bool(perf and increment and numeric),performance_gate=bool(perf),source_increment_gate=bool(increment),
        all_numerically_qualified=bool(numeric),paired_TD=paired)
