"""Deterministic leave-one-validation-block selection consensus."""
import numpy as np

def scores(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    if len(y)<5 or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Scoring input')
    yc=y-y.mean();pc=p-p.mean(0);den=np.sqrt((yc@yc)*np.sum(pc*pc,axis=0))
    if yc@yc<=0:raise ValueError('Constant response')
    return np.divide(yc@pc,den,out=np.full(p.shape[1],-2.),where=den>1e-14),np.mean((p-y[:,None])**2,axis=0)

def ordering(y,p,config,allowed):
    pc,ms=scores(y,p)
    return sorted(allowed,key=lambda k:(-float(pc[k]),float(ms[k]),config[k]['weight'],-config[k]['alpha'],k)),pc,ms

def select(y,p,config,blocks,allowed):
    y=np.asarray(y,float);p=np.asarray(p,float)
    cover=[int(i) for block in blocks for i in block]
    if sorted(cover)!=list(range(len(y))) or len(blocks)!=5:raise ValueError('Validation blocks must partition fit once')
    allowed=list(allowed);order,pc,ms=ordering(y,p,config,allowed)
    original=order[0];winners=[];votes=np.zeros(len(config),int)
    for block in blocks:
        keep=np.ones(len(y),bool);keep[np.asarray(block,int)]=False
        winner=ordering(y[keep],p[keep],config,allowed)[0][0];winners.append(winner);votes[winner]+=1
    rank={k:i for i,k in enumerate(order)};consensus=min(allowed,key=lambda k:(-int(votes[k]),rank[k]))
    return dict(original=int(original),selected=int(consensus),winners=[int(x) for x in winners],votes=votes.tolist(),
        original_support=int(votes[original]),unique_winners=len(set(winners)),
        original_top_PCC=float(pc[original]),original_margin=float(pc[order[0]]-pc[order[1]]),PCC=pc.tolist(),MSE=ms.tolist())
