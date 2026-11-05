"""Label-using empirical ceiling for a fixed score ordering; never a predictor evaluation."""
import numpy as np


def project(score, response):
    score,response=np.asarray(score,float),np.asarray(response,float)
    if score.ndim!=1 or response.shape!=score.shape or len(score)<3:
        raise ValueError('Matching one-dimensional arrays with at least three rows required')
    if not np.isfinite(score).all() or not np.isfinite(response).all():raise ValueError('Finite input required')
    yc=response-response.mean();yv=float(yc@yc)
    if yv<=1e-14:raise ValueError('Nonconstant response required')
    levels,group=np.unique(score,return_inverse=True)
    weights=np.bincount(group).astype(float);means=np.bincount(group,weights=response)/weights
    stack=[]
    for j,(weight,value) in enumerate(zip(weights,means)):
        stack.append([j,j+1,float(weight),float(weight*value)])
        while len(stack)>1 and stack[-2][3]/stack[-2][2]>stack[-1][3]/stack[-1][2]:
            right=stack.pop();left=stack.pop()
            stack.append([left[0],right[1],left[2]+right[2],left[3]+right[3]])
    fitted=np.empty(len(levels))
    for lo,hi,weight,total in stack:fitted[lo:hi]=total/weight
    projected=fitted[group]
    cumulative=np.cumsum(weights*(means-fitted))[:-1]
    feasible_dual=np.maximum(cumulative,0.)
    transpose=np.r_[feasible_dual,0.]-np.r_[0.,feasible_dual]
    within=.5*float(np.sum((response-means[group])**2))
    primal=.5*float(np.sum((response-projected)**2))
    dual=float(transpose@means-.5*np.sum(transpose**2/weights)+within)
    gap=primal-dual
    stationarity=float(np.max(np.abs(weights*(fitted-means)+transpose)))
    complement=float(np.max(np.abs(feasible_dual*np.diff(fitted)))) if len(fitted)>1 else 0.
    monotone_violation=max(0.,float(-np.min(np.diff(fitted)))) if len(fitted)>1 else 0.
    pc=projected-projected.mean();pv=float(pc@pc);sc=score-score.mean();sv=float(sc@sc)
    raw_pcc=float(yc@sc/np.sqrt(yv*sv)) if sv>1e-14 else None
    bound=float(np.sqrt(max(0.,min(1.,pv/yv))))
    projected_pcc=float(yc@pc/np.sqrt(yv*pv)) if pv>1e-14 else None
    slope=max(0.,float(sc@yc/sv)) if sv>1e-14 else 0.
    affine=response.mean()+slope*sc
    affine_mse=float(np.mean((affine-response)**2));mse=2*primal/len(score)
    tol=1e-9*max(1.,yv,float(np.max(weights)))
    qualified=bool(gap>=-tol and gap<=tol and stationarity<=tol and complement<=tol and monotone_violation<=tol)
    metrics=dict(raw_PCC=raw_pcc,raw_MSE=float(np.mean((score-response)**2)),upper_PCC=bound,
        upper_attained=bool(pv>1e-14),optimistic_projection_PCC=projected_pcc,optimistic_projection_MSE=mse,
        positive_affine_MSE=affine_mse,positive_affine_slope=slope,
        absolute_PCC_room=None if raw_pcc is None else bound-raw_pcc,
        optimistic_MSE_reduction_vs_affine_pct=100*(1-mse/affine_mse) if affine_mse>1e-14 else 0.,
        unique_score_count=len(levels),unique_fitted_levels=len(np.unique(fitted)))
    cert=dict(certified=qualified,primal=primal,dual=dual,gap=gap,threshold=tol,
        stationarity_max=stationarity,complementarity_max=complement,monotonicity_violation=monotone_violation,
        centered_projection_orthogonality=float((yc-pc)@pc),mean_difference=float(projected.mean()-response.mean()))
    state=dict(score_levels=levels,counts=weights,group_mean_response=means,fitted_levels=fitted,
        dual_multipliers=feasible_dual,block_ranges=[b[:2] for b in stack],metrics=metrics,certificate=cert,
        uses_evaluation_labels=True,deployable=False,new_OOF_predictions=False)
    return projected,state


def qualify(records, numeric):
    core=[r for r in records if r['condition']=='core_real']
    if len(core)!=3 or len({r['setting'] for r in core})!=3:raise ValueError('Three distinct primary pools required')
    pairs=[];room=[]
    for r in core:
        m=r['metrics'];td=next(x for x in records if x['setting']==r['setting'] and x['condition']=='TD_only')['metrics']
        room.append(bool(m['upper_attained'] and m['absolute_PCC_room'] is not None and m['absolute_PCC_room']>=.01
                         and m['optimistic_MSE_reduction_vs_affine_pct']>=1.))
        pairs.append(dict(setting=r['setting'],Core_upper_PCC=m['upper_PCC'],TD_upper_PCC=td['upper_PCC'],
                          delta_upper_PCC=m['upper_PCC']-td['upper_PCC']))
    roomgate=sum(room)>=2
    ordering=sum(p['delta_upper_PCC']>=.005 for p in pairs)>=2 and all(p['delta_upper_PCC']>=-.005 for p in pairs)
    return dict(qualified=bool(roomgate and ordering and numeric),calibration_room_gate=bool(roomgate),
        paired_ordering_gate=bool(ordering),all_numerically_qualified=bool(numeric),paired_ceilings=pairs,
        qualification_only=True,goal_achieved=False)
