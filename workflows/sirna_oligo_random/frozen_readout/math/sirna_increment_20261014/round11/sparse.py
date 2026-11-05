"""Fixed native ARD and deterministic policy; one linear deployable readout."""
import warnings
import numpy as np
from sklearn.linear_model import ARDRegression

PARAMETERS=dict(max_iter=300,tol=.001,alpha_1=1e-6,alpha_2=1e-6,lambda_1=1e-6,lambda_2=1e-6,
                threshold_lambda=10000.,fit_intercept=True,compute_score=True,copy_X=True,verbose=False)
ALPHAS=(.1,1.,10.,100.,1000.)

def fit(z,y):
    z=np.asarray(z,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    if z.ndim!=2 or z.shape[1] not in (24,40) or y.shape!=(len(z),) or len(z)<20:
        raise ValueError('Training contract')
    if not np.isfinite(z).all() or not np.isfinite(y).all() or np.var(y)<=0:
        raise ValueError('Nonfinite/constant response')
    with warnings.catch_warnings(record=True) as warning:
        warnings.simplefilter('always');m=ARDRegression(**PARAMETERS).fit(z,y)
    for value in (m.coef_,m.intercept_,m.alpha_,m.lambda_,m.sigma_,m.X_offset_):
        if not np.isfinite(value).all():raise FloatingPointError('Nonfinite native posterior')
    if m.alpha_<=0 or np.any(m.lambda_<=0):raise FloatingPointError('Invalid posterior precision')
    active=m.lambda_<PARAMETERS['threshold_lambda']
    assert np.count_nonzero(m.coef_[~active])==0 and m.sigma_.shape==(sum(active),sum(active))
    result=dict(coef=m.coef_,intercept=float(m.intercept_),noise_precision=float(m.alpha_),
        weight_precision=m.lambda_,posterior_covariance=m.sigma_,offset=m.X_offset_,iterations=int(m.n_iter_),
        iteration_cap_reached=bool(m.n_iter_==PARAMETERS['max_iter']),active_indices=np.flatnonzero(active),
        active_TD=int(np.count_nonzero(active[:24])),active_source=int(np.count_nonzero(active[24:])),
        evidence_scores=[float(s) if np.isfinite(s) else str(s) for s in m.scores_],
        warnings=[str(w.message) for w in warning],input_width=z.shape[1])
    np.testing.assert_allclose(predict(z,result),m.predict(z),atol=0,rtol=0)
    return result

def predict(z,state):return np.asarray(z)@np.asarray(state['coef'])+state['intercept']

def choose(pcc,mse,indices=range(6)):
    return min(indices,key=lambda k:(-float(pcc[k]),float(mse[k]),int(k==5),-ALPHAS[k] if k<5 else 0,k))

def collapse(t,state):
    coef=np.pad(np.asarray(state['coef']),(0,40-len(state['coef'])))
    td=coef[:24]/np.asarray(t['td_scale']);pc=coef[24:]/np.asarray(t['pc_scale'])
    source=(pc@np.asarray(t['components']))/np.asarray(t['src_scale'])
    offset=state['intercept']-td@np.asarray(t['td_mean'])-source@np.asarray(t['src_mean'])
    offset-=pc@(np.asarray(t['components'])@np.asarray(t['pca_mean'])+np.asarray(t['pc_mean']))
    return np.r_[td,source],float(offset)
