"""Full-system independent audits for the original five-alpha family."""
import importlib.util
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('_r55_math54',R.parent/'round54/verify.py')
prior=importlib.util.module_from_spec(s);s.loader.exec_module(prior)
profile=prior.profile;indicator_geometry=prior.indicator_geometry;close=prior.close


def check_source(z,y,labels,b,center,cert,alpha,cached=None):
    data,rhs,quad,zm,ym,G,w,meta=profile(z,y,labels) if cached is None else cached
    gram=data+alpha*np.eye(216);close(center,ym-zm@b,1e-10);close(w@(y-z@b-center),0,1e-8)
    td0=np.linalg.solve(gram[:24,:24],rhs[:24]);coupling=np.linalg.solve(gram[:24,:24],gram[:24,24:])
    h=gram[24:,24:]-gram[24:,:24]@coupling;h=(h+h.T)*.5;q=rhs[24:]-gram[24:,:24]@td0
    critical=float(np.linalg.norm(q.reshape(6,32),2));lam=.1*critical
    assert cert['alpha']==alpha and cert['lambda_ratio']==.1 and list(cert['matrix_shape'])==[6,32]
    assert cert['mode']=='native_trace_common_target_intercept' or (meta['reference_equal'] and cert['mode']=='native_trace')
    close(cert['lambda_critical'],critical,1e-9);close(cert['lambda_value'],lam,1e-9)
    singular=np.linalg.svd(b[24:].reshape(6,32),compute_uv=False);close(cert['singular_values'],singular,1e-10)
    assert cert['rank']==int(sum(singular>1e-8))
    normal=float(np.max(abs((gram@b-rhs)[:24])));assert normal<1e-8
    dual=q-h@b[24:];norm=float(np.linalg.norm(dual.reshape(6,32),2))
    if norm>lam:dual*=lam/norm
    close(cert['dual'],dual,1e-8);assert np.linalg.norm(dual.reshape(6,32),2)<=lam+1e-9
    primal=.5*b@gram@b-rhs@b+lam*singular.sum();dv=np.r_[np.zeros(24),dual]
    dualvalue=-.5*(rhs-dv)@np.linalg.solve(gram,rhs-dv);gap=float(primal-dualvalue)
    scale=max(1.,.5*q@np.linalg.solve(h,q));constant=.5*rhs[:24]@td0
    close(cert['primal_profile'],primal+constant,1e-9);close(cert['dual_profile'],dualvalue+constant,1e-9)
    close(cert['gap_scale'],scale,1e-9);close(cert['gap'],max(0.,gap),1e-9);assert gap>=-1e-8
    assert cert['certified'] and cert['gap']<=1e-9*cert['gap_scale'] and gap<=1e-9*scale+1e-9
    if 'maximum_iterations' in cert:
        assert cert['maximum_iterations']==20000 and cert['algorithm']=='strongly_convex_proximal_gradient';cap=20000
    else:cap=5000
    assert 0<=cert['iterations']<=cap
    if cert['iterations']:
        assert int(cert['history'][-1][0])==cert['iterations'];close(cert['history'][-1][1],cert['gap'],1e-12)
    else:assert critical==0
    close(cert['train_MSE'],(quad-2*rhs@b+b@data@b)/len(y),1e-10)
    return normal,abs(gap)/scale


def audit_TD(z,y,labels,b,center,cert,alpha,cached=None):
    data,rhs,quad,zm,ym,G,w,meta=profile(z,y,labels) if cached is None else cached
    n,k=G.shape;c=meta['normalizer'];rho=meta['common_group_penalty']
    aug=np.zeros((n+24+k,1+24+k));aug[:n,0]=np.sqrt(c);aug[:n,1:25]=np.sqrt(c)*z;aug[:n,25:]=np.sqrt(c)*G
    aug[n:n+24,1:25]=np.sqrt(alpha)*np.eye(24);aug[n+24:,25:]=np.sqrt(c*rho)*np.eye(k)
    sol=np.linalg.lstsq(aug,np.r_[np.sqrt(c)*y,np.zeros(24+k)],rcond=None)[0]
    close(center,sol[0],1e-8);close(b,sol[1:25],1e-8);close(center,ym-zm@b,1e-10)
    close(w@(y-z@b-center),0,1e-8)
    error=float(np.max(abs((data+alpha*np.eye(24))@b-rhs)));threshold=1e-8*(1+float(np.max(abs(rhs))))
    assert cert['alpha']==alpha and (cert['mode']=='TD_common_target_intercept_Ridge' or (meta['reference_equal'] and cert['mode']=='TD_partial_target_group_Ridge'))
    close(cert['normal_max'],error,1e-8);close(cert['normal_threshold'],threshold,1e-9)
    assert cert['certified'] and error<=threshold
    close(cert['train_MSE'],(quad-2*rhs@b+b@data@b)/n,1e-10)
    return error,float(np.max(abs(b-sol[1:25])))
