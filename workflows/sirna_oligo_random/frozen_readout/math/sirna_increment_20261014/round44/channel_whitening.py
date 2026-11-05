"""Fixed, full-rank shared-channel covariance transform; one linear head."""
import importlib.util
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_r44_native25', R.parent/'round25/native_trace.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
transform = native.transform
fit = native.fit
SHRINKAGE = .5


def design(z):
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] != 216 or not len(z) or not np.isfinite(z).all():
        raise ValueError('Finite nonempty TD24 + source192 required')
    return z


def fit_whitener(z):
    z = design(z)
    if np.max(np.abs(z.mean(0))) > 1e-8:
        raise ValueError('Original training-centered design required')
    source = np.sqrt(12.) * z[:,24:].reshape(-1,6,32)
    cov = np.einsum('nrc,nrd->cd', source, source)/(6*len(z))
    cov = (cov + cov.T)/2
    shrunk = .5*np.eye(32) + .5*cov
    eigen, vectors = np.linalg.eigh(shrunk)
    if eigen[0] < .5-1e-9:
        raise ValueError('Shrunken covariance must be positive definite')
    inverse_root = (vectors/np.sqrt(eigen)) @ vectors.T
    total = float(np.trace(cov))
    normalized = float(np.trace(inverse_root @ cov @ inverse_root))
    gamma = float(np.sqrt(total/normalized)) if total > 1e-24 else 1.
    weight = gamma*inverse_root if total > 1e-24 else np.eye(32)
    spectrum = np.linalg.eigvalsh(cov)
    transformed = weight.T @ cov @ weight
    effective_rank = total*total/float(np.sum(cov**2)) if total > 1e-24 else 0.
    return dict(shrinkage=SHRINKAGE, n_training_rows=len(z), region_count=6, channel_count=32,
        covariance=cov, covariance_spectrum=spectrum, shrunk_covariance=shrunk,
        shrunk_spectrum=eigen, inverse_root=inverse_root, energy_scale=gamma, weight=weight,
        energy_before=total, energy_after=float(np.trace(transformed)),
        effective_rank=effective_rank, shrunk_condition=float(eigen[-1]/eigen[0]),
        covariance_condition=float(spectrum[-1]/spectrum[0]) if spectrum[0]>1e-12 else None,
        zero_covariance=bool(total<=1e-24), dimensionality_reduction=False)


def transform_channels(z, state):
    z = design(z)
    weight = np.asarray(state['weight'],float)
    if weight.shape != (32,32) or not np.isfinite(weight).all():
        raise ValueError('Finite shared32x32 transform required')
    return np.c_[z[:,:24], (z[:,24:].reshape(-1,6,32) @ weight).reshape(-1,192)]


def fold_coefficients(beta, state):
    beta = np.asarray(beta,float)
    if beta.shape != (216,) or not np.isfinite(beta).all():
        raise ValueError('Finite216D coefficient required')
    weight = np.asarray(state['weight'],float)
    if weight.shape != (32,32) or not np.isfinite(weight).all():
        raise ValueError('Finite shared32x32 transform required')
    return np.r_[beta[:24], (beta[24:].reshape(6,32) @ weight.T).ravel()]


def deployment(scaler, beta, center, whitener):
    return native.deployment(scaler, fold_coefficients(beta,whitener), center)


def qualify(records, numeric):
    core = [r for r in records if r['source']=='core_real']
    if len(core)!=3 or len({r['setting'] for r in core})!=3:
        raise ValueError('Three distinct primary pools required')
    performance = (sum(r['delta_PCC_vs_R25']>=.005 and r['MSE_reduction_vs_R25_pct']>=0 for r in core)>=2
                   and all(r['delta_PCC_vs_R25']>=-.005 for r in core))
    pairs = [dict(setting=r['setting'], delta_PCC=r['metrics']['PCC']-r['TD_metrics']['PCC'],
                  MSE_reduction_pct=100*(1-r['metrics']['MSE']/r['TD_metrics']['MSE'])) for r in core]
    increment = sum(r['delta_PCC']>=0 for r in pairs)>=2 and np.mean([r['MSE_reduction_pct'] for r in pairs])>=0
    return dict(qualified=bool(performance and increment and numeric), performance_gate=bool(performance),
                source_increment_gate=bool(increment), all_numerically_qualified=bool(numeric), paired_TD=pairs)
