"""Independent SVD and geometry audit; does not import the R8 fitter."""
import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('_r8_previous_audit',Path(__file__).resolve().parents[1]/'round4/verify.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
close=previous.close

def expand(train,other,t):
    z=previous.project(train,train,t['projection']);zv=previous.project(train,other,t['projection'])
    td,source=z[:,:24],z[:,24:];u,s,vt=np.linalg.svd(td,full_matrices=False)
    keep=s>s[0]*1e-10;expected=(vt[keep].T/s[keep])@(u[:,keep].T@source)
    a=np.asarray(t['td_to_source']);close(a,expected,1e-9)
    assert t['rcond']==1e-10 and t['TD_rank']==int(keep.sum())
    close(s,t['TD_singular'],1e-10)
    remainder=source-td@a;mean=remainder.mean(0);close(mean,t['residual_mean'],1e-11)
    close(td.T@remainder/len(td),np.zeros((24,16)),1e-10)
    return np.c_[zv[:,:24],zv[:,24:]-zv[:,:24]@a-mean]
