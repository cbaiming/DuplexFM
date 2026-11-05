"""Create R8 files from the immutable bounded R5 lifecycle before registration."""
from pathlib import Path
R=Path(__file__).resolve().parent
assert not (R/'lock.json').exists()
def change(s,old,new):
    assert old in s,old
    return s.replace(old,new)

s=(R.parent/'round5/run.py').read_text(encoding='utf-8')
s=change(s,'from conditional import configs,fit_transform,transform,fit_heads,score,choose','from orthogonal import configs,fit_transform,transform,fit_heads,score,choose,collapse')
s=change(s,"VIEWS=('core_real','full_real','initial_core','TDQuadratic')","VIEWS=('core_real','full_real','initial_core')")
s=change(s,"CODE=('run.py','conditional.py','verify.py','test_conditional.py','PROTOCOL.md','CODE_AUDIT.md')",
    "CODE=('run.py','orthogonal.py','verify.py','audit_projection.py','test_orthogonal.py','build_io.py','PROTOCOL.md','CODE_AUDIT.md')")
s=change(s,"return view+'_'+mode+('Quadratic' if view=='TDQuadratic' else 'Bilinear')","return view+'_'+mode+'Orthogonal'")
s=change(s,"def kind(view):return 'td_quadratic' if view=='TDQuadratic' else 'bilinear'\n",'')
s=change(s,"==6\n","==5\n")
s=change(s,'core_real_AdaptiveBilinear','core_real_AdaptiveOrthogonal')
s=change(s,"    previous.validate();v=io.read(PRIOR/'verification.json')","""    dlock=io.read(HERE/'diagnostic_lock.json');io.check_files(HERE,dlock['files'])
    for p,h in dlock['dependencies'].items():assert io.sha(Path(p))==h,p
    diagnostic=io.read(HERE/'diagnostic.json')
    assert diagnostic['qualified'] and diagnostic['real_labels_used'] is False and diagnostic['new_prediction_metrics'] is False
    assert diagnostic['lock_sha256']==io.sha(HERE/'diagnostic_lock.json')
    previous.validate();v=io.read(PRIOR/'verification.json')""")
s=change(s,"    dependencies=dict(old['dependencies'])","    dependencies=dict(old['dependencies']);dependencies.update(dlock['dependencies'])")
s=change(s,"HERE.parent/'r4_training_diagnostic.json',HERE.parent/'r4_training_diagnostic.py'",
    "HERE/'diagnostic_lock.json',HERE/'diagnostic.json',HERE/'diagnose.py',HERE/'DIAGNOSTIC_PROTOCOL.md',HERE.parent/'round7/verification.json'")
s=change(s,"validate();rows,xx,_=previous.inputs();xx['TDQuadratic']=xx['core_real'][:,:24].copy()","validate();rows,xx,_=previous.inputs()")
s=change(s,"('core_real' if view=='TDQuadratic' else view)","view")
s=change(s,'fit_transform(x[fi[tr]],kind(view))','fit_transform(x[fi[tr]])')
s=change(s,'fit_transform(x[fi],kind(view))','fit_transform(x[fi])')
s=change(s,"if view!='TDQuadratic':same_projection","same_projection")
s=change(s,"b,c=fit_heads(z,y[fi],[cfg[selected[m]] for m in MODES]);p=transform(x[ti],t)@b.T+c",
    """b,c=fit_heads(z,y[fi],[cfg[selected[m]] for m in MODES]);p=transform(x[ti],t)@b.T+c
                collapsed,offset=collapse(t,b,c)
                np.testing.assert_allclose(x[ti]@collapsed.T+offset,p,atol=1e-10,rtol=1e-10)""")
s=change(s,'final_coef=b,final_intercept=np.array(c),test_pred=p)',
    'final_coef=b,final_intercept=np.array(c),test_pred=p,collapsed_coef=collapsed,collapsed_intercept=offset)')
s=change(s,'30384','22788');s=change(s,'len(selections)==128','len(selections)==96')
s=change(s,"no_source_control=view=='TDQuadratic'","no_source_control=False")
s=change(s,'views=4,inner_projection_blocks=320','views=3,inner_projection_blocks=240')
s=change(s,'inner_fits=8000,final_heads=128','inner_fits=6000,final_heads=96')
s=change(s,'ROUND5_COMPLETE','ROUND8_COMPLETE')
with (R/'run.py').open('x',encoding='utf-8',newline='\n') as f:f.write(s)

s=(R.parent/'round5/verify.py').read_text(encoding='utf-8')
s=change(s,'Independent R5 audit','Independent R8 audit')
s=change(s,'import importlib.util,json,time','import importlib.util,json,time,traceback,sys')
s=change(s,"VIEWS=('core_real','full_real','initial_core','TDQuadratic')","VIEWS=('core_real','full_real','initial_core')")
s=change(s,"return view+'_'+mode+('Quadratic' if view=='TDQuadratic' else 'Bilinear')","return view+'_'+mode+'Orthogonal'")
a=s.index('def expand(');b=s.index('\ndef equations(',a)
s=s[:a]+"sys.path.insert(0,str(R))\nfrom audit_projection import expand as reconstruct\ndef expand(train,other,t,view):\n    assert view in VIEWS\n    return reconstruct(train,other,t)\n\n"+s[b:]
s=change(s,"xx={'TDQuadratic':td}","xx={}")
s=change(s,'core_real_AdaptiveBilinear','core_real_AdaptiveOrthogonal')
s=change(s,'30384','22788');s=change(s,'len(summary)==32 and len(selection)==128','len(summary)==24 and len(selection)==96')
s=change(s,"width=324 if view=='TDQuadratic' else 424","width=40")
s=change(s,"            oof=np.full((len(fi),25),np.nan);coverage=[]","""            prior=read(P/'results/folds'/setting/f'fold{fold}'/view/'transforms.json')
            for nt,ot in zip(t['inner']+[t['final']],prior['inner']+[prior['final']]):
                for key in ot:close(nt['projection'][key],ot[key],1e-12)
            oof=np.full((len(fi),25),np.nan);coverage=[]""")
s=change(s,"                oof[va]=zv@a['inner_coef'][j].T+a['inner_intercept'][j]","""                for k in range(25):close(a['inner_coef'][j,k,:24],a['inner_coef'][j,k%5,:24],1e-10)
                oof[va]=zv@a['inner_coef'][j].T+a['inner_intercept'][j]""")
s=change(s,"            for k,mode in enumerate(MODES):","""            pstate=t['final']['projection'];coef=a['final_coef'];source_coef=coef[:,24:]/np.asarray(pstate['pc_scale'])
            td_coef=(coef[:,:24]-coef[:,24:]@np.asarray(t['final']['td_to_source']).T)/np.asarray(pstate['td_scale'])
            src_coef=(source_coef@np.asarray(pstate['components']))/np.asarray(pstate['src_scale'])
            offset=float(a['final_intercept'])-coef[:,24:]@np.asarray(t['final']['residual_mean'])
            offset-=td_coef@np.asarray(pstate['td_mean'])+src_coef@np.asarray(pstate['src_mean'])
            offset-=source_coef@(np.asarray(pstate['components'])@np.asarray(pstate['pca_mean'])+np.asarray(pstate['pc_mean']))
            close(np.c_[td_coef,src_coef],a['collapsed_coef'],1e-10);close(offset,a['collapsed_intercept'],1e-10)
            for k,mode in enumerate(MODES):""")
s=change(s,"                maxPrediction=max(maxPrediction,float(np.max(abs(pp-a['test_pred'][:,k]))))","""                maxPrediction=max(maxPrediction,float(np.max(abs(pp-a['test_pred'][:,k]))))
                close(x[ti]@a['collapsed_coef'][k]+a['collapsed_intercept'][k],pp,1e-10)""")
s=change(s,"                canonical.setdefault((setting,cond),[]).extend(zip(ti,pp))",
    "                canonical.setdefault((setting,cond),[]).extend(zip(ti,[float(r['prediction']) for r in part]))")
s=change(s,'assert count==8000 and final==128','assert count==6000 and final==96')
s=change(s,"    baseline={(r['setting'],r['condition']):r for r in table(OLD/'results/summary.csv')}","""    diagnostic=read(R/'diagnostic.json');assert diagnostic['qualified'] and not diagnostic['real_labels_used']
    assert diagnostic['lock_sha256']==sha(R/'diagnostic_lock.json') and len(diagnostic['rows'])==9
    diagnostics_checked=0
    for dr in diagnostic['rows']:
        setting,view=dr['setting'],dr['source'];item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0)
        assert item['fit_ids']==diagnostic['fit_ids'][setting]
        fi=np.array([ids[k] for k in item['fit_ids']]);x=xx[view][fi]
        pstate=read(P/'results/folds'/setting/'fold0'/view/'transforms.json')['final']
        z=audit4.project(x,x,pstate);T,S=z[:,:24],z[:,24:]
        U,s,V=np.linalg.svd(T,full_matrices=False);keep=s>s[0]*1e-10
        A=(V[keep].T/s[keep])@(U[:,keep].T@S);close(A,dr['projection'],1e-9)
        residual=S-T@A;value=1-np.sum(residual**2)/np.sum(S**2)
        close(value,dr['source_variance_explained_by_TD'],1e-10)
        raw=(x[:,24:]-np.asarray(pstate['src_mean']))/np.asarray(pstate['src_scale'])
        close(np.sum(pstate['eigenvalues'])/np.var(raw,axis=0).sum(),dr['original_PCA16_variance_retained'],1e-10)
        singular=np.linalg.svd(residual,compute_uv=False)
        assert int((singular>singular[0]*1e-10).sum())==dr['remaining_source_rank']
        assert int(keep.sum())==dr['TD_rank'] and len(T)==dr['n']
        assert np.max(abs(T.T@residual))/len(T)<1e-10;diagnostics_checked+=1
    core=[d for d in diagnostic['rows'] if d['source']=='core_real']
    assert sum(d['source_variance_explained_by_TD']>=.1 for d in core)>=2 and all(d['remaining_source_rank']==16 for d in core)
    assert diagnostics_checked==9
    assert complete['inner_fits']==6000 and complete['final_heads']==96 and complete['rows']==22788
    baseline={(r['setting'],r['condition']):r for r in table(OLD/'results/summary.csv')}""")
s=change(s,'assert len(goals)==8','assert len(goals)==6')
s=change(s,'source_models=3,views=4','source_models=3,views=3,feature_only_diagnostics_checked=diagnostics_checked')
s=change(s,'all_PCA_interaction_selection_predictions_metrics_checked','all_PCA_orthogonalization_deployment_selection_metrics_checked')
s=change(s,"if __name__=='__main__':main()","""if __name__=='__main__':
    begin=time.monotonic();rc=1;error=None
    try:main();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        with (R/f'verification_execution_{time.time_ns()}.json').open('x',encoding='utf-8') as f:
            json.dump(dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-begin,GPU_hours=0),f,indent=2)
""")
with (R/'verify.py').open('x',encoding='utf-8',newline='\n') as f:f.write(s)
print('Generated R8 controller and independent verifier; no scientific execution')
