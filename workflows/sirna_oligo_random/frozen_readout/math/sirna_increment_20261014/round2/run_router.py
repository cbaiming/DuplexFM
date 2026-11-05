"""One CPU hard-routing experiment reusing immutable cross-fitted branches."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,json,sys,time,traceback
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;R1=R.parent/'round1'
sys.path.insert(0,str(R));sys.path.insert(1,str(R1))
import run as previous
from core import transform,fit_heads
from router_core import targets,fit_gate,route
io=previous.io;OLD=previous.OLD;SETTINGS=previous.SETTINGS;METHODS=('InterceptGate','LinearGate')
CODE=('router_core.py','run_router.py','verify_router.py','test_router.py','PROTOCOL.md','CODE_AUDIT.md')

def validate():
    lock=io.read(R/'lock.json');io.check_files(R,lock['files']);previous.validate()
    assert io.sha(R1/'lock.json')==lock['prior_lock_sha256'] and io.versions()==lock['versions']
    assert io.sha(R1/'results/complete.json')==lock['prior_complete_sha256']
    assert io.sha(R1/'verification.json')==lock['prior_verification_sha256']
    io.check_files(R1/'results',io.read(R1/'results/complete.json')['files_sha256'])

def register():
    if (R/'lock.json').exists():raise FileExistsError('Registered')
    import xml.etree.ElementTree as ET
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==4
    assert not any(int(s.get('errors',0))+int(s.get('failures',0))+int(s.get('skipped',0)) for s in suites)
    previous.validate();assert io.read(R1/'verification.json')['passed']
    io.write(R/'lock.json',dict(files={n:io.sha(R/n) for n in CODE+('pytest.xml',)},
        prior_complete_sha256=io.sha(R1/'results/complete.json'),prior_verification_sha256=io.sha(R1/'verification.json'),
        prior_lock_sha256=io.sha(R1/'lock.json'),versions=io.versions(),source_seed=42,GPU_hours=0,
        excluded=['wang2009','alas1_patent'],gate_regularization=1.,gate_input='source_PCA20_only',gate_grid=False))
    print('REGISTERED',io.sha(R/'lock.json'),flush=True)

def run():
    validate();rows,x=previous.inputs();index={r['sample_id']:i for i,r in enumerate(rows)}
    y=np.array([float(r['label']) for r in rows]);plan=io.read(R1/'plan.json');oldcfg=io.read(OLD/'plan.json')['configs']
    root=R/'results';locksha=io.sha(R/'lock.json')
    if (root/'complete.json').exists():raise FileExistsError('Complete')
    allpred=[];usage=[]
    for item in plan['folds']:
        s,f=item['setting'],item['fold'];out=root/'folds'/s/f'fold{f}';prior=R1/'results/folds'/s/f'fold{f}'
        fi=np.array([index[z] for z in item['fit_ids']]);ti=np.array([index[z] for z in item['test_ids']])
        if (out/'complete.json').exists():
            rec=io.read(out/'complete.json');assert rec['lock_sha256']==locksha;io.check_files(out,rec['files_sha256'])
        else:
            if out.exists() and any(out.iterdir()):raise RuntimeError('Partial fold requires audit')
            started=time.monotonic();tr=io.read(prior/'transforms.json')['final'];sel=io.read(prior/'selection.json')
            fullalpha=plan['configs'][sel['selected']['PCA4_Full']]['alpha']
            oldsel=io.read(OLD/'results/folds'/s/f'fold{f}'/'selection.json');tdindex=oldsel['selected']['TD24_Ridge_CV']
            tdalpha=oldcfg[tdindex]['alpha'];cfg=[dict(mask=m,td=True,alpha=fullalpha) for m in (31,30,29,27,23,15)]
            cfg.append(dict(mask=0,td=True,alpha=tdalpha))
            with np.load(prior/'arrays.npz',allow_pickle=False) as a:
                ci=[next(k for k,c in enumerate(plan['configs']) if c==conf) for conf in cfg[:6]]
                cross=a['inner_oof'][:,ci]
            with np.load(OLD/'results/folds'/s/f'fold{f}'/'arrays.npz',allow_pickle=False) as a:
                cross=np.c_[cross,a['inner_oof'][:,tdindex]]
            q,tau=targets(y[fi],cross);z=transform(x[fi],tr);theta,receipt=fit_gate(z[:,24:],q)
            coef,intercept=fit_heads(z,y[fi],cfg)
            io.write(out/'training.json',dict(branch_configs=cfg,source_full_candidate=sel['selected']['PCA4_Full'],
                original_TD24_candidate=tdindex,tau=tau,gate=receipt,theta=theta,fit_ids=item['fit_ids'],
                outer_validation_unused=True,test_labels_used=False,lock_sha256=locksha))
            zt=transform(x[ti],tr);branch_pred=zt@coef.T+intercept
            routes=np.c_[np.full(len(ti),receipt['constant_route']),route(zt[:,24:],theta)]
            pred=np.take_along_axis(branch_pred,routes,axis=1)
            previous.npz_write(out/'arrays.npz',crossfit_branches=cross,soft_targets=q,theta=theta,
                coef=coef,intercept=np.array(intercept),branch_pred=branch_pred,routes=routes,test_pred=pred)
            part=[]
            for j,m in enumerate(METHODS):
                for pos,i in enumerate(ti):
                    part.append(dict(setting=s,fold=f,condition=m,sample_id=rows[i]['sample_id'],label=y[i],binary=int(rows[i]['y']),
                        prediction=pred[pos,j],route=int(routes[pos,j]),mask=cfg[int(routes[pos,j])]['mask']))
            io.write_csv(out/'predictions.csv',part)
            io.write(out/'complete.json',dict(lock_sha256=locksha,CPU_wall_seconds=time.monotonic()-started,new_linear_branches=7,
                linear_gate_fits=1,constant_gates=1,GPU_hours=0,source_updates=0,
                files_sha256={n:io.sha(out/n) for n in ('training.json','arrays.npz','predictions.csv')}))
            print(json.dumps(dict(setting=s,fold=f,complete=True)),flush=True)
        part=io.table(out/'predictions.csv');allpred+=part
        for m in METHODS:
            for k,mask in enumerate((31,30,29,27,23,15,0)):
                usage.append(dict(setting=s,fold=f,condition=m,route=k,mask=mask,count=sum(r['condition']==m and int(r['route'])==k for r in part)))
    assert len(allpred)==7596
    base={(r['setting'],r['condition']):r for r in io.table(OLD/'results/summary.csv') if r['setting'] in SETTINGS}
    summary=[]
    for s in SETTINGS:
        expected={r['sample_id'] for r in rows if r['cohort']==('Mixset' if s=='Inter_Hu_to_Mix' else s)}
        for m in METHODS:
            part=[r for r in allpred if r['setting']==s and r['condition']==m];assert len(part)==len(expected) and {r['sample_id'] for r in part}==expected
            metric=io.metric(part);b=base[s,'TD24_Ridge_CV']
            summary.append(dict(setting=s,condition=m,**metric,TD24_PCC=float(b['PCC']),PCC_delta=metric['PCC']-float(b['PCC']),
                PCC_relative_gain_pct=100*(metric['PCC']/float(b['PCC'])-1),MSE_relative_reduction_pct=100*(1-metric['MSE']/float(b['MSE']))))
    goal=[]
    for m in METHODS:
        part=[r for r in summary if r['condition']==m];g=float(np.mean([r['PCC_relative_gain_pct'] for r in part]))
        goal.append(dict(condition=m,mean_relative_PCC_gain_pct=g,reaches_development_5pct=g>=5,
            negative_settings=[r['setting'] for r in part if r['PCC_delta']<0]))
    io.write_csv(root/'summary.csv',summary);io.write_csv(root/'oof_predictions.csv',allpred);io.write_csv(root/'usage.csv',usage)
    io.write(root/'goal_summary.json',goal);validate()
    io.write(root/'complete.json',dict(lock_sha256=locksha,outer_folds=16,new_branches=112,new_linear_gates=16,new_constant_gates=16,
        rows=7596,GPU_hours=0,source_updates=0,excluded=['wang2009','alas1_patent'],
        files_sha256={p.relative_to(root).as_posix():io.sha(p) for p in sorted(root.rglob('*')) if p.is_file()}))
    print('ROUND2_COMPLETE',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('register','run'));a=p.parse_args()
    if a.mode=='register':register();return
    mutex=R/'execution.mutex'
    with mutex.open('x') as f:f.write(str(os.getpid()))
    started=time.monotonic();rc=1;error=None
    try:run();rc=0
    except Exception:error=traceback.format_exc();raise
    finally:
        io.write(R/f'execution_{time.time_ns()}.json',dict(returncode=rc,error=error,CPU_wall_seconds=time.monotonic()-started,GPU_hours=0,
            source_updates=0,lock_sha256=io.sha(R/'lock.json')));mutex.unlink()
if __name__=='__main__':main()
