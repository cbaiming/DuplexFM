"""One CPU-only preparation run: input verification, unit tests, actual encoders."""
import argparse,json,os,platform,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--assets',required=True)
    p.add_argument('--native-only',action='store_true');a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Preparation is CPU only')
    from duplexfm_mechanism.common import read,sha,verify_files,backup,check_disk
    from duplexfm_mechanism.durable_io import durable_json as write
    from duplexfm_mechanism.protocol import contract,OFFICIAL_TESTS
    root,assets=Path(a.root),Path(a.assets)
    envroot=root/'environment';envroot.mkdir(exist_ok=True)
    if a.native_only:
        from duplexfm_mechanism.engineering_three import smoke
        smoke(root,assets,envroot/'native_cpu');return
    if (envroot/'ready.json').exists():
        ready=read(envroot/'ready.json');verify_files(root,ready['files_sha256']);print(json.dumps(ready));return
    check_disk(root)
    from duplexfm_mechanism.reference_initial import reference
    reference(root)
    verify_files(root,read(root/'release_manifest.json')['files_sha256'])
    import numpy as np
    from duplexfm_joint.data import PairStore
    prep=root/'preparation';prep.mkdir(exist_ok=True)
    with PairStore(assets/'mirbench') as store:
        integrity=store.verify_integrity();train,val=store.indices('train'),store.indices('val')
        if (len(train),len(val))!=(2094107,401428):raise ValueError('Source row counts differ')
        refs=np.concatenate([val]+[store.indices(n) for n in OFFICIAL_TESTS])
        if np.intersect1d(train,refs).size:raise ValueError('Train/test row IDs overlap')
        for n,values in [('train',train),('validation',val)]:
            path=prep/(n+'.npy')
            if path.exists():
                if not np.array_equal(np.load(path,allow_pickle=False),values):raise ValueError('Preparation conflict')
            else:
                with path.open('xb') as f:np.save(f,values,allow_pickle=False)
        write(prep/'inputs.json',integrity,immutable=True)
        write(prep/'audit.json',{'train_rows':len(train),'validation_rows':len(val),'source_keep_overlap':True,
              'physics_arrays_opened':False,'no_source_label_shuffle':True,'source_seeds':[42],'fixed_masks':['no_pair','no_cross','no_access']},immutable=True)
    files={p.name:sha(p) for p in prep.iterdir() if p.is_file() and p.name!='complete.json'}
    write(prep/'complete.json',{'files_sha256':files,'data_manifest_sha256':integrity['manifest_sha256']},immutable=True)
    t=time.time()
    pip=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,timeout=120)
    write(envroot/'pip_check.json',{'returncode':pip.returncode,'stdout':pip.stdout,'stderr':pip.stderr},immutable=True)
    if pip.returncode:raise RuntimeError('pip check failed')
    packages=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True,timeout=120)
    (envroot/'requirements.lock.txt').write_text(packages.stdout,encoding='utf-8')
    run=subprocess.run([sys.executable,'-m','pytest','-q',str(root/'code/tests'),
                        '--junitxml='+str(envroot/'pytest.xml')],cwd=root/'code',capture_output=True,text=True,timeout=600)
    (envroot/'pytest.log').write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
    write(envroot/'pytest_receipt.json',{'returncode':run.returncode,'seconds':time.time()-t},immutable=True)
    if run.returncode:raise RuntimeError('CPU tests failed; see environment/pytest.log')
    if not (envroot/'native_cpu/report.json').exists():
        with (envroot/'native_cpu.log').open('ab') as log:
            native=subprocess.run([sys.executable,'-u',__file__,'--root',str(root),'--assets',str(assets),'--native-only'],
                                  stdout=log,stderr=subprocess.STDOUT,timeout=2400)
        if native.returncode:raise RuntimeError('Native CPU acceptance failed')
    report=read(envroot/'native_cpu/report.json')
    if report['status']!='passed':raise ValueError('Native check not passed')
    from duplexfm_mechanism.runtime import registration,environment_identity
    from duplexfm_mechanism.parallel import register
    registration(root,assets)
    if not (root/'parallel_ledger.json').exists():register(root,assets)
    identity=environment_identity()
    small=['environment/pip_check.json','environment/pytest_receipt.json','environment/pytest.xml',
           'environment/requirements.lock.txt','environment/native_cpu/report.json','registration.json',
           'parallel_registration.json','parallel_execution.json','preparation/complete.json']
    ready={'status':'passed','hostname':platform.node(),'environment':identity,'formal_training_started':False,
           'H100_GPU_acceptance_completed':False,'files_sha256':{n:sha(root/n) for n in small}}
    write(envroot/'ready.json',ready,immutable=True)
    payload=[p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'backups' not in p.parts
             and '.pytest_cache' not in p.parts and p.suffix in ('.py','.json','.sh','.txt','.md','.npy','.xml')]
    receipt=backup(root,payload,'prepared')
    write(envroot/'prepared_backup.json',receipt,immutable=True)
    print(json.dumps(ready,indent=2),flush=True)

if __name__=='__main__':main()
