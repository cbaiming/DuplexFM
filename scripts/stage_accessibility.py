"""Stage the fixed CNN code next to restored inputs, without starting training."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES=('common.py','models.py','train.py','precompute.py','linearfold_backend.py','accessibility_metrics.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True,help='Restored icshape_source/data directory')
    p.add_argument('--workdir',type=Path,required=True)
    p.add_argument('--index',type=Path,default=ROOT/'data/large/icshape_source/index.json',help='Dataset index; may reside in a separate reviewed data checkout')
    a=p.parse_args();data=a.data.resolve();work=a.workdir.resolve()
    if work.exists():raise FileExistsError('Use a new isolated workdir')
    expected=json.loads(a.index.read_text())
    members={Path(x['path']).name:x for x in expected['members'] if x['path'].startswith('data/')}
    for n in ['sequences.npy','lengths.npy','median_accessibility_profiles.npy','observed_mask.npy','splits.npy','transcript_components.npy']:
        if sha(data/n)!=members[n]['sha256']:raise ValueError('Training array identity differs: '+n)
    work.mkdir(parents=True);(work/'inputs').mkdir()
    for n in FILES:shutil.copyfile(ROOT/'workflows/accessibility/cnn'/n,work/n)
    contract={'schema':'portable_supervised_CNN_v1','seed':42,'files':{n:sha(work/n) for n in FILES},
              'train_loss':'per-window masked L1','selection':'original validation PCC composite / region-MAE tie',
              'source_arrays':{n:members[n]['sha256'] for n in members},'new_execution_registration':True}
    (work/'science_lock.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    (work/'inputs/input_package_manifest.json').write_text(json.dumps({'data/'+n:dict(sha256=v['sha256'],bytes=v['bytes']) for n,v in members.items()},indent=2),encoding='utf-8')
    print(json.dumps(dict(staged=True,training_started=False,workdir=str(work),
        instructions=['Set ICSHAPE_DATA to the restored data directory.','Set DUPLEXFM_LINEARFOLD_ROOT to the declared tool build.','Run precompute.py once; it refuses a different historical binary.','Run train.py sequence --device cpu or an explicitly selected free GPU.','Run train.py structure with the same fixed settings.']),indent=2))
if __name__=='__main__':main()
