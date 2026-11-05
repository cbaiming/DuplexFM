"""Independent local numerical verification; imports no model/backend/torch.

Consumes only the bounded small-result archive and already verified local input
rows/splits. It independently reconstructs every ranked validation CSV SHA and
MSE, 8→3→1 choice, test coverage, per-seed metric and macro mean/sample SD.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import tarfile
import time
import zipfile

import numpy as np
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import average_precision_score,f1_score,roc_auc_score

MAX_COMPRESSED=8<<20
MAX_RAW=16<<20
MODELS=('OligoFormer-corrected','BERT-siRNA','RNAiSpline')
SEEDS=(42,43,44)
GRID=('base','lr_half','lr_double','wd_zero','wd_ten','batch_half','batch_double','lr_double_batch_double')
MAIN4=('Huesken','Mixset','Takayuki','Huesken_to_Mixset')
EXTRA2=('Wang','ALAS1')
METRICS=('PCC','Spearman','MSE','MAE','AP','AUROC','F1')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')


def confined(root,name):
    if not name or '\\' in name or ':' in name or name.startswith('/') or any(p in ('','.','..') for p in name.split('/')):
        raise ValueError('Unsafe result member path')
    root=Path(root).resolve()
    path=(root/name).resolve()
    if path!=root and root not in path.parents:
        raise ValueError('Result member escapes extraction root')
    return path


def extract_small(archive,dest,expected_archive_sha=None,expected_manifest_sha=None):
    archive,dest=Path(archive),Path(dest)
    if archive.stat().st_size>MAX_COMPRESSED:
        raise ValueError('Result archive exceeds 8 MiB compressed bound')
    if expected_archive_sha and sha(archive)!=expected_archive_sha:
        raise ValueError('Transport archive SHA differs')
    dest.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive,'r:gz') as bundle:
        members=bundle.getmembers()
        names=[m.name for m in members]
        if len(set(names))!=len(names) or sum(m.size for m in members)>MAX_RAW:
            raise ValueError('Duplicate members or excessive uncompressed result payload')
        for member in members:
            if not member.isfile() or member.size<0 or member.size>MAX_RAW:
                raise ValueError('Only bounded regular result files are accepted')
            target=confined(dest,member.name)
            if target.suffix.lower() in ('.pt','.bin','.safetensors','.pkl','.pickle'):
                raise ValueError('Model/checkpoint files are not part of small-result transport')
            target.parent.mkdir(parents=True,exist_ok=True)
            with bundle.extractfile(member) as inp,target.open('xb') as out:
                data=inp.read(member.size+1)
                if len(data)!=member.size:
                    raise ValueError('Tar member size differs')
                out.write(data)
    manifest_path=dest/'manifest.json'
    if expected_manifest_sha and sha(manifest_path)!=expected_manifest_sha:
        raise ValueError('Transport manifest SHA differs')
    manifest=read(manifest_path)
    if manifest.get('contains_checkpoints') is not False:
        raise ValueError('Result manifest lacks checkpoint exclusion')
    actual={p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file()}
    if actual!=set(manifest['files'])|{'manifest.json'}:
        raise ValueError('Unexpected or missing result payload members')
    for name,value in manifest['files'].items():
        path=confined(dest,name)
        if path.stat().st_size!=value['bytes'] or sha(path)!=value['sha256']:
            raise ValueError('Result payload SHA/size differs: '+name)
    return manifest


def canonical_csv(ids,prediction,truth):
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=['sample_id','prediction','label'])
    writer.writeheader()
    for i,p,y in zip(ids,prediction,truth):
        writer.writerow(dict(sample_id=i,prediction=float(p),label=float(y)))
    return stream.getvalue().encode('utf-8')


def ranked(records,count):
    if len({r['candidate_id'] for r in records})!=len(records):
        raise ValueError('Repeated candidate score')
    return [r['candidate_id'] for r in sorted(records,key=lambda r:(r['mse'],GRID.index(r['candidate_id'])))[:count]]


def candidates_for(model,values):
    baseline={'OligoFormer-corrected':dict(lr=3e-4,wd=1e-4,batch=128),
              'BERT-siRNA':dict(lr=5e-5,wd=.01,batch=100),
              'RNAiSpline':dict(lr=1e-4,wd=1e-5,batch=16)}[model]
    canonical=[]
    for cid in GRID:
        hp=dict(baseline)
        if cid=='lr_half':hp['lr']*=.5
        if cid in ('lr_double','lr_double_batch_double'):hp['lr']*=2
        if cid=='wd_zero':hp['wd']=0.
        if cid=='wd_ten':hp['wd']*=10
        if cid=='batch_half':hp['batch']//=2
        if cid in ('batch_double','lr_double_batch_double'):hp['batch']*=2
        canonical.append(dict(id=cid,hyperparameters=hp))
    actual=[dict(id=v['id'],hyperparameters=v.get('hyperparameters') or {k:v[k] for k in ('lr','wd','batch')}) for v in values]
    if actual!=canonical:
        raise ValueError('Exported candidate grid is not the registered native-policy grid')
    return {v['id']:v for v in canonical}


def setting_name(value):
    if value in ('Huesken->Mixset','Huesken→Mixset','Huesken_to_Mixset','Huesken-Mixset','Inter_Hu_to_Mix'):
        return 'Huesken_to_Mixset'
    if value=='wang2009':
        return 'Wang'
    if value=='alas1_patent':
        return 'ALAS1'
    return value


def vector_archive(path,entries):
    with zipfile.ZipFile(path) as z:
        items=z.infolist()
        if (len(items)!=len(entries) or len({i.filename for i in items})!=len(items) or
                sum(i.file_size for i in items)>MAX_RAW or any(i.file_size>10000 for i in items)):
            raise ValueError('Compact validation archive is oversized or contains unexpected entries')
        expected={e['key']+'.npy' for e in entries}
        if {i.filename for i in items}!=expected:
            raise ValueError('Compact validation member names differ')
        counts={e['key']+'.npy':e['n'] for e in entries}
        for item in items:
            count=counts[item.filename]
            if not isinstance(count,int) or not 1<=count<=1000:
                raise ValueError('Compact validation sample count exceeds this fixed benchmark')
            with z.open(item) as stream:
                version=np.lib.format.read_magic(stream)
                if version==(1,0):
                    shape,order,dtype=np.lib.format.read_array_header_1_0(stream)
                elif version==(2,0):
                    shape,order,dtype=np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('Unregistered NPY format in compact validation archive')
                if shape!=(count,) or dtype!=np.dtype(np.float64) or order or stream.tell()+count*8!=item.file_size:
                    raise ValueError('NPY header/dtype/size differs before allocating its array')
    arrays=np.load(path,allow_pickle=False)
    return arrays


def verify_search(payload,cfg,plan,rows_by_id):
    groups=cfg['groups']
    blocks={(b['setting'],b['fold']):b for b in plan['folds']}
    entries=read(payload/'validation_index.json')
    lookup={(e['group_id'],e['rung'],e['candidate_id'],e['inner']):e for e in entries}
    if len(lookup)!=len(entries) or len(entries)!=78*14:
        raise ValueError('Expected 8 rung1 and 6 rung2 validation vectors per each of 78 groups')
    arrays=vector_archive(payload/'validation_predictions.npz',entries)
    used=set()
    winners=[]

    def score(group,rung,record,item):
        key=(group['id'],rung,record['candidate_id'],record['inner'])
        entry=lookup[key]
        pred=np.asarray(arrays[entry['key']])
        ids=[item['fit_ids'][i] for i in item['inner'][record['inner']]['valid']]
        truth=np.array([np.float32(rows_by_id[i]['label']) for i in ids],dtype=np.float64)
        if pred.dtype!=np.float64 or pred.shape!=(len(ids),) or not np.isfinite(pred).all():
            raise ValueError('Compact validation vector type/shape/finite check failed')
        if entry['n']!=len(ids) or record['validation_n']!=len(ids):
            raise ValueError('Validation sample-count weight differs')
        names=[n for n in record['files'] if n.endswith('.csv')]
        rebuilt=hashlib.sha256(canonical_csv(ids,pred,truth)).hexdigest()
        if len(names)!=1 or entry['source_csv']!=names[0] or rebuilt!=entry['source_csv_sha256'] or rebuilt!=record['files'][names[0]]:
            raise ValueError('Independent validation CSV reconstruction does not match original SHA')
        mse=float(np.mean((pred-truth)**2))
        if not math.isfinite(mse) or abs(mse-float(record['mse']))>1e-12:
            raise ValueError('Independent validation MSE differs')
        used.add(key)
        return mse,len(ids)

    for group in groups:
        item=blocks[(group['setting'],group['fold'])]
        source=payload/'groups'/group['id']
        rung1,rung2,winner,selection=[read(source/name) for name in ('rung1.json','rung2.json','winner.json','selection.json')]
        if (rung1.get('outer_test_used_for_selection') is not False or rung2.get('outer_test_used_for_selection') is not False or
                winner.get('outer_test_used_for_selection') is not False or selection.get('outer_test_used_for_selection') is not False):
            raise ValueError('Search output lacks explicit outer-test exclusion')
        cap={'OligoFormer-corrected':30,'BERT-siRNA':8,'RNAiSpline':13}[group['model']]
        if rung1['seed']!=42 or rung1['cap']!=cap or tuple(r['candidate_id'] for r in rung1['scores'])!=GRID:
            raise ValueError('Rung1 candidate grid/seed/cap differs')
        for record in rung1['scores']:
            if record['inner']!=0 or not isinstance(record['selected_epochs'],int) or not 1<=record['selected_epochs']<=cap:
                raise ValueError('Rung1 may use only inner0')
            score(group,1,record,item)
        promoted=ranked(rung1['scores'],3)
        if promoted!=rung1['promoted'] or [r['candidate_id'] for r in rung2['scores']]!=promoted or rung2['seed']!=42:
            raise ValueError('Rung1 promotion or rung2 candidate scope differs')
        for record in rung2['scores']:
            if [p['inner'] for p in record['parts']]!=[0,1] or any(p['candidate_id']!=record['candidate_id'] for p in record['parts']):
                raise ValueError('Rung2 is not exactly the first two inner folds of one candidate')
            values=[score(group,2,p,item) for p in record['parts']]
            if any(not isinstance(p['selected_epochs'],int) or not 1<=p['selected_epochs']<=
                   {'OligoFormer-corrected':120,'BERT-siRNA':30,'RNAiSpline':50}[group['model']] for p in record['parts']):
                raise ValueError('Rung2 selected epochs exceed the registered native budget')
            n=sum(v[1] for v in values)
            mse=sum(v[0]*v[1] for v in values)/n
            if record['validation_n']!=n or abs(mse-record['mse'])>1e-12:
                raise ValueError('Rung2 independent sample-weighted MSE differs')
        cid=ranked(rung2['scores'],1)[0]
        candidates=candidates_for(group['model'],cfg['candidates'][group['model']])
        if (rung2['winner']!=cid or winner['candidate']!=candidates[cid] or selection['winner']!=candidates[cid] or
                winner['rung1_sha256']!=sha(source/'rung1.json') or winner['rung2_sha256']!=sha(source/'rung2.json') or
                selection.get('search_seed')!=42 or selection.get('fixed_other_seeds')!=[43,44]):
            raise ValueError('Frozen winner/three-seed reuse is not bound to the exact search')
        for seed in SEEDS:
            result=selection['seed_selection'][str(seed)]
            epochs=result['inner_epochs']
            if len(epochs)!=5 or any(not isinstance(e,int) or e<1 for e in epochs) or int(np.median(epochs))!=result['final_epochs']:
                raise ValueError('Final epoch count is not the median of five selected inner epochs')
        winners.append(dict(group_id=group['id'],model=group['model'],setting=group['setting'],fold=group['fold'],candidate_id=cid))
    arrays.close()
    if used!=set(lookup):
        raise ValueError('Unused/extra compact validation predictions')
    return winners,len(entries)


def metrics(truth,pred,binary):
    truth,pred,binary=np.asarray(truth,float),np.asarray(pred,float),np.asarray(binary,int)
    if len(truth)<2 or not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError('Metric inputs need at least two finite paired rows')
    constant=np.ptp(truth)==0 or np.ptp(pred)==0
    return dict(PCC=None if constant else float(pearsonr(truth,pred).statistic),
        Spearman=None if constant else float(spearmanr(truth,pred).statistic),
        MSE=float(np.mean((truth-pred)**2)),MAE=float(np.mean(abs(truth-pred))),
        AP=float(average_precision_score(binary,pred)) if len(set(binary))==2 else None,
        AUROC=float(roc_auc_score(binary,pred)) if len(set(binary))==2 else None,
        F1=float(f1_score(binary,pred>=.5,zero_division=0)))


def metric_tables(predictions,cfg,plan,rows_by_id):
    blocks={(b['setting'],b['fold']):b for b in plan['folds']}
    expected={(g['model'],seed,g['setting'],g['fold'],sample_id)
              for g in cfg['groups'] for seed in SEEDS for sample_id in blocks[(g['setting'],g['fold'])]['test_ids']}
    if len(predictions)!=37413 or len(expected)!=37413:
        raise ValueError('Expected exactly 37413 final model/seed/test prediction rows')
    actual=set()
    for row in predictions:
        key=(row['model'],int(row['seed']),row['setting'],int(row['fold']),row['sample_id'])
        if key in actual or key not in expected:
            raise ValueError('Final prediction duplicates or unexpected test identity')
        actual.add(key)
        source=rows_by_id[row['sample_id']]
        if float(row['label'])!=float(source['label']) or int(row['binary'])!=int(source['y']) or not math.isfinite(float(row['prediction'])):
            raise ValueError('Final label/binary contract or finite score differs')
    if actual!=expected:
        raise ValueError('Final predictions are not per-seed exact-once across all outer folds')
    per_seed,fold_rows=[],[]
    settings=sorted({g['setting'] for g in cfg['groups']})
    for model in MODELS:
        for seed in SEEDS:
            for setting in settings:
                rows=[r for r in predictions if r['model']==model and int(r['seed'])==seed and r['setting']==setting]
                result=metrics([float(r['label']) for r in rows],[float(r['prediction']) for r in rows],[int(r['binary']) for r in rows])
                per_seed.append(dict(model=model,seed=seed,setting=setting_name(setting),n=len(rows),**result))
                for fold in sorted({int(r['fold']) for r in rows}):
                    part=[r for r in rows if int(r['fold'])==fold]
                    fold_rows.append(dict(model=model,seed=seed,setting=setting_name(setting),fold=fold,n=len(part),
                        **metrics([float(r['label']) for r in part],[float(r['prediction']) for r in part],[int(r['binary']) for r in part])))
    for model in MODELS:
        for seed in SEEDS:
            for title,pool in (('Macro4',MAIN4),('Macro2_extra',EXTRA2)):
                rows=[r for r in per_seed if r['model']==model and r['seed']==seed and r['setting'] in pool]
                if len(rows)!=len(pool):
                    raise ValueError('Primary-four/extra-two setting identity contract differs')
                per_seed.append(dict(model=model,seed=seed,setting=title,n=sum(r['n'] for r in rows),
                    **{key:None if any(r[key] is None for r in rows) else float(np.mean([r[key] for r in rows])) for key in METRICS}))
    summary=[]
    for model in MODELS:
        for setting in sorted({r['setting'] for r in per_seed}):
            rows=[r for r in per_seed if r['model']==model and r['setting']==setting]
            if sorted(r['seed'] for r in rows)!=list(SEEDS):
                raise ValueError('Summary must contain each of the three training seeds exactly once')
            values={}
            for key in METRICS:
                numbers=[r[key] for r in rows]
                values[key+'_mean']=None if None in numbers else float(np.mean(numbers))
                values[key+'_SD']=None if None in numbers else float(np.std(numbers,ddof=1))
            summary.append(dict(model=model,setting=setting,seeds=3,**values))
    return per_seed,summary,fold_rows


def write_csv(path,rows):
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def verify_allocation_costs(costs,config_sha):
    if costs.get('closed') is not True or costs['allocated_GPU_seconds']>128*3600+1e-5:
        raise ValueError('GPU allocation cost is unclosed or outside the fixed budget')
    attempts=costs.get('attempts',[])
    if not attempts or len({r['attempt'] for r in attempts})!=len(attempts):
        raise ValueError('Allocation attempts must be nonempty and unique')
    total_seconds=0.
    for record in attempts:
        cost=record['receipt']
        raw=(json.dumps(cost,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')
        if (cost.get('closed') is not True or cost.get('own_children_exited') is not True or
                abs(cost['allocated_GPU_seconds']-4*cost['wall_seconds'])>1e-5 or
                cost.get('attempt')!=record['attempt'] or cost.get('config_sha256')!=config_sha or
                cost.get('started_sha256')!=record['started_sha256'] or
                cost.get('allocated_GPU_seconds')!=record['allocated_GPU_seconds'] or
                not math.isfinite(cost['allocated_GPU_seconds']) or cost['allocated_GPU_seconds']<0 or
                hashlib.sha256(raw).hexdigest()!=record['closed_sha256']):
            raise ValueError('An exported allocation lacks actual closed child/accounting evidence')
        total_seconds+=cost['allocated_GPU_seconds']
    if abs(total_seconds-costs['allocated_GPU_seconds'])>1e-5 or not any(r['status']=='complete' for r in attempts):
        raise ValueError('Allocation total does not independently reproduce from all closed attempts')
    return total_seconds


def analyze(payload,input_rows,split_plan,output):
    began=time.monotonic()
    payload,input_rows,split_plan,output=map(Path,(payload,input_rows,split_plan,output))
    cfg,proof=read(payload/'hpo_config.json'),read(payload/'provenance.json')
    if sha(payload/'hpo_config.json')!=proof['config_sha256'] or cfg.get('task')!='oligo_random_six_hpo4_v1':
        raise ValueError('Export does not bind the new HPO configuration')
    if sha(input_rows)!=proof['input_rows_sha256'] or sha(split_plan)!=proof['split_plan_sha256']:
        raise ValueError('Local frozen input rows/splits SHA differ')
    if (proof.get('all_group_checkpoints_verified_remotely') is not True or
            proof.get('all_latest_backup_objects_second_SHA_readback') is not True or
            proof.get('checkpoints_in_transport') is not False or
            proof.get('closed_ledger_unchanged_after_export_audit') is not True or
            proof.get('final_process_evidence',{}).get('own_matching_processes')!=[]):
        raise ValueError('Remote checkpoint/backup audit evidence is missing')
    audits=read(payload/'group_audits.json')
    if (len(audits)!=78 or len({a['group']['id'] for a in audits})!=78 or
            sum(a['full_fit_count'] for a in audits)!=1716 or sum(a['predictions'] for a in audits)!=37413):
        raise ValueError('Remote completion audit does not bind 78 groups/1716 full fits/37413 rows')
    expected_groups={g['id']:g for g in cfg['groups']}
    if {a['group']['id'] for a in audits}!=set(expected_groups) or any(
            a['group']!=expected_groups[a['group']['id']] or a['full_fit_count']!=22 or
            a['backup_audit'].get('objects_second_SHA_readback') is not True for a in audits):
        raise ValueError('Remote per-group identity/full-fit/backup audit differs')
    costs=read(payload/'allocation_costs.json')
    verify_allocation_costs(costs,proof['config_sha256'])
    rows=read(input_rows)
    rows_by_id={r['sample_id']:r for r in rows}
    if len(rows_by_id)!=len(rows):
        raise ValueError('Frozen input sample IDs repeat')
    plan=read(split_plan)
    winners,n_vectors=verify_search(payload,cfg,plan,rows_by_id)
    with (payload/'predictions.csv').open(encoding='utf-8',newline='') as f:
        predictions=list(csv.DictReader(f))
    for audit in audits:
        group=audit['group']
        selected=[r for r in predictions if r['model']==group['model'] and r['setting']==group['setting'] and int(r['fold'])==group['fold']]
        if not selected or len(selected)!=audit['predictions']:
            raise ValueError('Combined final predictions differ from a source group row count')
        stream=io.StringIO(newline='')
        writer=csv.DictWriter(stream,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
        if hashlib.sha256(stream.getvalue().encode('utf-8')).hexdigest()!=audit['source_predictions_sha256']:
            raise ValueError('Combined predictions do not reconstruct the source group CSV SHA')
    per_seed,summary,fold_rows=metric_tables(predictions,cfg,plan,rows_by_id)
    output.mkdir(parents=True,exist_ok=False)
    write_csv(output/'per_seed.csv',per_seed)
    write_csv(output/'summary.csv',summary)
    write_csv(output/'per_fold_supplement.csv',fold_rows)
    write_csv(output/'selected_candidates.csv',winners)
    text=['# HPO six-setting numerical results','',
        'Primary metric: pooled exact-once OOF PCC per setting and training seed; mean ± sample SD over seeds 42/43/44.',
        'The four primary settings and two protocol-extension settings have separate arithmetic macros. Fold metrics are supplementary.','',
        '| Model | Huesken | Mixset | Takayuki | Huesken→Mixset | Main 4 | Wang | ALAS1 | Extra 2 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model in MODELS:
        values=[]
        for setting in ('Huesken','Mixset','Takayuki','Huesken_to_Mixset','Macro4','Wang','ALAS1','Macro2_extra'):
            row=next(r for r in summary if r['model']==model and r['setting']==setting)
            values.append('/' if row['PCC_mean'] is None else f"{row['PCC_mean']:.4f} ± {row['PCC_SD']:.4f}")
        text.append('| '+model+' | '+' | '.join(values)+' |')
    text+=['','Validation: all 78 inner-only searches, 1,092 immutable validation vectors, 1,716 remotely audited full-fit receipts and 37,413 final predictions.',
           'No source model or checkpoint was downloaded. Full weights/backup objects were audited remotely; local verification independently reconstructed every ranked validation CSV SHA and all numerical results.',
           'Source HPO uses seed 42 only. Seeds 43/44 reuse the selected hyperparameters; their SD does not represent independent HPO repetitions. Random splits may retain near-neighbor sequences; these results do not establish SOTA or a new blind external confirmation.',
           f"Closed allocation GPU hours: {costs['allocated_GPU_seconds']/3600:.6f}. Post-allocation CPU export/local analysis are separate; prior fixed-parameter CCI compute is excluded."]
    (output/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    result=dict(passed=True,analyzer_sha256=sha(__file__),input_rows_sha256=sha(input_rows),split_plan_sha256=sha(split_plan),
        groups=78,full_fits_remotely_verified=1716,validation_vectors_independently_recomputed=n_vectors,
        prediction_rows=len(predictions),per_seed_metric_rows=len(per_seed),
        full_checkpoint_bytes_verified_locally=False,new_GPU_seconds=0,CPU_wall_seconds=time.monotonic()-began,
        files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in output.iterdir() if p.is_file()})
    write_new(output/'local_verification.json',result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--archive-sha',required=True)
    p.add_argument('--manifest-sha',required=True)
    p.add_argument('--extract-to',type=Path,required=True)
    p.add_argument('--input-rows',type=Path,required=True)
    p.add_argument('--split-plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    extract_small(args.archive,args.extract_to,args.archive_sha,args.manifest_sha)
    print(json.dumps(analyze(args.extract_to,args.input_rows,args.split_plan,args.output)),flush=True)


if __name__=='__main__':
    main()
