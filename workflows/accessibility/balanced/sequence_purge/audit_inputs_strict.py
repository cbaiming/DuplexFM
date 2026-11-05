"""Audit input-only exclusion evidence before outcomes are loaded."""
from pathlib import Path
import csv,gzip,json,re,time
import numpy as np
from purge_sequences_strict import ROOT,PARENT,SOURCE,EXPECTED,normalize,revcomp,sha,write_json,check_lock

def alignment_proof(a,b):
    # Global alignment feasibility at >=90% identity: +1 match, -9 error.
    # This independent integer DP does not use CD-HIT's scores or word filters.
    prior=[-9*j for j in range(len(b)+1)]
    for i,x in enumerate(a,1):
        row=[-9*i]
        for j,y in enumerate(b,1):
            row.append(max(prior[j-1]+(1 if x==y else -9),prior[j]-9,row[-1]-9))
        prior=row
    return int(prior[-1])

def main():
    start=time.time();check_lock();frozen=json.loads((ROOT/'mask_frozen.json').read_text());assert frozen['passed']
    assert not (ROOT/'input_audit.json').exists()
    for n,h in frozen['files'].items():assert sha(ROOT/n)==h,n
    for n,h in EXPECTED.items():assert sha(SOURCE/n)==h,n
    seq=np.load(SOURCE/'sequences.npy',mmap_mode='r');lengths=np.load(SOURCE/'lengths.npy',mmap_mode='r');splits=np.load(SOURCE/'splits.npy',mmap_mode='r')
    with np.load(PARENT/'dataset_v1/dataset.npz') as z:
        query=[normalize(s,n) for s,n in zip(z['sequence'],z['lengths'])];ql=z['lengths'];wid=z['window_id'];genes=z['gene_id'];groups=z['evaluation_group']
    with np.load(ROOT/'retained_masks.npz') as z:masks={k:z[k] for k in z.files}
    with gzip.open(ROOT/'excluded_matches.csv.gz','rt') as f:rows=list(csv.DictReader(f))
    rng=np.random.default_rng(20261104);sample=set(rng.choice(len(rows),min(512,len(rows)),replace=False).tolist());proofs=[];strands={'+':0,'-':0};seen=set();boundary90=0
    for k,row in enumerate(rows):
        i=int(row['query_index']);j=int(row['source_row_index']);assert i not in seen;seen.add(i)
        assert splits[j]<2 and not masks['sequence_purged'][i]
        m=re.search(r'at (\d+):(\d+):(\d+):(\d+)/([+-])/([\d.]+)%',row['cdhit_evidence']);assert m,row
        a,b,c,d=map(int,m.groups()[:4]);strand=m[5];identity=float(m[6]);strands[strand]+=1
        assert identity>=90 and (abs(b-a)+1)*5>=len(query[i])*4 and (abs(d-c)+1)*5>=int(lengths[j])*4,row
        assert 1<=min(a,b)<=max(a,b)<=len(query[i]) and 1<=min(c,d)<=max(c,d)<=lengths[j]
        boundary90+=identity==90
        if k in sample:
            ref=normalize(seq[j],lengths[j]);x=query[i][min(a,b)-1:max(a,b)];y=ref[min(c,d)-1:max(c,d)]
            if strand=='-':x=revcomp(x)
            score=alignment_proof(x,y);assert score>=0,(i,j,score,row)
            proofs.append(dict(query_index=i,source_row_index=j,source_split=int(splits[j]),query_sequence=query[i],source_sequence=ref,
                               query_start1=a,query_end1=b,source_start1=c,source_end1=d,strand=strand,reported_identity_percent=identity,integer_alignment_score=score))
    assert seen==set(np.flatnonzero(~masks['sequence_purged']).tolist())
    with np.load(PARENT/'evaluation/exposure.npz') as z:
        assert not np.any(masks['sequence_purged']&z['sequence_seen'])
        assert np.array_equal(masks['sequence_purged_gene_unseen'],masks['sequence_purged']&~z['gene_seen'])
        np.savez_compressed(ROOT/'previous_exposure_masks.npz',sequence_seen=z['sequence_seen'],gene_seen=z['gene_seen'])
    np.savez_compressed(ROOT/'alignment_samples.npz',proofs=np.array([json.dumps(x,sort_keys=True) for x in proofs]))
    with gzip.open(ROOT/'window_selection.csv.gz','wt',encoding='utf-8',newline='') as f:
        w=csv.writer(f);w.writerow(['index','window_id','length','gene_id','evaluation_group','sequence_purged','sequence_purged_gene_unseen'])
        for i in range(len(query)):w.writerow([i,wid[i],ql[i],genes[i],groups[i],int(masks['sequence_purged'][i]),int(masks['sequence_purged_gene_unseen'][i])])
    v=dict(passed=True,all_hit_coverage_identity_partition_checks=len(rows),independent_integer_alignment_checks=len(proofs),orientation_counts=strands,
           exactly_90_percent_reported=boundary90,retained_counts={k:int(v.sum()) for k,v in masks.items()},
           gene_counts={k:int(len(set(genes[v]))) for k,v in masks.items()},group_counts={k:int(len(set(groups[v]))) for k,v in masks.items()},
           seconds=time.time()-start,labels_or_predictions_read=False,mask_frozen_sha256=sha(ROOT/'mask_frozen.json'),
           files={n:sha(ROOT/n) for n in ['alignment_samples.npz','window_selection.csv.gz','previous_exposure_masks.npz']})
    write_json(ROOT/'input_audit.json',v);print(json.dumps(v),flush=True)

if __name__=='__main__':main()
