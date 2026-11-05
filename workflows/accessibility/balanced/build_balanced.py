"""Build a length-balanced, fully measured mRNA cohort; no model execution."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib,gzip,csv,time,math
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'external'))
import build_mrna60 as source
OUT=HERE/'dataset_v1';LENGTHS=tuple(range(50,61));SEED=42

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def priority(kind,identifier):
 return hashlib.sha256(f'length-balanced-v1|{SEED}|{kind}|{identifier}'.encode()).hexdigest()

def maximum_coverage(n):
 k=(n+59)//60
 return n if 50*k<=n else 60*(n//60)

def allocate(n,counts):
 """Balance globally, retaining maximum measured-base coverage per run."""
 original=n;result=[]
 while n>=50:
  goal=maximum_coverage(n)
  allowed=[l for l in LENGTHS if l<=n and l+maximum_coverage(n-l)==goal]
  assert allowed
  chosen=min(allowed,key=lambda l:(counts[l],priority('length',str(l))))
  result.append(chosen);counts[chosen]+=1;n-=chosen
 assert sum(result)==maximum_coverage(original)
 return result

def observed_runs(seq,tokens):
 assert len(seq)==len(tokens);i=0
 while i<len(seq):
  if tokens[i]=='NULL' or seq[i] not in 'ACGU':i+=1;continue
  j=i+1
  while j<len(seq) and tokens[j]!='NULL' and seq[j] in 'ACGU':j+=1
  if j-i>=50:yield i,j
  i=j

def main():
 assert not OUT.exists(),'Refuse overwrite'
 protocol=json.loads((HERE/'BUILD_PROTOCOL.json').read_text())
 assert sha(Path(__file__))==protocol['code_sha256']
 assert sha(source.DATA)==protocol['measurement_sha256']==source.RAW_SHA
 assert sha(source.REF)==protocol['reference_sha256']
 assert sha(Path(source.__file__))==protocol['source_helper_sha256']
 started=time.time();refs,refcounts=source.reference_subset(source.REF,source.read_ids(source.DATA))
 runs=[];profiles={};filtering=Counter()
 with gzip.open(source.DATA,'rt') as f:
  for lineno,line in enumerate(f,1):
   parts=line.split();tx=parts[0];n=int(parts[1]);tokens=parts[3:];assert len(tokens)==n
   filtering['source_transcripts']+=1;ref=refs.get(tx)
   if ref is None:filtering['excluded_no_reference']+=1;continue
   if ref['gene_biotype']!='protein_coding' or ref['transcript_biotype']!='protein_coding':
    filtering['excluded_not_protein_coding']+=1;continue
   if len(ref['sequence'])!=n:filtering['excluded_length_mismatch']+=1;continue
   vals=np.array([np.nan if t=='NULL' else float(t) for t in tokens],dtype='float64')
   assert all(t=='NULL' or math.isfinite(float(t)) and 0<=float(t)<=1 for t in tokens)
   profiles[tx]=vals
   for a,b in observed_runs(ref['sequence'],tokens):
    runs.append(dict(transcript_id=tx,gene_id=ref['gene'],source_line=lineno,start0=a,end0_exclusive=b,length=b-a))
 runs.sort(key=lambda x:(x['length'],priority('run',f"{x['transcript_id']}:{x['start0']}")))
 counts=Counter({l:0 for l in LENGTHS});pool=[];max_bases=0
 for run in runs:
  tx=run['transcript_id'];cursor=run['start0'];max_bases+=maximum_coverage(run['length'])
  for length in allocate(run['length'],counts):
   seq=refs[tx]['sequence'][cursor:cursor+length];digest=hashlib.sha256(seq.encode()).hexdigest()
   pool.append(dict(window_id=f'{tx}:{cursor+1}-{cursor+length}',transcript_id=tx,gene_id=run['gene_id'],source_line=run['source_line'],
       start0=cursor,end0_exclusive=cursor+length,sequence=seq,sequence_sha256=digest,sequence_id='seq_'+digest,length=length))
   cursor+=length
  assert cursor<=run['end0_exclusive']
 assert sum(r['length'] for r in pool)==max_bases
 per_length=min(counts.values());assert per_length>0
 selected=[]
 for length in LENGTHS:
  candidates=[r for r in pool if r['length']==length]
  selected.extend(sorted(candidates,key=lambda x:priority('window',x['window_id']))[:per_length])
 selected.sort(key=lambda x:(x['transcript_id'],x['start0']));groups=source.add_groups(selected)
 assert Counter(r['length'] for r in selected)==Counter({l:per_length for l in LENGTHS})
 OUT.mkdir();keys=list(selected[0]);arrays={k:np.array([r[k] for r in selected]) for k in keys}
 values=np.full((len(selected),60),np.nan,dtype='float64');mask=np.zeros_like(values,dtype=bool)
 for i,r in enumerate(selected):
  y=profiles[r['transcript_id']][r['start0']:r['end0_exclusive']];assert np.isfinite(y).all()
  values[i,:r['length']]=y;mask[i,:r['length']]=True
 arrays.update(reactivity=values,observed_mask=mask,lengths=arrays.pop('length'))
 np.savez_compressed(OUT/'dataset.npz',**arrays)
 with gzip.open(OUT/'windows.csv.gz','wt',newline='',encoding='utf-8') as f:
  fields=keys+[f'reactivity_{j+1:02d}' for j in range(60)];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for i,row in enumerate(selected):
   record=dict(row);record.update({f'reactivity_{j+1:02d}':repr(float(v)) if j<row['length'] else '' for j,v in enumerate(values[i])});writer.writerow(record)
 unique={row['sequence_id']:row['sequence'] for row in selected}
 with (OUT/'unique_sequences.fasta').open('w',encoding='ascii',newline='\n') as f:
  for sid,seq in sorted(unique.items()):f.write(f'>{sid}\n{seq}\n')
 (OUT/'evaluation_groups.json').write_text(json.dumps(groups,sort_keys=True),encoding='utf-8')
 audit=dict(candidate_windows=len(pool),candidate_length_counts=dict(counts),candidate_covered_positions=max_bases,
    selected_windows=len(selected),selected_per_length=per_length,discarded_windows=len(pool)-len(selected),
    retained_window_fraction=len(selected)/len(pool),retained_position_fraction=int(mask.sum())/max_bases,
    eligible_runs=len(runs),eligible_transcripts=len(profiles),filtering=dict(filtering),reference_counts=refcounts,
    coverage='Maximum per run before balancing; deterministic heuristic, not a claim of globally optimal balanced size',
    sampling_uses_label_values=False,run_order='ascending run length, SHA256 tie',selection='minimum current length count among maximum-coverage-feasible choices; SHA256 tie; final equal-count SHA256 subset')
 (OUT/'selection_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
 manifest=dict(schema='fully_measured_nonoverlap_length_balanced_50_60_v1',seed=SEED,windows=len(selected),entries_per_length=per_length,
    length_counts=dict(Counter(map(int,arrays['lengths']))),positions=int(mask.sum()),true_zeros=int((values==0).sum()),
    unique_sequences=len(unique),genes=len(set(arrays['gene_id'])),transcripts=len(set(arrays['transcript_id'])),groups=len(groups),
    source_sha256=sha(source.DATA),reference_sha256=sha(source.REF),protocol_sha256=sha(HERE/'BUILD_PROTOCOL.json'),
    nonoverlap_scope='Within each transcript; isoforms may share genomic bases; same gene/sequence grouping retained',
    prior_test_exposure=True,blind_external=False,training_or_inference_run=False,
    cell_condition_ambiguity='GEO filename InVivo vs description InVitro unresolved',
    seconds=time.time()-started,files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in OUT.iterdir()})
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print(json.dumps(dict(manifest=manifest,selection=audit),indent=2))
if __name__=='__main__':main()
