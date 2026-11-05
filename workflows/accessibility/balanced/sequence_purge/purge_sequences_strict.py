"""CPU input-only purge. Does not load prediction or reactivity arrays."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json,os,re,subprocess,time
import numpy as np
BASE=Path(__file__).resolve().parent
ROOT=BASE/'strict_v2'
PARENT=BASE.parent
SOURCE=Path(os.environ.get('ICSHAPE_DATA',str(BASE.parents[3]/'external_data/icshape_source/data')))
DATA_SHA='1c6157293616224ecdacd9cfbf76387ac65f67243ff412fde251dff29a961719'
EXPECTED={
 'sequences.npy':'ca7cdcafb22ac6eb2b31017646ff37a8e642f5806021d105b835de5608c55bc7',
 'splits.npy':'7ec0d282d8b247926fdaa4f8d109e6c9f5dfc279ca89a9420263550cb87508ff',
 'lengths.npy':'309de0f791cc7eb67893a610e43fc8c6a4daedf86dd5f9212d550c413399e9bf'}
OPTIONS=['-c','0.90','-G','0','-aS','0.80','-aL','0.80','-n','8','-b','60','-s2','0.0','-S2','60','-r','1','-g','0','-p','1','-d','0','-T','8','-M','8192']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def write_json(p,v):
 p=Path(p);assert not p.exists(),str(p);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.flush();os.fsync(f.fileno())
 tmp.replace(p)

def check_lock():
 v=json.loads((BASE/'execution_lock_strict.json').read_text())
 for n,h in v['files'].items():assert sha(BASE/n)==h,n
 return sha(BASE/'execution_lock_strict.json')

def binary():
 x=json.loads((BASE/'tools/boundaryfix2_identity.json').read_text())
 assert x['source_sha256']=='f8bc3cdd7aebb432fcd35eed0093e7a6413f1e36bbd2a837ebc06e57cdb20b70'
 assert sha(x['binary'])==x['binary_sha256']
 return x['binary']

def normalize(s,n=None):
 s=s.decode('ascii') if isinstance(s,(bytes,np.bytes_)) else str(s)
 if n is not None:s=s[:int(n)]
 s=s.upper().replace('U','T');assert set(s)<=set('ACGT'),repr(s)
 return s

def revcomp(s):return s.translate(str.maketrans('ACGT','TGCA'))[::-1]

def fasta(p,pairs):
 with Path(p).open('x',encoding='ascii',newline='\n') as f:
  for name,s in pairs:f.write('>'+name+'\n'+s+'\n')

def run_tool(ref,query,out,log):
 cmd=[binary(),'-i',str(ref),'-i2',str(query),'-o',str(out)]+OPTIONS;start=time.time()
 with Path(log).open('xb') as f:q=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=7200)
 v=dict(command=cmd,returncode=q.returncode,wall_seconds=time.time()-start,log_sha256=sha(log),cpu_only=True)
 write_json(str(log)+'.json',v);assert q.returncode==0,str(log)
 return v

def read_fasta_ids(p):
 with Path(p).open() as f:return [s[1:].strip().split()[0] for s in f if s.startswith('>')]

def fixtures():
 check_lock();root=ROOT/'synthetic_compat';root.mkdir(exist_ok=False);rng=np.random.default_rng(92173)
 u=''.join(rng.choice(list('ACGT'),51))
 s=''.join(rng.choice(list('ACGT'),60));t=''.join(rng.choice(list('ACGT'),50));m=list(s)
 for j in [4,14,24,34,44,54]:m[j]=next(b for b in 'ACGT' if b!=m[j])
 q={'exact':s,'reverse_complement':revcomp(s),'six_substitutions':''.join(m),
 'shifted_10':s[10:]+''.join(rng.choice(list('ACGT'),10)),
 'insert_delete':s[:20]+'A'+s[20:45]+s[46:],
 'query_longer_than_reference':'GGACT'+t+'TCCGA','query_shorter_than_reference':s[5:55],
 'unrelated':''.join(rng.choice(list('ACGT'),60)),
 'too_little_overlap':s[25:]+''.join(rng.choice(list('ACGT'),25)),'low_complexity_negative':'A'*60,
 'coverage_41_of_51':u[10:]+'CGACTGCATA', 'coverage_40_of_51':u[11:]+'CGACTGCATAG', 'mixed_reference_47_of_60':s[13:]+'AGTCGTAGTCGAT', 'mixed_reference_48_of_60':s[12:]+'AGTCGTAGTCGA'}
 expected={'unrelated','too_little_overlap','low_complexity_negative','coverage_40_of_51','mixed_reference_47_of_60'}
 fasta(root/'reference.fa',[('ref60',s),('ref50',t),('ref51',u)]);fasta(root/'query.fa',q.items())
 r=run_tool(root/'reference.fa',root/'query.fa',root/'retained.fa',root/'run.log');actual=set(read_fasta_ids(root/'retained.fa'))
 v=dict(passed=actual==expected,tests=len(q),expected_retained=sorted(expected),actual_retained=sorted(actual),tool=r)
 write_json(root/'receipt.json',v);print(json.dumps(v),flush=True);assert v['passed'],v

def search():
 lock=check_lock();prep=json.loads((BASE/'prepared.json').read_text())
 for n,h in prep['files'].items():assert sha(BASE/n)==h,n
 assert not (ROOT/'retained.fa').exists()
 tool=run_tool(BASE/'reference.fa',BASE/'query.fa',ROOT/'retained.fa',ROOT/'search.log')
 retained=read_fasta_ids(ROOT/'retained.fa');assert len(set(retained))==len(retained);keep=np.zeros(53735,bool)
 for x in retained:assert re.fullmatch(r'Q\d+',x);keep[int(x[1:])]=True
 assert not np.any(keep&np.load(BASE/'exact_overlap.npy'));rows=[];ref=None
 with (ROOT/'retained.fa.clstr').open() as f:
  for line in f:
   if line.startswith('>Cluster'):ref=None;continue
   if '... *' in line:
    m=re.search(r'>(R\d+)\.\.\.',line);assert m,line;ref=m.group(1)
   elif '>Q' in line:
    m=re.search(r'>(Q\d+)\.\.\.',line);assert m and ref,line
    rows.append(dict(query_index=int(m.group(1)[1:]),source_row_index=int(ref[1:]),cdhit_evidence=line.strip()))
 ids=[r['query_index'] for r in rows];assert len(set(ids))==len(ids)==int((~keep).sum());assert set(ids)==set(np.flatnonzero(~keep).tolist())
 with gzip.open(ROOT/'excluded_matches.csv.gz','wt',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['query_index','source_row_index','cdhit_evidence']);w.writeheader();w.writerows(rows)
 with np.load(PARENT/'evaluation/exposure.npz') as z:genes=z['gene_seen']
 np.savez_compressed(ROOT/'retained_masks.npz',sequence_purged=keep,sequence_purged_gene_unseen=keep&~genes)
 with np.load(PARENT/'dataset_v1/dataset.npz') as z:
  lens=z['lengths'];counts={name:{str(n):int((mask&(lens==n)).sum()) for n in range(50,61)} for name,mask in [('sequence_purged',keep),('sequence_purged_gene_unseen',keep&~genes)]}
 assert all(n>0 for p in counts.values() for n in p.values())
 v=dict(passed=True,query_rows=53735,excluded=int((~keep).sum()),retained=int(keep.sum()),gene_unseen_retained=int((keep&~genes).sum()),per_length_counts=counts,
 exact_overlaps_removed=466,tool=tool,labels_or_predictions_read=False,execution_lock_sha256=lock,prepared_sha256=sha(BASE/'prepared.json'),
 files={n:sha(ROOT/n) for n in ['retained_masks.npz','excluded_matches.csv.gz','retained.fa','retained.fa.clstr','search.log']})
 write_json(ROOT/'mask_frozen.json',v);print(json.dumps(v),flush=True)

if __name__=='__main__':
 ROOT.mkdir(exist_ok=True)
 a=argparse.ArgumentParser();a.add_argument('command',choices=['test','search']);x=a.parse_args()
 {'test':fixtures,'search':search}[x.command]()
