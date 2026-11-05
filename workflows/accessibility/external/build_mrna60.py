"""Build complete-observation 60-nt mRNA windows, with no model execution.

Coordinates are on the released 5'-to-3' transcript sequence, never genomic
coordinates. Duplicate sequences retain distinct measurement observations.
"""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np

ROOT=Path(__file__).resolve().parent/'gse74353'
DATA=ROOT/'GSE74353_HS_293T_icSHAPE_InVivo_BaseReactivities.txt.gz'
REF=ROOT/'reference'/'Homo_sapiens.GRCh37.74.cdna.all.fa.gz'
OUT=ROOT/'mrna60_complete_v1'
RAW_SHA='4fd439daad58556b5cc5668fedea8d8c0f18e15722927eb2724400c0dfbc47c9'

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def fasta_records(path):
    with gzip.open(path,'rt',encoding='ascii') as f:
        header=None; chunks=[]
        for line in f:
            line=line.strip()
            if not line: continue
            if line.startswith('>'):
                if header is not None: yield header,''.join(chunks).upper().replace('T','U')
                header=line[1:];chunks=[]
            else:
                if header is None: raise ValueError('Sequence before FASTA header')
                chunks.append(line)
        if header is not None: yield header,''.join(chunks).upper().replace('T','U')

def reference_subset(path, needed):
    records={};seen=set();counts=Counter()
    for header,sequence in fasta_records(path):
        fields=header.split();tx=fields[0]
        if tx in seen: raise ValueError('Ambiguous reference transcript ID: '+tx)
        seen.add(tx);counts['reference_transcripts']+=1
        if tx in needed:
            meta=dict(x.split(':',1) for x in fields[1:] if ':' in x)
            for key in ['gene','gene_biotype','transcript_biotype']:
                if key not in meta: raise ValueError('Missing reference annotation: '+key)
            records[tx]=dict(sequence=sequence,header=header,**meta)
    counts['matched_input_transcript_ids']=len(records)
    return records,dict(counts)

def complete_windows(sequence,tokens):
    if len(sequence)!=len(tokens): raise ValueError('Sequence / profile length mismatch')
    counts=Counter();windows=[]
    counts['tail_bases_dropped']=len(sequence)%60
    for start in range(0,len(sequence)-59,60):
        counts['full_length_grid_windows']+=1
        seq=sequence[start:start+60]
        if set(seq)-set('ACGU'):
            counts['windows_excluded_non_acgu']+=1;continue
        parts=tokens[start:start+60]
        if any(x=='NULL' for x in parts):
            counts['windows_excluded_any_missing']+=1;continue
        vals=tuple(float(x) for x in parts)
        if not all(math.isfinite(x) and 0<=x<=1 for x in vals):
            raise ValueError('Invalid measured reactivity')
        windows.append((start,seq,vals));counts['accepted_windows']+=1
    return windows,dict(counts)

def add_groups(rows):
    """All same-gene rows join; identical sequences also join across genes."""
    parent={r['gene_id']:r['gene_id'] for r in rows}
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:
            small,large=sorted([a,b]);parent[large]=small
    seq_gene={}
    for r in rows:
        if r['sequence_sha256'] in seq_gene: union(r['gene_id'],seq_gene[r['sequence_sha256']])
        else: seq_gene[r['sequence_sha256']]=r['gene_id']
    members=defaultdict(list)
    for gene in sorted(parent): members[find(gene)].append(gene)
    group={root:'group_'+hashlib.sha256('\n'.join(genes).encode()).hexdigest()[:20]
           for root,genes in members.items()}
    for r in rows: r['evaluation_group']=group[find(r['gene_id'])]
    return {group[root]:genes for root,genes in members.items()}

def read_ids(path):
    ids=set()
    with gzip.open(path,'rt') as f:
        for line in f:
            if not line.strip(): continue
            tx=line.split(maxsplit=1)[0]
            if tx in ids: raise ValueError('Duplicate source transcript: '+tx)
            ids.add(tx)
    return ids

def collect(data,reference):
    rows=[];audit=[];counts=Counter();biotypes=Counter()
    with gzip.open(data,'rt') as f:
        for line_number,line in enumerate(f,1):
            if not line.strip(): continue
            p=line.split();tx=p[0];length=int(p[1]);tokens=p[3:]
            if len(tokens)!=length: raise ValueError('Invalid source column count')
            counts['source_transcripts']+=1
            entry=dict(transcript_id=tx,source_line=line_number,declared_length=length)
            if tx not in reference:
                reason='excluded_no_exact_reference_id'
            else:
                ref=reference[tx];entry.update(gene_id=ref['gene'],reference_length=len(ref['sequence']),
                    gene_biotype=ref['gene_biotype'],transcript_biotype=ref['transcript_biotype'])
                biotypes[ref['gene_biotype']+'/'+ref['transcript_biotype']]+=1
                if ref['gene_biotype']!='protein_coding' or ref['transcript_biotype']!='protein_coding':
                    reason='excluded_non_protein_coding_transcript'
                elif len(ref['sequence'])!=length:
                    reason='excluded_reference_length_mismatch'
                else:
                    reason='eligible_protein_coding_transcript'
                    windows,wcounts=complete_windows(ref['sequence'],tokens)
                    entry.update(wcounts);counts.update(wcounts)
                    for start,seq,vals in windows:
                        digest=hashlib.sha256(seq.encode('ascii')).hexdigest()
                        rows.append(dict(window_id=f'{tx}:{start+1}-{start+60}',
                            transcript_id=tx,gene_id=ref['gene'],source_line=line_number,
                            start0=start,end0_exclusive=start+60,start1=start+1,end1_inclusive=start+60,
                            sequence=seq,sequence_sha256=digest,sequence_id='seq_'+digest,
                            reactivity=vals))
                    if windows: counts['transcripts_with_retained_windows']+=1
            entry['status']=reason;counts[reason]+=1;audit.append(entry)
    rows.sort(key=lambda r:(r['transcript_id'],r['start0']))
    groups=add_groups(rows)
    return rows,audit,dict(counts),dict(biotypes),groups

def write_dataset(out,rows,audit,groups):
    out.mkdir(exist_ok=False)
    fields=['window_id','sequence_id','transcript_id','gene_id','evaluation_group','source_line',
            'start0','end0_exclusive','start1','end1_inclusive','sequence','sequence_sha256']
    with gzip.open(out/'windows.csv.gz','wt',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields+[f'reactivity_{i:02d}' for i in range(1,61)])
        w.writeheader()
        for r in rows:
            values={key:r[key] for key in fields}
            values.update({f'reactivity_{i:02d}':repr(v) for i,v in enumerate(r['reactivity'],1)})
            w.writerow(values)
    arrays={key:np.asarray([r[key] for r in rows]) for key in fields}
    arrays['reactivity']=np.asarray([r['reactivity'] for r in rows],dtype=np.float64).reshape(-1,60)
    arrays['observed_mask']=np.ones((len(rows),60),dtype=bool)
    arrays['lengths']=np.full(len(rows),60,dtype=np.int16)
    np.savez_compressed(out/'dataset.npz',**arrays)
    unique={r['sequence_id']:r['sequence'] for r in rows}
    with (out/'unique_sequences.fasta').open('x',encoding='ascii',newline='\n') as f:
        for sid,seq in sorted(unique.items()): f.write(f'>{sid}\n{seq}\n')
    with gzip.open(out/'transcript_audit.jsonl.gz','wt',encoding='utf-8') as f:
        for row in audit: f.write(json.dumps(row,sort_keys=True)+'\n')
    (out/'evaluation_groups.json').write_text(json.dumps(groups,indent=2,sort_keys=True),encoding='utf-8')
    return arrays

def main():
    if OUT.exists(): raise FileExistsError('Dataset exists; never overwrite it')
    protocol=ROOT/'MRNA60_BUILD_PROTOCOL.json'
    p=json.loads(protocol.read_text(encoding='utf-8'))
    assert p['window_length']==60 and p['minimum_observed_positions']==60 and p['stride']==60
    assert sha(DATA)==RAW_SHA==p['inputs']['measurement_sha256']
    assert sha(REF)==p['inputs']['reference_sha256']
    assert sha(Path(__file__))==p['build_code_sha256']
    start=time.monotonic()
    needed=read_ids(DATA)
    reference,refcounts=reference_subset(REF,needed)
    rows,audit,counts,biotypes,groups=collect(DATA,reference)
    if not rows: raise ValueError('No eligible fully measured windows; do not relax criteria')
    arrays=write_dataset(OUT,rows,audit,groups)
    unique=len({r['sequence_id'] for r in rows})
    metrics=dict(counts,**refcounts,windows=len(rows),unique_sequences=unique,
                 repeated_sequence_observations=len(rows)-unique,
                 genes=len({r['gene_id'] for r in rows}),evaluation_groups=len(groups),
                 observed_positions=len(rows)*60,missing_positions=0,
                 real_zero_measurements=int(np.count_nonzero(arrays['reactivity']==0)))
    manifest=dict(schema='gse74353_mrna60_complete_v1',status='built_pending_independent_audit',
        protocol_sha256=sha(protocol),inputs=p['inputs'],rules=p,counts=metrics,
        source_biotype_counts=biotypes,build_wall_seconds=time.monotonic()-start,
        external_independence_verified=False,original_training_membership_known=False,
        new_model_execution=False,
        files={f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(OUT.iterdir())})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
