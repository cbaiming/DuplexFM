"""Check the current miRBench and Oligo-random tables without model fitting."""
import argparse, csv, hashlib, json, math, statistics, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from reproduce_oligo_results import run as check_oligo

def table(p):
    with p.open(encoding='utf-8-sig', newline='') as f:return list(csv.DictReader(f))

def equal(a,b,name):
    if not math.isfinite(a) or not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError('Inconsistent table: '+name)

def mirna(root):
    base=root/'results/miRNA';rows=table(base/'mirna_per_test.csv')
    conditions={'Full'};seeds={'42','43','44'}
    tests={'AGO2_CLASH_Hejret2023_test','AGO2_eCLIP_Klimentova2022_test',
           'AGO2_eCLIP_Manakov2022_test','AGO2_eCLIP_Manakov2022_leftout'}
    keys=[(r['condition'],r['seed'],r['test']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!={(c,s,t)for c in conditions for s in seeds for t in tests}:
        raise ValueError('miRNA condition/seed/test inventory differs')
    groups=defaultdict(list)
    for r in rows:
        value=float(r['AP'])
        if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid AP')
        groups[r['condition'],r['seed']].append(value)
    macros={k:statistics.mean(v) for k,v in groups.items()}
    per_seed=table(base/'mirna_per_seed_macro.csv')
    if len(per_seed)!=3 or {(r['condition'],r['seed'])for r in per_seed}!=set(macros):
        raise ValueError('miRNA per-seed inventory differs')
    for r in per_seed:equal(macros[r['condition'],r['seed']],float(r['macro_AP']),'miRNA per-seed AP')
    summary=table(base/'mirna_mean.csv')
    if len(summary)!=1 or {r['condition']for r in summary}!=conditions:raise ValueError('miRNA summary inventory differs')
    for r in summary:
        values=[macros[r['condition'],s] for s in seeds]
        equal(statistics.mean(values),float(r['mean_macro_AP']),'miRNA mean')
        equal(statistics.stdev(values),float(r['sample_SD']),'miRNA sample SD')
    return dict(passed=True,test_metrics=len(rows),source_seeds=[42,43,44],
                scope='Saved AP reaggregation; recompute_mirna_ap.py checks available row-level predictions')

def run(root):
    provenance=json.loads((root/'results/provenance.json').read_text('utf-8'))
    for r in provenance['tables']:
        p=(root/'results'/r['path']).resolve()
        if not p.is_relative_to((root/'results').resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:
            raise ValueError('Result table SHA/path differs: '+r['path'])
    return dict(passed=True,miRNA=mirna(root),siRNA=check_oligo(root),
                current_siRNA_protocol=provenance['primary_siRNA_protocol'],historical_results_excluded=True,new_model_fits=0)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();print(json.dumps(run(args.root.resolve()),indent=2,ensure_ascii=False))
