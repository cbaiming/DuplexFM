"""No model selection: paired source-seed deltas for every ablation/test."""
import numpy as np
from .protocol import CONDITIONS, SOURCE_SEEDS, TRAIN_VARIANTS, OFFICIAL_TESTS


def summarize(records,full):
    observed={(r['variant'],r['test']):r for r in records}
    reference={(r['variant'],r['test']):r for r in full['records']}
    if len(observed)!=len(records) or set(observed)!={(v,t) for v in TRAIN_VARIANTS for t in OFFICIAL_TESTS}:
        raise ValueError('Ablation metrics must cover every endpoint/test exactly once')
    if set(reference)!={(f'seed{s}',t) for s in SOURCE_SEEDS for t in OFFICIAL_TESTS}:
        raise ValueError('Full reference test coverage differs')
    result={'conditions':{},'prediction_ensemble':False,'source_keep_overlap':True,'records':records,
            'positive_delta_definition':'ablation AP minus paired Full AP; positive means removal improved AP',
            'source_seed_count':3,'single_seed_selection':False}
    for condition in CONDITIONS:
        per_seed={};deltas={};tests={}
        for seed in SOURCE_SEEDS:
            values=[];diff=[]
            for test in OFFICIAL_TESTS:
                row=observed[(f'{condition}_seed{seed}',test)];base=reference[(f'seed{seed}',test)]
                if row['rows']!=base['rows']:raise ValueError('Full/ablation test row counts differ')
                ap=float(row['metrics']['aps']);bp=float(base['metrics']['aps'])
                if not np.isfinite([ap,bp]).all():raise ValueError('Nonfinite AP')
                values.append(ap);diff.append(ap-bp)
                tests.setdefault(test,{})[str(seed)]={'AP':ap,'Full_AP':bp,'delta_AP':ap-bp}
            per_seed[str(seed)]=float(np.mean(values));deltas[str(seed)]=float(np.mean(diff))
        result['conditions'][condition]={'per_seed_macro_AP':per_seed,'per_seed_delta_macro_AP':deltas,
            'mean_macro_AP':float(np.mean(list(per_seed.values()))),
            'sample_SD_macro_AP':float(np.std(list(per_seed.values()),ddof=1)),
            'mean_delta_macro_AP':float(np.mean(list(deltas.values()))),
            'sample_SD_delta_macro_AP':float(np.std(list(deltas.values()),ddof=1)),
            'tests':tests}
    return result
