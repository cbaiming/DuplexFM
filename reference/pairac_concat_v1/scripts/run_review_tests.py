"""Run independent CPU-only synthetic suites; no weights/data/network required."""
import os
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', nargs='+', help='Run named suites only; default is all suites')
    args = parser.parse_args()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
    plans = [
        ('full', ROOT/'source_full/code', ['tests'], ''),
        ('ablation42', ROOT/'ablations/seed42/code', ['tests'], ''),
        ('ablation43_44', ROOT/'ablations/seeds43_44/code', ['tests'], ''),
        ('transfer_full', ROOT/'sirna_transfer/full', ['test_readout.py', 'test_export.py'], 'not fixed_counts_and_scope'),
        ('transfer_ablation', ROOT/'sirna_transfer/ablations', ['test_readout.py'], 'not fixed_counts_and_scope'),
        ('backup_fix', ROOT/'execution_reference', ['test_backup_compat.py', 'test_backup_compat_v2.py'], ''),
    ]
    out = ROOT/'review_runs'
    out.mkdir(exist_ok=True)
    if args.only and set(args.only) - {p[0] for p in plans}:
        parser.error('Unknown suite name')
    for name, cwd, targets, selection in plans:
        if args.only and name not in args.only:
            continue
        local = env.copy()
        local['PYTHONPATH'] = os.pathsep.join([str(cwd), str(ROOT/'ablations/seeds43_44/code')]) if name == 'backup_fix' else str(cwd)
        # Separate processes prevent same-name modules from different snapshots mixing.
        with tempfile.TemporaryDirectory(prefix='pairac_') as temp:
            args = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *targets,
                    '--basetemp', str(Path(temp)/'pytest'), '--junitxml', str(out/(name+'.xml'))]
            if selection: args += ['-k', selection]
            print('\nRunning '+name, flush=True)
            result = subprocess.run(args, cwd=cwd, env=local, timeout=600)
            if result.returncode:
                return result.returncode
    print('All requested CPU suites passed; real-split tests explicitly deselected.', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
