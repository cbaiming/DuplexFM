"""CPU preparation/recovery; formal run is two independent H100 workers."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['register', 'run', 'worker', 'recover', 'status'])
    p.add_argument('--root', required=True)
    p.add_argument('--assets')
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--lease')
    p.add_argument('--previous-stopped-at')
    a = p.parse_args()
    if a.action != 'worker':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    from duplexfm_mechanism import parallel
    from duplexfm_mechanism.common import read
    root = Path(a.root).resolve()
    if a.action == 'status':
        result = {n:read(root/n) for n in ['parallel_status.json','parallel_ledger.json','source_training_complete.json'] if (root/n).exists()}
    elif not a.assets:
        p.error('--assets is required')
    elif a.action == 'register':
        result = parallel.register(root, a.assets)
    elif a.action == 'recover':
        if not a.previous_stopped_at:
            p.error('Abrupt-stop accounting needs --previous-stopped-at from the platform event')
        result = parallel.recover(root, a.assets, a.previous_stopped_at)
    elif a.action == 'run':
        if a.previous_stopped_at:
            parallel.recover(root, a.assets, a.previous_stopped_at)
        result = parallel.controller(root, a.assets, a.gpus)
    else:
        if not a.lease:
            p.error('--lease is required')
        result = parallel.worker(root, a.assets, a.lease)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result.get('status') in ('parallel_registered','source_training_complete','complete','already_closed','recovered') or a.action=='status' else 75


if __name__ == '__main__':
    raise SystemExit(main())
