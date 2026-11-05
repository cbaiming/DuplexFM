import os, platform, subprocess, time, threading, importlib.metadata
from pathlib import Path
from .common import read, sha, verify_files, write
from .protocol import contract

def code_hashes(code):
    return {p.relative_to(code).as_posix(): sha(p) for p in sorted(Path(code).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".json")}

def environment_identity():
    return {"python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in
            ("torch", "numpy", "scipy", "scikit-learn", "transformers", "multimolecule", "safetensors", "tokenizers")}}

def gpu_inventory():
    def query(args):
        p = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                           text=True, capture_output=True, timeout=20, check=True)
        return [tuple(v.strip() for v in line.split(",")) for line in p.stdout.splitlines() if line.strip()]
    return query(["--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu"]), query(["--query-compute-apps=gpu_uuid,pid,used_memory"])

def choose_gpu(gpus, processes, requested, required_model):
    candidates = [g for g in gpus if requested in (g[0], g[1])]
    if len(candidates) != 1:
        raise ValueError("Requested GPU identity must resolve exactly once")
    gpu = candidates[0]
    if required_model not in gpu[2]:
        raise ValueError(f"This entry requires {required_model}, got {gpu[2]}")
    if any(p[0] == gpu[1] for p in processes) or int(gpu[3]) > 64 or int(gpu[5]) != 0:
        raise RuntimeError("Requested GPU is not idle; no eviction or automatic card switch")
    return {"index": int(gpu[0]), "uuid": gpu[1], "name": gpu[2], "memory_mib": int(gpu[4])}

def dispatch_gpu(requested, previous_exit=None, inventory=gpu_inventory, wait=None):
    """At most one bounded driver cleanup after our own successful child exit.

    Never waits for an external process, high memory use, or an old/failed child.
    A second full strict guard remains mandatory. No polling or GPU work.
    """
    gpus, processes = inventory()
    try:
        return choose_gpu(gpus, processes, requested, "H100")
    except RuntimeError:
        candidates = [g for g in gpus if requested in (g[0], g[1])]
        if len(candidates) != 1:
            raise
        g = candidates[0]
        if (previous_exit is None or not 0 <= time.monotonic() - previous_exit <= 15
                or any(p[0] == g[1] for p in processes) or int(g[3]) > 64):
            raise
        (wait or threading.Event().wait)(3)
        return choose_gpu(*inventory(), requested, "H100")

def thread_env(cpu_threads=4):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(cpu_threads), MKL_NUM_THREADS=str(cpu_threads),
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    return env

def registration(root,assets):
    from duplexfm_joint.backbones import _source_manifest
    from duplexfm_joint.data import PairStore
    root,assets=Path(root).resolve(),Path(assets).resolve()
    from .reference_initial import reference
    reference(root)
    prepared=read(root/'preparation/complete.json')
    verify_files(root/'preparation',prepared['files_sha256'])
    with PairStore(assets/'mirbench') as store:integrity=store.verify_integrity()
    if integrity!=read(root/'preparation/inputs.json'):raise ValueError('Prepared input changed')
    reference=read(root/'code/foundation_reference.json');bases={}
    for name,kind in [('mrnabert','target'),('rnafm','guide')]:
        bases[name]=_source_manifest(assets/name)['sha256']
        if any(bases[name].get(f)!=v for f,v in reference[kind]['sha256'].items()):raise ValueError('Foundation differs')
    value={'contract':contract(),'code_files':code_hashes(root/'code'),'public_assets':bases,
           'preparation_sha256':sha(root/'preparation/complete.json'),'source_manifest_sha256':integrity['manifest_sha256'],
           'environment':environment_identity(),'release_manifest_sha256':sha(root/'release_manifest.json')}
    write(root/'protocol.json',contract(),immutable=True)
    write(root/'registration.json',value,immutable=True)
    write(root/'ledger.json',{'active':None,'limit_seconds':172800,'phases':[]},immutable=True)
    return value

def verify_registration(root,assets):
    root,assets=Path(root),Path(assets)
    from .reference_initial import reference
    reference(root)
    value=read(root/'registration.json')
    if value['environment']!=environment_identity():raise ValueError('Environment changed')
    if value['contract']!=contract() or value['code_files']!=code_hashes(root/'code'):raise ValueError('Code or protocol changed')
    if sha(root/'release_manifest.json')!=value['release_manifest_sha256']:raise ValueError('Release changed')
    verify_files(root,read(root/'release_manifest.json')['files_sha256'])
    if sha(root/'preparation/complete.json')!=value['preparation_sha256']:raise ValueError('Preparation changed')
    verify_files(root/'preparation',read(root/'preparation/complete.json')['files_sha256'])
    if sha(assets/'mirbench/manifest.json')!=value['source_manifest_sha256']:raise ValueError('Manifest changed')
    for name,files in value['public_assets'].items():verify_files(assets/name,files)
    return value
