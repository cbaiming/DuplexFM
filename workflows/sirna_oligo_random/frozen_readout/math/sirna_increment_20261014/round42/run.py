"""Read-only coefficient qualification; no fit, prediction, or label parsing."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import argparse
import csv
import hashlib
import json
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_info
sys.path.insert(0, str(Path(__file__).resolve().parent))
from coherence import energy, aggregate, qualify

R = Path(__file__).resolve().parent
B = R.parent
N = B / "round25"
SETTINGS = ("Huesken", "Mixset", "Takayuki")
SOURCES = ("core_real", "full_real", "initial_core")
FILES = ("run.py", "coherence.py", "verify.py", "test_coherence.py", "PROTOCOL.md", "CODE_AUDIT.md", "pytest.xml",
         "PRELOCK_TEST_REPAIR.md", "prelock_v1/coherence.py", "prelock_v1/test_coherence.py", "prelock_v1/pytest.xml",
         "prelock_v2/run.py", "prelock_v2/registration_failure.json")


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def write(p, data):
    assert not p.exists(), p
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    tmp.rename(p)


def register():
    assert not (R / "lock.json").exists()
    suites = list(ET.parse(R / "pytest.xml").getroot().iter("testsuite"))
    assert sum(int(s.get("tests", 0)) for s in suites) == 8
    assert not any(int(s.get(k, 0)) for s in suites for k in ("errors", "failures", "skipped"))
    prior = read(N / "lock.json")
    assert read(N / "verification.json")["passed"]
    old_manifest = read(N / "diagnostic.json")["pilot_files_sha256"]
    deps = dict(prior["dependencies"])
    for name, digest in prior["files"].items():
        assert sha(N / name) == digest, name
        deps[str((N / name).resolve())] = digest
    for name, digest in old_manifest.items():
        assert sha(N / "pilot" / name) == digest, name
        deps[str((N / "pilot" / name).resolve())] = digest
    closed = read(B / "round41/closure_receipt.json")
    assert closed["status"] == "closed" and closed["readback_passed"]
    for name, digest in closed["files_sha256"].items():
        assert sha(B / name) == digest, name
    for p in [N / "lock.json", N / "verification.json", N / "diagnostic.json",
              B / "round4/plan.json", B / "round41/closure_receipt.json"]:
        deps[str(p.resolve())] = sha(p)
    for name, digest in deps.items():
        assert sha(Path(name)) == digest, name
    write(R / "lock.json", dict(files={n: sha(R / n) for n in FILES}, dependencies=deps,
        CPU_only=True, CPU_limit_seconds=60, numpy_version=np.__version__,
        settings=SETTINGS, sources=SOURCES, source_seed=42, primary="core_real",
        read_old_coefficient_heads=45, summaries=9, mean_ratio_max=.9,
        required_heads_below_one=4, required_primary_pools=2,
        edges=[[0, 2], [2, 4], [1, 3], [3, 5]],
        new_heads=0, new_OOF_rows=0, new_source_forward=0, new_PCA=0,
        numeric_labels_read=False, old_coefficients_depend_on_training_labels=True))
    print("REGISTERED", sha(R / "lock.json"), flush=True)


def validate():
    lock = read(R / "lock.json")
    for n, digest in lock["files"].items():
        assert sha(R / n) == digest, n
    for n, digest in lock["dependencies"].items():
        assert sha(Path(n)) == digest, n
    assert lock["numpy_version"] == np.__version__
    return lock


def run(start):
    validate()
    assert not (R / "diagnostic.json").exists() and not (R / "diagnostic_rows.csv").exists()
    threads = threadpool_info()
    assert all(t["num_threads"] == 1 for t in threads)
    plan = read(B / "round4/plan.json")
    records, detail = [], []
    for setting in SETTINGS:
        item = next(x for x in plan["folds"] if x["setting"] == setting and x["fold"] == 0)
        assert set(item["fit_ids"]).isdisjoint(item["test_ids"])
        for source in SOURCES:
            blocks = []
            for j, split in enumerate(item["inner"]):
                if time.monotonic() - start > 60:
                    raise TimeoutError("Coefficient diagnostic CPU cap")
                root = N / "pilot" / setting / source / f"inner{j}"
                state, receipt = read(root / "state.json"), read(root / "complete.json")
                assert state["train_indices"] == split["train"] and state["valid_indices"] == split["valid"]
                assert state["setting"] == setting and state["source"] == source and state["inner"] == j
                assert state["numerically_qualified"] and state["certificate"]["certified"]
                assert receipt["lock_sha256"] == sha(N / "lock.json")
                for n, digest in receipt["files_sha256"].items():
                    assert sha(root / n) == digest
                with np.load(root / "model.npz", allow_pickle=False) as a:
                    beta = a["beta"]
                assert beta.shape == (216,) and np.isfinite(beta).all()
                result = energy(beta[24:].reshape(6, 32))
                blocks.append(dict(inner=j, **result, model_sha256=sha(root / "model.npz"),
                                   state_sha256=sha(root / "state.json")))
                detail.append(dict(setting=setting, source=source, inner=j,
                    active=result["active"], smoothness_ratio=result["smoothness_ratio"],
                    adjacent_energy=result["adjacent_energy"], all_pair_energy=result["all_pair_energy"],
                    permutation_mean_energy=result["permutation_mean_energy"]))
            rec = dict(setting=setting, source=source, **aggregate(blocks), inner_blocks=blocks)
            records.append(rec)
            print(setting, source, rec["mean_smoothness_ratio"], rec["qualified"], flush=True)
    assert len(detail) == 45 and len(records) == 9
    tmp = R / "diagnostic_rows.csv.tmp"
    with tmp.open("x", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0]))
        w.writeheader(); w.writerows(detail)
    tmp.rename(R / "diagnostic_rows.csv")
    gates = qualify(records)
    write(R / "diagnostic.json", dict(**gates, records=records, read_old_heads=45,
        new_efficiency_heads=0, new_OOF_rows=0, new_source_forward=0, new_PCA=0, GPU_hours=0,
        numeric_labels_read=False, old_coefficients_depend_on_training_labels=True,
        outer_test_metrics_used=False, threadpool=threads,
        lock_sha256=sha(R / "lock.json"), diagnostic_rows_sha256=sha(R / "diagnostic_rows.csv")))
    validate()
    print("QUALIFIED", gates["qualified"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=("register", "run"))
    if ap.parse_args().mode == "register":
        register()
    else:
        mutex = R / "execution.mutex"
        with mutex.open("x") as f:
            f.write(str(os.getpid()))
        start = time.monotonic(); rc = 1; error = None
        try:
            run(start); rc = 0
        except Exception:
            error = traceback.format_exc(); raise
        finally:
            write(R / f"execution_{time.time_ns()}.json", dict(returncode=rc, error=error,
                CPU_wall_seconds=time.monotonic() - start, GPU_hours=0, new_efficiency_heads=0))
            mutex.unlink()
