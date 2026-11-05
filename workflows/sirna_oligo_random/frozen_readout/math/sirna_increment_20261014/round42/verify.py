"""Independent scalar readback of the fixed regional-order diagnosis."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import csv
import hashlib
import itertools
import json
import time
import traceback
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent
B = R.parent


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1024 * 1024), b""):
            h.update(c)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def write(p, d):
    assert not p.exists(), p
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("x", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
    tmp.rename(p)


def close(a, b):
    assert np.allclose(a, b, rtol=1e-11, atol=1e-13), (a, b)


def verify(start):
    lock = read(R / "lock.json")
    for name, digest in lock["files"].items():
        assert sha(R / name) == digest
    for name, digest in lock["dependencies"].items():
        assert sha(Path(name)) == digest
    assert not (R / "execution.mutex").exists()
    runs = list(R.glob("execution_*.json")); assert len(runs) == 1
    assert read(runs[0])["returncode"] == 0
    d = read(R / "diagnostic.json")
    assert d["lock_sha256"] == sha(R / "lock.json")
    assert d["diagnostic_rows_sha256"] == sha(R / "diagnostic_rows.csv")
    with (R / "diagnostic_rows.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 45 and len(d["records"]) == 9
    expected = {(s, c, j) for s in lock["settings"] for c in lock["sources"] for j in range(5)}
    assert {(x["setting"], x["source"], int(x["inner"])) for x in rows} == expected
    plan = read(B / "round4/plan.json")
    qualified = []; count = 0; max_error = 0.
    for record in d["records"]:
        setting, source = record["setting"], record["source"]
        item = next(x for x in plan["folds"] if x["setting"] == setting and x["fold"] == 0)
        assert set(item["fit_ids"]).isdisjoint(item["test_ids"])
        ratios = []
        assert [x["inner"] for x in record["inner_blocks"]] == list(range(5))
        for block in record["inner_blocks"]:
            if time.monotonic() - start > 60:
                raise TimeoutError("Independent audit CPU cap")
            j = block["inner"]; split = item["inner"][j]
            root = B / "round25/pilot" / setting / source / f"inner{j}"
            assert block["model_sha256"] == sha(root / "model.npz")
            assert block["state_sha256"] == sha(root / "state.json")
            st = read(root / "state.json")
            assert st["train_indices"] == split["train"] and st["valid_indices"] == split["valid"]
            assert st["numerically_qualified"]
            receipt = read(root / "complete.json")
            assert receipt["lock_sha256"] == sha(B / "round25/lock.json")
            for name, digest in receipt["files_sha256"].items():
                assert sha(root / name) == digest
            with np.load(root / "model.npz", allow_pickle=False) as a:
                beta = a["beta"]
            assert beta.shape == (216,) and np.isfinite(beta).all()
            c = beta[24:].reshape(3, 2, 32)
            pair = []
            # Order of saved pairs: (mean01,mean02,mean12,max01,max02,max12).
            for t in range(2):
                for u in range(3):
                    for v in range(u + 1, 3):
                        pair.append(sum(float(c[u, t, k] - c[v, t, k]) ** 2 for k in range(32)))
            total = sum(pair)
            orders = []
            for pi in itertools.permutations(range(3)):
                orders.append(sum(float(c[pi[u], t, k] - c[pi[u + 1], t, k]) ** 2
                                  for t in range(2) for u in range(2) for k in range(32)))
            e, null = orders[0], sum(orders) / 6
            close(null, total * 2 / 3)
            close(pair, block["pair_energies"])
            close(orders, block["permuted_path_energies"])
            close(e, block["adjacent_energy"]); close(total, block["all_pair_energy"])
            close(null, block["permutation_mean_energy"])
            active = total > 1e-24
            ratio = e / null if active else None
            assert active == block["active"]
            if active:
                close(ratio, block["smoothness_ratio"])
                max_error = max(max_error, abs(ratio - block["smoothness_ratio"]))
            else:
                assert block["smoothness_ratio"] is None
            row = next(x for x in rows if x["setting"] == setting and x["source"] == source and int(x["inner"]) == j)
            assert row["active"] == str(active)
            close(float(row["adjacent_energy"]), e); close(float(row["all_pair_energy"]), total)
            close(float(row["permutation_mean_energy"]), null)
            if active:
                close(float(row["smoothness_ratio"]), ratio)
            else:
                assert row["smoothness_ratio"] == ""
            ratios.append(ratio); count += 1
        active = all(x is not None for x in ratios)
        mean = sum(ratios) / 5 if active else None
        below = sum(x is not None and x < 1 for x in ratios)
        passed = active and mean <= .9 and below >= 4
        assert record["all_active"] == active and record["qualified"] == passed
        assert record["below_permutation_mean_count"] == below
        if active:
            close(record["mean_smoothness_ratio"], mean)
        else:
            assert record["mean_smoothness_ratio"] is None
        if source == "core_real":
            qualified.append(passed)
    assert count == 45 and len(qualified) == 3
    assert d["qualified"] == (sum(qualified) >= 2)
    assert d["qualified_core_pools"] == sum(qualified)
    assert d["new_efficiency_heads"] == d["new_OOF_rows"] == d["new_source_forward"] == d["GPU_hours"] == 0
    assert not d["numeric_labels_read"] and d["old_coefficients_depend_on_training_labels"]
    assert not d["outer_test_metrics_used"]
    for name, digest in lock["files"].items():
        assert sha(R / name) == digest
    for name, digest in lock["dependencies"].items():
        assert sha(Path(name)) == digest
    result = dict(passed=True, qualified=d["qualified"], verified_old_coefficient_heads=count,
        independent_scalar_energy_checks=45, permutation_energies_verified=270,
        max_ratio_difference=max_error, new_efficiency_heads=0, new_OOF_rows=0,
        new_source_forward=0, GPU_hours=0, numeric_labels_read=False,
        old_coefficients_depend_on_training_labels=True, outer_test_metrics_used=False,
        lock_sha256=sha(R / "lock.json"), diagnostic_sha256=sha(R / "diagnostic.json"),
        CPU_wall_seconds=time.monotonic() - start)
    write(R / "verification.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    start = time.monotonic(); rc = 1; error = None
    try:
        verify(start); rc = 0
    except Exception:
        error = traceback.format_exc(); raise
    finally:
        write(R / f"verification_execution_{time.time_ns()}.json", dict(returncode=rc, error=error,
              CPU_wall_seconds=time.monotonic() - start, GPU_hours=0, new_efficiency_heads=0))
