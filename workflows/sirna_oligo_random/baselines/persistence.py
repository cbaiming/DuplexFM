"""One GPU process, one immutable model/seed/setting/outer-fold task.

Training/checkpoint semantics belong to the SHA-pinned gpu_models backend.
This module orchestrates five common inner folds, a fresh final fit, independent
test prediction, audited completion reuse, and independent persistent backups.
It imports no torch; its persistence and orchestration are testable on CPU.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import time
import uuid

import numpy as np


MODELS = ("OligoFormer-corrected", "BERT-siRNA", "RNAiSpline")
SEEDS = (42, 43, 44)
_STOP = False


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def value_sha(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_bytes(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def write(path, value):
    atomic_bytes(path, json_bytes(value))


def immutable_json(path, value):
    path, raw = Path(path), json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Immutable identity mismatch: {path}")
        return
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def confined(root, relative):
    root = Path(root).resolve()
    result = (root / relative).resolve()
    if result != root and root not in result.parents:
        raise ValueError(f"Path escapes intended root: {relative}")
    return result


def descriptor(config_dir, value):
    if not isinstance(value, dict) or set(("path", "sha256")) - set(value):
        raise ValueError("File descriptor needs path and sha256")
    path = Path(value["path"])
    if not path.is_absolute():
        path = Path(config_dir) / path
    path = path.resolve()
    if sha(path) != value["sha256"]:
        raise ValueError(f"Pinned file SHA mismatch: {path}")
    return path


class JobLease:
    """Exclusive local lease; an absent PID on this host permits stale recovery."""
    def __init__(self, root, identity_digest, worker_id):
        self.root = Path(root)
        self.path = self.root / ".worker.lock"
        self.value = dict(pid=os.getpid(), host=socket.gethostname(), worker_id=worker_id,
                          identity_sha256=identity_digest, token=uuid.uuid4().hex,
                          created_unix=time.time())

    def __enter__(self):
        self.root.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                with self.path.open("xb") as f:
                    f.write(json_bytes(self.value))
                    f.flush()
                    os.fsync(f.fileno())
                return self
            except FileExistsError:
                old = read(self.path)
                if old.get("identity_sha256") != self.value["identity_sha256"]:
                    raise RuntimeError("Existing worker lease has a different identity")
                if old.get("host") != self.value["host"]:
                    raise RuntimeError("Cannot prove worker on another host has exited")
                try:
                    os.kill(int(old["pid"]), 0)
                except ProcessLookupError:
                    archive = self.root / "stale_locks" / (old.get("token", "unknown") + ".json")
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    if archive.exists():
                        raise RuntimeError("Stale lease was already archived")
                    os.replace(self.path, archive)
                except (PermissionError, OSError) as error:
                    raise RuntimeError("Cannot establish that previous worker exited") from error
                else:
                    raise RuntimeError("Task is already held by a live worker")
        raise RuntimeError("Could not obtain task lease")

    def __exit__(self, *_):
        if self.path.exists() and read(self.path).get("token") == self.value["token"]:
            self.path.unlink()


def regular_files(root):
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError(f"Symlinks are not accepted in task state: {p}")
        if p.is_file() and not p.name.endswith(".tmp") and p.name not in (".worker.lock", "latest_backup.json"):
            yield p


def verify_initial_reference(fit, initial_root=None):
    path = Path(fit) / "initial_reference.json"
    if not path.exists():
        raise ValueError(f"Missing initialization reference: {path}")
    ref = read(path)
    target = Path(ref["path"]).resolve()
    if initial_root is not None:
        initial_root = Path(initial_root).resolve()
        if initial_root not in target.parents:
            raise ValueError("Initialization reference escapes configured initial object root")
    if sha(target) != ref["sha256"]:
        raise ValueError("Initialization object SHA mismatch")
    return target, ref["sha256"]


def verify_fit(fit, expected_execution_identity, initial_root=None):
    fit = Path(fit)
    receipt = read(fit / "complete.json")
    if receipt.get("status") != "complete" or receipt.get("test_used_for_selection") is not False:
        raise ValueError("Fit is not a complete, test-independent fit")
    ident = read(fit / "identity.json")
    if ident.get("execution_identity") != expected_execution_identity:
        raise ValueError("Fit execution identity mismatch")
    if receipt.get("identity_sha256") != sha(fit / "identity.json"):
        raise ValueError("Fit identity SHA mismatch")
    if not receipt.get("files"):
        raise ValueError("Complete fit has no file digest manifest")
    for name, digest in receipt["files"].items():
        if sha(confined(fit, name)) != digest:
            raise ValueError(f"Completed fit file SHA mismatch: {name}")
    checkpoint = confined(fit, receipt["checkpoint_file"])
    if sha(checkpoint) != receipt["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint SHA mismatch")
    verify_initial_reference(fit, initial_root)
    return receipt


class BackupStore:
    """Content-addressed independent copies, verified after writing and on reuse."""
    def __init__(self, backup_root, job_root, identity_digest, initial_root, interval=3600):
        self.backup_root = Path(backup_root).resolve()
        self.job_root = Path(job_root).resolve()
        self.identity_digest = identity_digest
        self.initial_root = Path(initial_root).resolve()
        self.job_backup = self.backup_root / "jobs" / self.job_root.name
        self.interval = float(interval)
        if self.interval <= 0:
            raise ValueError("Backup interval must be positive")
        self.latest = self.job_backup / "latest.json"
        self.last_time = 0.0
        if self.latest.exists():
            pointer = read(self.latest)
            if pointer["identity_sha256"] != identity_digest:
                raise ValueError("Backup identity mismatch")
            self.last_time = float(pointer["created_unix"])

    def object_copy(self, source):
        source = Path(source).resolve()
        objects = self.backup_root / "objects"
        objects.mkdir(parents=True, exist_ok=True)
        expected_digest = sha(source)
        existing = objects / expected_digest
        if existing.exists():
            if sha(existing) != expected_digest or existing.stat().st_nlink != 1:
                raise ValueError("Existing backup object failed verification")
            st_source, st_target = source.stat(), existing.stat()
            if (st_source.st_dev, st_source.st_ino) == (st_target.st_dev, st_target.st_ino):
                raise ValueError("Backup object shares its source inode")
            return dict(sha256=expected_digest, bytes=st_source.st_size,
                        object=f"objects/{expected_digest}")
        temp = objects / ("copy." + uuid.uuid4().hex + ".tmp")
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as inp, temp.open("xb") as out:
            for block in iter(lambda: inp.read(1 << 20), b""):
                digest.update(block)
                size += len(block)
                out.write(block)
            out.flush()
            os.fsync(out.fileno())
        h = digest.hexdigest()
        if h != expected_digest or sha(temp) != h or temp.stat().st_nlink != 1:
            raise ValueError("Independent backup copy failed read-back verification")
        target = objects / h
        if target.exists():
            if sha(target) != h or target.stat().st_nlink != 1:
                raise ValueError("Existing backup object failed verification")
            temp.unlink()
        else:
            os.replace(temp, target)
        # No hardlink or shared source inode is accepted, including same-filesystem tests.
        st_source, st_target = source.stat(), target.stat()
        if (st_source.st_dev, st_source.st_ino) == (st_target.st_dev, st_target.st_ino):
            raise ValueError("Backup object shares its source inode")
        if sha(target) != h:
            raise ValueError("Backup object failed final read-back")
        return dict(sha256=h, bytes=size, object=f"objects/{h}")

    def snapshot(self, reason, force=False):
        now = time.time()
        if not force and now - self.last_time < self.interval:
            return None
        # Once a fit is complete, its selected weights and receipt replace the
        # optimizer-bearing last as the retained payload. An active fit retains last.
        files = {p.relative_to(self.job_root).as_posix(): self.object_copy(p)
                 for p in regular_files(self.job_root)
                 if not (p.name in ("last.pt", "last.sha256.json") and (p.parent / "complete.json").exists())}
        external = {}
        for ref in sorted(self.job_root.rglob("initial_reference.json")):
            target, digest = verify_initial_reference(ref.parent, self.initial_root)
            value = self.object_copy(target)
            if value["sha256"] != digest:
                raise ValueError("Initialization changed during backup")
            external[str(target)] = value
        manifest = dict(schema=1, job_id=self.job_root.name, identity_sha256=self.identity_digest,
                        local_job_root=str(self.job_root), initial_root=str(self.initial_root),
                        reason=reason, created_unix=now, files=files, external_files=external,
                        payload_policy="completed_fit_selected_weights_metadata_initial_objects_plus_active_fit_resume_state",
                        independent_copy=True, SHA_readback_verified=True)
        name = f"snapshots/{time.time_ns()}_{re.sub('[^A-Za-z0-9_-]', '_', reason)}.json"
        path = self.job_backup / name
        immutable_json(path, manifest)
        pointer = dict(schema=1, identity_sha256=self.identity_digest, created_unix=now,
                       snapshot=name, sha256=sha(path), reason=reason)
        if read(path) != manifest:
            raise ValueError("Backup manifest failed read-back")
        write(self.latest, pointer)
        if read(self.latest) != pointer:
            raise ValueError("Latest backup pointer failed read-back")
        write(self.job_root / "latest_backup.json", pointer)
        self.last_time = now
        return pointer

    def checkpoint_callback(self, _fit, reason):
        return self.snapshot(str(reason), force=reason in ("stopped", "complete") or not self.last_time)

    def restore_latest(self):
        if not self.latest.exists():
            raise ValueError("No persistent backup exists for explicit restore")
        pointer = read(self.latest)
        path = confined(self.job_backup, pointer["snapshot"])
        if sha(path) != pointer["sha256"]:
            raise ValueError("Backup manifest SHA mismatch")
        manifest = read(path)
        if (manifest["identity_sha256"] != self.identity_digest or
                Path(manifest["local_job_root"]).resolve() != self.job_root or
                Path(manifest["initial_root"]).resolve() != self.initial_root):
            raise ValueError("Backup does not belong to this exact local task identity/path")
        targets = [(confined(self.job_root, n), value) for n, value in manifest["files"].items()]
        for name, value in manifest.get("external_files", {}).items():
            target = Path(name).resolve()
            if self.initial_root not in target.parents:
                raise ValueError("External restore path escapes initialization object root")
            targets.append((target, value))
        # Verify everything before the first write. Existing differing local files are preserved.
        for target, value in targets:
            source = confined(self.backup_root, value["object"])
            if sha(source) != value["sha256"] or source.stat().st_nlink != 1:
                raise ValueError("Backup object is missing, changed, or hardlinked")
            if target.exists() and sha(target) != value["sha256"]:
                raise ValueError(f"Refusing to replace divergent local state during restore: {target}")
        for target, value in targets:
            if not target.exists():
                source = confined(self.backup_root, value["object"])
                # Restore copies also remain independent from persistent backup objects.
                target.parent.mkdir(parents=True, exist_ok=True)
                temp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
                with source.open("rb") as inp, temp.open("xb") as out:
                    for block in iter(lambda: inp.read(1 << 20), b""):
                        out.write(block)
                    out.flush()
                    os.fsync(out.fileno())
                if sha(temp) != value["sha256"]:
                    raise ValueError("Restored file failed read-back")
                os.replace(temp, target)
        return pointer


def load_backend(path):
    spec = importlib.util.spec_from_file_location("oligo_random_locked_gpu_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_task(config_path, expected_config_sha, job_id):
    config_path = Path(config_path).resolve()
    if sha(config_path) != expected_config_sha:
        raise ValueError("Worker config SHA mismatch")
    cfg = read(config_path)
    if cfg.get("schema") != 1:
        raise ValueError("Unsupported worker config schema")
    root = config_path.parent
    paths = {k: descriptor(root, cfg[k]) for k in ("input_rows", "split_plan", "backend")}
    if "input_manifest" in cfg:
        paths["input_manifest"] = descriptor(root, cfg["input_manifest"])
        manifest = read(paths["input_manifest"])
        for key, name in (("input_rows", "input_rows.json"), ("split_plan", "native_split_plan.json")):
            if manifest["generated"][name]["sha256"] != cfg[key]["sha256"]:
                raise ValueError("Config inputs disagree with qualification manifest")
    if "execution_lock" in cfg:
        paths["execution_lock"] = descriptor(root, cfg["execution_lock"])
    if "oligo_features" in cfg:
        paths["oligo_features"] = descriptor(root, cfg["oligo_features"])
    jobs = cfg["jobs"]
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate task identifiers in config")
    matches = [j for j in jobs if j["id"] == job_id]
    if len(matches) != 1 or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", job_id) or job_id in (".", ".."):
        raise ValueError("Unknown or unsafe task identifier")
    job = {k: matches[0][k] for k in ("id", "model", "seed", "setting", "fold")}
    if job["model"] not in MODELS or job["seed"] not in SEEDS:
        raise ValueError("Unregistered model or seed")
    plan = read(paths["split_plan"])
    items = [x for x in plan["folds"] if x["setting"] == job["setting"] and x["fold"] == job["fold"]]
    if len(items) != 1:
        raise ValueError("Task does not name exactly one fixed outer fold")
    item = items[0]
    fit, test = item["fit_ids"], item["test_ids"]
    if len(set(fit)) != len(fit) or len(set(test)) != len(test) or set(fit) & set(test):
        raise ValueError("Outer fit/test identities overlap or repeat")
    universe = set(range(len(fit)))
    coverage = np.zeros(len(fit), dtype=np.int8)
    if len(item["inner"]) != 5:
        raise ValueError("Exactly five common inner folds are required")
    for split in item["inner"]:
        tr, va = split["train"], split["valid"]
        if (len(set(tr)) != len(tr) or len(set(va)) != len(va) or set(tr) & set(va) or
                set(tr) | set(va) != universe):
            raise ValueError("Inner fold is not a disjoint full partition of outer fit")
        coverage[va] += 1
    if not np.all(coverage == 1):
        raise ValueError("Each outer fit input must be held out in exactly one inner fold")
    identity = dict(schema=1, job=job, config_sha256=expected_config_sha,
                    worker_source_sha256=sha(__file__), backend_sha256=cfg["backend"]["sha256"],
                    input_rows_sha256=cfg["input_rows"]["sha256"], split_plan_sha256=cfg["split_plan"]["sha256"],
                    fit_ids_sha256=value_sha(fit), test_ids_sha256=value_sha(test),
                    inner_sha256=value_sha(item["inner"]),
                    protocol="five_inner_validation_MSE_median_epoch_full_outer_fit_v1",
                    test_used_for_selection=False)
    if "execution_lock" in cfg:
        identity["execution_lock_sha256"] = cfg["execution_lock"]["sha256"]
    if "input_manifest" in cfg:
        identity["input_manifest_sha256"] = cfg["input_manifest"]["sha256"]
    return cfg, paths, job, item, identity


def verify_job_complete(job_root, expected_identity, initial_root):
    job_root = Path(job_root)
    receipt = read(job_root / "complete.json")
    if receipt.get("status") != "complete" or receipt.get("identity_sha256") != value_sha(expected_identity):
        raise ValueError("Completed job identity mismatch")
    for name, digest in receipt["files"].items():
        if sha(confined(job_root, name)) != digest:
            raise ValueError(f"Completed task file SHA mismatch: {name}")
    for name, expected in receipt["fit_execution_identities"].items():
        verify_fit(confined(job_root, name), expected, initial_root)
    if len(receipt["fit_execution_identities"]) != 6:
        raise ValueError("A complete task requires five inner fits and one final fit")
    return receipt


def csv_save(path, rows):
    import io
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, buffer.getvalue().encode("utf-8"))


def prune_completed_resume_payload(out, receipt, store, stage):
    """Remove only this completed fit's redundant last after durable final backup."""
    out = Path(out).resolve()
    if store.job_root not in out.parents:
        raise ValueError("Refusing to prune outside the current job")
    names = ("last.pt", "last.sha256.json")
    if any(name in receipt["files"] for name in names):
        raise ValueError("Backend completed manifest still requires redundant last state")
    plan_path = out / "completed_state_pruning_plan.json"
    if plan_path.exists():
        plan = read(plan_path)
        durable_path = confined(store.job_backup, plan["durable_backup"]["snapshot"])
        if (plan["completed_fit_receipt_sha256"] != sha(out / "complete.json") or
                sha(durable_path) != plan["durable_backup"]["sha256"]):
            raise ValueError("Interrupted pruning plan identity or backup differs")
        for name, value in plan["removed"].items():
            if name not in names:
                raise ValueError("Pruning plan contains an unexpected target")
            path = out / name
            if path.exists():
                if sha(path) != value["sha256"]:
                    raise ValueError("Completed last changed after pruning was registered")
                path.unlink()
        immutable_json(out / "completed_state_pruning.json", dict(plan, pruning_completed=True))
        return
    found = [out / name for name in names if (out / name).exists()]
    if not found:
        return
    last = out / "last.pt"
    pointer = out / "last.sha256.json"
    if not last.exists() or not pointer.exists():
        raise ValueError("Completed last payload/pointer pair is incomplete")
    last_digest = sha(last)
    pointer_value = read(pointer)
    if (last_digest != receipt.get("last_checkpoint_sha256") or pointer_value.get("sha256") != last_digest or
            pointer_value.get("identity_sha256") != receipt["identity_sha256"]):
        raise ValueError("Completed last payload does not match its recorded checkpoint")
    removed = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in found}
    # This snapshot omits completed last but includes every final fit artifact and
    # the shared initialization object, with independent copy/read-back verification.
    durable = store.snapshot(f"{stage}_complete_before_prune", force=True)
    plan = dict(schema=1, completed_fit_receipt_sha256=sha(out / "complete.json"), removed=removed,
        durable_backup=durable, retained="selected_weights_history_identity_normalization_initial_reference",
        completion_verified_before_prune=True, independent_backup_readback_verified=True)
    immutable_json(plan_path, plan)
    for path in found:
        if sha(path) != removed[path.name]["sha256"]:
            raise ValueError("Completed last changed before pruning")
        path.unlink()
    immutable_json(out / "completed_state_pruning.json", dict(plan, pruning_completed=True))


def run_job(config_path, expected_config_sha, job_id, run_root, backup_root, worker_id="worker",
            device="cuda", stop_requested=None, restore_latest=False, backend_override=None):
    cfg, paths, job, item, identity = load_task(config_path, expected_config_sha, job_id)
    run_root = Path(run_root).resolve()
    job_root = confined(run_root / "jobs", job_id)
    stop_requested = stop_requested or (lambda: _STOP or (run_root / "STOP").exists() or (job_root / "STOP").exists())
    initial_root = run_root / "initial_objects"
    digest = value_sha(identity)
    store = BackupStore(backup_root, job_root, digest, initial_root, cfg.get("backup_interval_seconds", 3600))
    with JobLease(job_root, digest, worker_id):
        if restore_latest:
            store.restore_latest()
        immutable_json(job_root / "job_identity.json", identity)
        if (job_root / "complete.json").exists():
            receipt = verify_job_complete(job_root, identity, initial_root)
            store.snapshot("complete_reuse", force=True)
            return dict(status="complete", job_id=job_id, reused=True, receipt_sha256=sha(job_root / "complete.json"),
                        predictions=receipt["predictions"])
        if stop_requested():
            store.snapshot("stopped", force=True)
            return dict(status="stopped", job_id=job_id)
        rows = read(paths["input_rows"])
        if not isinstance(rows, list):
            raise ValueError("input_rows must be an ordered array")
        ids = [r["sample_id"] for r in rows]
        if len(set(ids)) != len(ids):
            raise ValueError("Input sample identifiers must be unique")
        index = {sample_id: i for i, sample_id in enumerate(ids)}
        fi = np.array([index[x] for x in item["fit_ids"]], dtype=np.int64)
        ti = np.array([index[x] for x in item["test_ids"]], dtype=np.int64)
        backend = backend_override or load_backend(paths["backend"])
        features = np.load(paths["oligo_features"], mmap_mode="r", allow_pickle=False) if "oligo_features" in paths else None
        if features is not None and (features.shape != (len(rows), 280) or not np.isfinite(features).all()):
            raise ValueError("Corrected Oligo feature cache must be finite n by 280")
        arrays = backend.prepare_arrays(rows, oligo_features=features)
        assets = dict(cfg.get("assets", {}))
        assets["initial_objects_dir"] = str(initial_root)
        assets["device"] = device
        fit_identities = {}
        chosen_epochs = []
        started = time.time()
        try:
            for j in range(6):
                if stop_requested():
                    store.snapshot("stopped", force=True)
                    return dict(status="stopped", job_id=job_id, completed_inner=len(chosen_epochs))
                stage = f"inner{j}" if j < 5 else "final"
                out = job_root / "fits" / stage
                if j < 5:
                    split = item["inner"][j]
                    tr, va = fi[np.array(split["train"], dtype=int)], fi[np.array(split["valid"], dtype=int)]
                    fixed_epochs = None
                else:
                    tr, va = fi, None
                    fixed_epochs = int(statistics.median(chosen_epochs))
                    if fixed_epochs < 1:
                        raise ValueError("Selected epoch budget must be positive")
                execution_identity = dict(job=identity, stage=stage,
                    train_sample_ids_sha256=value_sha([ids[i] for i in tr]),
                    validation_sample_ids_sha256=value_sha([] if va is None else [ids[i] for i in va]),
                    fixed_epochs=fixed_epochs)
                fit_identities[f"fits/{stage}"] = execution_identity
                if (out / "complete.json").exists():
                    receipt = verify_fit(out, execution_identity, initial_root)
                else:
                    receipt = backend.run_fit(job["model"], arrays, tr, job["seed"], out, assets,
                        validation_indices=va, fixed_epochs=fixed_epochs,
                        execution_identity=execution_identity, stop_requested=stop_requested,
                        checkpoint_callback=store.checkpoint_callback, device=device)
                    if receipt.get("status") == "stopped":
                        store.snapshot("stopped", force=True)
                        return dict(status="stopped", job_id=job_id, completed_inner=len(chosen_epochs))
                    receipt = verify_fit(out, execution_identity, initial_root)
                if j < 5:
                    epoch = int(receipt["selected_epochs"])
                    if epoch < 1 or not math.isfinite(float(receipt["best_validation_mse"])):
                        raise ValueError("Inner selection requires positive epoch and finite validation MSE")
                    chosen_epochs.append(epoch)
                elif receipt.get("fixed_epochs") != fixed_epochs:
                    raise ValueError("Final fit does not use the registered median epoch budget")
                prune_completed_resume_payload(out, receipt, store, stage)
                write(job_root / "progress.json", dict(job_id=job_id, worker_id=worker_id,
                    stage=stage, status="fit_complete", completed_inner=len(chosen_epochs),
                    selected_inner_epochs=chosen_epochs, updated_unix=time.time()))
                store.snapshot(f"{stage}_complete", force=True)
            final_identity = fit_identities["fits/final"]
            if stop_requested():
                store.snapshot("stopped", force=True)
                return dict(status="stopped", job_id=job_id, completed_inner=5, final_fit_complete=True)
            predictions = np.asarray(backend.predict_fit(job["model"], arrays, ti,
                job_root / "fits/final", assets, expected_identity=final_identity, device=device), dtype=np.float64)
            if predictions.shape != (len(ti),) or not np.isfinite(predictions).all():
                raise ValueError("Final prediction must be one finite continuous score per test input")
            output_rows = [dict(model=job["model"], seed=job["seed"], setting=job["setting"], fold=job["fold"],
                sample_id=ids[i], label=rows[i]["label"], binary=rows[i]["y"], prediction=float(predictions[k]))
                for k, i in enumerate(ti)]
            csv_save(job_root / "predictions.csv", output_rows)
            immutable_json(job_root / "selection.json", dict(inner_selected_epochs=chosen_epochs,
                final_epochs=int(statistics.median(chosen_epochs)), fit_ids=item["fit_ids"],
                test_ids=item["test_ids"], outer_test_used_for_selection=False,
                rule="median_of_five_inner_validation_MSE_selected_epochs"))
            write(job_root / "progress.json", dict(job_id=job_id, worker_id=worker_id,
                stage="complete", status="complete", completed_inner=5,
                selected_inner_epochs=chosen_epochs, updated_unix=time.time()))
            files = {p.relative_to(job_root).as_posix(): sha(p) for p in regular_files(job_root)
                     if p.name != "complete.json" or p.parent != job_root}
            receipt = dict(schema=1, status="complete", job=job, identity_sha256=digest,
                predictions=len(output_rows), inner_fits=5, final_fits=1, selected_inner_epochs=chosen_epochs,
                final_epochs=int(statistics.median(chosen_epochs)), test_used_for_selection=False,
                fit_execution_identities=fit_identities, files=files, local_elapsed_seconds=time.time() - started)
            immutable_json(job_root / "complete.json", receipt)
            verify_job_complete(job_root, identity, initial_root)
            store.snapshot("job_complete", force=True)
            return dict(status="complete", job_id=job_id, reused=False,
                        receipt_sha256=sha(job_root / "complete.json"), predictions=len(output_rows))
        except BaseException as error:
            # A backend may signal graceful stop by exception after saving its update boundary.
            status = "stopped" if stop_requested() else "failed"
            write(job_root / f"{status}.json", dict(status=status, job_id=job_id,
                exception_type=type(error).__name__, message=str(error), timestamp=time.time()))
            store.snapshot(status, force=True)
            if status == "stopped":
                return dict(status="stopped", job_id=job_id, completed_inner=len(chosen_epochs))
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-config-sha", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--backup-root", required=True, type=Path)
    parser.add_argument("--worker-id", default="worker")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--restore-latest", action="store_true")
    args = parser.parse_args()
    def stop_handler(_signum, _frame):
        global _STOP
        _STOP = True
    signal.signal(signal.SIGINT, stop_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_handler)
    result = run_job(args.config, args.expected_config_sha, args.job_id, args.run_root,
                     args.backup_root, args.worker_id, args.device, restore_latest=args.restore_latest)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if result["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
