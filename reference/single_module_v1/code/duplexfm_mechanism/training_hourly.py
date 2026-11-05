"""Fixed-update training, exact initial, resumable validation, no test selection."""
from __future__ import annotations

import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from duplexfm_joint.io_utils import (autocast_context, binary_metrics, load_checkpoint,
    load_trainable_state, restore_rng, rng_state, seed_all, to_device, trainable_state)
from .common import backup, check_disk, read, sha, write
from .data_plan import BatchSchedule
from .durable_io import save_checkpoint, valid_checkpoint, durable_json
from .efficiency import backward as micro_backward


def category(name):
    return "lora" if "lora_" in name.lower() else "fusion" if name.startswith("fusion.") else "cross"


def optimizer_for(model, options):
    groups = []
    for name, lr in (("lora", options.get("lora_learning_rate", 2e-5)),
                     ("other", options.get("learning_rate", 3e-4))):
        params = [p for n, p in model.named_parameters() if p.requires_grad and
                  ((category(n) == "lora") == (name == "lora"))]
        if params:
            groups.append({"params": params, "lr": lr, "group_name": name})
    if not groups:
        raise ValueError("No active parameters")
    return torch.optim.AdamW(groups, weight_decay=options.get("weight_decay", 1e-4))


class StopRequested(Exception):
    pass


class StopGuard:
    def __init__(self, deadline_unix=None, stop_file=None):
        self.deadline = deadline_unix
        self.stop_file = Path(stop_file) if stop_file else None
        self.signal_received = False

    def requested(self):
        return (self.signal_received or (self.stop_file is not None and self.stop_file.exists())
                or (self.deadline is not None and time.time() >= self.deadline))

    def check(self):
        if self.requested():
            raise StopRequested("checkpointed stop requested")


@torch.inference_mode()
def validate(model, store, indices, device, batch_size, amp, guard, boundary=None):
    model.eval()
    truth, scores = [], []
    for lo in range(0, len(indices), batch_size):
        guard.check()
        selected = indices[lo:lo + batch_size]
        raw = store.collate(selected)
        if not np.array_equal(raw["row_indices"], selected):
            raise ValueError("Validation row order changed")
        batch = to_device(raw, device)
        with autocast_context(device, amp):
            logits = model(batch).float().reshape(-1)
        labels = batch["labels"].float().reshape(-1)
        if logits.shape != labels.shape:
            raise ValueError("Validation output shape differs")
        truth.append(labels.cpu().numpy())
        scores.append(logits.sigmoid().cpu().numpy())
        if boundary is not None:
            boundary()
    return binary_metrics(np.concatenate(truth), np.concatenate(scores))


def run_engine(model, store, fit, val, folder, options, signature, *, resume=False,
               shuffled_labels=None, backup_root=None, guard=None, stop_after_step=None, backup_extras=()):
    """Engineering tests call this same optimizer/cursor/checkpoint implementation.

    stop_after_step simulates interruption and never changes the scientific
    endpoint in signature. Production CLI does not expose a shorter endpoint.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    guard = guard or StopGuard()
    device = next(model.parameters()).device
    amp = options.get("amp", "fp32")
    autocast_context(device, amp)
    schedule = BatchSchedule(fit, options["reference_rows"], options["batch_size"], options.get("seed", 42))
    max_steps = int(options["max_updates"])
    if max_steps <= 0 or max_steps % schedule.steps_per_epoch:
        raise ValueError("Endpoint must contain complete reference epochs")
    if len(val) == 0 or np.intersect1d(fit, val).size:
        raise ValueError("Invalid train/validation isolation")
    if shuffled_labels is not None:
        if not np.isin(shuffled_labels[np.asarray(fit)], (0, 1)).all():
            raise ValueError("Shuffled labels do not cover every fit row")
    signature = {"inputs": signature, "options": options}
    start = time.monotonic()
    optimizer = optimizer_for(model, options)
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    masked = model.masked_parameter_names() if hasattr(model, "masked_parameter_names") else set()
    if not masked <= set(parameters):
        raise ValueError("Masked parameter declaration differs from optimizer eligibility")
    state = {"step": 0, "seen_rows": 0, "last_validation_step": 0, "history": [],
             "epoch_loss_sum": 0.0, "epoch_rows": 0, "gradient_audit": None,
             "elapsed_seconds": 0.0, "last_backup_unix": 0.0}
    elapsed_prior = 0.0

    def payload():
        state["elapsed_seconds"] = elapsed_prior + time.monotonic() - start
        return {"format_version": 1, **trainable_state(model), "optimizer": optimizer.state_dict(),
                "rng": rng_state(device), "state": dict(state), "signature": signature,
                "scientific_checkpoint_selection": "fixed_terminal_not_best_validation"}

    def save():
        save_checkpoint(folder / "last.pt", payload())
        write(folder / "history.json", state["history"])
        write(folder / "progress.json", {k: v for k, v in state.items() if k != "history"})

    def snapshot(force=False):
        if backup_root is None:
            return
        now = time.time()
        if not force and now - state["last_backup_unix"] < options.get("backup_seconds", 7200):
            return
        check_disk(backup_root)
        state["last_backup_unix"] = now
        save()
        root = Path(backup_root)
        files = list(folder.glob("*.pt")) + list(folder.glob("*.json"))
        files += [p for name in ("code", "preparation") for p in (root / name).rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts and p.suffix in (".py", ".json", ".npy")]
        files += [p for p in (root / "registration.json", root / "ledger.json", root / "protocol.json",
                              root / "parallel_execution.json", root / "parallel_registration.json") if p.exists()]
        # Lease is immutable. The shared live ledger is deliberately excluded;
        # only its sole controller backs it up at a stable heartbeat boundary.
        files += list(backup_extras)
        receipt = backup(root, files, folder.name + f"_step{state['step']}")
        durable_json(folder / "latest_backup.json", receipt)

    if resume:
        if read(folder / "signature.json") != signature:
            raise ValueError("Resume code, settings, rows, data, or public assets differ")
        initial_receipt = read(folder / "initial.json")
        if sha(folder / "initial.pt") != initial_receipt["sha256"]:
            raise ValueError("Initial checkpoint changed")
        checkpoint = valid_checkpoint(folder / "last.pt", signature)
        if checkpoint["signature"] != signature:
            raise ValueError("Checkpoint signature mismatch")
        load_trainable_state(model, checkpoint)
        optimizer.load_state_dict(checkpoint["optimizer"])
        state = checkpoint["state"]
        elapsed_prior = state["elapsed_seconds"]
        if (not 0 <= state["step"] <= max_steps or state["last_validation_step"] > state["step"]
                or state["last_validation_step"] % schedule.steps_per_epoch):
            raise ValueError("Invalid saved update/validation cursor")
        expected_rows = (state["step"] // schedule.steps_per_epoch * schedule.reference_rows
                         + state["step"] % schedule.steps_per_epoch * schedule.batch_size)
        if state["seen_rows"] != expected_rows:
            raise ValueError("Saved training row cursor does not match update cursor")
        restore_rng(checkpoint["rng"], device)
    else:
        if any((folder / n).exists() for n in ("initial.pt", "last.pt", "signature.json", "complete.json")):
            raise FileExistsError("Existing run is never silently replaced")
        durable_json(folder / "signature.json", signature, immutable=True)
        if any(torch.count_nonzero(p).item() for n, p in model.named_parameters() if "lora_B" in n):
            raise ValueError("Task LoRA must have fresh zero B before the first update")
        save_checkpoint(folder / "initial.pt", payload())
        durable_json(folder / "initial.json", {"sha256": sha(folder / "initial.pt"), "before_update": 0,
                                        "exact_for_this_run": True}, immutable=True)
        save()
        snapshot(force=True)

    try:
        while True:
            guard.check()
            # A saved epoch end with unfinished validation resumes that validation.
            if state["step"] and state["step"] % schedule.steps_per_epoch == 0 and state["last_validation_step"] < state["step"]:
                saved_rng = rng_state(device)
                def validation_boundary():
                    current_rng = rng_state(device)
                    restore_rng(saved_rng, device)
                    try:
                        snapshot()
                    finally:
                        restore_rng(current_rng, device)
                try:
                    metrics = validate(model, store, val, device, options.get("eval_batch_size", schedule.batch_size), amp, guard,
                                       boundary=validation_boundary)
                finally:
                    restore_rng(saved_rng, device)
                record = {"reference_epoch": state["step"] // schedule.steps_per_epoch,
                          "step": state["step"], "seen_rows": state["seen_rows"],
                          "train_loss": state["epoch_loss_sum"] / max(state["epoch_rows"], 1),
                          "validation": metrics}
                state["history"].append(record)
                state["last_validation_step"] = state["step"]
                state["epoch_loss_sum"], state["epoch_rows"] = 0.0, 0
                print(record, flush=True)
                save()
                snapshot()
            if state["step"] == max_steps:
                save_checkpoint(folder / "terminal.pt", payload())
                save()
                snapshot(force=True)
                result = {"status": "complete", "updates": state["step"], "seen_rows": state["seen_rows"],
                          "reference_epochs": len(state["history"]), "selection": "terminal_update",
                          "initial_sha256": sha(folder / "initial.pt"), "terminal_sha256": sha(folder / "terminal.pt"),
                          "signature_sha256": sha(folder / "signature.json"),
                          "history_sha256": sha(folder / "history.json"),
                          "elapsed_seconds": elapsed_prior + time.monotonic() - start}
                durable_json(folder / "complete.json", result, immutable=True)
                if backup_root is not None:
                    receipt=backup(Path(backup_root), [folder/"complete.json", folder/"terminal.commit.json",
                                  folder/"initial.json", folder/"signature.json"], folder.name+"_completion_receipt")
                    durable_json(folder/"completion_backup.json",receipt)
                return result
            if stop_after_step is not None and state["step"] >= stop_after_step:
                raise StopRequested("engineering interruption")
            selected = schedule.at(state["step"])
            raw = store.collate(selected)
            if not np.array_equal(raw["row_indices"], selected):
                raise ValueError("Training row identity changed")
            batch = to_device(raw, device)
            labels = batch["labels"].float().reshape(-1)
            if shuffled_labels is not None:
                labels = torch.as_tensor(np.array(shuffled_labels[selected]), device=device)
            if not torch.isfinite(labels).all() or not ((labels == 0) | (labels == 1)).all():
                raise ValueError("Invalid training labels")
            model.train()
            optimizer.zero_grad(set_to_none=True)
            before = {n: p.detach().clone() for n, p in parameters.items()} if state["gradient_audit"] is None else None
            loss = micro_backward(model, batch, labels, options.get("micro_batch_size", 512), device, amp)
            # A permanently masked expert must not update through AdamW decay.
            # Keep it in the optimizer; None records its absent task path.
            for n in masked:
                if parameters[n].grad is not None and bool(torch.count_nonzero(parameters[n].grad)):
                    raise ValueError(f"Masked expert leaked a gradient: {n}")
                parameters[n].grad = None
            torch.nn.utils.clip_grad_norm_(list(parameters.values()), options.get("clip_grad_norm", 1.0), error_if_nonfinite=True)
            audit = {}
            if before is not None:
                for n, p in parameters.items():
                    row = audit.setdefault(category(n), {"nonzero_gradients": 0, "updated_tensors": 0})
                    row["nonzero_gradients"] += int(p.grad is not None and bool(torch.count_nonzero(p.grad)))
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if before is not None:
                for n, p in parameters.items():
                    audit[category(n)]["updated_tensors"] += int(not torch.equal(before[n], p.detach()))
                inactive_groups = {category(n) for n in masked} - {category(n) for n in set(parameters) - masked}
                if any((v["nonzero_gradients"] == 0 or v["updated_tensors"] == 0) for g, v in audit.items() if g not in inactive_groups):
                    raise ValueError(f"Active parameter group failed gradient/update audit: {audit}")
                if any(not torch.equal(before[n], parameters[n].detach()) for n in masked):
                    raise ValueError("Permanently masked expert changed under the optimizer")
                audit["fixed_mask"] = {"variant": getattr(model, "variant", "synthetic"),
                    "nominal_trainable_parameters": sum(p.numel() for p in parameters.values()),
                    "permanently_masked_parameter_names": sorted(masked),
                    "inactive_groups": sorted(inactive_groups), "masked_tensors_unchanged": True}
                state["gradient_audit"] = audit
                write(folder / "gradient_audit.json", audit, immutable=True)
            state["step"] += 1
            state["seen_rows"] += len(selected)
            state["epoch_rows"] += len(selected)
            state["epoch_loss_sum"] += float(loss.detach()) * len(selected)
            if state["step"] % options.get("checkpoint_steps", 500) == 0:
                check_disk(folder)
                save()
            if state["step"] % 100 == 0 or state["step"] == 1:
                print({"step": state["step"], "target": max_steps, "loss": float(loss.detach())}, flush=True)
            snapshot()
    except StopRequested as exc:
        save()
        snapshot(force=True)
        result = {"status": "interrupted", "reason": str(exc), "step": state["step"],
                  "last_sha256": sha(folder / "last.pt"), "exact_resume_supported": True}
        write(folder / ("interruption_" + str(time.time_ns()) + ".json"), result, immutable=True)
        return result
