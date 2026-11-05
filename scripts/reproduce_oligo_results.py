#!/usr/bin/env python3
"""Recompute the Oligo-random siRNA tables; no model fitting or GPU use.

The default check needs only Python's standard library.  The optional raw
prediction check additionally needs NumPy and SciPy and the data supplement.
Reported uncertainty is the sample SD of single-source/seed PCC values;
predictions from different seeds are never averaged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

SETTINGS = ("Huesken", "Mixset", "Takayuki", "Inter_Hu_to_Mix", "wang2009", "alas1_patent")
COUNTS = dict(zip(SETTINGS, (2196, 469, 664, 469, 131, 228)))
MACROS = {"Macro4": SETTINGS[:4], "Macro2_extra": SETTINGS[4:]}
SEEDS = ("42", "43", "44")
FAMILIES = ("full",)
BENCHMARK_MODELS = {
    "TD24": "TD24_R55", "OligoFormer-corrected": "OligoFormer-corrected",
    "BERT-siRNA": "BERT-siRNA", "RNAiSpline": "RNAiSpline",
    "DSIR（官方固定权重）": "DSIR_official_fixed", "DuplexFM＋TD24": "DuplexFM_TD24",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Non-finite value: " + context)
    return result


def close(actual: float, expected: float, context: str, tolerance: float = 1e-11) -> None:
    if abs(actual - expected) > tolerance:
        raise ValueError(f"{context}: {actual!r} differs from {expected!r}")


def mean_sd(values: list[float]) -> tuple[float, float | None]:
    if not values:
        raise ValueError("No single-seed values")
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def source_parts(source: str) -> tuple[str, str]:
    match = re.fullmatch(r"(.+)_seed(42|43|44)", source)
    return (match.group(1), match.group(2)) if match else (source, "")


def collect_benchmark(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], float]:
    seen = set()
    values = {}
    for row in rows:
        key = (row["model"], row["seed"], row["setting"], row["aggregation"])
        if key in seen:
            raise ValueError("Duplicate benchmark metric: " + repr(key))
        seen.add(key)
        if row["model"] not in BENCHMARK_MODELS.values() or row["aggregation"] not in ("pooled", "fold_mean"):
            raise ValueError("Unknown benchmark model or aggregation: " + repr(key))
        if row["aggregation"] == "pooled" and row["setting"] in (*SETTINGS, *MACROS):
            value = number(row["PCC"], repr(key))
            if row["setting"] in COUNTS and int(row["n"]) != COUNTS[row["setting"]]:
                raise ValueError("Incorrect benchmark sample count: " + repr(key))
            values[key[:3]] = value
    expected = set()
    for model in BENCHMARK_MODELS.values():
        seeds = ("",) if model in ("TD24_R55", "DSIR_official_fixed") else SEEDS
        settings = SETTINGS[1:] if model == "DSIR_official_fixed" else SETTINGS
        macros = ("Macro2_extra",) if model == "DSIR_official_fixed" else tuple(MACROS)
        expected.update((model, seed, setting) for seed in seeds for setting in (*settings, *macros))
    if set(values) != expected:
        raise ValueError(f"Benchmark group inventory mismatch; missing={expected - set(values)}, unexpected={set(values) - expected}")
    for model, seed, setting in expected:
        if setting in MACROS:
            close(values[model, seed, setting], statistics.mean(values[model, seed, s] for s in MACROS[setting]), "Benchmark macro " + repr((model, seed, setting)))
    return values


def collect_modules(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], float]:
    values = {}
    for row in rows:
        key = (row["source"], row["mode"], row["setting"])
        if key in values:
            raise ValueError("Duplicate module metric: " + repr(key))
        if row["setting"] not in (*SETTINGS, *MACROS):
            raise ValueError("Unknown module setting: " + repr(key))
        expected_n = COUNTS.get(row["setting"], sum(COUNTS[s] for s in MACROS.get(row["setting"], ())))
        if row["n"] == "" and row["source"] == "CNN_M_direct" and row["setting"] in MACROS:
            pass  # This accepted extension leaves macro n blank; six base counts are explicit.
        elif int(row["n"]) != expected_n:
            raise ValueError("Incorrect module sample count: " + repr(key))
        values[key] = number(row["PCC"], repr(key))
    sources_modes = {(f"{family}_seed{seed}", mode) for family in FAMILIES for seed in SEEDS for mode in ("source", "TD24")}
    sources_modes.update((f"Full50_seed{seed}", "pca_td") for seed in SEEDS)
    sources_modes.update({("CNN_M", "pca_td"), ("CNN_M_direct", "direct160"), ("TD24_R55", "TD24"), ("TD24_common", "ridge_td")})
    expected = {(source, mode, setting) for source, mode in sources_modes for setting in (*SETTINGS, *MACROS)}
    if set(values) != expected:
        raise ValueError(f"Module group inventory mismatch; missing={expected - set(values)}, unexpected={set(values) - expected}")
    for source, mode in sources_modes:
        for macro, settings in MACROS.items():
            close(values[source, mode, macro], statistics.mean(values[source, mode, s] for s in settings), "Module macro " + repr((source, mode, macro)))
    return values


def check_benchmark_table(rows: list[dict[str, str]], values: dict) -> dict:
    if len(rows) != len(BENCHMARK_MODELS) or {r["model"] for r in rows} != set(BENCHMARK_MODELS):
        raise ValueError("Benchmark comparison inventory mismatch")
    result = {}
    for row in rows:
        model = BENCHMARK_MODELS[row["model"]]
        seeds = ("",) if model in ("TD24_R55", "DSIR_official_fixed") else SEEDS
        result[model] = {}
        for setting in (*SETTINGS, *MACROS):
            available = [values[model, seed, setting] for seed in seeds if (model, seed, setting) in values]
            if not available:
                if row[setting + "_mean"] != "/" or row[setting + "_SD"] != "/":
                    raise ValueError("Unsupported DSIR result in comparison table")
                continue
            avg, sd = mean_sd(available)
            close(avg, number(row[setting + "_mean"], model), f"Benchmark table {model}/{setting}")
            if sd is None:
                if row[setting + "_SD"] != "/":
                    raise ValueError("SD invented for deterministic reference")
            else:
                close(sd, number(row[setting + "_SD"], model), f"Benchmark SD {model}/{setting}")
            result[model][setting] = {"PCC_mean": avg, "PCC_SD": sd}
    return result


def check_module_table(rows: list[dict[str, str]], values: dict) -> dict:
    groups = {}
    for source, mode, setting in values:
        family, seed = source_parts(source)
        groups.setdefault((family, mode), set()).add(source)
    if len(rows) != len(groups) or {(r["model"], r["mode"]) for r in rows} != set(groups):
        raise ValueError("Module comparison inventory mismatch")
    result = {}
    for row in rows:
        family, mode = row["model"], row["mode"]
        result[family + "/" + mode] = {}
        for setting in (*SETTINGS, *MACROS):
            avg, sd = mean_sd([values[source, mode, setting] for source in sorted(groups[family, mode])])
            shown = row[setting].split(" ± ")
            close(avg, number(shown[0], family), f"Module table {family}/{mode}/{setting}", 0.000050001)
            if sd is not None:
                if len(shown) != 2:
                    raise ValueError("Missing module SD")
                close(sd, number(shown[1], family), f"Module SD {family}/{mode}/{setting}", 0.000050001)
            elif len(shown) != 1:
                raise ValueError("SD invented for single-source comparison")
            result[family + "/" + mode][setting] = {"PCC_mean": avg, "PCC_SD": sd}
    return result


def prediction_pcc(rows: list[dict[str, str]], labels: dict, plan: dict, expected_groups: set[tuple]) -> dict:
    """Independently check one exact-once raw file and return pooled PCCs.

    Keys are (model/source, seed/mode, setting); benchmark caller translates
    source identifiers before comparing.  No fitting or prediction averaging.
    """
    from scipy.stats import pearsonr
    import numpy as np

    folds = {}
    expected_by_setting = {}
    for fold in plan["folds"]:
        key = (fold["setting"], str(fold["fold"]))
        ids = fold["test_ids"]
        if key in folds or len(ids) != len(set(ids)):
            raise ValueError("Duplicate fold or test IDs in plan")
        if set(fold["fit_ids"]) & set(ids):
            raise ValueError("Train/test overlap in plan")
        folds[key] = set(ids)
        previous = expected_by_setting.setdefault(key[0], set())
        if previous & set(ids):
            raise ValueError("Test sample appears in two outer folds")
        previous.update(ids)
    expected_fold_keys = {(s, str(i)) for s in SETTINGS for i in range(1 if s == "Inter_Hu_to_Mix" else 5)}
    if set(folds) != expected_fold_keys or {s: len(ids) for s, ids in expected_by_setting.items()} != COUNTS:
        raise ValueError("Wrong Oligo-random six-setting plan")

    grouped = {}
    seen = set()
    for row in rows:
        name = row.get("source", row.get("model"))
        variant = row["mode"] if "mode" in row else row.get("seed", "")
        setting, sample = row["setting"], row["sample_id"]
        key = (name, variant, setting)
        identity = (*key, sample)
        if key not in expected_groups:
            raise ValueError("Unexpected prediction group: " + repr(key))
        if identity in seen:
            raise ValueError("Duplicate prediction sample: " + repr(identity))
        seen.add(identity)
        if sample not in folds.get((setting, row["fold"]), set()):
            raise ValueError("Prediction assigned to wrong test fold")
        if sample not in labels:
            raise ValueError("Unknown sample ID")
        label = number(row["label"], repr(identity))
        close(label, float(labels[sample]["label"]), "Prediction label " + repr(identity), 1e-12)
        if "binary" in row and row["binary"] != labels[sample]["y"]:
            raise ValueError("Binary label changed")
        prediction = number(row["prediction"], repr(identity))
        grouped.setdefault(key, {})[sample] = (label, prediction)
    if set(grouped) != expected_groups:
        raise ValueError("Missing prediction groups")
    output = {}
    for key, data in grouped.items():
        if set(data) != expected_by_setting[key[2]]:
            raise ValueError("Incomplete exact-once prediction coverage: " + repr(key))
        pairs = np.asarray([data[sample] for sample in sorted(data)], dtype=np.float64)
        value = float(pearsonr(pairs[:, 0], pairs[:, 1]).statistic)
        if not math.isfinite(value):
            raise ValueError("Undefined prediction PCC: " + repr(key))
        output[key] = value
    return output


def verify_predictions(root: Path, benchmark: dict, modules: dict) -> dict:
    data = root / "data/sirna_oligo_random"
    pinned = {"input_rows.json": "a928975aa974a20bf3303ee46257336390a1e1a90b643a1d92960539c38c4980",
              "native_split_plan.json": "d4e2ca306f7074ff643d5ec60dc0f6c72fe692e6ca60ec0211e701c7c0c4de5b"}
    for filename, expected_sha in pinned.items():
        if hashlib.sha256((data / filename).read_bytes()).hexdigest() != expected_sha:
            raise ValueError("Wrong fixed Oligo input or split SHA: " + filename)
    input_rows = json.loads((data / "input_rows.json").read_text(encoding="utf-8-sig"))
    labels = {row["sample_id"]: row for row in input_rows}
    if len(labels) != len(input_rows) or len(labels) != 3688:
        raise ValueError("Input identity inventory mismatch")
    plan = json.loads((data / "native_split_plan.json").read_text(encoding="utf-8-sig"))
    hpo_expected = {(model, seed, s) for model in ("OligoFormer-corrected", "BERT-siRNA", "RNAiSpline") for seed in SEEDS for s in SETTINGS}
    hpo_rows = read_csv(data / "predictions/hpo.csv")
    hpo = prediction_pcc(hpo_rows, labels, plan, hpo_expected)
    for key, value in hpo.items():
        close(value, benchmark[key], "Raw HPO PCC " + repr(key))

    cpu_expected = {(model, "", s) for model in ("TD24_R55", "PairAC42", "PairAC43", "PairAC44") for s in SETTINGS}
    cpu_rows = read_csv(data / "predictions/cpu_reference.csv")
    cpu = prediction_pcc(cpu_rows, labels, plan, cpu_expected)
    for (model, seed, s), value in cpu.items():
        target = ("DuplexFM_TD24", model[-2:], s) if model.startswith("PairAC") else (model, "", s)
        close(value, benchmark[target], "Raw CPU-reference PCC " + repr(target))

    modules_expected = {key for key in modules if key[2] in SETTINGS and key[0] != "CNN_M_direct"}
    module_rows = read_csv(data / "predictions/pairac_modules.csv")
    module_values = prediction_pcc(module_rows, labels, plan, modules_expected)
    for key, value in module_values.items():
        close(value, modules[key], "Raw module PCC " + repr(key))
    cnn_rows = read_csv(data / "predictions/cnn_direct.csv")
    cnn = prediction_pcc(cnn_rows, labels, plan, {("CNN_M", "direct160", s) for s in SETTINGS})
    for (_, mode, setting), value in cnn.items():
        close(value, modules["CNN_M_direct", mode, setting], "Raw CNN-direct PCC " + setting)
    dsir_rows = read_csv(data / "predictions/dsir_reference.csv")
    dsir = prediction_pcc(dsir_rows, labels, plan, {("DSIR_official_fixed", "", s) for s in SETTINGS[1:]})
    for key, value in dsir.items():
        close(value, benchmark[key], "Raw fixed-DSIR PCC " + repr(key))
    return {"passed": True, "prediction_rows": {"HPO": len(hpo_rows), "CPU_reference": len(cpu_rows), "PairAC_modules": len(module_rows), "CNN_direct": len(cnn_rows), "DSIR_fixed": len(dsir_rows)},
            "pooled_groups_independently_recomputed": len(hpo) + len(cpu) + len(module_values) + len(cnn) + len(dsir),
            "aggregate_only": [], "DSIR_note": "Official fixed Huesken-trained scores; no Huesken estimate; Hu-to-Mix aliases the same Mixset scores",
            "new_model_fits": 0, "GPU_used": False}


def run(root: Path, raw: bool = False) -> dict:
    base = root / "results/siRNA_oligo_random"
    benchmark = collect_benchmark(read_csv(base / "benchmark/per_seed.csv"))
    modules = collect_modules(read_csv(base / "modules/PCC_per_seed.csv"))
    benchmark_summary = check_benchmark_table(read_csv(base / "benchmark/PCC_comparison.csv"), benchmark)
    module_summary = check_module_table(read_csv(base / "modules/PCC_tables.csv"), modules)
    # The full-precision original module table must agree with the compact table.
    original_rows = read_csv(base / "modules/per_seed.csv")
    original_modules = {(r["source"], r["mode"], r["setting"]): number(r["PCC"], r["source"]) for r in original_rows}
    expected_original = {k for k in modules if k[0] != "CNN_M_direct"}
    if len(original_rows) != len(original_modules) or set(original_modules) != expected_original:
        raise ValueError("Original module table inventory mismatch")
    for key, value in original_modules.items():
        close(value, modules[key], "Original/compact module table " + repr(key))
    for setting in (*SETTINGS, *MACROS):
        close(modules["TD24_R55", "TD24", setting], benchmark["TD24_R55", "", setting], "Shared TD24 control " + setting)
        for seed in SEEDS:
            close(modules[f"full_seed{seed}", "TD24", setting], benchmark["DuplexFM_TD24", seed, setting], "Shared Full control " + setting + "/" + seed)
    result = {"passed": True, "protocol": "Oligo-random six settings; training-only inner selection", "aggregation": "pooled PCC per seed, then arithmetic mean and sample SD; no prediction averaging",
              "benchmark": benchmark_summary, "modules": module_summary, "new_model_fits": 0, "GPU_used": False}
    if raw:
        result["raw_predictions"] = verify_predictions(root, benchmark, modules)
    else:
        result["raw_predictions"] = "not checked; add --verify-predictions with the data supplement"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verify-predictions", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.verify_predictions), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
