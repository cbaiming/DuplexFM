"""Focused checks for table aggregation and fail-closed raw-result validation."""
import copy
import importlib.util
from pathlib import Path
import statistics

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/reproduce_oligo_results.py"
SPEC = importlib.util.spec_from_file_location("oligo_results", PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def benchmark_fixture():
    rows = []
    for model in m.BENCHMARK_MODELS.values():
        seeds = ("",) if model in ("TD24_R55", "DSIR_official_fixed") else m.SEEDS
        settings = m.SETTINGS[1:] if model == "DSIR_official_fixed" else m.SETTINGS
        for seed in seeds:
            scores = {setting: 0.2 + 0.03 * m.SETTINGS.index(setting) + (int(seed or 42) - 42) * 0.02 for setting in settings}
            for setting in settings:
                rows.append(dict(model=model, seed=seed, setting=setting, aggregation="pooled", n=str(m.COUNTS[setting]), PCC=str(scores[setting])))
            for macro, subset in m.MACROS.items():
                if all(s in scores for s in subset):
                    rows.append(dict(model=model, seed=seed, setting=macro, aggregation="pooled", n=str(sum(m.COUNTS[s] for s in subset)), PCC=str(statistics.mean(scores[s] for s in subset))))
    return rows


def test_seed_metrics_averaged_without_prediction_voting():
    # Opposite offset/scale changes alter prediction averaging but not each PCC.
    a = [0, 1, 2, 3]
    b = [30, 20, 10, 0]
    y = [0, 1, 2, 3]
    pccs = [statistics.correlation(y, a), statistics.correlation(y, b)]
    avg, sd = m.mean_sd(pccs)
    assert avg == 0
    assert sd == pytest.approx(2 ** 0.5)
    assert statistics.correlation(y, [(x + z) / 2 for x, z in zip(a, b)]) == -1
    assert m.collect_benchmark(benchmark_fixture())["DuplexFM_TD24", "42", "Macro4"] == pytest.approx(0.245)


def test_duplicate_metric_rejected():
    rows = benchmark_fixture()
    with pytest.raises(ValueError, match="Duplicate"):
        m.collect_benchmark(rows + [rows[0]])


def test_missing_seed_and_forged_macro_rejected():
    rows = benchmark_fixture()
    with pytest.raises(ValueError, match="inventory"):
        m.collect_benchmark(rows[1:])
    forged = copy.deepcopy(rows)
    next(row for row in forged if row["setting"] == "Macro4")["PCC"] = "0.99"
    with pytest.raises(ValueError, match="macro"):
        m.collect_benchmark(forged)


def tiny_predictions(monkeypatch):
    # One group on all six settings; five tiny CV folds plus one cross-domain fold.
    monkeypatch.setattr(m, "COUNTS", dict.fromkeys(m.SETTINGS, 5))
    labels = {f"{setting}_{i}": {"label": str(i / 4), "y": str(int(i >= 2))} for setting in m.SETTINGS for i in range(5)}
    plan = {"folds": []}
    predictions = []
    for setting in m.SETTINGS:
        for fold in range(1 if setting == "Inter_Hu_to_Mix" else 5):
            tests = [f"{setting}_{i}" for i in range(5)] if setting == "Inter_Hu_to_Mix" else [f"{setting}_{fold}"]
            plan["folds"].append({"setting": setting, "fold": fold, "test_ids": tests, "fit_ids": ["unrelated_training_input"]})
            for sample in tests:
                predictions.append({"model": "example", "seed": "42", "setting": setting, "fold": str(fold), "sample_id": sample, "label": labels[sample]["label"], "binary": labels[sample]["y"], "prediction": labels[sample]["label"]})
    expected = {("example", "42", setting) for setting in m.SETTINGS}
    return predictions, labels, plan, expected


def test_raw_exact_once_and_labels(monkeypatch):
    predictions, labels, plan, expected = tiny_predictions(monkeypatch)
    assert all(value == pytest.approx(1) for value in m.prediction_pcc(predictions, labels, plan, expected).values())
    with pytest.raises(ValueError, match="Duplicate prediction"):
        m.prediction_pcc(predictions + [predictions[0]], labels, plan, expected)
    forged = copy.deepcopy(predictions)
    forged[0]["label"] = "0.1"
    with pytest.raises(ValueError, match="Prediction label"):
        m.prediction_pcc(forged, labels, plan, expected)
    with pytest.raises(ValueError, match="coverage"):
        m.prediction_pcc(predictions[1:], labels, plan, expected)


def test_wrong_fold_and_missing_protocol_groups(monkeypatch):
    predictions, labels, plan, expected = tiny_predictions(monkeypatch)
    changed = copy.deepcopy(predictions)
    changed[0]["fold"] = "1"
    with pytest.raises(ValueError, match="wrong test fold"):
        m.prediction_pcc(changed, labels, plan, expected)
    changed_plan = copy.deepcopy(plan)
    changed_plan["folds"].pop()
    with pytest.raises(ValueError, match="Wrong Oligo"):
        m.prediction_pcc(predictions, labels, changed_plan, expected)


def test_undefined_raw_pcc_rejected(monkeypatch):
    predictions, labels, plan, expected = tiny_predictions(monkeypatch)
    for row in predictions:
        row["prediction"] = "0.5"
    with pytest.warns():
        with pytest.raises(ValueError, match="Undefined prediction PCC"):
            m.prediction_pcc(predictions, labels, plan, expected)
