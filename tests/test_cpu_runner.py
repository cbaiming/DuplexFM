"""Bounded wrapper regression checks; subprocess pytest execution is mocked."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_cpu_runner", ROOT / "scripts/run_cpu_checks.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_all_nine_suites_iterate_without_overwriting_options(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    summary = runner.run_suites()
    assert summary["passed"]
    assert [row["suite"] for row in summary["completed_suites"]] == [row[0] for row in runner.SUITES]
    assert len(calls) == 9
    for command, kwargs in calls:
        assert command[:3] == [runner.sys.executable, "-m", "pytest"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert kwargs["timeout"] == 600
    assert "-k" in calls[3][0] and "-k" in calls[4][0]


def test_selection_and_first_failure_are_reported(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=3)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    summary = runner.run_suites(only=["transfer_full", "public_interface"])
    assert summary["passed"] is False
    assert summary["requested_suites"] == ["transfer_full", "public_interface"]
    assert summary["completed_suites"] == [{"suite": "transfer_full", "returncode": 3}]
    assert len(calls) == 1
