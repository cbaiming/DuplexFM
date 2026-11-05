"""Run isolated offline CPU suites; write optional reports outside the checkout."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference" / "pairac_concat_v1"
SUITES = (
    ("full", REFERENCE / "source_full/code", ["tests"], ""),
    ("ablation42", REFERENCE / "ablations/seed42/code", ["tests"], ""),
    ("ablation43_44", REFERENCE / "ablations/seeds43_44/code", ["tests"], ""),
    ("transfer_full", REFERENCE / "sirna_transfer/full", ["test_readout.py", "test_export.py"], "not fixed_counts_and_scope"),
    ("transfer_ablation", REFERENCE / "sirna_transfer/ablations", ["test_readout.py"], "not fixed_counts_and_scope"),
    ("backup_fix", REFERENCE / "execution_reference", ["test_backup_compat.py", "test_backup_compat_v2.py"], ""),
    ("accessibility_cnn", ROOT / "workflows/accessibility/cnn", ["test_models.py", "test_training.py", "test_linear.py"], ""),
    ("single_module", ROOT / "reference/single_module_v1", ["test_fastpath.py"], ""),
    ("public_interface", ROOT, ["tests"], ""),
)


def report_counts(path: Path) -> dict:
    """Read only aggregate counts; do not copy JUnit paths or host metadata."""
    if not path.is_file():
        return {}
    xml = ET.parse(path).getroot()
    suites = [xml] if xml.tag == "testsuite" else list(xml.findall("testsuite"))
    return {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }


def run_suites(only=None, report_dir=None) -> dict:
    names = {suite[0] for suite in SUITES}
    if only is not None and (not only or set(only) - names):
        raise ValueError("Select at least one known suite")
    selected = [suite for suite in SUITES if only is None or suite[0] in only]
    report_root = Path(report_dir).expanduser().resolve() if report_dir is not None else None
    if report_root is not None:
        if report_root.is_relative_to(ROOT):
            raise ValueError("Keep local JUnit reports outside the public checkout")
        report_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTHONDONTWRITEBYTECODE="1")
    records = []
    for name, cwd, targets, selection in selected:
        local_env = env.copy()
        paths = [str(cwd)]
        if name == "backup_fix":
            paths.append(str(REFERENCE / "ablations/seeds43_44/code"))
        local_env["PYTHONPATH"] = os.pathsep.join(paths)
        with tempfile.TemporaryDirectory(prefix="pairac_cpu_") as temp:
            temporary_root = Path(temp)
            report_path = (report_root if report_root is not None else temporary_root) / (name + ".xml")
            command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *targets,
                       "--basetemp", str(temporary_root / "pytest"), "--junitxml", str(report_path)]
            if selection:
                command += ["-k", selection]
            print("Running " + name, flush=True)
            try:
                completed = subprocess.run(command, cwd=cwd, env=local_env, timeout=600)
                returncode = completed.returncode
                record = {"suite": name, "returncode": returncode, **report_counts(report_path)}
            except subprocess.TimeoutExpired:
                returncode = 124
                record = {"suite": name, "returncode": returncode, "error": "suite_timeout"}
            records.append(record)
        if returncode:
            break
    summary = {
        "scope": "offline_cpu_synthetic_checks",
        "passed": len(records) == len(selected) and all(row["returncode"] == 0 for row in records),
        "requested_suites": [suite[0] for suite in selected],
        "completed_suites": records,
        "real_split_dependent_tests": "explicitly deselected in selected transfer suites",
        "real_data_evaluation_or_training": False,
    }
    if report_root is not None:
        (report_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=[suite[0] for suite in SUITES])
    parser.add_argument("--list", action="store_true", help="List suite names without running tests")
    parser.add_argument("--report-dir", type=Path, help="Optional local report directory outside this checkout")
    options = parser.parse_args()
    if options.list:
        print("\n".join(suite[0] for suite in SUITES))
        return 0
    try:
        summary = run_suites(options.only, options.report_dir)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
