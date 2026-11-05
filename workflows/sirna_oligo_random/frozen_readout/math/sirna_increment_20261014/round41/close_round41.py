"""Archive the completed joint-head experiment without changing its selection."""
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

R = Path(__file__).resolve().parent
B = R.parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    started = time.monotonic()
    assert not (R / "execution.mutex").exists()
    report = B / "ROUND41_RESULTS.md"
    for p in [R / "closure_receipt.json", R / "cost_receipt.json", report,
              B / "GOAL_TURN_R41_AUDIT.json"]:
        assert not p.exists(), p
    lock = read(R / "lock.json")
    d = read(R / "diagnostic.json")
    v = read(R / "verification.json")
    assert v["passed"] and v["all_numerically_qualified"]
    assert not d["qualified"] and not v["qualified"]
    assert not d["primary_native_gate"] and not d["paired_kernel_gate"]
    assert d["source_increment_gate"]
    assert d["lock_sha256"] == v["lock_sha256"] == sha(R / "lock.json")
    assert v["diagnostic_sha256"] == sha(R / "diagnostic.json")
    for name, digest in lock["files"].items():
        assert sha(R / name) == digest, name
    for name, digest in lock["dependencies"].items():
        assert sha(Path(name)) == digest, name
    for name, digest in d["pilot_files_sha256"].items():
        assert sha(R / "pilot" / name) == digest, name
    assert sha(R / "diagnostic_rows.csv") == d["diagnostic_rows_sha256"]
    assert d["new_heads"] == v["verified_new_joint_heads"] == 60
    assert d["new_nuclear_heads"] == 45 and d["new_TD_heads"] == 15
    assert d["new_internal_OOF_rows"] == v["new_internal_OOF_rows"] == 9196
    assert d["new_proximal_updates"] == v["new_proximal_updates"] == 5650
    assert len(d["records"]) == 12
    assert all(len(row["inner_blocks"]) == 5 for row in d["records"])
    runs = list(R.glob("execution_*.json"))
    audits = list(R.glob("verification_execution_*.json"))
    assert len(runs) == len(audits) == 1
    e, a = read(runs[0]), read(audits[0])
    assert e["returncode"] == a["returncode"] == 0
    tests = list(ET.parse(R / "pytest.xml").getroot().iter("testsuite"))
    assert sum(int(t.get("tests", 0)) for t in tests) == 9
    assert not any(int(t.get(k, 0)) for t in tests
                   for k in ("errors", "failures", "skipped"))
    synthetic = sum(float(t.get("time", 0)) for t in tests)
    cost = dict(
        CPU_numeric_seconds=e["CPU_wall_seconds"],
        CPU_independent_audit_outer_seconds=a["CPU_wall_seconds"],
        CPU_independent_audit_inner_seconds_included=v["CPU_wall_seconds"],
        CPU_numeric_and_audit_seconds=e["CPU_wall_seconds"] + a["CPU_wall_seconds"],
        CPU_synthetic_seconds=synthetic,
        GPU_hours=0, new_source_forward=0, new_PCA=0, new_scaler_fits=0,
        source_training_original_cache_startup_imports_development_and_close_not_included=True,
        execution_sha256=sha(runs[0]), verification_execution_sha256=sha(audits[0]))

    lines = [
        "# R41：联合原生线性与固定非线性，未通过采用门槛", "",
        "结论：不采用本轮方法，不新增外层测试。主Core的收益太小，Mixset稍退步；相对R40也未保住Hu/Mix的收益。完整四设置的主结果仍是R26平均相对PCC增益+3.2771%，未达到5%。", "",
        "本轮把原生TD24＋source192读出的核范数约束，与一个固定非线性函数放进同一个凸损失联合求解。没有使用既有模型的效率预测作训练输入，也没有平均两套预测。线性部分的缩放、alpha=100及原R25 lambda定义均保持；去掉核函数中的解析线性项后，辅助偶核仅按训练数据中心化、平均对角固定为16。有限样本中不保证它与线性特征经验正交。", "",
        "只用三个既定outer fold0的fit池（1564、319、416条）及原五个purged inner折。以下是内部OOF开发结果，不是完整四设置结果，也不是新的盲外部确认。MSE下降为正表示改善。", "",
        "|训练池|来源|新PCC|新MSE|ΔPCC vs原生R25|MSE下降 vsR25|ΔPCC vs原生核R40|MSE下降 vsR40|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in d["records"]:
        if row["source"] == "TD_only":
            continue
        lines.append(
            f"|{row['setting']}|{row['source']}|{row['metrics']['PCC']:.6f}|"
            f"{row['metrics']['MSE']:.6f}|{row['delta_PCC_vs_R25']:+.6f}|"
            f"{row['MSE_reduction_vs_R25_pct']:+.4f}%|{row['delta_PCC_vs_R40']:+.6f}|"
            f"{row['MSE_reduction_vs_R40_pct']:+.4f}%|")
    lines += ["",
        "主Core相对R25的三个PCC变化为+0.000818、−0.000761、+0.002848，均未达到至少两个池增加0.005的要求；相对R40，Hu/Mix退步，Tak改善。Full在Hu/Tak相对R25改善更大，但不能在看过结果后改成主来源。精确初始化也有局部改善，因此不能把本轮非线性收益都归因于miRNA监督。", "",
        f"预定门槛：原生主对照={d['primary_native_gate']}，核配对对照={d['paired_kernel_gate']}，同类TD增量={d['source_increment_gate']}，数值资格={d['all_numerically_qualified']}，最终qualified={d['qualified']}。不改阈值、核权重、lambda、步数或来源，不追加网格。", "",
        "|训练池|新TD-only PCC|TD相对原固定TD的ΔPCC|TD的MSE下降|Core相对新TD的ΔPCC|Core的MSE下降|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for pair in d["paired_TD"]:
        row = next(x for x in d["records"]
                   if x["setting"] == pair["setting"] and x["source"] == "TD_only")
        lines.append(
            f"|{pair['setting']}|{row['metrics']['PCC']:.6f}|{row['delta_PCC_vs_R4']:+.6f}|"
            f"{row['MSE_reduction_vs_R4_pct']:+.4f}%|{pair['delta_PCC']:+.6f}|"
            f"{pair['MSE_reduction_pct']:+.4f}%|")
    lines += ["",
        "Mixset的Core仍弱于同类TD-only。该TD控制是本轮新拟合的15个联合头，不是旧R33/R40核控制。报告TD控制的退步，避免因为换了较弱的对照而误称迁移达标。", "",
        "计数：45新来源头＋15新TD头＝60新头，5650次近端更新、9196内部OOF行、2299唯一训练池样本。45旧R25、45旧R40和15唯一旧R4 TD头仅复用；验收另核查45旧R4来源头。没有新source forward、PCA、scaler拟合或GPU计算。核展开存储35268次训练行引用，不能把它当作独立样本数，也不能把整个联合读出仅称为217系数模型。", "",
        f"验收：9项纯合成通过；60新联合目标的全局可行对偶证书、60核驻点方程、独立LU求解、原生lambda、部署、覆盖和SHA均通过。最大相对对偶间隙{v['max_relative_joint_gap']:.3g}，核驻点误差{v['max_kernel_stationarity_error']:.3g}，部署差{v['max_deployment_difference']:.3g}，原始与消元目标差{v['max_full_vs_profiled_objective_difference']:.3g}。数值通过不等于性能门槛通过。", "",
        "|训练池|来源|内折|新PCC|新MSE|R4参考PCC|R25参考PCC|R40参考PCC|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in d["records"]:
        for x in row["inner_blocks"]:
            r25 = f"{x['R25_reference']['PCC']:.6f}" if "R25_reference" in x else "/"
            r40 = f"{x['R40_reference']['PCC']:.6f}" if "R40_reference" in x else "/"
            lines.append(f"|{row['setting']}|{row['source']}|{x['inner']}|"
                         f"{x['metrics']['PCC']:.6f}|{x['metrics']['MSE']:.6f}|"
                         f"{x['R4_reference']['PCC']:.6f}|{r25}|{r40}|")
    lines += ["",
        f"成本：数值控制器{e['CPU_wall_seconds']:.6f}秒，独立验收{a['CPU_wall_seconds']:.6f}秒（报告内层{v['CPU_wall_seconds']:.6f}秒已包含），合计{cost['CPU_numeric_and_audit_seconds']:.6f}CPU秒；合成{synthetic:.6f}秒另列。旧来源训练、原始缓存、进程启动导入、开发及收尾不包含，不能当作端到端时间。", "",
        "数学参考：[Cho与Saul2009](https://proceedings.neurips.cc/paper_files/paper/2009/file/5751ec3e9a4feab575962e78e006250d-Paper.pdf)的一阶核及[Schölkopf等2001](https://alex.smola.org/papers/2001/SchHerSmo01.pdf)的有限表示结果。具体偶核分解与联合消元为本轮代数推导；文献不提供siRNA性能保证。", "",
        "本轮在已知R25/R40后提出，仍是自适应开发。单source42、源keep-overlap/context暴露、purged非gene隔离、Mixset两个设置非独立、缺少新鲜盲外部等限制保持。不能据此判断更复杂读出必然无效，也不能据辅助Full的局部正结果宣布达标。", "",
        f"科学锁：{sha(R / 'lock.json')}。诊断：{sha(R / 'diagnostic.json')}。独立验收：{sha(R / 'verification.json')}。",
    ]
    write(R / "cost_receipt.json", cost)
    with report.open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    files = [R / "lock.json", R / "diagnostic.json", R / "diagnostic_rows.csv",
             R / "verification.json", R / "cost_receipt.json", report, Path(__file__)]
    files += [R / n for n in lock["files"]] + runs + audits
    manifest = {p.relative_to(B).as_posix(): sha(p) for p in files}
    for name, digest in manifest.items():
        assert sha(B / name) == digest, name
    write(R / "closure_receipt.json", dict(
        status="closed", readback_passed=True, goal_complete=False, qualified=False,
        no_live_execution_mutex=True, all_pilot_files_verified=len(d["pilot_files_sha256"]),
        files_sha256=manifest, CPU_close_seconds=time.monotonic() - started))
    write(B / "GOAL_TURN_R41_AUDIT.json", dict(
        previous_turn="progress", this_turn="progress", goal_status="active",
        objective_achieved=False,
        concrete_progress="One jointly fitted partially linear nuclear-regularized kernel head completed: 60 new heads and independent convex/deployment verification. Core gains too small versus native head and Hu/Mix lose versus R40; qualification false.",
        new_outer_test_evaluation=False, new_heads=60, no_active_handle=True,
        genuine_external_blocker=False,
        closure_sha256=sha(R / "closure_receipt.json"), report_sha256=sha(report)))
    with (B / "PROGRESS.md").open("a", encoding="utf-8") as f:
        f.write("\n\nR41原生线性＋固定偶核的单凸目标联合读出闭合：45新来源头＋15新TD头、5650近端更新、9196内部OOF；全60完整凸证书、核驻点、独立LU/原lambda/部署/覆盖/SHA通过。Core相对R25ΔPCC +.000818/−.000761/+.002848；相对R40的Hu/Mix退步、Tak改善。primary_native与paired_kernel=false、source_increment=true、最终gate=false，不挑Full、不改门槛/核权重/正则、不做新外层。"
                f"CPU数值＋验收{cost['CPU_numeric_and_audit_seconds']:.6f}s，合成{synthetic:.6f}s另计，新GPU/forward/PCA/scaler0。完整四设置主结果仍R26+3.2771%，goal active未达5%；上一轮及本轮progress，无活句柄/外部阻断，详见ROUND41_RESULTS.md。\n")
    print(json.dumps(dict(status="closed", qualified=False, goal_complete=False,
                          closure_sha256=sha(R / "closure_receipt.json"),
                          report_sha256=sha(report),
                          CPU_numeric_and_audit_seconds=cost["CPU_numeric_and_audit_seconds"])))


if __name__ == "__main__":
    main()
