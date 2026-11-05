"""Close already verified R26 artifacts; no fitting or metric recomputation."""
import csv
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

R = Path(__file__).resolve().parent
ROOT = R.parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


start = time.monotonic()
assert not (R / "execution.mutex").exists()
assert not (R / "closure_receipt.json").exists()
lock = read(R / "lock.json")
done = read(R / "results/complete.json")
v = read(R / "verification.json")
goals = read(R / "results/goal_summary.json")
assert v["passed"] and v["no_new_dense_fit"]
assert v["lock_sha256"] == done["lock_sha256"] == sha(R / "lock.json")
assert v["complete_sha256"] == sha(R / "results/complete.json")
assert goals == v["goal_summary"]
for name, digest in lock["files"].items():
    assert sha(R / name) == digest, name
for name, digest in lock["dependencies"].items():
    assert sha(Path(name)) == digest, name
for name, digest in done["files_sha256"].items():
    assert sha(R / "results" / name) == digest, name
expected = {"total_inner_heads": 1200, "new_inner_heads": 1155,
            "new_final_heads": 48, "new_scientific_heads": 1203,
            "new_OOF_rows": 11394, "reused_dense_OOF_rows": 11394,
            "full_joint_certificates": 1248, "reused_R25_heads": 45}
for key, value in expected.items():
    assert v[key] == value, key
ex = [(p, read(p)) for p in R.glob("execution_*.json")]
vx = [(p, read(p)) for p in R.glob("verification_execution_*.json")]
assert len(ex) == len(vx) == 1
assert all(item["returncode"] == 0 and item["error"] is None for _, item in ex + vx)
suites = list(ET.parse(R / "pytest.xml").getroot().iter("testsuite"))
assert sum(int(s.get("tests", 0)) for s in suites) == 10
assert not any(int(s.get("failures", 0)) + int(s.get("errors", 0)) + int(s.get("skipped", 0)) for s in suites)
fit = ex[0][1]["CPU_wall_seconds"]
audit = vx[0][1]["CPU_wall_seconds"]
tests = sum(float(s.get("time", 0)) for s in suites)
cost = dict(CPU_controller_seconds=fit, CPU_audit_outer_seconds=audit,
            CPU_audit_inner_seconds_included=v["CPU_wall_seconds"],
            CPU_fit_and_audit_seconds=fit + audit, synthetic_seconds_separate=tests,
            **expected, reused_R22_dense_heads=1248, new_source_forward=0,
            new_PCA_fits=0, GPU_hours=0,
            excluded="Earlier source training/export, R25 pilot, R22 dense, startup, development and report/closure are separate.",
            execution_sha256={p.name: sha(p) for p, _ in ex + vx})
save(R / "cost_receipt.json", cost)
rows = table(R / "results/summary.csv")
paired = table(R / "results/paired_dense.csv")
assert len(rows) == len(paired) == 12
assert len([g for g in goals if g["primary"]]) == 1
main = next(g for g in goals if g["primary"])
assert main["condition"] == "core_real_native_trace" and not main["reaches_development_5pct"]
lines = ["# R26：原生系数矩阵收缩的四设置结果", "", "2026-10-15。主Core方案相对固定TD24 Ridge的四设置PCC相对增益均值为+3.2771%，仍未达到5%。Mixset及Huesken→Mixset仍为负增益，不能称稳定迁移已解决。", "",
         "唯一方法变化沿用R25：把原生六个池化统计块×32通道的来源系数作为矩阵，加入核范数收缩。模型仍是TD24+source192的单线性头，不增加手工乘积、不改变冻结网络或池化。217个部署系数包含截距；有效秩不是名义参数量。标准化逐列进行，因此秩的解释限定在该坐标系。", "",
         "使用原五个alpha [.1,1,10,100,1000]与内层五折PCC/MSE/更强正则的固定选择；lambda比例0.1，不搜索秩。原样本、划分、来源seed42和TD24基线均不变。", "",
         "|设置|来源|TD24 PCC|方法PCC|相对PCC增益|MSE下降|", "|---|---|---:|---:|---:|---:|"]
for row in rows:
    lines.append(f"|{row['setting']}|{row['source']}|{float(row['TD24_PCC']):.6f}|{float(row['PCC']):.6f}|{float(row['PCC_relative_gain_pct']):+.4f}%|{float(row['MSE_relative_reduction_pct']):+.4f}%|")
lines += ["", "|来源|四设置相对PCC增益等权均值|达到5%|", "|---|---:|---|"]
for g in goals:
    lines.append(f"|{g['condition']}|{g['mean_relative_PCC_gain_pct']:+.4f}%|否|")
lines += ["", "相对上一轮主Core稀疏读出的+2.7840%，本轮增加约0.4931个百分点。这是反复开发中的数值变化，未据此宣称统计显著性或独立确认。Full不是主方案；不按队列在Core/Full之间挑选。", "",
          "|设置|来源|相对R22同216D dense的ΔPCC|相对dense MSE下降|", "|---|---|---:|---:|"]
for row in paired:
    lines.append(f"|{row['setting']}|{row['source']}|{float(row['delta_PCC']):+.6f}|{float(row['MSE_reduction_pct']):+.4f}%|")
lines += ["", "12项配对dense变化均为正，包括未做miRNA任务训练的初始化控制，因此不能将正则化收益全部解释为miRNA监督。当前Core相对精确初始化在Huesken/Takayuki/跨队列更好，Mixset反而更差。训练过的表示存在有用信号，但尚不具备跨队列一致增益。", "",
          "计数：1200内层头中1155新拟合、45复用R25；48新最终头，共1203新科学头。新11394行覆盖三来源各3798个四设置测试样本出现；独立样本3329，两个Mixset设置不是两个独立队列。另复用R22 dense的1248头及11394行，未重训或合并其预测。", "",
          f"验收：十项合成测试通过；1248完整联合系统证书、所有行序/来源/输入/选择/部署/结果SHA通过。最大相对对偶gap {v['max_relative_dual_gap']:.3g}，最大TD正规方程差{v['max_TD_normal_equation_error']:.3g}，最大部署预测差{v['max_prediction_difference']:.3g}，TD基线重建差{v['max_TD_baseline_difference']:.3g}。未重新拟合标准化/PCA或来源模型。", "",
          f"成本：CPU控制器{fit:.6f}s，独立验收外层{audit:.6f}s，合计{fit+audit:.6f}s；内层验收{v['CPU_wall_seconds']:.6f}s已包含，不重复累计。合成测试{tests:.6f}s另列；旧来源训练/导出、R25/R22、启动、开发与收尾不含在本次数值阶段时间中。GPU及新来源forward均0。", "",
          "限制：source keep-overlap/context暴露、单来源训练种子、反复使用开发队列、purged_sequence并非gene隔离仍存在。不把本结果叫SOTA或盲外部验证，也不根据它扩展lambda/秩/种子网格。source43范围问题尚未收到回复，未执行导出。目标保持active、未完成；本轮无活执行句柄，不构成外部阻断。", "",
          f"科学锁：{sha(R / 'lock.json')}。完成清单：{sha(R / 'results/complete.json')}。独立验收：{sha(R / 'verification.json')}。", ""]
report = ROOT / "ROUND26_RESULTS.md"
with report.open("x", encoding="utf-8") as f:
    f.write("\n".join(lines))
paths = [R / "lock.json", R / "results/complete.json", R / "verification.json",
         R / "cost_receipt.json", report, Path(__file__)]
paths += [R / name for name in lock["files"]] + [p for p, _ in ex + vx]
files = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
for name, digest in files.items():
    assert sha(ROOT / name) == digest
save(R / "closure_receipt.json", dict(status="closed", goal_complete=False,
     readback_passed=True, no_live_execution_mutex=True, files_sha256=files,
     result_files_verified=len(done["files_sha256"]), dependency_files_verified=len(lock["dependencies"]),
     CPU_close_seconds=time.monotonic() - start))
save(ROOT / "GOAL_TURN_R26_AUDIT.json", dict(previous_turn="progress", this_turn="progress",
     goal_status="active", objective_achieved=False, new_outer_test_evaluation=True,
     primary_mean_gain_pct=main["mean_relative_PCC_gain_pct"], source43_scope_reply_received=False,
     no_active_handle=True, genuine_external_blocker=False,
     closure_sha256=sha(R / "closure_receipt.json"), report_sha256=sha(report)))
with (ROOT / "PROGRESS.md").open("a", encoding="utf-8") as f:
    f.write("\n\nR26四设置原生系数矩阵收缩已闭合：1155新内层+48新最终头、45个R25旧头，11394新OOF；R22 dense控制原SHA复用，未重训。1248全局证书与全部SHA/部署/选择验收通过。主Core相对TD24四设置+3.4542/−1.1414/+11.6925/−.8968%，均值+3.2771%；Full+2.4076%、初始化−.5156%，均未达到5%。相对R22主方案增加.4931个百分点，不称显著；Mix及跨队列负增益保留。CPU拟合+验收30.796s，合成.279s另计；GPU/forward/PCA0。详见ROUND26_RESULTS.md。目标active未完成，无活进程/外部阻断，本轮progress。不扩lambda/秩网格或按队列选来源；source43范围回复尚待。\n")
print(json.dumps(dict(closed=True, primary_mean_gain_pct=main["mean_relative_PCC_gain_pct"],
                      closure_sha256=sha(R / "closure_receipt.json"), cost=cost)))
