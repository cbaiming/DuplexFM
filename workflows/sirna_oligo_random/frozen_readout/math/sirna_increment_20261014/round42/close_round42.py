"""Close a failed structural qualification without fitting its proposed head."""
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

R = Path(__file__).resolve().parent
B = R.parent
sys.path.insert(0, str(R))
from run import read, sha, write, validate


def main():
    started = time.monotonic()
    assert not (R / "execution.mutex").exists()
    report = B / "ROUND42_RESULTS.md"
    for p in [R / "closure_receipt.json", R / "cost_receipt.json", report, B / "GOAL_TURN_R42_AUDIT.json"]:
        assert not p.exists(), p
    lock = validate()
    d, v = read(R / "diagnostic.json"), read(R / "verification.json")
    assert v["passed"] and not d["qualified"] and not v["qualified"]
    assert d["lock_sha256"] == v["lock_sha256"] == sha(R / "lock.json")
    assert v["diagnostic_sha256"] == sha(R / "diagnostic.json")
    assert d["diagnostic_rows_sha256"] == sha(R / "diagnostic_rows.csv")
    assert d["read_old_heads"] == v["verified_old_coefficient_heads"] == 45
    runs = list(R.glob("execution_*.json")); audits = list(R.glob("verification_execution_*.json"))
    assert len(runs) == len(audits) == 1
    e, a = read(runs[0]), read(audits[0])
    assert e["returncode"] == a["returncode"] == 0
    suites = list(ET.parse(R / "pytest.xml").getroot().iter("testsuite"))
    failed = list(ET.parse(R / "prelock_v1/pytest.xml").getroot().iter("testsuite"))
    assert sum(int(s.get("tests", 0)) for s in suites) == 8
    assert not any(int(s.get(k, 0)) for s in suites for k in ("errors", "failures", "skipped"))
    assert sum(int(s.get("failures", 0)) for s in failed) == 1
    synthetic = sum(float(s.get("time", 0)) for s in suites + failed)
    import_failure = read(R / "prelock_v2/registration_failure.json")
    assert import_failure["returncode"] == 1 and not import_failure["real_coefficients_read"]
    cost = dict(CPU_diagnostic_seconds=e["CPU_wall_seconds"],
        CPU_independent_audit_outer_seconds=a["CPU_wall_seconds"],
        CPU_independent_audit_inner_seconds_included=v["CPU_wall_seconds"],
        CPU_diagnostic_and_audit_seconds=e["CPU_wall_seconds"] + a["CPU_wall_seconds"],
        CPU_synthetic_both_attempts_seconds=synthetic,
        CPU_failed_registration_process_wall_seconds=import_failure["CPU_process_wall_seconds"],
        failed_registration_clock_source="exec_command wall_time_seconds, separate from controller clock",
        GPU_hours=0, new_efficiency_heads=0, new_OOF_rows=0, new_source_forward=0,
        old_training_cache_startup_development_registration_success_and_close_not_included=True,
        execution_sha256=sha(runs[0]), verification_execution_sha256=sha(audits[0]))
    lines = ["# R42：固定区域平滑假设未通过资格", "",
        "结论：不训练图平滑读出，不调整区域顺序、尺度或门槛。主Core三个池均不合格。四设置完整主结果仍是R26平均相对PCC提升+3.2771%，5%目标尚未达到。", "",
        "本轮只检查已核验R25内折头的来源系数。固定将seed、中央区、其余区按顺序连成路径，分别比较mean和max块。同一通道的相邻系数平方差，除以所有区域排列的平均路径能量；比值1表示与随机顺序平均相同，越低表示越平滑。资格为五头平均比值≤0.9且至少4/5头低于1，Core至少两个池满足。", "",
        "|训练池|来源|平均平滑比值|低于随机顺序平均的内折数|资格|",
        "|---|---|---:|---:|---|",
    ]
    for row in d["records"]:
        value = f"{row['mean_smoothness_ratio']:.6f}" if row["mean_smoothness_ratio"] is not None else "/"
        lines.append(f"|{row['setting']}|{row['source']}|{value}|{row['below_permutation_mean_count']}/5|{row['qualified']}|")
    lines += ["",
        "Core在Mixset和Takayuki的比值约1.15，五个内折都高于1；Huesken约0.927，虽4/5折低于1，但没有达到预先固定的0.9平均阈值。这不支持在当前尺度和现有线性头上额外要求相邻区域系数趋同。它不是对所有图正则的否定，也不是估计新的预测性能。", "",
        "精确初始化在Hu/Mix满足该顺序资格，训练后的Core/Full不满足。系数还受TD共线性、标准化、原核范数约束和训练样本影响，不能把差别直接解释为miRNA产生了区域生物机制；更不能因此改用初始化作为主模型。第三分区包含位置1及13–19，不是纯3′端。", "",
        "与旧实验的区别：R10考察无监督输入方差及PCA截断，本次只看旧监督系数的固定邻接关系。未重做PCA、位置特征、源训练或任何旧性能指标；未读取NPZ的prediction键。旧系数依赖历史fit标签，因此这不是完全无监督证据，只是本轮没有重新读取效值或拟合。五个内折共享训练样本，不能作为五个独立来源种子或显著性检验。", "",
        "|训练池|来源|内折|平滑比值|邻边能量|随机顺序平均能量|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in d["records"]:
        for x in row["inner_blocks"]:
            ratio = f"{x['smoothness_ratio']:.6f}" if x["smoothness_ratio"] is not None else "/"
            lines.append(f"|{row['setting']}|{row['source']}|{x['inner']}|{ratio}|"
                         f"{x['adjacent_energy']:.9g}|{x['permutation_mean_energy']:.9g}|")
    lines += ["",
        f"验收：45个旧系数头、270个排列能量、全部行序/源码/模型SHA和固定资格已独立逐通道复算，最大比值差{v['max_ratio_difference']:.3g}。新增效率头、OOF预测、source forward、PCA及GPU均为0。", "",
        "登记前首次合成7/8通过，错误是手写线性序列参考比值0.75，应为0.5；修正后8/8通过。原测试、实现和失败XML保留。第一次CLI登记在读取真实系数前发生隔离Python导入路径错误，补显式脚本目录后登记成功；原入口及失败回执保留。统计实现和0.9/4-of-5/2-of-3阈值未因真实数据修改。", "",
        f"成本：诊断{e['CPU_wall_seconds']:.6f}秒＋独立验收{a['CPU_wall_seconds']:.6f}秒，共{cost['CPU_diagnostic_and_audit_seconds']:.6f}CPU秒；两次合成{synthetic:.6f}秒及登记导入失败进程{import_failure['CPU_process_wall_seconds']:.6f}秒另列。旧模型训练、原缓存、其余启动/登记/开发和收尾不在该计时内，不称端到端成本。", "",
        "结构正则的背景参考为[Evgeniou等2005](https://jmlr.csail.mit.edu/papers/volume6/evgeniou05a/evgeniou05a.pdf)。本文仅借鉴用正则表达关系的思路；六池化块不是六个监督任务，文献没有保证RNA区域应平滑。", "",
        "本轮属于既有开发池上的自适应诊断，没有新增外层评价。单source42、源keep-overlap/context暴露、purged非gene隔离、两个Mixset设置依赖及缺少新鲜盲外部的限制保持。", "",
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
        assert sha(B / name) == digest
    write(R / "closure_receipt.json", dict(status="closed", readback_passed=True,
        qualified=False, goal_complete=False, new_efficiency_heads=0,
        no_live_execution_mutex=True, files_sha256=manifest,
        CPU_close_seconds=time.monotonic() - started))
    write(B / "GOAL_TURN_R42_AUDIT.json", dict(previous_turn="progress", this_turn="progress",
        goal_status="active", objective_achieved=False, genuine_external_blocker=False,
        concrete_progress="45 old coefficient heads independently diagnosed: predetermined regional smoothness qualification failed for every primary pool, so no new smoothing head will be fitted.",
        new_efficiency_heads=0, new_outer_test_evaluation=False, no_active_handle=True,
        closure_sha256=sha(R / "closure_receipt.json"), report_sha256=sha(report)))
    with (B / "PROGRESS.md").open("a", encoding="utf-8") as f:
        f.write("\n\nR42固定区域系数路径资格闭合：45旧R25系数头/270排列能量及SHA独立标量核验通过，未读取预测键或新的效值。Core平均平滑比值Hu .926840/Mix 1.152582/Tak 1.159169，低于随机平均4/0/0折，三池资格均false；初始化Hu/Mix合格但不替换主方案。不实施图平滑头、不改邻接/门槛/尺度，不新增外层。登记前合成参考值与CLI路径两失败完整保留，修正后8/8及真实验收通过。"
                f"CPU诊断＋验收{cost['CPU_diagnostic_and_audit_seconds']:.6f}s，两次合成{synthetic:.6f}s、导入失败0.083150s另计；新头/OOF/forward/PCA/GPU0。完整主结果仍R26+3.2771%，goal active未达5%；上一轮及本轮progress，无活句柄/阻断，详见ROUND42_RESULTS.md。\n")
    print({"status": "closed", "qualified": False, "goal_complete": False,
           "closure_sha256": sha(R / "closure_receipt.json"), "report_sha256": sha(report),
           "CPU_diagnostic_and_audit_seconds": cost["CPU_diagnostic_and_audit_seconds"]})


if __name__ == "__main__":
    main()
