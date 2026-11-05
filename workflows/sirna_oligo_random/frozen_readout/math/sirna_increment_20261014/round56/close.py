"""Publish all fixed R55 sensitivity results without choosing a new model."""
import sys,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import run as io


def span(x):return '/' if x is None else f'[{x[0]:+.3f}, {x[1]:+.3f}]'


def main():
    start=time.monotonic();io.validate();v=io.read(R/'verification.json');assert v['passed'] and not (R/'execution.mutex').exists()
    root=R/'results';c=io.read(root/'complete.json');d=io.read(root/'diagnostic.json');primary=d['macro'][0]
    assert v['complete_sha256']==io.sha(root/'complete.json') and v['lock_sha256']==io.sha(R/'lock.json')
    for name,h in c['files_sha256'].items():assert io.sha(root/name)==h,name
    runs=list(R.glob('execution_*.json'));audits=list(R.glob('verification_execution_*.json'));assert len(runs)==len(audits)==1
    e,a=io.read(runs[0]),io.read(audits[0]);assert e['returncode']==a['returncode']==0
    report=io.B/'ROUND56_RESULTS.md';assert not report.exists() and not (R/'closure_receipt.json').exists()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==4 and not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
    synthetic=sum(float(s.get('time',0)) for s in suites)
    failed_synthetic=sum(float(s.get('time',0)) for s in ET.parse(R/'pytest_attempt1_failed.xml').getroot().iter('testsuite'))
    cost=dict(CPU_controller_seconds=e['CPU_wall_seconds'],CPU_independent_audit_seconds=a['CPU_wall_seconds'],
        CPU_controller_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],synthetic_seconds_separate=synthetic,
        synthetic_failed_setup_seconds_separate=failed_synthetic,
        GPU_hours=0,new_fits=0,source_forward=0,new_graphs=0,new_random_draws=0,new_condition_statistics=4,
        reused_reference_condition_statistics=2,prior_costs_excluded=True,not_end_to_end=True)
    lines=['# R56：R55的5.94%开发增益稳定性审计','',
        f"固定Core方案四设置平均相对PCC仍为 **{primary['mean_relative_PCC_pct']:+.6f}%**。本轮无新模型、选参或分数提升，只审计已固定预测的样本组成敏感性。联合四设置区间资格 **{d['joint_interval_eligible']}**，因此不能宣称在未知独立数据上稳定超过5%。",'',
        '原R51的319全局输入关联闭包和2000次抽样权重按SHA复用。原TD24及Core50的整体/删组/bootstrap单模型统计复用，只为R55四条件新算，再形成七配对；独立验收从逐行预测用显式中心化复算所有新配对。两个Mix设置共享469响应和相同权重。','',
        '|设置|样本数|非空组|有效组数|最大组占比|条件区间资格|','|---|---:|---:|---:|---:|---|']
    for s,q in d['qualification'].items():lines.append(f"|{s}|{q['n']}|{q['components']}|{q['effective_components']:.3f}|{100*q['largest_fraction']:.2f}%|{q['eligible']}|")
    lines+=['','|设置|方法|对照|ΔPCC|相对PCC增益|相对增益条件95%范围|MSE下降|删组相对增益范围|','|---|---|---|---:|---:|---|---:|---|']
    for r in d['summaries']:
        leave=None if r['leave_component_min'] is None else [r['leave_component_min'],r['leave_component_max']]
        lines.append(f"|{r['setting']}|{r['method']}|{r['reference']}|{r['delta_PCC']:+.6f}|{r['relative_PCC_pct']:+.3f}%|{span(r['relative_PCC_interval']['percentile95'])}|{r['MSE_reduction_pct']:+.3f}%|{span(leave)}|")
    lines+=['','|方法|对照|四设置平均相对PCC|条件95%范围|删组均值范围|有效/无定义删组数|','|---|---|---:|---|---|---:|']
    for r in d['macro']:
        leave=None if r['leave_component_min'] is None else [r['leave_component_min'],r['leave_component_max']]
        lines.append(f"|{r['method']}|{r['reference']}|{r['mean_relative_PCC_pct']:+.6f}%|{span(r['interval']['percentile95'])}|{span(leave)}|{r['leave_component_valid']}/{r['leave_component_undefined']}|")
    lines+=['',
        '区间仅条件于固定拟合模型和给定关联定义，不能覆盖来源初始化、重训或反复开发选择不确定性。Takayuki的结构性缺失不是2000次算法失败；该设置及总体均不输出区间。组数资格是原保守操作规则，不是统计有效性定理。删组范围不是置信区间，也不能通过删组达到或证明目标。',
        '原分组仅表示序列关联，不是确认的gene/assay；本轮不改变原purged划分。四设置仍全保留，Wang/ALAS1仍排除；不把三个合格设置均值替代原目标。全部28设置配对、7宏配对、112原折比较和逐抽样/删组结果保存，不用某个有利比较替换Core主条件。',
        f"验收：4项新复用合同合成通过；{v['verified_prediction_appearances']}预测出现、{v['bootstrap_setting_metric_blocks']}合格draw-setting块、{v['leave_component_setting_blocks']}删组-setting块及112折比较通过独立显式统计、身份和SHA核验，最大数值差{v['maximum_absolute_statistic_difference']:.3g}。原图有R51独立验收，本轮通过原输入/组SHA复用，新建图0。",'',
        f"成本：CPU控制器{e['CPU_wall_seconds']:.6f}s＋独立验收{a['CPU_wall_seconds']:.6f}s＝{cost['CPU_controller_and_audit_seconds']:.6f}s；4项合成{synthetic:.6f}s、首次全局临时目录权限导致setup失败{failed_synthetic:.6f}s另列。失败XML保留，仅改为工作区内新的临时目录后4项通过，科学代码与测试断言未改。新拟合、source forward、PCA、GPU均0，历史训练缓存、启动/开发/登记/收尾不含，不称端到端耗时。",'',
        '开发数值门槛已达到；独立确认仍未证实。source keep-overlap/context暴露、source42单种子、反复开发和两个Mix设置非独立的限制保持，不宣称普遍5%或SOTA。完成审计应将数值达标与确认缺口分列，不能以重复旧数据抽样补足新数据证据。',
        f"科学锁SHA {io.sha(R/'lock.json')}；结果complete SHA {io.sha(root/'complete.json')}；独立验收SHA {io.sha(R/'verification.json')}。"]
    io.write(R/'cost_receipt.json',cost)
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    io.write(R/'bootstrap_qualification_notes.json',dict(Takayuki_bootstrap_metrics='Not evaluated: one input-dependence component.',
        four_setting_bootstrap_metrics='Not evaluated: the fixed four-setting eligibility fails.',
        ineligible_invalid_draws='Structural missingness, not numerical bootstrap failures.',
        numerical_failures=0,actual_evaluated_draw_setting_blocks=6000,reference_statistics_reused=True,new_fits=0))
    files=[R/'lock.json',root/'complete.json',R/'verification.json',R/'cost_receipt.json',R/'bootstrap_qualification_notes.json',report]+[R/n for n in io.CODE]+runs+audits
    manifest={p.relative_to(io.B).as_posix():io.sha(p) for p in files}
    for name,h in manifest.items():assert io.sha(io.B/name)==h,name
    io.write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,files_sha256=manifest,result_files_verified=len(c['files_sha256']),
        goal_complete=False,development_threshold_met=True,confirmation_complete=False,no_live_execution_mutex=True,CPU_close_seconds=time.monotonic()-start))
    io.write(io.B/'GOAL_TURN_R56_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        genuine_external_blocker=False,no_active_handle=True,new_efficiency_heads=0,new_stability_evidence=True,
        primary_relative_PCC_gain_pct=primary['mean_relative_PCC_pct'],joint_interval_eligible=d['joint_interval_eligible'],
        closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)))
    with (io.B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write(f"\n\nR56固定R55预测稳定性闭合：沿用319组/2000权重，四新条件统计与两旧参考复用；22788预测出现、6000合格抽样块/1276删组块/112原折比较全部独立验收。主增益{primary['mean_relative_PCC_pct']:+.6f}%，四设置联合区间资格{d['joint_interval_eligible']}；不替换分组/样本/目标。无新拟合/forward/GPU，CPU控制器+验收{cost['CPU_controller_and_audit_seconds']:.6f}s。数值已达标、独立确认仍不足，goal active；上一轮及本轮progress。见ROUND56_RESULTS.md。\n")
    print(dict(status='closed',primary_gain=primary['mean_relative_PCC_pct'],joint_interval_eligible=d['joint_interval_eligible'],
        CPU_seconds=cost['CPU_controller_and_audit_seconds'],closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)),flush=True)


if __name__=='__main__':main()
