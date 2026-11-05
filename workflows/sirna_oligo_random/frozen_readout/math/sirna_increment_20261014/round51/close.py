"""Publish the complete audited sensitivity result without changing the goal."""
import sys,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import run as io

def value(x,d=3):return '/' if x is None else f'{x:.{d}f}'
def span(x):return '/' if x is None else f'[{x[0]:+.3f}, {x[1]:+.3f}]'

def main():
    start=time.monotonic();io.validate();v=io.read(R/'verification.json');assert v['passed'] and not (R/'execution.mutex').exists()
    report=io.B/'ROUND51_RESULTS.md';assert not report.exists() and not (R/'closure_receipt.json').exists()
    c=io.read(R/'results/complete.json')
    for name,h in c['files_sha256'].items():assert io.sha(R/'results'/name)==h,name
    runs=list(R.glob('execution_*.json'));audits=list(R.glob('verification_execution_*.json'))
    assert len(runs)==len(audits)==1
    e,a=io.read(runs[0]),io.read(audits[0]);assert e['returncode']==a['returncode']==0
    d=io.read(R/'results/diagnostic.json');primary=d['macro'][0]
    synthetic=sum(float(x.get('time',0)) for x in ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    cost=dict(CPU_controller_seconds=e['CPU_wall_seconds'],CPU_independent_audit_seconds=a['CPU_wall_seconds'],
        CPU_controller_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],synthetic_seconds_separate=synthetic,
        GPU_hours=0,new_fits=0,source_forward=0,prior_costs_excluded=True,not_end_to_end=True)
    lines=['# R51：4.19%增量的固定预测稳定性审计','',
        f"原预定Core方案四设置平均相对PCC仍为 **{primary['mean_relative_PCC_pct']:+.4f}%**，没有新增模型或提高成绩，5%目标尚未达到。本轮完成输入关联闭包的配对稳定性审计，用于区分点估计与可支持的稳定性结论。",'',
        f"三个队列3329独立输入构成{d['global_components']}个全局序列闭包。沿用guide19 Hamming≤2或共享canonical target25的关系后取传递闭包，仅用于本轮依赖敏感性；不修改原训练或划分、不等同于gene/assay。两个Mix设置共享同一469条响应及每次权重，不能当作两个独立外部队列。",'',
        '|设置|样本数|非空闭包数|有效组数|最大组占比|条件bootstrap资格|',
        '|---|---:|---:|---:|---:|---|']
    for s,q in d['qualification'].items():lines.append(f"|{s}|{q['n']}|{q['components']}|{q['effective_components']:.3f}|{100*q['largest_fraction']:.2f}%|{q['eligible']}|")
    lines+=['',
        '资格固定为至少10闭包、有效组数≥10、最大组占比≤50%，仅是保守操作标准而非有效性的理论保证。组数过少或大小极不均衡时不退回逐行bootstrap。组内及跨组仍可能存在未识别的实验依赖。',
        f"四设置联合区间资格：**{d['joint_interval_eligible']}**。不合格时总体区间留空，不能用其他三个设置代替原四设置目标。",'',
        '|设置|方法|参照|ΔPCC|相对PCC增益|条件95%百分位范围|逐闭包删除增益范围|删组无定义数|',
        '|---|---|---|---:|---:|---|---|---:|']
    for r in d['summaries']:
        leave='/' if r['leave_component_min'] is None else f"[{r['leave_component_min']:+.3f}, {r['leave_component_max']:+.3f}]"
        lines.append(f"|{r['setting']}|{r['method']}|{r['reference']}|{r['delta_PCC']:+.6f}|{r['relative_PCC_pct']:+.3f}%|{span(r['relative_PCC_interval']['percentile95'])}|{leave}|{r['leave_component_undefined']}|")
    lines+=['',
        '|方法|参照|四设置平均相对PCC|条件95%范围|逐闭包删除均值范围|有效/无定义删组数|',
        '|---|---|---:|---|---|---:|']
    for r in d['macro']:
        leave='/' if r['leave_component_min'] is None else f"[{r['leave_component_min']:+.3f}, {r['leave_component_max']:+.3f}]"
        lines.append(f"|{r['method']}|{r['reference']}|{r['mean_relative_PCC_pct']:+.4f}%|{span(r['interval']['percentile95'])}|{leave}|{r['leave_component_valid']}/{r['leave_component_undefined']}|")
    lines+=['',
        '百分位范围仅条件于已固定的模型/预测，不能覆盖重新训练、来源初始化或反复开发选择的不确定性；本轮在R50结果已知后登记，不报告p或显著性。逐闭包删除范围是样本组成敏感性，不是置信区间；删除使均值跨过5%也不构成目标达成，不允许借此删除样本、选择来源或调参。',
        '若某设置由很少的序列闭包支撑，缺少的是能够支撑独立组泛化的证据；无法通过反复抽取同一批行补足。这不证明模型无效，也不说明该闭包是真实同一基因或实验批次。下一步改进必须仍在训练数据内固定方案，并保留全四设置，不能把本轮外层敏感性用作参数搜索。',
        'Full、精确initial、同处理TD和上一版Core完整报告；不能只展示Core有利比较。112行原外折配对指标及全部2000次共享组权重、有效/无定义统计、所有逐组删除结果保存在results中，不把16折视为16次独立实验。','',
        f"验收：10项纯合成首次通过；原benchmark密集邻接＋SciPy闭包独立复核，{v['verified_prediction_appearances']}个预测出现、{v['bootstrap_setting_metric_blocks']}个有效资格draw-setting块、{v['leave_component_setting_blocks']}个删组-setting块、112折比较及所有身份/摘要/SHA通过。最大独立数值差{v['maximum_absolute_statistic_difference']:.3g}。",'',
        f"CPU控制器{cost['CPU_controller_seconds']:.6f}s＋独立验收{cost['CPU_independent_audit_seconds']:.6f}s＝{cost['CPU_controller_and_audit_seconds']:.6f}s；合成{synthetic:.6f}s另列。新拟合、source forward、GPU均0；历史训练/缓存、启动/登记/开发/收尾不含，不称端到端耗时。",'',
        'source keep-overlap/context暴露、单来源seed42、原purged非gene隔离、反复开发及两Mix设置不独立仍保留。不宣称SOTA、稳定达到5%或独立外部确认。目标active、未完成，本轮为新增稳定性证据而非新的性能模型。',
        f"科学锁SHA {io.sha(R/'lock.json')}；完整结果SHA {io.sha(R/'results/complete.json')}；独立验收SHA {io.sha(R/'verification.json')}。"]
    io.write(R/'cost_receipt.json',cost)
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    files=[R/'lock.json',R/'results/complete.json',R/'verification.json',R/'cost_receipt.json',report]+[R/n for n in io.CODE]+runs+audits
    manifest={p.relative_to(io.B).as_posix():io.sha(p) for p in files}
    for name,h in manifest.items():assert io.sha(io.B/name)==h,name
    io.write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,files_sha256=manifest,result_files_verified=len(c['files_sha256']),
        goal_complete=False,no_live_execution_mutex=True,CPU_close_seconds=time.monotonic()-start))
    io.write(io.B/'GOAL_TURN_R51_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        genuine_external_blocker=False,no_active_handle=True,new_efficiency_heads=0,new_stability_evidence=True,
        current_primary_relative_PCC_gain_pct=primary['mean_relative_PCC_pct'],joint_interval_eligible=d['joint_interval_eligible'],
        closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)))
    with (io.B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write(f"\n\nR51固定R50预测稳定性审计闭合：3329输入/{d['global_components']}全局关联闭包、六条件22788预测出现；2000次共享组权重，{v['bootstrap_setting_metric_blocks']}资格draw-setting、{v['leave_component_setting_blocks']}删组-setting、112原折比较，原benchmark独立图与显式中心化/全SHA通过。联合四设置区间资格{d['joint_interval_eligible']}，不退回逐条抽样或排除设置；点估计仍+4.190823%，未达到5%。无新模型/来源forward/GPU，CPU数值+验收{cost['CPU_controller_and_audit_seconds']:.6f}s、合成另列。前轮与本轮progress，无活句柄/真实外部阻断；不据删组结果选择样本或参数。见ROUND51_RESULTS.md。\n")
    print(dict(status='closed',mean_relative_PCC=primary['mean_relative_PCC_pct'],joint_interval_eligible=d['joint_interval_eligible'],
        CPU_seconds=cost['CPU_controller_and_audit_seconds'],closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)),flush=True)

if __name__=='__main__':main()
