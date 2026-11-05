"""Close the completed bounded pilot; no fits, predictions or test evaluation."""
import hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
start=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'closure_receipt.json').exists()
d=read(R/'diagnostic.json');v=read(R/'verification.json');lock=read(R/'lock.json')
assert v['passed'] and not v['qualified'] and not d['qualified'] and v['all_numerically_qualified']
assert v['lock_sha256']==sha(R/'lock.json') and v['diagnostic_sha256']==sha(R/'diagnostic.json')
for name,h in lock['files'].items():assert sha(R/name)==h,name
for name,h in lock['dependencies'].items():assert sha(Path(name))==h,name
for name,h in d['pilot_files_sha256'].items():assert sha(R/'pilot'/name)==h,name
assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
executions=[(p,read(p)) for p in sorted(R.glob('execution_*.json'))]
audits=[(p,read(p)) for p in sorted(R.glob('verification_execution_*.json'))]
assert len(executions)==len(audits)==1 and all(x['returncode']==0 for p,x in executions+audits)
suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==10 and not any(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites)
run_seconds=executions[0][1]['CPU_wall_seconds'];audit_seconds=audits[0][1]['CPU_wall_seconds']
test_seconds=sum(float(s.get('time',0)) for s in suites)
cost=dict(CPU_controller_seconds=run_seconds,CPU_audit_outer_seconds=audit_seconds,CPU_audit_inner_seconds_included=v['CPU_wall_seconds'],
          CPU_controller_plus_audit_seconds=run_seconds+audit_seconds,synthetic_tests_seconds_separate=test_seconds,
          new_heads=120,scientific_weighted_linear_solves=d['weighted_ridge_solves'],independent_augmented_audits=120,
          GPU_hours=0,source_forward=0,new_PCA_fits=0,new_KMeans_fits=0,
          excluded_cost='Earlier source training/export, earlier rounds, process startup, coding, and report/closure are separate.',
          execution_sha256={p.name:sha(p) for p,x in executions+audits})
save(R/'cost_receipt.json',cost)
core=[r for r in d['records'] if r['source']=='core_real' and r['mode']=='robust']
lines=['# R24：最坏输入群体风险未改善整体迁移',
       '', '2026-10-15。固定损失对照已完成并独立验收；资格false，结束该分支，不运行新的外层四设置评价，不调整群数/正则/权重或门槛。此前完整四设置主结果仍为R22的+2.784%，目标5%未达到。',
       '', '本轮复用旧R23分区，只在训练时重加权；部署仍一个40D头，没有新MoE、路由分支或来源前向。对照balanced为两群等权，robust为最坏群体损失。下表是三个outer-fold0 fit池内部OOF相对原R4固定alpha100/weight1的变化，PCC差为绝对值，不是目标相对百分比。',
       '', '|训练池|Core robust ΔPCC|MSE相对下降|', '|---|---:|---:|']
for r in core:lines.append(f"|{r['setting']}|{r['delta_PCC_vs_R4']:+.6f}|{r['MSE_reduction_vs_R4_pct']:+.4f}%|")
lines += ['', 'Core在Huesken和Mixset下降，Takayuki改善；预定性能门槛和相对balanced的配对门槛均未通过。Takayuki的TD-only稳健控制也改善且ΔPCC更大，所以不能将该队列变化单独归因为miRNA来源监督。',
          '', '|训练池|表示|损失|内部PCC|ΔPCC相对各自R4固定参考|MSE相对下降|', '|---|---|---|---:|---:|---:|']
for r in d['records']:
    lines.append(f"|{r['setting']}|{r['source']}|{r['mode']}|{r['metrics']['PCC']:.6f}|{r['delta_PCC_vs_R4']:+.6f}|{r['MSE_reduction_vs_R4_pct']:+.4f}%|")
lines += ['', 'balanced的微小改善也不是单独可采用的结论：它是预定辅助条件，没有通过本轮主条件门槛，不事后挑选有利条件或来源。',
          '', '解释边界：输入分区稳定，并不意味着它们代表可纠正的测量偏移。最坏群体训练可能牺牲平均拟合、放大不可约噪声；本结果与这些解释相容，但未证明哪一个是原因。不能因此断言所有group DRO或MoE无效。',
          '', f"验收：120科学头、{d['weighted_ridge_solves']}次内部带权线性求解、18392条内部OOF，全部数值合格；另120次独立增广最小二乘为审计，不称新增科学条件。最大系数差{v['max_independent_coefficient_difference']:.3g}，最大预测差{v['max_prediction_difference']:.3g}，最大相对primal/dual gap {v['max_relative_duality_gap']:.3g}。全部源码/输入/旧投影/分区/产物SHA读回通过。",
          '', f"CPU控制器{run_seconds:.6f}s，独立验收外层{audit_seconds:.6f}s（已包含内部{v['CPU_wall_seconds']:.6f}s），两阶段合计{run_seconds+audit_seconds:.6f}s。10项合成验收{test_seconds:.6f}s另计。开发/启动/关闭另计；旧源训练和导出不在上述秒数中。新GPU、source forward、PCA、KMeans拟合均0。",
          '', '所有控制器/验收返回码0，无本轮未结算任务或执行互斥文件。原goal active未完成。前一解释轮对数值优化无新进展；本轮新增可复核证据并关闭假设分支，属于progress，无外部阻断。source43导出范围问题仍待用户回复，本轮未启动新导出。',
          '', '限制仍包括三个fit池反复开发、单source42、keep-overlap/context暴露、非gene隔离，不能据训练内部变化宣称外部泛化、普遍迁移或SOTA。',
          '', '方法背景：[Sagawa等，Group DRO](https://arxiv.org/abs/1911.08731)。本轮借用最坏群体风险目标，使用已固定输入分区和凸Ridge，不复现原文神经网络，也不把分区等同生物机制。',
          '', f"科学锁{sha(R/'lock.json')}；诊断{sha(R/'diagnostic.json')}；独立验收{sha(R/'verification.json')}。", '']
report=ROOT/'ROUND24_RESULTS.md'
with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
paths=[R/'lock.json',R/'diagnostic.json',R/'diagnostic_rows.csv',R/'verification.json',R/'cost_receipt.json',Path(__file__),report]
paths += [R/n for n in lock['files']]+[p for p,x in executions+audits]
files={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
for name,h in files.items():assert sha(ROOT/name)==h
save(R/'closure_receipt.json',dict(status='closed',qualified=False,goal_complete=False,readback_passed=True,
    no_live_execution_mutex=True,files_sha256=files,all_pilot_files_verified=len(d['pilot_files_sha256']),CPU_close_seconds=time.monotonic()-start))
save(ROOT/'GOAL_TURN_R24_AUDIT.json',dict(previous_turn='no_progress_explanation_only',this_turn='progress',
    concrete_progress='Completed and independently verified bounded worst-group-risk pilot; rejected by fixed criteria.',
    objective_achieved=False,goal_status='active',new_outer_test_evaluation=False,source43_scope_reply_received=False,
    closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report),CPU_only=True,no_active_handle=True,
    genuine_external_blocker=False,do_not_repeat='No rerun or grid expansion of R24; R23 rejected 56D MoE remains unrun.'))
with (ROOT/'PROGRESS.md').open('a',encoding='utf-8') as f:
    f.write('\n\nR24最坏输入群体风险单头已闭合：120头/526带权线性求解/18392内部OOF，全部全局数值证书与独立增广lstsq通过。Core在Hu/Mix/Tak相对R4固定参考ΔPCC −.025069/−.015765/+.024050，MSE下降−4.4043%/−5.0769%/+3.2052%；主与paired gate均false，不运行新外层、不调群体/权重/正则。Tak TD-only也改善，不归因为miRNA监督。CPU成本见ROUND24_RESULTS.md与cost_receipt；新GPU/forward/PCA/KMeans0。前一解释轮no progress，本轮progress，无活进程/阻断，goal active未达5%。\n')
print(json.dumps(dict(closed=True,qualified=False,closure_sha256=sha(R/'closure_receipt.json'),cost=cost)))
