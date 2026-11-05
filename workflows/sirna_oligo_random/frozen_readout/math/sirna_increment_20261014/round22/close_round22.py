"""Freeze the fully audited four-setting result and all failure costs."""
import csv,hashlib,json,time
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
start=time.monotonic();lock=read(R/'lock.json');v=read(R/'verification.json');complete=read(R/'results_v2/complete.json')
assert v['passed'] and v['complete_sha256']==sha(R/'results_v2/complete.json') and v['lock_sha256']==sha(R/'lock.json')
assert not (R/'execution.mutex').exists() and not (R/'verify_compat.mutex').exists()
for n,h in lock['files'].items():assert sha(R/n)==h,n
for n,h in lock['dependencies'].items():assert sha(Path(n))==h,n
for n,h in complete['files_sha256'].items():assert sha(R/'results_v2'/n)==h,n
for name in ('execution_lock.json','verification_metadata_lock.json'):
    for n,h in read(R/name)['files_sha256'].items():assert sha(R/n)==h,n
mr=read(R/'verification_metadata_receipt.json');assert mr['verification_sha256']==sha(R/'verification.json')
attempts=[]
for pat in ('execution_*.json','compat_execution_*.json','compat_verification_execution_*.json'):
    for p in R.glob(pat):
        c=read(p)
        if c.get('returncode') is not None:attempts.append(dict(file=p.name,**c))
assert len(attempts)==4 and sorted(a['returncode'] for a in attempts)==[0,0,1,1]
summary=table(R/'results_v2/summary.csv');goals=read(R/'results_v2/goal_summary.json')
primary=next(g for g in goals if g['primary']);assert primary['condition']=='core_real_sparse' and not primary['reaches_development_5pct']
assert not any(g['reaches_development_5pct'] for g in goals)
lines=['# R22：四设置平均增益2.784%，仍未达到5%','',
 '2026-10-15。R21训练内初筛通过后，以固定五alpha和原内部PCC选择协议完成四设置评价。完整来源192D采用联合坐标L1+L2，TD24不加L1；单216D头，全部CPU，无新来源训练或前向。主条件始终Core sparse，未根据测试换源或组合输出。','',
 '|设置|TD24 PCC|Core sparse PCC|PCC相对增益|MSE相对下降|','|---|---:|---:|---:|---:|']
for r in summary:
    if r['condition']=='core_real_sparse':lines.append(f"|{r['setting']}|{float(r['TD24_PCC']):.6f}|{float(r['PCC']):.6f}|{float(r['PCC_relative_gain_pct']):+.4f}%|{float(r['MSE_relative_reduction_pct']):+.4f}%|")
lines+=['', '四设置相对PCC的等权均值为+2.7840008700%，原R4预定Core Adaptive为+2.6811767109%，仅高0.102824个百分点，不能解释成显著提升或已达5%。Huesken改善，Takayuki的增益比R4小；Mixset及Hu→Mix仍为负。内部资格通过没有转化为足够大的四设置增量。','',
 '|固定条件|四设置相对PCC均值|负增益设置|','|---|---:|---|']
for g in goals:lines.append(f"|{g['condition']}|{g['mean_relative_PCC_gain_pct']:+.4f}%|{', '.join(g['negative_settings']) or '无'}|")
lines+=['', '配对dense的Core均值+2.2403%，sparse+2.7840%；Full均值也由+0.4615%到+1.4616%；初始化由−1.5728%到+1.0542%。稀疏读出减少了一部分估计问题，但收益并非miRNA训练独有。不能根据六条件中个别较优项替换主方案。', '',
 '|设置|来源/读出|PCC|Spearman|MSE|MAE|PCC相对TD24|','|---|---|---:|---:|---:|---:|---:|']
for r in summary:lines.append(f"|{r['setting']}|{r['condition']}|{float(r['PCC']):.6f}|{float(r['Spearman']):.6f}|{float(r['MSE']):.6f}|{float(r['MAE']):.6f}|{float(r['PCC_relative_gain_pct']):+.4f}%|")
lines+=['', '该结果不支持继续扩大这一L1/alpha网格来寻找5%。每个条件严格复用原样本和标签预算，候选仅原五alpha；所有标准化只用对应训练分区，原outer validation未使用，初筛与本轮都属于重复开发。来源keep-overlap/context暴露、单来源42、非gene隔离、两个Mixset设置不独立等限制仍在，不宣称盲确认或SOTA。','',
 '相同读出下的训练来源对精确初始化只作描述，不替代来源多种子确认：','',
 '|设置|Core sparse − initial sparse PCC（绝对）|','|---|---:|']
for r in summary:
    if r['condition']=='core_real_sparse':
        c=next(x for x in summary if x['setting']==r['setting'] and x['condition']=='initial_core_sparse')
        lines.append(f"|{r['setting']}|{float(r['PCC'])-float(c['PCC']):+.6f}|")
lines+=['',
 f"独立验收：2400内层候选、96最终头、22788逐行预测；所有数值合格。1248个dense独立联合solve，最大系数差{v['max_dense_coef_difference']:.3g}；最大完整KKT{v['max_KKT']:.3g}；预测重建差{v['max_prediction_difference']:.3g}；原TD24候选匹配差{v['max_TD_baseline_difference']:.3g}。源/数据/划分/原锁和全部新输出SHA读回一致。PCC/Spearman/MSE/MAE/AP/AUROC/F1及每折alpha均完整保存于results_v2。",'',
 '两个实现错误及修复均保留：首次控制器因重复mode字段退出，只有一个候选certificate、无测试预测；独立兼容修改元数据构造并使用新输出results_v2。首次验收在所有检查后因support/s​​upports变量拼写错误退出，第二兼容只修复输出变量名。原代码/科学锁未改，容差/指标未改，失败未被删除。成功2496头之外另有1个首次失败前已计算候选，单列成本。','',
 '|执行阶段|返回码|CPU墙时秒|','|---|---:|---:|']
for a in attempts:lines.append(f"|{a['file']}|{a['returncode']}|{a['CPU_wall_seconds']:.6f}|")
lines+=['',f"上述四阶段合计{sum(a['CPU_wall_seconds'] for a in attempts):.6f}秒。成功验收内部{v['CPU_wall_seconds']:.6f}秒已经包含在外层，不重复相加；metadata receipt也是包裹计时。9合成0.90秒、3兼容0.05秒、2字段修复0.05秒另计，开发/进程启动/报告保存另计。R21初筛成本独立列于ROUND21_RESULTS.md，旧来源训练导出不含在这些CPU秒内。新GPU=0、新来源forward=0。",'',
 '当前没有本轮活进程。目标仍active，未完成，不因阶段结束标complete。原论文source43的CPU桥接范围回复尚未到来，本轮未导出新表示；当前结果不依赖该等待。后续需以可重复的新证据改变下一因素，不从已看过的测试结果选择队列专属方法。','',
 '方法背景：[Zou与Hastie，Elastic Net](https://doi.org/10.1111/j.1467-9868.2005.00503.x)。本轮以该思想构造带TD24不稀疏约束的凸读出，不是完整原论文算法复现。', '',
 f"科学锁{sha(R/'lock.json')}；完整结果{sha(R/'results_v2/complete.json')}；验收{sha(R/'verification.json')}。"]
report=ROOT/'ROUND22_RESULTS.md'
with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
cost=dict(attempts=attempts,total_CPU_wall_seconds=sum(a['CPU_wall_seconds'] for a in attempts),successful_scientific_heads=2496,discarded_failed_candidate_heads=1,
    independent_dense_verification_solves_per_attempt=1248,verification_attempts=2,GPU_hours=0,new_source_forward=0)
write(R/'cost_receipt.json',cost)
paths=[R/n for n in ('lock.json','execution_lock.json','verification_metadata_lock.json','verification_metadata_receipt.json','verification.json','results_v2/complete.json','cost_receipt.json','pytest.xml','compat_pytest.xml','metadata_pytest.xml')]
paths += [R/a['file'] for a in attempts]+[report,Path(__file__)]
write(R/'closure_receipt.json',dict(closed=True,goal_complete=False,goal_remains_active=True,primary_mean_relative_PCC_gain_pct=primary['mean_relative_PCC_gain_pct'],
    own_live_process=False,inner_heads=2400,final_heads=96,OOF_rows=22788,all_six_conditions_below_goal=True,GPU_hours=0,new_source_forward=0,
    files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},CPU_close_seconds=time.monotonic()-start))
for n,h in read(R/'closure_receipt.json')['files_sha256'].items():assert sha(ROOT/n)==h,n
write(ROOT/'GOAL_TURN_R22_AUDIT.json',dict(previous_goal_turn='no progress',current_goal_turn='progress',goal_status_read='active',goal_complete=False,
    blocker=False,own_live_process=False,evidence='R21 qualification and R22 all four-setting evaluation completed with independent numerical/selection/metric audit. Primary +2.784%, below5%.',
    pilot_closure_sha256=sha(ROOT/'round21/closure_receipt.json'),closure_sha256=sha(R/'closure_receipt.json')))
with (ROOT/'PROGRESS.md').open('a',encoding='utf-8') as f:f.write('\n\nR22在R21资格通过后完整四设置评价并验收：2400内层/96最终头/22788行，原五alpha同内层PCC选择及TD24基线不变。主Core sparse均值+2.7840%，逐设置+3.1431/−1.4706/+10.7119/−1.2485%，仍未达5%；Core dense+2.2403，Full dense/sparse+.4615/+1.4616，initial dense/sparse−1.5728/+1.0542，全低于5%。相比R4主方案仅+0.1028个百分点，不称显著改善。两次元数据/输出错误的原失败、费用及显式兼容锁均保留，科学代码/目标/容差不变，全部数值通过。四阶段CPU合计42.454秒，新GPU/forward0，详见ROUND22_RESULTS.md。目标active，当前无活进程/外部阻断，本轮R21+R22为progress；不继续本L1网格，不挑辅助来源或拼队列。\n')
print(json.dumps(dict(closed=True,goal_complete=False,primary=primary,closure_sha256=sha(R/'closure_receipt.json'))))
