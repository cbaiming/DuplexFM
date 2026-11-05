"""Summarize only independently verified R5 artifacts, without new fits."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;R=ROOT/'round5';D=R/'results'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,text):
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
v=read(R/'verification.json');assert v['passed'] and v['complete_sha256']==sha(D/'complete.json')
compat=read(R/'artifact_metric_execution_lock.json')
assert compat['original_lock_sha256']==sha(R/'lock.json') and compat['original_result_sha256']==sha(D/'complete.json')
for name,h in compat['files'].items():assert sha(R/name)==h,name
for name,h in read(D/'complete.json')['files_sha256'].items():assert sha(D/name)==h,name
execution=[read(p) for p in R.glob('execution_*.json')];assert len(execution)==1 and execution[0]['returncode']==0
verified=read(R/'artifact_metric_execution_receipt.json');assert verified['returncode']==0 and verified['verification_sha256']==sha(R/'verification.json')
summary=table(D/'summary.csv');goals=read(D/'goal_summary.json');selections=table(D/'selection.csv')
assert len(summary)==32 and len(goals)==8 and len(selections)==128
lookup={(r['setting'],r['condition']):r for r in summary}
settings=('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix')
primary='core_real_AdaptiveBilinear';secondary='full_real_UniformBilinear';tdcontrol='TDQuadratic_UniformQuadratic'
diagnostic=read(ROOT/'r4_training_diagnostic.json')
report=['# R5：条件交互有局部收益，固定主方案未达标','',
    '2026-10-15。预定主方案为core_real_AdaptiveBilinear。只使用CPU和既有冻结表示，新增固定TD24×PCA16交互，单424D有正则回归头；未重训来源、未占GPU、未使用其他模型预测。预定四设置与TD24_Ridge_CV基线不变。', '',
    '主方案四设置PCC相对增益均值 **+1.3915%**，未达到5%，也弱于R4预定主方案的+2.6812%。所有条件均未达到5%；目标保持active，不选测试赢家替换主方案。', '',
    '| 条件 | Huesken | Mixset | Takayuki | Huesken→Mixset | 相对PCC均值 |', '|---|---:|---:|---:|---:|---:|']
for g in goals:
    values=[float(lookup[s,g['condition']]['PCC_relative_gain_pct']) for s in settings]
    report.append('| '+g['condition']+' | '+' | '.join(f'{x:+.4f}%' for x in values)+f" | {g['mean_relative_PCC_gain_pct']:+.4f}% |")
report += ['', '可讨论的次要结果：Full＋固定强度二阶交互在四项PCC均为正，平均+3.1633%。但Mixset仅+0.7347%、跨队列仅+0.1521%，没有区间或多来源种子确认，不能称已稳定非劣，更不能改称本轮预定主方案。', '',
    '| 设置 | 固定TD24 PCC | 预定Core主方案 | Full固定强度对照 | TD24自身二阶对照 |', '|---|---:|---:|---:|---:|']
for s in settings:
    report.append(f"| {s} | {float(lookup[s,primary]['TD24_PCC']):.6f} | {float(lookup[s,primary]['PCC']):.6f} | {float(lookup[s,secondary]['PCC']):.6f} | {float(lookup[s,tdcontrol]['PCC']):.6f} |")
report += ['',
    'TD24自身二阶的固定强度对照平均+1.3448%，表明部分收益可由TD24非线性解释。Full固定交互在本次四项均高于这个对照，但容量也不同（425 vs 325系数），且没有Full精确初始化的同轮二阶对照，所以不能把全部差异单独归因miRNA监督。Core有精确初始化对照，真实训练的优势仍有队列依赖。', '',
    '自适应选择并不总优于固定强度：Core从Uniform的+2.6221%降至Adaptive的+1.3915%，Full从+3.1633%降至+1.2433%；TD24自身二阶也从+1.3448%降至−0.3329%。更大的候选集合在训练内必然不差，却没有转化成外层优势。这支持把选参方差作为下一项诊断，不证明任何特定路由或正则一定更好。', '',
    '固定强度仍有alpha五候选，由原fit内五折pooled PCC选择；自适应有25候选。不能混淆“固定辅助强度”和“不使用验证标签”。全部模型使用同一fit/内折，原outer validation未使用。没有将各测试集最好的来源拼成模型。', '',
    '训练内证据：R4固定alpha100、weight1的Core线性补充在Huesken为22/25内折MSE改善、Mixset为6/25、Takayuki18/25；跨队列训练池的5/5内折均改善，但外层Mixset下降。该诊断来自已保存OOF，未训练新模型；各outer fit有交叠，因此不是独立队列证据。它提示训练内一致性与跨域有效性需要分开看待。', '',
    '验收已闭合：8000个内层线性解、128最终头、30384 OOF行全部通过独立正规方程、PCA/交互列、训练统计、选参、覆盖及指标核对。TD-only原五候选最大逐行差8.33e−15，最终重建预测最大差8.88e−16，原科学与数据SHA不变。', '',
    '首次独立验收在Mixset/Core Uniform的Spearman失败：两条精确并列预测被独立浮点重建的约1e−16尾差拆开，Spearman相差6.589e−6。已保留原验收器、失败与原锁；新增显式入口保持所有原数值检查/容差，对通过逐行核验的实际CSV预测精确复算指标。没有放宽阈值、改预测、重训或更改主PCC指标。6项原合成测试及2项秩兼容测试通过。', '',
    f"CPU拟合控制器{execution[0]['CPU_wall_seconds']:.3f}s；成功独立验收子进程{v['CPU_wall_seconds']:.3f}s、包裹全过程{verified['CPU_wall_seconds']:.3f}s（重叠，不相加）。首次失败验收完整耗时未记录，仅知工具已等待至少10.002s，不能当0；另外有诊断/测试及启动开销。全部新GPU耗时0、来源前向0。这不含历史来源训练/缓存导出成本。", '',
    f"科学锁：`{sha(R/'lock.json')}`。",f"完成回执：`{sha(D/'complete.json')}`。",
    f"显式验收执行锁：`{sha(R/'artifact_metric_execution_lock.json')}`。",f"独立验收：`{sha(R/'verification.json')}`。", '',
    '后续保持5%目标与基线，先检查训练内部候选排序是否对少数样本/组敏感、以及可复核的选择稳健性；不据本轮测试再扩大权重/alpha/PCA网格，也不自动启用最佳Full来源作为确认主方案。若下一轮采用它，必须明示是看过本轮后的新探索性候选，另锁协议。', '',
    '限制：单source42；源keep-overlap及context暴露；当前purged_sequence不是gene隔离；同一Mixset进入两个设置；这些数据反复开发。已有正结果不等于新外部确认或SOTA。全部PCC/Spearman/MSE/MAE/AP/AUROC/F1在round5/results/summary.csv，逐行预测及所有选择一并保留。','']
save(ROOT/'ROUND5_RESULTS.md','\n'.join(report))
derived=dict(kind='verified_post_result_description',primary_condition=primary,
    all_eight_conditions=goals,no_test_based_model_choice=True,
    original_lock_sha256=sha(R/'lock.json'),result_sha256=sha(D/'complete.json'),verification_sha256=sha(R/'verification.json'),
    reported_secondary_not_new_primary=secondary,goal_met=False,
    cost=dict(fit_CPU_wall_seconds=execution[0]['CPU_wall_seconds'],success_verification_outer_wall_seconds=verified['CPU_wall_seconds'],
              failed_verifier_exact_wall_seconds=None,failed_verifier_observed_lower_bound_seconds=10.0018653,GPU_hours=0))
save(R/'derived_description.json',json.dumps(derived,ensure_ascii=False,indent=2)+'\n')
save(ROOT/'GOAL_TURN_R5_AUDIT.json',json.dumps(dict(previous_turn='progress',current_turn='progress',goal_status='active',
    objective_satisfied=False,reason='R5 bounded experiment and independent audit completed; no condition reaches 5%. New evidence separates conditional source information from nonlinear TD and selection effects.',
    next_action='training-internal candidate selection stability diagnosis before another locked experiment',
    running_processes_owned_by_this_round=False,requires_user_input=False,
    scientific_result_sha256=sha(D/'complete.json'),report_sha256=sha(ROOT/'ROUND5_RESULTS.md')),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(report='ROUND5_RESULTS.md',primary_gain_pct=next(x['mean_relative_PCC_gain_pct'] for x in goals if x['primary']),goal_met=False)))
