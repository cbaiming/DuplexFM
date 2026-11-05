"""Close the fully verified R6 description, without new selection or training."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;R=ROOT/'round6';D=R/'results'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,text):
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
v=read(R/'verification.json');assert v['passed'] and v['complete_sha256']==sha(D/'complete.json')
c=read(D/'complete.json')
for name,h in c['files_sha256'].items():assert sha(D/name)==h,name
fit=[read(p) for p in R.glob('execution_*.json')];verify=[read(p) for p in R.glob('verification_execution_*.json')]
assert len(fit)==len(verify)==1 and fit[0]['returncode']==verify[0]['returncode']==0
summary=table(D/'summary.csv');goals=read(D/'goal_summary.json');oldgoals=read(ROOT/'round5/results/goal_summary.json')
old={r['condition']:r for r in oldgoals};lookup={(r['setting'],r['condition']):r for r in summary}
settings=('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix');primary='core_real_AdaptiveBilinear_Consensus'
report=['# R6：共识选择改善部分负增益，但不足以达到目标','',
    '2026-10-15。R5之后的独立有界开发轮；只改变训练内选参规则，候选模型/特征/损失/来源/划分和5%目标全部不变。没有占用GPU或新增来源前向。', '',
    '预定Core主方案平均相对PCC增益从R5的+1.3915%提高到 **+2.2061%**，仍低于5%，也不超过R4主方案的+2.6812%。全部新条件未达门槛，目标保持active。', '',
    '| 条件 | 原R5均值 | 共识选择均值 | 变化（百分点） |', '|---|---:|---:|---:|']
for g in goals:
    name=g['condition']
    if name=='TD24_Consensus':before=0.
    else:before=old[name.removesuffix('_Consensus')]['mean_relative_PCC_gain_pct']
    after=g['mean_relative_PCC_gain_pct'];report.append(f'| {name} | {before:+.4f}% | {after:+.4f}% | {after-before:+.4f} |')
report += ['', '| 设置 | 固定及匹配TD24 PCC | 共识Core PCC | 相对增益 | MSE相对下降 |', '|---|---:|---:|---:|---:|']
for s in settings:
    r=lookup[s,primary];report.append(f"| {s} | {float(r['TD24_PCC']):.6f} | {float(r['PCC']):.6f} | {float(r['PCC_relative_gain_pct']):+.4f}% | {float(r['MSE_relative_reduction_pct']):+.4f}% |")
report += ['',
    'Core主方案的变化集中于Mixset：原−3.5589%改善到−0.3007%，其余三个设置相同。但Full固定强度从+3.1633%下降到+2.2667%，Mixset由+0.7347%变成−2.8516%。共识只说明参数更常被训练内子视图选中，并不保证外推更准确；不能隐藏这个反例。', '',
    '同选择协议的TD24对照已逐折验证：五个alpha的共识选择在16外折全部等于原选择，原预测逐行相同，全部指标完全一致。因此此次主增益变化不是分母变弱。', '',
    '本轮仅重新拟合12个真正变更配置的最终头；116个R5最终头和16个TD24头严格复用。历史8000内层拟合未重跑。共识选择144项、最终条件144项、34182 OOF行全部独立验收；原样复用的系数与预测按字节核对，新头最大正规残差7.80e−17、独立预测重建最大差3.33e−16。5项合成测试通过，真实运行与首次独立验收均rc0。', '',
    f"新控制器CPU墙时{fit[0]['CPU_wall_seconds']:.3f}s，独立验收含回执{verify[0]['CPU_wall_seconds']:.3f}s（内部数值验收{v['CPU_wall_seconds']:.3f}s，互相重叠不能相加），另有启动/诊断/合成验收开销；新GPU0、来源前向0。这不包含历史8000拟合或来源训练/导出成本。", '',
    '训练内诊断表明，R5自适应Core原胜者平均在5个删除块视图中获3票，Uniform约4.75票；这一敏感性真实存在。不过改变共识选择后没有跨条件稳定改善，因此不继续扩大票数规则、重采样数或alpha/weight网格。R6保留为方法边界证据，不取代更好的已有参考。', '',
    '下一方向：检查固定PCA截断和当前加性/二阶读出是否限制了来源互补信息，再选一个训练内自足、有限的非线性读出对照。继续保留TD24自身同类读出及初始化控制，避免把一般非线性收益都归因DuplexFM。尚未登记或运行下一轮，不能声称已有改进结果。', '',
    '限制：5个删除块视图不是独立生物学重复、不是置信区间；Takayuki相似分量退化不能伪做component bootstrap。仍是单source42、反复开发四设置，源keep-overlap/context暴露及Mixset双设置依赖存在，当前purged_sequence非gene隔离。不能据本轮宣称SOTA、普遍非劣或已达到5%。', '',
    f"科学锁：`{sha(R/'lock.json')}`。",f"完成回执：`{sha(D/'complete.json')}`。",f"独立验收：`{sha(R/'verification.json')}`。", '',
    '全部九条件四设置PCC/Spearman/MSE/MAE/AP/AUROC/F1和逐行OOF保存在round6/results；源码、原五视图胜者、票数、选择及复用来源完整保留。','']
save(ROOT/'ROUND6_RESULTS.md','\n'.join(report))
save(ROOT/'GOAL_TURN_R6_AUDIT.json',json.dumps(dict(previous_turn='progress',current_turn='progress',goal_status='active',
    objective_satisfied=False,blocking_condition=None,running_processes_owned_by_this_round=False,
    completed='training-internal sensitivity diagnosis, registered consensus test, independent selection/reuse/normal-equation/metric audit',
    next_action='new bounded representation/readout hypothesis; do not extend consensus selection grid',
    primary_gain_pct=next(x['mean_relative_PCC_gain_pct'] for x in goals if x['primary']),
    result_sha256=sha(D/'complete.json'),verification_sha256=sha(R/'verification.json'),report_sha256=sha(ROOT/'ROUND6_RESULTS.md')),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(report='ROUND6_RESULTS.md',primary_gain_pct=next(x['mean_relative_PCC_gain_pct'] for x in goals if x['primary']),goal_met=False)))
