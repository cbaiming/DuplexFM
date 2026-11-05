"""Publish verified R7 results; no scientific fitting or recomputation."""
import csv,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent;D=R/'round7'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def create(p,text):
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
def table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

v=read(D/'verification.json');c=read(D/'results/complete.json')
assert v['passed'] and v['complete_sha256']==sha(D/'results/complete.json')
assert v['lock_sha256']==sha(D/'lock.json') and not (D/'execution.mutex').exists()
executions=[read(p) for p in sorted(D.glob('execution_*.json'))]
audits=[read(p) for p in sorted(D.glob('verification_execution_*.json'))]
assert executions[-1]['returncode']==audits[-1]['returncode']==0
assert len(executions)==len(audits)==1
goals=read(D/'results/goal_summary.json');rows=table(D/'results/summary.csv')
settings=['Huesken','Mixset','Takayuki','Inter_Hu_to_Mix']
assert len(goals)==8 and len(rows)==32
g={x['condition']:x for x in goals};primary=g['core_real_AdaptiveKernel']
assert primary['primary'] and not any(x['reaches_development_5pct'] for x in goals)
cost=executions[0]['CPU_wall_seconds'];verifycost=audits[0]['CPU_wall_seconds']
lines=['# R7：平滑非线性读出未恢复稳定的迁移增益','',
    '2026-10-15。预定主方案Core Adaptive核读出的四设置相对PCC均值为'
    f" **{primary['mean_relative_PCC_gain_pct']:+.4f}%**，未达到5%；所有八个条件均未达到门槛。仅CPU、来源冻结、原数据划分及强TD24基线不变。",'',
    '| 条件 | Huesken | Mixset | Takayuki | Huesken→Mixset | 相对PCC均值 |',
    '|---|---:|---:|---:|---:|---:|']
for goal in goals:
    vals=[next(float(x['PCC_relative_gain_pct']) for x in rows if x['condition']==goal['condition'] and x['setting']==s) for s in settings]
    lines.append('| '+goal['condition']+' | '+' | '.join(f'{x:+.4f}%' for x in vals)+f" | {goal['mean_relative_PCC_gain_pct']:+.4f}% |")
lines += ['',
    '这轮没有支持“只要换成平滑非线性，就能恢复论文整体增益”。Core固定强度+1.7217%、自适应+0.9755%，均低于此前R4主方案+2.6812%。Full两条件也弱于R5的二阶固定强度条件；不据结果改带宽、基点数、正则或主来源。', '',
    '同类TDKernel固定强度在Huesken/Takayuki已有+2.7184%/+7.1276%，但Mixset−8.0304%、跨队列−0.8906%。因此部分非线性收益来自TD24自身，不能全部归因miRNA迁移。Core固定强度高于精确初始化的前三个设置，但跨队列更差；自适应Core在Mixset也低于初始化。两种模式下均缺普遍非劣证据。', '',
    '比较仍是一个联合88D头及同一个25候选集合；不是外部模型预测堆叠、残差集成或来源预测平均。所有64个核基点、带宽、中心化与方差只来自该块训练数据；三种来源均复用R4的训练内PCA16，另有同容量TD-only核控制。新来源前向/来源更新/GPU均为0。', '',
    f"验收：{c['inner_fits']}内层候选解、{c['final_heads']}最终头、{c['rows']}行全部核验；原TD24候选最大逐行差{v['max_TD24_candidate_prediction_difference']:.3g}，最终预测独立重建最大差{v['max_final_prediction_reconstruction_difference']:.3g}，归一正规方程残差{v['max_normal_residual']:.3g}。独立审计重建距离、谱阈值、训练归一化、PCA、选择与实际CSV完整指标，未重训头。", '',
    f"CPU控制器{cost:.3f}s；独立验收包裹全过程{verifycost:.3f}s（含内部{v['CPU_wall_seconds']:.3f}s，不相加）。6项先前数学合成+3项新增审计合成通过；新增3项0.87s，合成资格/开发/进程启动另计。历史来源训练/导出未计入这些小头成本，不能称端到端只需这些时间。", '',
    f"科学锁：{sha(D/'lock.json')}。完成回执：{sha(D/'results/complete.json')}。验收：{sha(D/'verification.json')}。", '',
    '结果是已反复开发数据上的单source42诊断，源keep-overlap/context暴露保留，purged_sequence不是gene隔离；两个Mixset设置相关，Takayuki相似分量退化不允许伪造分量bootstrap区间。没有新置信区间、多来源稳定性或盲外部确认，目标active未达到。', '',
    '下一步优先回到来源身份/旧协议桥接问题，避免继续扩读出网格。旧source43权重存在，但当前缺直接可复用的同输入已验收160D缓存；不重复盘点或冒称模型不存在。应先固定最小桥接协议与CPU可执行性，再考虑新特征提取。此次未启动新来源前向、远程任务或GPU工作。']
create(R/'ROUND7_RESULTS.md','\n'.join(lines)+'\n')
record=dict(goal_status='active',goal_achieved=False,previous_goal_turn='progress',this_goal_turn='progress',
    new_round='R7',CPU_only=True,GPU_hours=0,new_source_forward=0,
    primary_mean_relative_PCC_gain_pct=primary['mean_relative_PCC_gain_pct'],
    lock_sha256=sha(D/'lock.json'),complete_sha256=sha(D/'results/complete.json'),verification_sha256=sha(D/'verification.json'),
    controller_CPU_seconds=cost,verification_CPU_seconds=verifycost,blocking_condition=None,
    no_running_process=True,next_direction='Separate original-source/protocol bridge from additional readout search; not registered yet')
create(R/'GOAL_TURN_R7_AUDIT.json',json.dumps(record,ensure_ascii=False,indent=2)+'\n')
with (R/'PROGRESS.md').open('a',encoding='utf-8',newline='\n') as f:
    f.write('\nR7固定64基点低秩RBF单头已完成并独立验收。预定Core自适应均值+0.9755%，Core固定+1.7217%；Full固定+0.8190%/自适应+0.6621%；初始化固定−0.5300%/自适应+0.0912%；TD-only核固定+0.2312%/自适应−0.3733%。所有条件未达到5%，不扩核网格、不更换主方案。8000内层解/128最终头/30384行通过，原TD24候选逐行一致，源/划分/投影SHA不变。\n')
    f.write(f'控制器CPU{cost:.3f}s、独立验收总{verifycost:.3f}s（含内部计时），新GPU/来源前向0；6+3合成验收通过，无当前活进程。锁{sha(D/"lock.json")}，complete{sha(D/"results/complete.json")}。见ROUND7_RESULTS.md及GOAL_TURN_R7_AUDIT.json。用户关于原论文增益的详细对齐见PAPER_GAIN_COMPARISON_20261015.md。目标active未完成，本轮与上一轮均progress；下步优先最小旧来源/协议桥接，不自动展开读出网格。\n')
print(json.dumps(record,ensure_ascii=False,indent=2))
