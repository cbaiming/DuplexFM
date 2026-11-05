"""Close already independently verified results; no fitting or prediction replay."""
from pathlib import Path
import csv,hashlib,json,time
R=Path(__file__).resolve().parent;ROOT=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
def save(p,v):
    with p.open('x',encoding='utf-8') as f:f.write(v)
start=time.monotonic()
assert not (R/'execution.mutex').exists() and not (R/'closure_receipt.json').exists()
v=read(R/'verification.json');assert v['passed'] and v['complete_sha256']==sha(R/'results/complete.json')
assert v['lock_sha256']==sha(R/'lock.json')
summary=table(R/'results/summary.csv');goals=read(R/'results/goal_summary.json')
old=table(ROOT/'round4/results/summary.csv');oldg=read(ROOT/'round4/results/goal_summary.json')
exe=read(next(R.glob('execution_*.json')));ve=read(next(R.glob('verification_execution_*.json')))
diag=read(R/'diagnostic.json');dv=read(R/'diagnostic_verification.json')
assert exe['returncode']==0 and ve['returncode']==0 and dv['passed']
comparisons=[]
for row in summary:
    cond=row['condition'];basecond=cond.removesuffix('Spectral')+'40'
    ref=next(q for q in old if q['setting']==row['setting'] and q['condition']==basecond)
    comparisons.append(dict(setting=row['setting'],condition=cond,R4_condition=basecond,
        R12_PCC=float(row['PCC']),R4_PCC=float(ref['PCC']),PCC_change=float(row['PCC'])-float(ref['PCC']),
        R12_relative_gain_pct=float(row['PCC_relative_gain_pct']),R4_relative_gain_pct=float(ref['PCC_relative_gain_pct']),
        relative_gain_change_percentage_points=float(row['PCC_relative_gain_pct'])-float(ref['PCC_relative_gain_pct'])))
assert len(comparisons)==24
write(R/'postresult_comparison.json',dict(comparisons=comparisons,
    diagnostic_only_no_performance_guarantee=True,new_numeric_fits=0,primary_unchanged=True))
primary='core_real_AdaptiveSpectral';main=next(q for q in goals if q['condition']==primary)
prior_main=next(q for q in oldg if q['condition']=='core_real_Adaptive40')
core_d=[d for d in diag['rows'] if d['source']=='core_real']
lines=['# R12：保留PCA相对方差没有提高平均迁移增益','',
    '2026-10-15，预登记CPU对照与独立验收已完成。仍用全部16个PC，只将逐方向单位方差改为统一尺度，保持来源块总方差16。原PCA轴、TD24、单40D头、全部候选和划分不变。',
    '',f"主Core Adaptive四设置相对TD24 PCC平均 **{main['mean_relative_PCC_gain_pct']:+.4f}%**，与R4 **{prior_main['mean_relative_PCC_gain_pct']:+.4f}%** 几乎相同，差{main['mean_relative_PCC_gain_pct']-prior_main['mean_relative_PCC_gain_pct']:+.4f}个百分点。未达5%，不采用为改进，不追加尺度指数或参数网格。",'',
    '| 设置 | 固定TD24 PCC | R12 Core PCC | 相对PCC变化 | MSE相对下降 | R4相对PCC变化 |',
    '|---|---:|---:|---:|---:|---:|']
for r in summary:
    if r['condition']!=primary:continue
    c=next(q for q in comparisons if q['condition']==primary and q['setting']==r['setting'])
    lines.append(f"| {r['setting']} | {float(r['TD24_PCC']):.6f} | {float(r['PCC']):.6f} | {float(r['PCC_relative_gain_pct']):+.4f}% | {float(r['MSE_relative_reduction_pct']):+.4f}% | {c['R4_relative_gain_pct']:+.4f}% |")
lines+=['','Mixset和跨队列退步减轻、Huesken略增，但Takayuki收益减少。跨队列PCC接近TD24并不意味着误差不变，其MSE仍恶化。不能只展示改善的队列。','',
    '| 条件 | R12四设置相对PCC均值 | R4同条件均值 |','|---|---:|---:|']
for g in goals:
    o=next(q for q in oldg if q['condition']==g['condition'].removesuffix('Spectral')+'40')
    lines.append(f"| {g['condition']} | {g['mean_relative_PCC_gain_pct']:+.4f}% | {o['mean_relative_PCC_gain_pct']:+.4f}% |")
lines+=['','全6条件均未达5%。Core精确初始化对照的收益仍弱，当前Core真实训练有一定平均增量；但Mixset依然负迁移，不能推广为每队列都受益。Full也未稳定超越TD24。','',
    '## 训练内诊断与解释','',
    '| Core训练池 | 第一/第16方向方差比 | 两半top8子空间重合 | 两半next8子空间重合 |','|---|---:|---:|---:|']
for d in core_d:lines.append(f"| {d['setting']} | {d['variance_ratio_first_to_16']:.3f} | {d['half_top8_overlap']:.4f} | {d['half_next8_overlap']:.4f} |")
lines+=['','诊断证明逐方向缩放的影响不可忽略、靠后的子空间更不稳定；它没有证明低方差方向无用。实际受控结果几乎没有平均改善，故不能把当前增益减弱主要归因白化。该变换仅改变正则化的方向偏好，不增加表示信息。','',
    '## 验收与成本','',
    f"6000内层候选解、96最终头、22788行OOF全部通过独立验收。新增共同尺度计算240内层/48最终；原PCA全部复用，新增PCA拟合0。独立诊断18半池SVD已完成，不算新的效率训练。正规方程最大残差{v['max_normal_residual']:.3g}，原TD候选逐行最大差{v['max_TD24_candidate_prediction_difference']:.3g}。",'',
    f"控制器本体CPU {exe['CPU_wall_seconds']:.3f}秒，独立验收总本体{ve['CPU_wall_seconds']:.3f}秒（包含其内部{v['CPU_wall_seconds']:.3f}秒，不重复相加），独立诊断验收{dv['CPU_wall_seconds']:.3f}秒。合成6项0.220秒，最初诊断计时另保留在diagnostic_execution回执。进程启动、开发及既有来源训练/导出另计，不是端到端几秒。新GPU/来源forward/来源训练均0。",'',
    f"科学锁 `{sha(R/'lock.json')}`；complete `{sha(R/'results/complete.json')}`；独立验收 `{sha(R/'verification.json')}`。",'',
    '仍为反复开发证据：源keep-overlap/context暴露、单源42、非gene隔离、Mix与跨队列共享数据限制不变。目标未完成。下一步应优先解决旧source43与当前source42、旧/新输入划分和读出之间的可比性；桥接尚未登记，不自动下载或新增encoder推理，也不把同一批数据反复优化后称独立验证。','']
save(ROOT/'ROUND12_RESULTS.md','\n'.join(lines))
progress=f'''\n\nR12共同PC尺度已闭合并独立验收：原PCA16不逐方向白化、改为总方差保持16的共同尺度，主Core Adaptive均值+2.6765%，与R4+2.6812%几乎相同。逐设置+2.3541/−1.4827/+9.9368/−0.1023%，Mix与跨队列改善但Tak收益减少，未达5%。其他五条件Core固定+2.4801、Full固定/自适应+0.8356/−0.0631、初始化固定/自适应+0.4913/+0.3115%，全部未达目标；不采用、不扩缩放网格。训练内9池诊断/18半池SVD资格通过不等于性能改善。\n6000内层解/96最终头/22788行、原PCA/基线/选择/部署与所有SHA通过。CPU控制器{exe['CPU_wall_seconds']:.3f}s、独立验收总{ve['CPU_wall_seconds']:.3f}s（内部计时不叠加）、独立诊断验收{dv['CPU_wall_seconds']:.3f}s，6合成已通过；新GPU/来源forward0、新PCA拟合0。lock {sha(R/'lock.json')}；complete {sha(R/'results/complete.json')}。见ROUND12_RESULTS.md与round12/closure_receipt.json。当前无活进程/外部阻断，旧goal不变，新goal active未达标。旧source43协议桥接仍为建议，未登记、不自动突破已核验缓存/CPU/单源42范围。用户对论文增益的追问以PAPER_GAIN_COMPARISON_20261015.md为依据，不能把跨协议差异单独归因某个模块。\n'''
with (ROOT/'PROGRESS.md').open('a',encoding='utf-8') as f:f.write(progress)
with (ROOT/'PAPER_GAIN_COMPARISON_20261015.md').open('a',encoding='utf-8') as f:
    f.write('\n补充截至R12：正交化、核读出、自动收缩和共同PC尺度均未恢复稳定的5%平均收益。R12主方案+2.6765%与R4+2.6812%基本相同，进一步说明不能单独把增益减弱归因PC白化。它们仍是当前来源和评价下的开发试验，不能代替旧source43固定条件桥接。\n')
receipt=dict(closed=True,goal_complete=False,goal_remains_active=True,has_live_process=False,
    new_GPU_hours=0,new_source_forward=0,primary_condition=primary,mean_relative_PCC_gain_pct=main['mean_relative_PCC_gain_pct'],
    files_sha256={str(p.relative_to(ROOT)):sha(p) for p in [R/'lock.json',R/'results/complete.json',R/'verification.json',R/'diagnostic_verification.json',R/'postresult_comparison.json',ROOT/'ROUND12_RESULTS.md']},
    code_sha256=sha(Path(__file__)),CPU_postresult_wall_seconds=time.monotonic()-start)
write(R/'closure_receipt.json',receipt)
write(ROOT/'GOAL_TURN_R12_AUDIT.json',dict(status='progress',goal_status_read='active',objective_achieved=False,
    blocker=False,previous_turn_progress=True,own_active_process=False,CPU_only=True,
    closure_receipt_sha256=sha(R/'closure_receipt.json'),new_method_expansion_authorized=False))
print(json.dumps(receipt,ensure_ascii=False))
