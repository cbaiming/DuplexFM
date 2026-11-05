"""Close and report the preregistered negative low-rank pilot."""
import hashlib,json,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;ROOT=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
started=time.monotonic();d=read(R/'diagnostic.json');v=read(R/'verification.json');lock=read(R/'lock.json')
assert not d['qualified'] and not v['qualified'] and v['passed'] and v['all_certified'] and not (R/'execution.mutex').exists()
assert v['diagnostic_sha256']==sha(R/'diagnostic.json') and v['lock_sha256']==sha(R/'lock.json')
for n,h in lock['files'].items():assert sha(R/n)==h
for n,h in lock['dependencies'].items():assert sha(Path(n))==h
for n,h in d['pilot_files_sha256'].items():assert sha(R/'pilot'/n)==h
ex=[read(p) for p in R.glob('execution_*.json')];ve=[read(p) for p in R.glob('verification_execution_*.json')]
assert len(ex)==len(ve)==1 and ex[0]['returncode']==ve[0]['returncode']==0
rows=[]
for r in d['records']:
    ranks=[a for a in v['rank_records'] if a['setting']==r['setting'] and a['view']==r['view']];assert len(ranks)==5
    rows.append(dict(setting=r['setting'],view=r['view'],rank_min=min(a['rank'] for a in ranks),rank_max=max(a['rank'] for a in ranks),
        rank_mean=float(np.mean([a['rank'] for a in ranks])),total_iterations=sum(a['iterations'] for a in ranks),
        delta_PCC_vs_R4=r['delta_PCC_vs_R4'],MSE_reduction_vs_R4_pct=r['MSE_reduction_vs_R4_pct'],
        MSE_reduction_vs_paired_pct=r['MSE_reduction_vs_paired_pct']))
write(R/'rank_summary.json',dict(rows=rows,all_nuclear_certified=True,qualified=False))
lines=['# R17：低秩交互减轻部分过拟合，但没有通过性能资格','',
       '2026-10-15，60个普通交互Ridge和60个核范数交互头完成。所有凸解证书通过独立验收；主Core性能资格false，不运行新的四设置外层评价，不调整核范数比例或改用较好来源。','',
       '本轮保留原TD24＋PCA16线性项，在双线性交互矩阵上增加核范数。每个训练分区固定alpha100、lambda/critical=0.25；配对普通Ridge只有核范数项为0，映射/标签/输入相同。交互整块采用共同尺度，保留矩阵秩；所以与R5不完全同一特征合同，不能直接拿R5当唯一对照。','',
       '| 训练池 | 来源/控制 | 核范数对原R4 ΔPCC（绝对） | 对原R4 MSE下降 | 对配对交互Ridge MSE下降 | 交互秩范围 |',
       '|---|---|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['setting']} | {r['view']} | {r['delta_PCC_vs_R4']:+.6f} | {r['MSE_reduction_vs_R4_pct']:+.4f}% | {r['MSE_reduction_vs_paired_pct']:+.4f}% | {r['rank_min']}–{r['rank_max']} |")
core=[r for r in rows if r['view']=='core_real'];paired_mean=float(np.mean([r['MSE_reduction_vs_paired_pct'] for r in core]))
lines += ['',
    f"Core对同特征普通交互Ridge的等权MSE改善为{paired_mean:.4f}%，在Huesken/Mixset方向为正，Takayuki为负；该配对门槛通过。但相对更简单的R4固定40D线性读出，Huesken/Mixset均退步，只有Takayuki改善；Huesken绝对ΔPCC低于−0.005下限，故主资格失败。不能以‘比更复杂基线好一点’替代用户要求的相对强TD24平均≥5%。", '',
    '这些只是在三个outer fold0 fit池内部的五折OOF，参考原R4固定alpha100/weight1，而非四设置调参后TD24_Ridge_CV；没有计算新的四设置测试指标。Full、初始化与TD-only控制全部报告，不能按测试或资格结果更换主来源。TD-only同样能从核范数收缩得到部分收益，因此收缩效果本身不证明miRNA监督增量。', '',
    f"数值审计：120头、9196样本行（两预测列共18392个新值）、60个独立联合Ridge线性求解、60个可行对偶证书，全部通过。最大相对原始—对偶gap {v['max_relative_dual_gap']:.3g}（门槛1e-9），最大预测重建差{v['max_prediction_difference']:.3g}。没有数值达限，不能将此负结果归为求解未收敛。", '',
    '秩是学习到的系数矩阵性质，不是生物学机制证据。原存储系数为425（来源）/601（TD自交互），TD自交互对称有效二次项300；模型容量并非严格相同，PCA及来源模型仍有额外训练/提取成本。', '',
    f"CPU控制器墙时{ex[0]['CPU_wall_seconds']:.6f}s，独立验收外层{ve[0]['CPU_wall_seconds']:.6f}s（其中内部{v['CPU_wall_seconds']:.6f}s不重复相加）；合成测试0.24s、开发/启动/本文件保存另计。新来源forward0、新PCA0、新GPU小时0，旧来源训练与缓存导出不包含在这几秒内。", '',
    '原5%目标仍active且未达到。R4预定Core Adaptive四设置平均+2.6812%；R5次要Full Uniform二阶+3.1633%尚未确认。本轮没有新的外层提升，不扩大网格继续找正结果。源keep-overlap/context暴露、单来源seed42、数据反复开发、purged不是gene隔离及两个Mixset设置非独立等限制不变。', '',
    '参考数学工具：[Parikh与Boyd，Proximal Algorithms](https://web.stanford.edu/~boyd/papers/prox_algs.html)，核范数近端为奇异值软阈值；本轮应用及固定参数并非文献保证的siRNA最优方法。', '',
    f"科学锁：{sha(R/'lock.json')}；诊断：{sha(R/'diagnostic.json')}；独立验收：{sha(R/'verification.json')}。", '',
    '上一轮R16和本轮均产生新证据，属于progress；当前无本任务活进程或外部阻断。下一因素尚未登记，不将已失败低秩分支扩成rank/比例搜索。']
report=ROOT/'ROUND17_RESULTS.md'
with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
paths=[R/'lock.json',R/'diagnostic.json',R/'verification.json',R/'rank_summary.json',R/'pytest.xml',report,Path(__file__)]
paths+=list(R.glob('execution_*.json'))+list(R.glob('verification_execution_*.json'))
write(R/'closure_receipt.json',dict(closed=True,qualified=False,goal_complete=False,goal_remains_active=True,own_live_process=False,
    all_convex_certified=True,new_efficiency_heads=120,independent_Ridge_audit_solves=60,
    new_PCA_fits=0,new_source_forward=0,GPU_hours=0,outer_evaluation_started=False,
    files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},CPU_close_seconds=time.monotonic()-started))
for name,h in read(R/'closure_receipt.json')['files_sha256'].items():assert sha(ROOT/name)==h
write(ROOT/'GOAL_TURN_R17_AUDIT.json',dict(previous_goal_turn='progress',current_goal_turn='progress',goal_status_read='active',
    goal_complete=False,blocker=False,own_live_process=False,
    evidence='120 paired heads with 60 independent convex certificates; low-rank penalty improves paired complexity control but fails against simpler fixed Core reference.',
    closure_sha256=sha(R/'closure_receipt.json')))
with (ROOT/'PROGRESS.md').open('a',encoding='utf-8') as f:
    f.write(f"\n\nR17凸核范数交互固定比较闭合：120新头（60普通Ridge+60核范数）/9196内部行，全部凸证书通过；独立60联合Ridge求解另计。Core对原R4固定40D的绝对ΔPCC为−0.006334/−0.002065/+0.007360，MSE下降−1.2054%/−0.1654%/+1.4121%；对本轮同特征普通交互Ridge等权MSE改善{paired_mean:.4f}%，但原R4门槛失败，qualified=false。最大相对gap{v['max_relative_dual_gap']:.3g}，并非未收敛导致负结果。不扩大rank/比例网格、不做新外层。CPU控制器{ex[0]['CPU_wall_seconds']:.3f}s、验收外层{ve[0]['CPU_wall_seconds']:.3f}s，新GPU/来源forward0。见ROUND17_RESULTS.md；上一轮和本轮progress，无活进程/外部阻断，目标active未达5%。\n")
print(json.dumps(dict(closed=True,qualified=False,all_certified=True,core_paired_MSE_reduction_mean=paired_mean,closure_sha256=sha(R/'closure_receipt.json'))))
