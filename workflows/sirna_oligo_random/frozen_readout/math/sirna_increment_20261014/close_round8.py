"""Publish the completed R8 result without additional fitting or testing."""
import csv,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;D=R/'round8';P=R/'round4'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def create(p,text):
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(text)

v=read(D/'verification.json');c=read(D/'results/complete.json');diagnostic=read(D/'diagnostic.json')
assert v['passed'] and v['complete_sha256']==sha(D/'results/complete.json')
assert v['lock_sha256']==sha(D/'lock.json') and not (D/'execution.mutex').exists()
execution=[read(p) for p in D.glob('execution_*.json')];audit=[read(p) for p in D.glob('verification_execution_*.json')]
diagcost=[read(p) for p in D.glob('diagnostic_execution_*.json')]
assert len(execution)==len(audit)==len(diagcost)==1
assert execution[0]['returncode']==audit[0]['returncode']==diagcost[0]['returncode']==0
goals=read(D/'results/goal_summary.json');old=read(P/'results/goal_summary.json');rows=table(D/'results/summary.csv')
assert len(goals)==6 and not any(g['reaches_development_5pct'] for g in goals)
settings=['Huesken','Mixset','Takayuki','Inter_Hu_to_Mix'];comparison=[]
for g in goals:
    parent=g['condition'].replace('Orthogonal','40');original=next(x for x in old if x['condition']==parent)
    comparison.append(dict(condition=g['condition'],parent_condition=parent,
        R4_mean_relative_PCC_gain_pct=original['mean_relative_PCC_gain_pct'],
        R8_mean_relative_PCC_gain_pct=g['mean_relative_PCC_gain_pct'],
        delta_percentage_points=g['mean_relative_PCC_gain_pct']-original['mean_relative_PCC_gain_pct']))
assert all(x['delta_percentage_points']<0 for x in comparison)
record=dict(derived_from_verified_summaries=True,R4_complete_sha256=sha(P/'results/complete.json'),
    R4_verification_sha256=sha(P/'verification.json'),R8_complete_sha256=sha(D/'results/complete.json'),
    R8_verification_sha256=sha(D/'verification.json'),comparison=comparison,
    no_new_fitting=True,no_posthoc_condition_selection=True)
create(D/'postresult_comparison.json',json.dumps(record,ensure_ascii=False,indent=2)+'\n')
primary=next(x for x in goals if x['primary']);cost=execution[0]['CPU_wall_seconds'];verifycost=audit[0]['CPU_wall_seconds']
lines=['# R8：特征正交化没有改善迁移，保留负结果','',
    f"2026-10-15。预定主方案Core Adaptive的四设置平均相对PCC为 **{primary['mean_relative_PCC_gain_pct']:+.4f}%**，未达到5%。与原R4同源/同模式比较，六个条件的平均增益全部下降，因此不采用，不增加投影强度或正则网格。",'',
    '| 条件 | Huesken | Mixset | Takayuki | Huesken→Mixset | 平均相对PCC | R4对应平均 |',
    '|---|---:|---:|---:|---:|---:|---:|']
for g,cp in zip(goals,comparison):
    vals=[next(float(r['PCC_relative_gain_pct']) for r in rows if r['condition']==g['condition'] and r['setting']==s) for s in settings]
    lines.append('| '+g['condition']+' | '+' | '.join(f'{x:+.4f}%' for x in vals)+f" | {g['mean_relative_PCC_gain_pct']:+.4f}% | {cp['R4_mean_relative_PCC_gain_pct']:+.4f}% |")
lines += ['', '| 设置 | TD24 PCC | Core主方案PCC | 相对提升 | MSE相对下降 |', '|---|---:|---:|---:|---:|']
for s in settings:
    r=next(x for x in rows if x['setting']==s and x['condition']==primary['condition'])
    lines.append(f"| {s} | {float(r['TD24_PCC']):.6f} | {float(r['PCC']):.6f} | {float(r['PCC_relative_gain_pct']):+.4f}% | {float(r['MSE_relative_reduction_pct']):+.4f}% |")
lines += ['',
    '资格诊断：三个固定outer fold0训练池Core源PC16有24.69%/28.02%/31.15%训练方差可被TD24解释；Full为27.09%/30.85%/33.84%，初始化为14.69%/19.55%/23.33%。各残余秩16，原PCA16保留source192方差约51%–58%。它只说明训练输入有重叠，未说明重叠方向无用或可保证去重后泛化更好。', '',
    '此方法把S变为S−TA后仍训练一个40D联合Ridge，无额外预测器；所有来源、标签、候选和划分固定。它在无正则时保持同一个线性函数空间，在L2下改变惩罚的坐标和对共享信号的偏好。因此负结果否定本次固定正交化的实用收益，不是“共享信息全部无用”或“DuplexFM完全没有信号”的证据。', '',
    '已验证每个alpha下正交坐标的TD系数与同alpha TD-only一致；但最终alpha也由训练内选择，且外部样本不必正交，不能宣称保护了所有测试预测。实际Takayuki的Core主方案由R4的+12.3177%下降到+4.8462%，削弱整体增益；Mixset和跨队列虽较R4略缓和，仍为负。', '',
    '初始Core自适应在Mixset略好于TD24（+0.1601%），训练Core为−2.0157%；因此训练miRNA监督并未在所有队列带来正增量。Full各模式平均为负，六条件完整保存，未按测试挑源、挑模式或组合不同设置的赢家。', '',
    f"验收：{v['inner_normal_equations']}内层正规方程、{v['final_heads']}最终头、{v['OOF_rows']} OOF行与9个无标签诊断映射全部通过。TD24五候选最大逐行差{v['max_TD24_candidate_prediction_difference']:.3g}；最终预测独立重建最大差{v['max_final_prediction_reconstruction_difference']:.3g}；最大归一正规方程残差{v['max_normal_residual']:.3g}。部署折成单个原输入216维系数及截距，预测同40D头；不是新增216个独立监督系数。", '',
    f"成本：无标签诊断CPU {diagcost[0]['CPU_wall_seconds']:.3f}s，头拟合控制器{cost:.3f}s，独立验收全过程{verifycost:.3f}s（含内部{v['CPU_wall_seconds']:.3f}s，不相加）。5项合成0.95s，开发/进程启动另计。新GPU/来源前向/来源更新均为0；历史来源训练和特征提取不包含在上述秒数内。", '',
    f"科学锁：{sha(D/'lock.json')}。完成：{sha(D/'results/complete.json')}。独立验收：{sha(D/'verification.json')}。", '',
    '当前目标保持active，5%未达到；R7和R8均产出改变下一动作的证据，属于progress，不是外部阻断。所有本轮进程已经正常退出，未启动新后台任务。', '',
    '后续结束本条正交化分支。原source43桥接需另外的输入身份与缓存资格，而当前目标固定source42，不偷偷换源。下一步应优先利用训练内部证据检查样本/响应异质性与来源有效性，决定是否值得改变效率损失；尚未登记或运行新因素，不反复扩读出网格制造5%。', '',
    '限制：单source42、miRNA keep-overlap/context暴露、当前数据已反复开发、purged_sequence非gene隔离、Mixset两设置相关、没有新独立外部或来源初始化稳定性确认。无新显著性/置信区间或SOTA声明。']
create(R/'ROUND8_RESULTS.md','\n'.join(lines)+'\n')
turn=dict(goal_status='active',goal_achieved=False,previous_goal_turn='progress',this_goal_turn='progress',
    round='R8',primary_mean_relative_PCC_gain_pct=primary['mean_relative_PCC_gain_pct'],
    all_six_means_lower_than_matched_R4=True,CPU_only=True,GPU_hours=0,new_source_forward=0,
    lock_sha256=sha(D/'lock.json'),complete_sha256=sha(D/'results/complete.json'),verification_sha256=sha(D/'verification.json'),
    no_running_process=True,blocking_condition=None,next_factor_registered=False)
create(R/'GOAL_TURN_R8_AUDIT.json',json.dumps(turn,ensure_ascii=False,indent=2)+'\n')
with (R/'PROGRESS.md').open('a',encoding='utf-8',newline='\n') as f:
    f.write('\nR8在固定训练池无标签冗余资格通过后，完成来源PC16对TD24特征正交化；单40D联合头、同25候选/划分，预定Core自适应均值+0.8905%，Core固定+1.4714%；Full固定−1.0274%/自适应−0.3640%；初始化固定−0.5814%/自适应+0.1698%。六对应条件均低于R4，全部未达到5%，不采用、不扩大投影网格。\n')
    f.write(f'6000内层解/96最终头/22788行、9个无标签映射、部署单向量和全部SHA通过独立验收。CPU诊断{diagcost[0]["CPU_wall_seconds"]:.3f}s、控制器{cost:.3f}s、验收{verifycost:.3f}s，5项合成0.95s；新GPU/来源前向0。科学锁{sha(D/"lock.json")}，complete{sha(D/"results/complete.json")}。见ROUND8_RESULTS.md。无活进程/外部阻断，上一轮与本轮progress，goal保持active未完成。旧source43桥接与当前固定source42目标分开；下一因素尚未登记，优先训练内部响应异质性资格，不再继续正交化或核网格。\n')
print(json.dumps(turn,ensure_ascii=False,indent=2))
