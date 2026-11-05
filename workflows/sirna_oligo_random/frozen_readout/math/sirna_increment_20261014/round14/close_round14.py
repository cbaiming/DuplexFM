"""Close audited Core+Access comparison, preserving every positive/negative result."""
from pathlib import Path
import csv,hashlib,json,time
R=Path(__file__).resolve().parent;ROOT=R.parent;P=ROOT/'round4'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
start=time.monotonic();assert not (R/'closure_receipt.json').exists() and not (R/'execution.mutex').exists()
v=read(R/'verification.json');assert v['passed'] and v['complete_sha256']==sha(R/'results/complete.json')
assert v['lock_sha256']==sha(R/'lock.json')
plan=read(R/'plan.json');assert plan['reference_R4_complete_sha256']==sha(P/'results/complete.json')
assert read(P/'verification.json')['complete_sha256']==sha(P/'results/complete.json')
e=read(next(R.glob('execution_*.json')));ve=read(next(R.glob('verification_execution_*.json')))
assert e['returncode']==ve['returncode']==0
new=table(R/'results/summary.csv');prior=table(P/'results/summary.csv');goals=read(R/'results/goal_summary.json');pg=read(P/'results/goal_summary.json')
selections=table(R/'results/selection.csv');comparisons=[];usage=[]
for row in new:
    mode=row['condition'].split('_real_')[1]
    for src in ('core_real','full_real'):
        ref=next(r for r in prior if r['setting']==row['setting'] and r['condition']==src+'_'+mode)
        assert float(row['TD24_PCC'])==float(ref['TD24_PCC'])
        comparisons.append(dict(setting=row['setting'],condition=row['condition'],reference_condition=ref['condition'],
            PCC_delta=float(row['PCC'])-float(ref['PCC']),relative_gain_change_percentage_points=float(row['PCC_relative_gain_pct'])-float(ref['PCC_relative_gain_pct'])))
for setting in ('Huesken','Mixset','Takayuki','Inter_Hu_to_Mix'):
    chosen=[r for r in selections if r['setting']==setting and r['condition']=='core_access_real_Adaptive40']
    off=[int(r['fold']) for r in chosen if float(r['weight'])==0]
    active_rows=sum(len(f['test_ids']) for f in plan['folds'] if f['setting']==setting and f['fold'] not in off)
    usage.append(dict(setting=setting,folds=len(chosen),TD_only_folds=off,source_active_test_rows=active_rows))
write(R/'postresult_comparison.json',dict(source_comparisons=comparisons,usage=usage,
    old_control_refits=0,no_matching_CoreAccess_initial_features=True))
main=next(r for r in goals if r['primary']);assert not main['reaches_development_5pct']
lines=['# R14：Core＋开放性未改善当前四设置目标','',
    '2026-10-15，既有缓存一致性、固定读出计算与独立验收均完成。仅替换来源为已训练Core+Access seed42第8轮终点，保持R4的PCA16、40D Ridge、候选和输入/划分。没有新encoder推理或源训练。','',
    f"预定自适应主方案四设置相对TD24 PCC均值 **{main['mean_relative_PCC_gain_pct']:+.4f}%**。固定source权重1的Uniform对照为 **{goals[0]['mean_relative_PCC_gain_pct']:+.4f}%**；均未达5%，不采用为主方案，不据测试切换到次要方案或改变候选。",'',
    '| 设置 | TD24 PCC | Core+Access Adaptive PCC | 相对PCC变化 | MSE相对下降 | Core+Access Uniform相对PCC变化 |',
    '|---|---:|---:|---:|---:|---:|']
for row in new:
    if row['condition']!='core_access_real_Adaptive40':continue
    uniform=next(r for r in new if r['setting']==row['setting'] and r['condition']=='core_access_real_Uniform40')
    lines.append(f"| {row['setting']} | {float(row['TD24_PCC']):.6f} | {float(row['PCC']):.6f} | {float(row['PCC_relative_gain_pct']):+.4f}% | {float(row['MSE_relative_reduction_pct']):+.4f}% | {float(uniform['PCC_relative_gain_pct']):+.4f}% |")
lines+=['','## 与既有来源的同读出对照','',
    '| 来源 | Uniform平均相对PCC | Adaptive平均相对PCC |','|---|---:|---:|']
for source,items in [('Core',pg[:2]),('Full',pg[2:4]),('Core初始化',pg[4:6]),('Core+Access',goals)]:
    lines.append(f"| {source} | {items[0]['mean_relative_PCC_gain_pct']:+.4f}% | {items[1]['mean_relative_PCC_gain_pct']:+.4f}% |")
lines+=['','原Core/Full/initial_Core的六条件按完整SHA复用，未新增拟合。Core+Access固定强度在Mixset相对TD24有+1.0598%，而Core的固定强度在该队列为负；但其Takayuki收益明显小于Core，不能把局部改善概括为整体更好。','',
    '## 选择行为与解释','',
    '| 设置 | TD-only折/总折 | 使用来源的测试行 |','|---|---:|---:|']
for r in usage:lines.append(f"| {r['setting']} | {len(r['TD_only_folds'])}/{r['folds']} | {r['source_active_test_rows']} |")
lines+=['','Takayuki有3/5折选择TD-only，但另外两折分别使用来源权重0.1和10；因此不能一概称为过度保守。固定保留来源确实改善了这个队列，却仍只有+1.6031%的四设置均值。该现象支持继续区分“表示能提供什么”与“有限训练池能否稳定选择到它”，不证明任何特定gate损失已经是解法。','',
    'Core+Access的准确初始权重文件已核验，但没有对应初始化表示缓存；不能把initial_Core当成匹配初始化。新旧源均为8轮32728更新，但仅同seed不等于全部初始化字节相同。本轮不能单独归因miRNA监督或开放性的生物机制。','',
    '## 验收与成本','',
    f"80个新内层PCA+16最终PCA、2000内层Ridge解、32最终头、7596行OOF及32选择行均核验。原TD24五候选逐行最大差{v['max_TD24_candidate_prediction_difference']:.3g}。所有标签/划分和source192身份不变；旧对照96头/22788行只按SHA复用，不算新训练。",'',
    f"控制器CPU本体{e['CPU_wall_seconds']:.3f}秒；独立验收本体{ve['CPU_wall_seconds']:.3f}秒；新接口4合成0.87秒。资产资格时间记录于asset_verification.json。登记前生成器一次参数遗漏失败0.01733秒已保留，补参数后编译通过，没有影响训练/结果；详见BUILD_REPAIR.md。开发、进程启动和已有H100源训练/导出成本另计，不声称端到端几秒。新GPU/来源forward均0。",'',
    f"科学锁 `{sha(R/'lock.json')}`；complete `{sha(R/'results/complete.json')}`；独立验收 `{sha(R/'verification.json')}`。",'',
    '当前目标仍未完成。来源keep-overlap/context暴露、同一Mixset两设置、单来源42和四设置反复开发限制保持。不能据此宣称源架构普遍更差/更好、全新外部验证或SOTA。后续不重复该源/读出的同一比较，不追加按测试选择的来源组合。','']
with (ROOT/'ROUND14_RESULTS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
receipt=dict(closed=True,goal_complete=False,goal_remains_active=True,own_live_process=False,
    primary_condition=main['condition'],mean_relative_PCC_gain_pct=main['mean_relative_PCC_gain_pct'],
    new_Ridge_solves=2032,new_PCA_fits=96,old_control_refits=0,new_source_forward=0,GPU_hours=0,
    files_sha256={str(p.relative_to(ROOT)):sha(p) for p in (R/'asset_verification.json',R/'lock.json',R/'results/complete.json',R/'verification.json',R/'postresult_comparison.json',ROOT/'ROUND14_RESULTS.md')},
    CPU_postresult_wall_seconds=time.monotonic()-start)
write(R/'closure_receipt.json',receipt)
with (ROOT/'PROGRESS.md').open('a',encoding='utf-8') as f:
    f.write(f'\n\nR14既有Core+Access seed42缓存身份/3329输入对齐通过后，按同R4数学、相同25候选完成单因素比较。主Adaptive均值−1.1008%，逐设置+1.2498/−3.9707/−1.6752/−0.0070%；固定Uniform+1.6031%，均低于5%和Core相应读出。Tak自适应3/5折退回TD-only，但固定来源均值仍不达标，不能仅归因保守gate。没有匹配Core+Access初始化表示缓存，不做监督因果归因。\n新80内层/16最终PCA、2000内层解/32最终头/7596行全部独立核验，旧R4六条件只按SHA引用。CPU控制器{e["CPU_wall_seconds"]:.3f}s、验收{ve["CPU_wall_seconds"]:.3f}s；新GPU/encoder forward0。生成器登记前失败及修复保留，科学锁{sha(R/"lock.json")}，complete{sha(R/"results/complete.json")}。见ROUND14_RESULTS.md。本目标轮R13资格失败闭合+R14实测闭合均为progress，没有活进程或外部阻断，goal active未达5%。后续应围绕训练内可重复的补充信号定位，不能继续把测试较好来源或网格改为主方案。\n')
write(ROOT/'GOAL_TURN_R13_R14_AUDIT.json',dict(previous_goal_turn='progress',current_goal_turn='progress',
    goal_status_read='active',goal_complete=False,blocker=False,own_live_process=False,CPU_only=True,
    R13_closure_sha256=sha(ROOT/'round13/closure_receipt.json'),R14_closure_sha256=sha(R/'closure_receipt.json'),
    no_new_model_averaging=True,no_new_source_forward=True))
print(json.dumps(receipt,ensure_ascii=False))
