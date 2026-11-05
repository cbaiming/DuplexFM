"""Summarize a closed training-internal qualification; no additional fits."""
from pathlib import Path
import hashlib,json
R=Path(__file__).resolve().parent;ROOT=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')
d=read(R/'diagnostic.json');v=read(R/'diagnostic_verification.json')
assert v['passed'] and not d['qualified'] and not v['qualified']
assert v['diagnostic_sha256']==sha(R/'diagnostic.json') and v['lock_sha256']==sha(R/'diagnostic_lock.json')
assert not (R/'diagnostic.mutex').exists() and not (R/'closure_receipt.json').exists()
e=read(next(R.glob('diagnostic_execution_*.json')));ve=read(next(R.glob('verification_execution_*.json')))
assert e['returncode']==ve['returncode']==0
lines=['# R13：单方向曲率未通过训练内资格','',
    '2026-10-15，CPU资格试验及独立验收完成。给每个来源PC增加一个固定自然立方样条曲率项，同时保留TD24和原PC线性项。三个Core训练池均未改善；按预登记规则停止，不运行新的四设置外层测试，不扩大结点或正则网格。','',
    '| 训练池 | 来源/控制 | 新OOF PCC | 原线性OOF PCC | 绝对ΔPCC | MSE相对下降 |',
    '|---|---|---:|---:|---:|---:|']
for r in d['records']:
    lines.append(f"| {r['setting']} | {r['view']} | {r['new']['PCC']:.6f} | {r['old']['PCC']:.6f} | {r['delta_PCC']:+.6f} | {r['MSE_relative_reduction_pct']:+.4f}% |")
lines+=['','门槛为至少两个Core池ΔPCC≥0.005且MSE不恶化、全部Core池ΔPCC≥−0.005。本轮资格false。TD自身非线性只在Huesken训练池改善；Full/初始化也没有一致改善，不能改用某个次要条件宣布通过。','',
    '这些是原outer fold0 fit池内的五折OOF诊断，不是新的外层测试成绩。全四设置的5%目标没有重新计算或达到。它仅排除了本轮固定的小型加性曲率方案，不能证明全部非线性都无用。','',
    f"新拟合60个Ridge（3池×4视图×5内折）、{v['verified_internal_OOF_rows']}行内部OOF；独立SciPy自然CubicSpline重建全部映射、60个正规方程与资格，最大预测差{v['max_prediction_reconstruction_difference']:.3g}。原PCA和原线性对照复用，无新encoder forward、GPU或源训练。",'',
    f"控制器CPU本体{e['CPU_wall_seconds']:.3f}秒、独立验收总本体{ve['CPU_wall_seconds']:.3f}秒（不与其内部时间相加），8合成测试0.40秒。开发、进程启动及既有源模型成本另计。",'',
    '本方法借鉴[加性模型](https://doi.org/10.1214/ss/1177013604)的低维平滑思路；三结点、单曲率和Ridge配置为本轮选择，不是文献给出的siRNA最优配置。','',
    f"科学锁 `{sha(R/'diagnostic_lock.json')}`；诊断 `{sha(R/'diagnostic.json')}`；独立验收 `{sha(R/'diagnostic_verification.json')}`。",'',
    '数据重复开发、源keep-overlap/context暴露、单源42、purged_sequence非gene隔离及队列依赖限制保持。本轮取得筛查证据，目标仍active且未达成。下一步只读核对已存在Core＋开放性seed42缓存的身份与当前输入对齐，避免继续扩读出复杂度。','']
with (ROOT/'ROUND13_RESULTS.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
receipt=dict(closed=True,qualified=False,goal_complete=False,goal_remains_active=True,own_live_process=False,
    new_fits=60,new_outer_test_evaluations=0,GPU_hours=0,new_source_forward=0,
    files_sha256={str(p.relative_to(ROOT)):sha(p) for p in (R/'diagnostic_lock.json',R/'diagnostic.json',R/'diagnostic_verification.json',ROOT/'ROUND13_RESULTS.md')})
write(R/'closure_receipt.json',receipt)
with (ROOT/'PROGRESS.md').open('a',encoding='utf-8') as f:
    f.write(f'\n\nR13单方向自然样条曲率训练内资格已闭合：Core三池绝对ΔPCC −0.000489/−0.019396/−0.014046，均未改善，qualification=false；停止该路线，不运行新外层评价、不扩结点/网格。Full/初始化/TD控制全部报告。60新Ridge/9196内部OOF、独立CubicSpline/正规方程/来源身份和资格通过；CPU{e["CPU_wall_seconds"]:.3f}s+验收{ve["CPU_wall_seconds"]:.3f}s，8合成0.40s，GPU/encoder forward0。见ROUND13_RESULTS.md。上一轮R12与本轮均progress，无活进程/阻断，goal active未达5%。下一步核对既有Core＋开放性seed42缓存，旧source43桥接仍不自动实施。\n')
print(json.dumps(receipt,ensure_ascii=False))
