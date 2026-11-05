"""Close the failed geometry gate without launching a supervised pilot."""
import hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent; ROOT=R.parent
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f: json.dump(x,f,indent=2,ensure_ascii=False); f.write('\n')

start=time.monotonic()
d=read(R/'geometry_result.json'); v=read(R/'geometry_verification.json'); lock=read(R/'geometry_lock.json')
assert not d['qualified'] and v['passed'] and not v['qualified'] and not (R/'geometry.mutex').exists()
assert not (R/'pilot').exists() and not (R/'results').exists()
assert v['result_sha256']==sha(R/'geometry_result.json') and v['lock_sha256']==sha(R/'geometry_lock.json')
for name,h in lock['files'].items(): assert sha(R/name)==h
for name,h in lock['dependencies'].items(): assert sha(Path(name))==h
ex=[read(p) for p in R.glob('geometry_execution_*.json')]; ve=[read(p) for p in R.glob('geometry_verification_execution_*.json')]
assert len(ex)==len(ve)==1 and ex[0]['returncode']==ve[0]['returncode']==0
suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
tests=sum(int(s.get('tests',0)) for s in suites); test_seconds=sum(float(s.get('time',0)) for s in suites)
assert tests==6 and not any(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites)
lines=['# R15：表示长度归一化未通过训练输入资格','',
       '2026-10-15。本轮完成固定训练池的幅度/方向几何诊断及独立验收，资格为false。未启动监督pilot、四设置新外层评价或参数搜索。', '',
       '唯一候选是将每条样本的native source192归一为单位L2长度，TD24不动，再训练原PCA16/Ridge。此候选与R12的PC方向尺度调整不同。资格要求Core三个池至少两个CV≥10%且与原PC1相关平方≥0.20，所有Core范数非零；该门槛在查看统计前固定。','',
       '| 训练池 | 来源 | 长度CV | 长度与PC1的相关平方 | 资格 |','|---|---|---:|---:|---|']
for r in d['records']:
    lines.append(f"| {r['setting']} | {r['source']} | {100*r['norm_CV']:.4f}% | {r['norm_PC1_r2']:.6f} | {'通过' if r['pool_qualified'] else '未通过'} |")
lines += ['',
    'Core三池长度CV只有3.83%–4.90%，全部低于10%，因此本轮按协议结束。长度与PC1的相关平方仍为0.337–0.695：它参与表示变化，不能把资格失败解释为“幅度没有信息”，也不能证明单位归一化绝不可能有效。这里只是预先约定的有界筛查，不根据结果降低门槛或改成另一归一化网格。', '',
    '本轮未使用效率标签进行几何计算；原CSV因身份核验/哈希被读取，标签字段未数值化或参与资格。仅三个outer fold0 fit池，九个source×pool映射，6897来源行（同输入对应三个来源），独立math.hypot/math.fsum与原PCA特征方程均通过。全部原输入/代码/权重SHA核对通过，旧PCA未重新拟合，旧预测未重新评价。', '',
    f"6项CPU合成验收通过，XML记录{test_seconds:.6f}s；诊断控制器本体{ex[0]['CPU_wall_seconds']:.6f}s；独立验收外层{ve[0]['CPU_wall_seconds']:.6f}s（内部{v['CPU_wall_seconds']:.6f}s包含于其中，不重复相加）。进程启动、开发和旧模型训练/导出成本另计。新效率拟合0、新来源forward0、新GPU小时0。", '',
    '当前5%目标仍未达到：R4预定Core Adaptive四设置平均+2.6812%；R5次要Full Uniform二阶方案+3.1633%未确认。本轮不更新性能排名、不更换主来源或降低TD24基线。源keep-overlap/context暴露、外层反复开发、单来源种子及两个Mixset设置非独立的限制保持。', '',
    '参考：Chen等的[作者代码distLinear](https://github.com/wyharveychen/CloserLookFewShot/blob/master/backbone.py)使用样本特征L2归一化；本轮只借鉴幅度/方向分离的思路，不复现其分类器，不将视觉结果推断为RNA迁移证据。', '',
    f"科学锁：{sha(R/'geometry_lock.json')}；诊断：{sha(R/'geometry_result.json')}；独立验收：{sha(R/'geometry_verification.json')}。", '',
    '后续因素尚未登记，不将旧source43桥接或任何未运行模型描述为新结果。当前无本轮活进程或外部阻断。']
report=ROOT/'ROUND15_RESULTS.md'
with report.open('x',encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')
paths=[R/'geometry_lock.json',R/'geometry_result.json',R/'geometry_verification.json',R/'pytest.xml',Path(__file__),report]
paths+=list(R.glob('geometry_execution_*.json'))+list(R.glob('geometry_verification_execution_*.json'))
receipt=dict(closed=True,goal_complete=False,goal_remains_active=True,qualified=False,
    own_live_process=False,new_efficiency_fits=0,new_PCA_fits=0,new_source_forward=0,GPU_hours=0,
    supervised_pilot_started=False,outer_evaluation_started=False,
    files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},CPU_close_seconds=time.monotonic()-start)
write(R/'closure_receipt.json',receipt)
for name,h in read(R/'closure_receipt.json')['files_sha256'].items(): assert sha(ROOT/name)==h
write(ROOT/'GOAL_TURN_R15_AUDIT.json',dict(previous_goal_turn='no progress in numerical optimization',
    previous_explanation='The preceding turn answered the paper-comparison question from existing results and wrote a bridge checklist; it did not add new scientific evidence.',
    current_goal_turn='progress',evidence='New preregistered radial geometry hypothesis tested and independently verified; fixed gate false, so supervised normalization branch stops.',
    goal_status_read='active',goal_complete=False,blocker=False,own_live_process=False,
    closure_sha256=sha(R/'closure_receipt.json'),CPU_only=True,new_source_forward=0))
with (ROOT/'PROGRESS.md').open('a',encoding='utf-8') as f:
    f.write('\n\nR15训练输入幅度资格完成并独立验收：Core三池长度CV为4.6186%/4.9050%/3.8346%，PC1相关平方0.673552/0.694876/0.336655；均未达到预登记CV≥10%的联合门槛，qualified=false。9个映射/6897来源行、独立范数/相关性/PCA方程与SHA通过。未数值使用标签、新效率拟合/新PCA/来源forward/GPU均0；不降低门槛、不运行L2归一化外层实验。不能将相关性强而CV较小解释为幅度无用。见ROUND15_RESULTS.md。上一解释轮对数值优化无新进展，本轮新增证据并关闭该假设分支，属于progress；无活进程/外部阻断，goal active未达5%。\n')
print(json.dumps(dict(closed=True,qualified=False,closure_sha256=sha(R/'closure_receipt.json'),goal_complete=False),ensure_ascii=False))
