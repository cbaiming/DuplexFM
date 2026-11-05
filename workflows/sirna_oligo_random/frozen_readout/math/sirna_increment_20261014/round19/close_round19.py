"""Close the preregistered negative supervised-screening pilot."""
import hashlib,json,time
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
start=time.monotonic();d=read(R/'diagnostic.json');v=read(R/'verification.json');lock=read(R/'lock.json')
assert not d['qualified'] and not v['qualified'] and v['passed'] and v['all_rank_and_response_valid'] and not (R/'execution.mutex').exists()
assert v['diagnostic_sha256']==sha(R/'diagnostic.json') and v['lock_sha256']==sha(R/'lock.json')
for n,h in lock['files'].items():assert sha(R/n)==h
for n,h in lock['dependencies'].items():assert sha(Path(n))==h
for n,h in d['pilot_files_sha256'].items():assert sha(R/'pilot'/n)==h
previous=read(ROOT/'round18/closure_receipt.json')
for n,h in previous['files_sha256'].items():assert sha(ROOT/n)==h
ex=[read(p) for p in R.glob('execution_*.json')];ve=[read(p) for p in R.glob('verification_execution_*.json')]
assert len(ex)==len(ve)==1 and ex[0]['returncode']==ve[0]['returncode']==0
lines=['# R19：条件监督筛选未改善Core的训练内泛化','',
    '2026-10-15。90个PCA与90个CPU轻量读出已完成并通过独立验收。主Core conditional性能资格false、配对资格false；停止本轮，不运行新的四设置测试，不改变筛选维度、正则或随机种子。','',
    '本轮把经典监督主成分改为条件筛选：只在inner train扣除TD24线性关系后，根据效率相关性从source192选32个坐标，再用原选中坐标训练PCA16，拼TD24形成40D单Ridge。固定alpha100/weight1。随机对照以固定seed42选相同数量坐标，其他步骤一致。部署没有TD预测或其它模型预测叠加。','',
    '| 训练池 | 来源 | 筛选 | 对原R4 ΔPCC（绝对） | MSE下降 |',
    '|---|---|---|---:|---:|']
for r in d['records']:
    lines.append(f"| {r['setting']} | {r['source']} | {r['mode']} | {r['delta_PCC_vs_R4']:+.6f} | {r['MSE_reduction_vs_R4_pct']:+.4f}% |")
lines += ['',
    'Core条件监督筛选在Huesken、Mixset、Takayuki的PCC和MSE均退步，未通过相对原R4门槛；同时三个池均弱于随机筛选对照。Full只有Huesken改善，初始化在Takayuki改善，不能挑某个来源/队列替换预定主方案。', '',
    '| Core配对比较 | 条件筛选相对随机 ΔPCC | MSE下降 |',
    '|---|---:|---:|']
for r in d['paired_core_records']:
    lines.append(f"| {r['setting']} | {r['delta_PCC']:+.6f} | {r['MSE_reduction_pct']:+.4f}% |")
lines += ['',
    '固定随机对照在Mixset/Takayuki好于原R4固定读出，但Huesken稍差。这个现象只提示压缩方向值得进一步定位，不能事后把random改为本轮成功方法；它也未在此轮经过四设置的完整内层选参与外层评价。若将来研究随机压缩，需另立可复核的固定规则与稳定性控制，不能搜索随机种子寻找5%。', '',
    '监督筛选估计了192个条件相关并执行变量选择，额外监督容量不能被“仍是40D头”掩盖。选到的单坐标相关也不代表生物机制。失败可能涉及选择方差或互补信号分散；本轮没有分离这些原因，不能断言全局PCA已经保留全部有效信息。', '',
    '参照都是三个outer fold0 fit池内部的旧R4固定alpha100/weight1 OOF，不是四设置调参后TD24_Ridge_CV。全部筛选、标准化与PCA在每inner train完成，没有先用整个outer fit筛选再交叉验证；没有新的测试数值统计。', '',
    f"验收覆盖90个新PCA/90个头、13794内部OOF行，独立SVD偏相关、PCA谱和正交条件、Ridge正规方程及单向量部署均通过。最大偏相关差{v['max_partial_score_difference']:.3g}，正规方程归一残差{v['max_normal_equation_residual']:.3g}，预测重建差{v['max_prediction_difference']:.3g}；全部PCA秩与响应残差有效。8项纯合成测试0.23s通过。", '',
    f"CPU控制器墙时{ex[0]['CPU_wall_seconds']:.6f}s；独立验收外层{ve[0]['CPU_wall_seconds']:.6f}s（含内部{v['CPU_wall_seconds']:.6f}s，不重复相加）。启动、开发及报告保存另计。新来源forward/GPU小时0；既有来源训练/缓存提取成本不包含在这几秒中。", '',
    '参考：[Bair等，Prediction by Supervised Principal Components，JASA 2006](https://web.stanford.edu/~hastie/Papers/spca_JASA.pdf)。本轮是带TD条件化、固定32列/16PC的应用变体，不是该文保证的siRNA性能提升。', '',
    '目标保持active，尚未达到四设置平均相对PCC≥5%。现有预定主方案R4 Core Adaptive仍为+2.6812%；R5次要Full Uniform为+3.1633%，没有把它事后替换为主方案。源seed42/keep-overlap/context暴露、反复开发、purged非gene隔离及两Mixset设置非独立的限制继续保留。', '',
    f"科学锁{sha(R/'lock.json')}；结果{sha(R/'diagnostic.json')}；验收{sha(R/'verification.json')}。"]
report=ROOT/'ROUND19_RESULTS.md'
with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
paths=[R/'lock.json',R/'diagnostic.json',R/'verification.json',R/'pytest.xml',report,Path(__file__)]
paths+=list(R.glob('execution_*.json'))+list(R.glob('verification_execution_*.json'))
write(R/'closure_receipt.json',dict(closed=True,qualified=False,goal_complete=False,goal_remains_active=True,
    own_live_process=False,verified_heads=90,new_PCA_fits=90,new_source_forward=0,GPU_hours=0,outer_evaluation_started=False,
    files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},CPU_close_seconds=time.monotonic()-start))
for n,h in read(R/'closure_receipt.json')['files_sha256'].items():assert sha(ROOT/n)==h
write(ROOT/'GOAL_TURN_R19_AUDIT.json',dict(previous_goal_turn='progress',current_goal_turn='progress',goal_status_read='active',
    goal_complete=False,blocker=False,own_live_process=False,
    evidence='90 paired screened-PCA heads and 13794 internal OOF rows independently verified; supervised screen fails and random control motivates only future fixed-rule unsupervised compression qualification.',
    previous_closure_sha256=sha(ROOT/'round18/closure_receipt.json'),closure_sha256=sha(R/'closure_receipt.json')))
with (ROOT/'PROGRESS.md').open('a',encoding='utf-8') as f:
    f.write(f"\n\nR19条件监督筛选闭合：90新PCA/90头/13794内部OOF通过独立SVD/谱/正规方程/部署验收；Core conditional相对R4固定读出的ΔPCC为−0.012479/−0.012639/−0.020527，MSE下降−2.3706%/−2.2262%/−3.7679%，全部退步，performance/paired gate均false。固定random对照在Mix/Tak改善、Hu稍差，但不事后换为主方案，不据此直接开外层或搜索seed/维度。全部18组合完整报告，CPU{ex[0]['CPU_wall_seconds']:.3f}s+验收{ve[0]['CPU_wall_seconds']:.3f}s；来源forward/GPU0。见ROUND19_RESULTS.md。下一方向应先固定无监督压缩规则/稳定性资格，尚未登记执行；旧监督筛选分支结束。上一轮和本轮均progress，无活进程或阻断，目标active未达5%。\n")
print(json.dumps(dict(closed=True,qualified=False,verified_heads=90,closure_sha256=sha(R/'closure_receipt.json'))))
