"""Close a failed routing qualification without fitting a new efficiency head."""
import hashlib,json,time
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
start=time.monotonic();d=read(R/'diagnostic.json');v=read(R/'verification.json');lock=read(R/'lock.json')
assert v['passed'] and not d['qualified'] and not v['qualified'] and not (R/'execution.mutex').exists()
assert v['diagnostic_sha256']==sha(R/'diagnostic.json') and v['lock_sha256']==sha(R/'lock.json')
for n,h in lock['files'].items():assert sha(R/n)==h,n
for n,h in lock['dependencies'].items():assert sha(Path(n))==h,n
for n,h in d['gates_files_sha256'].items():assert sha(R/'gates'/n)==h,n
ex=[read(p) for p in R.glob('execution_*.json')];ve=[read(p) for p in R.glob('verification_execution_*.json')]
assert len(ex)==len(ve)==1 and ex[0]['returncode']==ve[0]['returncode']==0
lines=['# R23：TD24两分区路由没有足够跨队列资格','',
 '2026-10-15。15个无标签两质心路由和独立验收完成。三个输入分区均较稳定，但只有Takayuki的Core条件损失差距通过预定门槛，整体qualified=false。结束该两分区路由定义，不训练下一步56D MoE，不搜索更多K/seed/阈值。此结果不是所有MoE无效的证明。','',
 '|训练池|分区ARI中位数|所有块平衡/收敛|Core条件收益差距/TD MSE|同方向内折|资格|','|---|---:|---|---:|---:|---|']
for g in d['gate_records']:
    r=next(x for x in d['records'] if x['setting']==g['setting'] and x['source']=='core_real')
    a=next(x for x in v['diagnostics'] if x['setting']==g['setting'] and x['source']=='core_real')
    lines.append(f"|{g['setting']}|{g['median_ARI']:.4f}|{g['balanced']}/{g['all_converged']}|{100*r['summary']['normalized_gain_gap']:+.4f}%|{a['consistent_inner_blocks']}/5|{r['qualified']}|")
lines+=['', '表中差距是route1与route0平均平方误差收益之差，以同池TD MSE归一；不是PCC提升比例，也不是5%目标。route编号来自质心字典序，不预设哪个区域更好或对应某种生物机制。','',
 'Huesken的区域收益差仅0.7170%且方向只在2/5内折一致；Mixset为−2.0989%、4/5同向但幅度未达5%工程门槛。Takayuki为−6.0650%、4/5同向，通过该池资格；其精确初始化也呈现更大区域差异。因此这不是普遍稳定的、miRNA监督特有的路由信号，不能据Tak单项启动跨四设置复杂化。','',
 '|训练池|来源|区域0平均收益|区域1平均收益|归一差距|资格|','|---|---|---:|---:|---:|---|']
for r in d['records']:
    s=r['summary'];lines.append(f"|{r['setting']}|{r['source']}|{s['groups'][0]['mean_gain']:+.7f}|{s['groups'][1]['mean_gain']:+.7f}|{100*s['normalized_gain_gap']:+.4f}%|{r['qualified']}|")
lines+=['', '正平均收益表示来源相对TD减少平方误差，负值表示增加。区域PCC、MSE、计数和每内折方向均完整保存在diagnostic.json；全部2299条路由OOF及6897条来源OOF在CSV中逐行可核对，没有按有利区域删除行。','',
 '判断边界：无标签分区稳定不等于效应异质性真实，也不保证监督gate无法学习其它模式。这里仅排除了预先指定的TD24两质心定义作为下一分支的足够依据；不调整K/距离/门槛寻找合格分区。复用R4固定alpha100/weight1内部OOF，没有新的外层测试指标或效率模型。原四设置基线和R22主结果+2.784%不变，5%目标未达到。','',
 f"验收：15无标签gate及15个原生确定性重放，质心最大差{v['max_centroid_replay_difference']:.3g}；独立2×2列联表ARI最大差{v['max_ARI_difference']:.3g}。训练内ID、TD训练均值/SD、距离路由、全部误差统计及门槛与原SHA核验通过。7项合成0.95s通过，首6项0.98s回执保留。",'',
 f"CPU诊断{ex[0]['CPU_wall_seconds']:.6f}s，独立验收外层{ve[0]['CPU_wall_seconds']:.6f}s（含内部{v['CPU_wall_seconds']:.6f}s，不重复计）。进程启动/开发/报告另计；15个验收重放另列为无标签计算，来源forward/效率头拟合/新PCA/GPU均0。",'',
 '背景：[Jordan与Jacobs，Hierarchical Mixtures of Experts](https://www2.bcs.rochester.edu/sites/robbie/jordanjacobs.nc94.pdf)。该原文的监督概率gate与EM没有在此复现，本轮只是固定无标签硬路由资格。','',
 '限制：反复开发的三个训练池、单source42、keep-overlap/context暴露、非gene隔离、Mix两个设置非独立。当前无本任务活进程，上一轮与本轮均为progress，不因失败阶段标记目标complete/blocked。source43导出范围问题仍待回复，本轮没有执行任何新导出。', '',
 f"科学锁{sha(R/'lock.json')}；诊断{sha(R/'diagnostic.json')}；验收{sha(R/'verification.json')}。"]
report=ROOT/'ROUND23_RESULTS.md'
with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
paths=[R/'lock.json',R/'diagnostic.json',R/'verification.json',R/'pytest.xml',R/'pytest_first6.xml',report,Path(__file__)]+list(R.glob('execution_*.json'))+list(R.glob('verification_execution_*.json'))
write(R/'closure_receipt.json',dict(closed=True,qualified=False,goal_complete=False,goal_remains_active=True,own_live_process=False,
    new_unsupervised_gates=15,verification_unsupervised_replays=15,new_efficiency_fits=0,new_source_forward=0,GPU_hours=0,
    files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},CPU_close_seconds=time.monotonic()-start))
for n,h in read(R/'closure_receipt.json')['files_sha256'].items():assert sha(ROOT/n)==h,n
write(ROOT/'GOAL_TURN_R23_AUDIT.json',dict(previous_goal_turn='progress',current_goal_turn='progress',goal_status_read='active',goal_complete=False,
    blocker=False,own_live_process=False,evidence='Fifteen unsupervised gates and 6897 source OOF rows audited; only one of three Core pools qualifies, so no new MoE fit is justified under fixed protocol.',
    closure_sha256=sha(R/'closure_receipt.json')))
with (ROOT/'PROGRESS.md').open('a',encoding='utf-8') as f:f.write(f"\n\nR23两质心无标签路由资格闭合：15gate/15独立无标签重放、2299路由OOF/6897来源OOF通过。分区ARI中位.8858/.9019/.8308稳定，但Core条件损失差距为+.7170%/−2.0989%/−6.0650%（TD MSE单位），方向一致2/4/4内折，仅Tak合格，整体gate=false；初始化Tak也合格。不训练后续56D MoE、不扩K/seed/门槛。新效率拟合/forward/GPU0，CPU{ex[0]['CPU_wall_seconds']:.3f}s+验收{ve[0]['CPU_wall_seconds']:.3f}s。见ROUND23_RESULTS.md。目标active未达5%，无活进程/阻断，上一轮及本轮progress。\n")
print(json.dumps(dict(closed=True,qualified=False,closure_sha256=sha(R/'closure_receipt.json'))))
