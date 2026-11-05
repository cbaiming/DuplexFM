"""Close the fixed random-subspace pilot without another fit or seed search."""
import hashlib,json,time
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
start=time.monotonic();d=read(R/'diagnostic.json');v=read(R/'verification.json');lock=read(R/'lock.json')
assert not d['qualified'] and not v['qualified'] and v['passed'] and v['all_rank16_valid'] and not (R/'execution.mutex').exists()
assert v['diagnostic_sha256']==sha(R/'diagnostic.json') and v['lock_sha256']==sha(R/'lock.json')
for n,h in lock['files'].items():assert sha(R/n)==h
for n,h in lock['dependencies'].items():assert sha(Path(n))==h
for n,h in d['pilot_files_sha256'].items():assert sha(R/'pilot'/n)==h
previous=read(ROOT/'round19/closure_receipt.json')
for n,h in previous['files_sha256'].items():assert sha(ROOT/n)==h
ex=[read(p) for p in R.glob('execution_*.json')];ve=[read(p) for p in R.glob('verification_execution_*.json')]
assert len(ex)==len(ve)==1 and ex[0]['returncode']==ve[0]['returncode']==0
assert d['core_map_gates']=={'42':False,'43':False,'44':False}
lines=['# R20：固定随机压缩未能稳定改善迁移读出','',
    '2026-10-15。135个CPU读出及独立审计完成，三个预定Core投影全部未过性能资格。停止此分支，不搜索更多投影种子、维数或正则，不开展新的外层评价。','',
    'source192先按原R4训练尺度标准化，再映射到固定随机16维子空间，训练内白化保持协方差约I16，拼TD24后仍为单40D Ridge、alpha100/weight1。投影42为主，43/44只作稳定性，不平均预测，不代表三个来源训练种子；源模型始终seed42。','',
    '| 训练池 | 来源 | 投影种子 | 对原R4 ΔPCC（绝对） | MSE下降 |',
    '|---|---|---:|---:|---:|']
for r in d['records']:
    lines.append(f"| {r['setting']} | {r['source']} | {r['map_seed']} | {r['delta_PCC_vs_R4']:+.6f} | {r['MSE_reduction_vs_R4_pct']:+.4f}% |")
lines += ['',
    '主Core投影42在Huesken/Mixset退步，Takayuki只小幅改善；投影43在Mixset改善但另两池退步，投影44三个池均退步。三个投影各自都未达到原资格，不是仅因为稳定性要求更严格而被否决。因此不挑某个投影/队列组合，不能把上轮随机坐标控制的局部正结果推广成随机压缩有效。', '',
    'Full与精确Core初始化完整报告，部分正方向也随投影改变。输入子空间的随机性会影响内部泛化，但本轮没有证明全部来源信息都已由PCA保留，也没有独立区分信号分散、有限样本方差和正则先验等原因。投影本身不使用标签，当前训练协方差白化仍依赖训练输入；不能称所有预处理完全数据无关。', '',
    '这仍是三个outer fold0 fit池内部五折OOF，相对原R4固定alpha100/weight1，而非四设置完整调参后的TD24_Ridge_CV。原四设置5%目标和基线保持不变，本轮未产生新的外层测试提升。', '',
    f"验收：135个新协方差白化/135个头、20691内部OOF行；三固定投影的登记SHA、原训练预处理、独立样本矩阵SVD白化、正规方程、单向量部署及全部逐行预测通过。白化最大差{v['max_whitening_difference']:.3g}，正规方程归一残差{v['max_normal_equation_residual']:.3g}，预测重建差{v['max_prediction_difference']:.3g}。所有135个秩有效，无数值失败。8项合成测试0.21s通过。", '',
    f"CPU控制器墙时{ex[0]['CPU_wall_seconds']:.6f}s；独立验收外层{ve[0]['CPU_wall_seconds']:.6f}s（含内部{v['CPU_wall_seconds']:.6f}s，不重复相加）。进程启动/开发/报告保存另计。新PCA/来源forward/GPU小时均0；135个协方差白化另计，既有来源训练/导出成本不含在这几秒。", '',
    '参考：[Dasgupta与Gupta，随机投影的JL理论](https://cseweb.ucsd.edu/~dasgupta/papers/jl.pdf)。固定16维及训练白化并未验证该定理的距离失真条件，不将其作为预测改善保证。', '',
    '目标仍active、未完成。R4预定Core Adaptive四设置平均相对增益仍为+2.6812%；保留所有负结果。单来源42、源keep-overlap/context暴露、反复开发、purged非gene隔离及两个Mixset设置非独立限制不变。当前无活进程，上一轮与本轮均为新证据进展，没有外部资源阻断。', '',
    '已向用户询问是否允许在下一阶段增加原论文source43的CPU表示导出，以实施固定条件桥接。当前只是范围问题，未获得回复前仍遵守现有seed42缓存约束；没有因此派发新forward或改目标状态，也不将待回复视作当前长期阻断。', '',
    f"科学锁{sha(R/'lock.json')}；结果{sha(R/'diagnostic.json')}；验收{sha(R/'verification.json')}。"]
report=ROOT/'ROUND20_RESULTS.md'
with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
paths=[R/'lock.json',R/'projections.npz',R/'diagnostic.json',R/'verification.json',R/'pytest.xml',report,Path(__file__)]
paths+=list(R.glob('execution_*.json'))+list(R.glob('verification_execution_*.json'))
write(R/'closure_receipt.json',dict(closed=True,qualified=False,goal_complete=False,goal_remains_active=True,
    own_live_process=False,verified_heads=135,new_covariance_whitenings=135,new_PCA_fits=0,new_source_forward=0,
    GPU_hours=0,outer_evaluation_started=False,prediction_averaging=False,
    files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},CPU_close_seconds=time.monotonic()-start))
for n,h in read(R/'closure_receipt.json')['files_sha256'].items():assert sha(ROOT/n)==h
write(ROOT/'GOAL_TURN_R20_AUDIT.json',dict(previous_goal_turn='progress',current_goal_turn='progress',goal_status_read='active',
    goal_complete=False,blocker=False,own_live_process=False,
    evidence='135 fixed random-subspace heads and 20691 internal rows verified; all three Core maps fail, ending this compression branch without seed selection.',
    previous_closure_sha256=sha(ROOT/'round19/closure_receipt.json'),closure_sha256=sha(R/'closure_receipt.json')))
with (ROOT/'PROGRESS.md').open('a',encoding='utf-8') as f:
    f.write(f"\n\nR20固定随机子空间闭合：135个新协方差白化/135头/20691内部OOF，三固定投影与独立SVD/正规方程/部署全部通过；Core主map42相对R4固定读出的ΔPCC为−0.017610/−0.018042/+0.013866，43/44也各自未过原资格，三gate全部false。结束随机投影分支，不挑seed/平均预测/扩维网格，不做新外层。CPU{ex[0]['CPU_wall_seconds']:.3f}s+验收{ve[0]['CPU_wall_seconds']:.3f}s；GPU/来源forward0。见ROUND20_RESULTS.md。已询问下一阶段是否允许CPU导出论文source43；回复前仍仅现有seed42缓存，未派发/改goal，不把等待可选范围回复当作阻断。上一轮与本轮均progress，当前无活进程，目标active未达5%。\n")
print(json.dumps(dict(closed=True,qualified=False,verified_heads=135,closure_sha256=sha(R/'closure_receipt.json'))))
