"""Record the bounded pilot outcome without rerunning its fits or statistics."""
import hashlib,json,time
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
start=time.monotonic();d=read(R/'diagnostic.json');v=read(R/'verification.json');lock=read(R/'lock.json')
assert not d['qualified'] and not v['qualified'] and v['passed'] and not (R/'execution.mutex').exists()
assert v['diagnostic_sha256']==sha(R/'diagnostic.json') and v['lock_sha256']==sha(R/'lock.json')
for n,h in lock['files'].items():assert sha(R/n)==h
for n,h in lock['dependencies'].items():assert sha(Path(n))==h
for n,h in d['pilot_files_sha256'].items():assert sha(R/'pilot'/n)==h
ex=[read(p) for p in R.glob('execution_*.json')];ve=[read(p) for p in R.glob('verification_execution_*.json')]
assert len(ex)==len(ve)==1 and ex[0]['returncode']==ve[0]['returncode']==0
nr=v['numerical_records'];n_good=sum(r['numerically_qualified'] for r in nr)
assert len(nr)==60 and n_good==49 and all(r['optimizer_success'] for r in nr)
lines=['# R18：两块自动收缩没有形成稳定提升','',
    '2026-10-15。本轮60个CPU轻量读出已完成，独立验收通过；性能资格false，数值资格也未全部满足。不进行新的外层测试、不改变门槛或继续拟合此分支。','',
    '唯一方法变化是用当前inner train的高斯剖面边际似然连续估计TD24和来源PCA16两个精度。保持原40D线性模型与训练内投影，TD-only控制只有一个精度。它不同于R11逐系数ARD，也没有新增网络/非线性或别的模型预测。','',
    '| 训练池 | 来源/控制 | 对固定R4 ΔPCC（绝对） | MSE下降 | 数值合格头 |',
    '|---|---|---:|---:|---:|']
for r in d['records']:
    part=[a for a in nr if a['setting']==r['setting'] and a['view']==r['view']]
    lines.append(f"| {r['setting']} | {r['view']} | {r['delta_PCC_vs_R4']:+.6f} | {r['MSE_reduction_vs_R4_pct']:+.4f}% | {sum(a['numerically_qualified'] for a in part)}/5 |")
lines += ['',
    'Core在Mixset改善PCC与MSE，在Huesken轻微退步，在Takayuki较明显退步。只有一个池符合绝对ΔPCC≥0.005且MSE不恶化，并且Takayuki低于−0.005下限；即使不考虑数值资格，性能门槛也失败。不能把单个池的正结果当成四设置平均≥5%的证据。', '',
    '参照为三个outer fold0训练池内部OOF中的旧R4固定alpha100/weight1，TD-only参照alpha100。不是四设置调参后强TD24的完整外层评价，不与R4外层+2.6812%直接作数值相减。其他来源与TD控制全报告，未选某个测试集较好的源拼接。', '',
    f"数值审计：60头、9196内部OOF行、105个独立梯度差分、所有原投影/数据与代码SHA均通过；预测重建最大差{v['max_prediction_difference']:.3g}，差分梯度最大误差{v['max_finite_difference_error']:.3g}。优化器60次均报告success，但只有{n_good}/60通过更严格的投影梯度≤1e-6条件；其余11个最大为{v['max_projected_gradient']:.3g}，不是1000步达限。共{sum(r['iterations'] for r in nr)}接受步，单头最多{max(r['iterations'] for r in nr)}步。保留全部末端，不延长/重启以换取通过。", '',
    '数学驻点不保证全局证据最优，更不保证PCC最优。源精度大小是给定特征尺度与高斯先验下的收缩，不代表专家的生物重要性。TD-only也在Takayuki退步，不能把这轮负结果全部归为DuplexFM表示质量或miRNA监督。', '',
    f"CPU控制器墙时{ex[0]['CPU_wall_seconds']:.6f}s、独立验收外层{ve[0]['CPU_wall_seconds']:.6f}s（内部计时重叠不再相加），9项合成0.35s；进程启动、开发与报告保存另计。新来源前向/新PCA/GPU小时均0，既有来源训练与导出不包含在这几秒中。", '',
    '参考：[MacKay, Bayesian Interpolation (1992)](https://authors.library.caltech.edu/records/r7qgh-q6g10)。仅借鉴边际似然确定正则/噪声的框架，不声称该文证明本应用有效。', '',
    '当前5%目标仍active且未达到。本轮是受限负结果，不是阻断。源seed42/keep-overlap/context暴露、反复开发、purged非gene隔离和两个Mixset设置非独立等限制不变。没有重新训练旧来源或调用旧外层统计。', '',
    '用户关于论文旧增益的问题与此轮分开：旧source43/P12/P13、行级划分、过滤与基线不同。要分摊原因需固定条件桥接，不能从本轮正则负结果声称已证明旧模型无可迁移信息。详见PAPER_GAIN_COMPARISON_20261015.md。', '',
    f"科学锁{sha(R/'lock.json')}；结果{sha(R/'diagnostic.json')}；验收{sha(R/'verification.json')}。"]
report=ROOT/'ROUND18_RESULTS.md'
with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
paths=[R/'lock.json',R/'diagnostic.json',R/'verification.json',R/'pytest.xml',report,Path(__file__)]
paths+=list(R.glob('execution_*.json'))+list(R.glob('verification_execution_*.json'))
write(R/'closure_receipt.json',dict(closed=True,qualified=False,goal_complete=False,goal_remains_active=True,
    own_live_process=False,verified_heads=60,numerically_qualified_heads=n_good,new_PCA_fits=0,
    new_source_forward=0,GPU_hours=0,outer_evaluation_started=False,
    files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},CPU_close_seconds=time.monotonic()-start))
for n,h in read(R/'closure_receipt.json')['files_sha256'].items():assert sha(ROOT/n)==h
write(ROOT/'GOAL_TURN_R18_AUDIT.json',dict(previous_goal_turn='progress',current_goal_turn='progress',goal_status_read='active',
    goal_complete=False,blocker=False,own_live_process=False,
    evidence='60 two-block empirical Bayes pilot heads, independent audit; mixed internal gains fail fixed Core gate. User paper comparison supported by original table and same-protocol readout controls.',
    closure_sha256=sha(R/'closure_receipt.json')))
with (ROOT/'PROGRESS.md').open('a',encoding='utf-8') as f:
    f.write(f"\n\nR18两块经验Bayes收缩已闭合：60头/9196内部OOF/105独立梯度差分，原投影与全部SHA通过。Core对R4固定alpha100的ΔPCC为−0.000623/+0.018276/−0.021965，MSE下降−0.1319%/+2.6279%/−7.5299%，只有Mixset改善，性能资格false。优化器全success但49/60通过预定严格投影梯度；11个未通过、最大8.634e-6，不调容差/重启。CPU{ex[0]['CPU_wall_seconds']:.3f}s+验收{ve[0]['CPU_wall_seconds']:.3f}s，GPU/来源forward/PCA均0，无新外层。见ROUND18_RESULTS.md。用户论文增益追问已按原Table4核对，旧新来源/划分/基线不同，不作单因素归因。上一轮及本轮progress，无活进程或阻断，目标active未达5%。\n")
print(json.dumps(dict(closed=True,qualified=False,numerically_qualified_heads=n_good,closure_sha256=sha(R/'closure_receipt.json'))))
