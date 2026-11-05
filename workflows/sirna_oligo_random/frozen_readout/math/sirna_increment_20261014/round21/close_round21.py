"""Save all positive/negative pilot evidence without another fit."""
import hashlib,json,time
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
start=time.monotonic();d=read(R/'diagnostic.json');v=read(R/'verification.json');lock=read(R/'lock.json')
assert d['qualified'] and v['qualified'] and v['passed'] and not (R/'execution.mutex').exists()
assert v['diagnostic_sha256']==sha(R/'diagnostic.json') and v['lock_sha256']==sha(R/'lock.json')
for n,h in lock['files'].items():assert sha(R/n)==h
for n,h in lock['dependencies'].items():assert sha(Path(n))==h
for n,h in d['pilot_files_sha256'].items():assert sha(R/'pilot'/n)==h
ex=[read(p) for p in R.glob('execution_*.json')];ve=[read(p) for p in R.glob('verification_execution_*.json')]
assert len(ex)==len(ve)==1 and ex[0]['returncode']==ve[0]['returncode']==0
lines=['# R21：联合坐标稀疏通过训练内初筛，尚未证明四设置增益','',
 '2026-10-15。现有source42冻结192D、无新来源推理。保留全部坐标后用同总方差尺度、单216D联合L1+L2读出；普通同维Ridge为配对控制。完整90头及独立审计通过。','',
 '|训练池|来源|读出|相对R4绝对ΔPCC|MSE下降|', '|---|---|---|---:|---:|']
for r in d['records']:lines.append(f"|{r['setting']}|{r['source']}|{r['mode']}|{r['delta_PCC_vs_R4']:+.6f}|{r['MSE_reduction_vs_R4_pct']:+.4f}%|")
lines+=['', 'Core sparse在Mixset/Takayuki的绝对PCC分别改善0.008407/0.017349，Huesken为−0.000167，近乎持平且仍完整披露负值。至少两池达到原0.005门槛，全部不低于−0.005；同维dense配对MSE条件也通过。因此这一方案获得完整四设置评价资格，不能将内部结果称为目标达成。','',
 '稀疏条件相对同维dense并非全面改善：Takayuki反而较弱；保留192D本身已经对Tak有益。Full在Hu和Tak改善，Mix仍小退；精确初始化在Mix/Tak也可改善。不能把收益全部归因miRNA监督、L1或某个专家。这不是旧group-lasso/旧full192 Ridge重新命名：前者整组关闭，本轮联合坐标L1；dense只是配对控制。','',
 '参考固定R4 alpha100/weight1的三个outer fold0 fit池内部五折OOF，尚未使用四设置外层测试结果。接下来固定原五alpha [.1,1,10,100,1000]与lambda/lambda_max=.1，进行相同内层PCC选择；新四设置协议须先锁定，不能根据本轮辅助结果换主来源或挑队列。','',
 f"验收：90新头、13794内部OOF，全部收敛；45个dense独立联合求解，最大系数差{v['max_dense_coefficient_difference']:.3g}，最大完整KKT{v['max_joint_KKT']:.3g}，部署/预测最大差{v['max_prediction_difference']:.3g}。原source/输入/划分/训练尺度SHA均核验；8项合成0.22s通过。",'',
 f"CPU控制器{ex[0]['CPU_wall_seconds']:.6f}s，独立验收外层{ve[0]['CPU_wall_seconds']:.6f}s（包含内部{v['CPU_wall_seconds']:.6f}s）。开发/启动/报告保存和既有来源训练导出另计；新GPU、source forward、PCA均0。",'',
 '依据：[Zou与Hastie，Elastic Net](https://doi.org/10.1111/j.1467-9868.2005.00503.x)。本实现对TD24只加L2，不是直接复现其全部训练算法，也不把稀疏系数解释成生物因果作用。','',
 '目标仍active未完成。训练池反复开发、单来源42、keep-overlap及context暴露、purged非gene隔离、两Mix设置非独立等限制保留。source43新导出仍未获范围回复，未执行；没有当前资源阻断。', '',
 f"锁{sha(R/'lock.json')}；结果{sha(R/'diagnostic.json')}；验收{sha(R/'verification.json')}。"]
report=ROOT/'ROUND21_RESULTS.md'
with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
paths=[R/'lock.json',R/'diagnostic.json',R/'verification.json',R/'pytest.xml',report,Path(__file__)]+list(R.glob('execution_*.json'))+list(R.glob('verification_execution_*.json'))
write(R/'closure_receipt.json',dict(closed=True,qualified=True,goal_complete=False,goal_remains_active=True,own_live_process=False,
    verified_heads=90,new_PCA_fits=0,new_source_forward=0,GPU_hours=0,outer_evaluation_started=False,
    files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths},CPU_close_seconds=time.monotonic()-start))
for n,h in read(R/'closure_receipt.json')['files_sha256'].items():assert sha(ROOT/n)==h
write(ROOT/'GOAL_TURN_R21_AUDIT.json',dict(previous_goal_turn='no progress',current_goal_turn='progress',goal_status_read='active',goal_complete=False,
    blocker=False,own_live_process=False,evidence='90 coordinate-L1/dense heads and 13794 internal predictions independently verified; fixed Core pilot qualifies for four-setting evaluation.',closure_sha256=sha(R/'closure_receipt.json')))
with (ROOT/'PROGRESS.md').open('a',encoding='utf-8') as f:f.write(f"\n\nR21完整坐标联合L1+L2初筛闭合：90新头/13794内部OOF，Core对R4固定参考ΔPCC为−.000167/+.008407/+.017349，性能与dense配对门槛均通过，全部数值通过。Full/initial/dense全报告，不把训练内资格当5%达标。CPU{ex[0]['CPU_wall_seconds']:.3f}s+验收{ve[0]['CPU_wall_seconds']:.3f}s，GPU/source forward/PCA0。见ROUND21_RESULTS.md。前一解释轮no progress，本轮progress，无活进程/阻断，goal active未完成。下一阶段需锁原五alpha、固定L1比例和四设置选择协议，不扩网格或改主来源。\n")
print(json.dumps(dict(closed=True,qualified=True,closure_sha256=sha(R/'closure_receipt.json'))))
