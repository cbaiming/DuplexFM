"""Record a negative pilot and post-hoc train/OOF fit-gap description."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import importlib.util,json,time,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent; ROOT=R.parent; P=ROOT/'round4'
spec=importlib.util.spec_from_file_location('_r16_close_io',R/'run.py'); run=importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
spec2=importlib.util.spec_from_file_location('_r16_close_audit',R/'verify.py'); audit=importlib.util.module_from_spec(spec2); spec2.loader.exec_module(audit)
io=run.io; started=time.monotonic(); lock=run.validate(); d=io.read(R/'diagnostic.json'); v=io.read(R/'verification.json')
assert v['passed'] and not d['qualified'] and not v['qualified'] and not (R/'execution.mutex').exists()
assert v['diagnostic_sha256']==io.sha(R/'diagnostic.json') and v['lock_sha256']==io.sha(R/'lock.json')
io.check_files(R/'pilot',d['pilot_files_sha256'])
ex=[io.read(p) for p in R.glob('execution_*.json')]; ve=[io.read(p) for p in R.glob('verification_execution_*.json')]
assert len(ex)==len(ve)==1 and ex[0]['returncode']==ve[0]['returncode']==0
rows,xx,_=run.old.inputs(); idx={r['sample_id']:i for i,r in enumerate(rows)}; plan=io.read(P/'plan.json'); gaps=[]; states=[]
for setting in run.SETTINGS:
    item=next(f for f in plan['folds'] if f['setting']==setting and f['fold']==0); fi=np.array([idx[s] for s in item['fit_ids']]); y=np.array([float(rows[i]['label']) for i in fi])
    for view in run.VIEWS:
        source='core_real' if view=='TD24' else view; x=xx[source][fi]
        reference=io.read(P/'results/folds'/setting/'fold0'/source/'transforms.json')
        with np.load(P/'results/folds'/setting/'fold0'/source/'arrays.npz',allow_pickle=False) as prior:
            for j,split in enumerate(item['inner']):
                tr=np.array(split['train']); va=np.array(split['valid']); path=R/'pilot'/setting/view/f'inner{j}'; state=io.read(path/'state.json'); states.append(state)
                z=run.old.transform(x[tr],reference['inner'][j]); k=3 if view=='TD24' else 18
                oldpred=z@prior['inner_coef'][j,k]+prior['inner_intercept'][j]
                if view=='TD24': z=z[:,:24]
                with np.load(path/'model.npz',allow_pickle=False) as a:
                    newpred=state['y_mean']+state['y_scale']*audit.evaluate(z,a['theta']); held=a['prediction']
                oldm=run.stats(y[tr],oldpred); newm=run.stats(y[tr],newpred); heldm=run.stats(y[va],held)
                gaps.append(dict(setting=setting,view=view,inner=j,n_train=len(tr),n_valid=len(va),
                    train_MLP_PCC=newm['PCC'],train_Ridge_PCC=oldm['PCC'],train_MLP_MSE=newm['MSE'],train_Ridge_MSE=oldm['MSE'],
                    valid_MLP_PCC=heldm['PCC'],valid_MLP_MSE=heldm['MSE'],
                    n_iter=state['n_iter'],numerically_qualified=state['qualified_numerically']))
assert len(states)==len(gaps)==60
io.write_csv(R/'posthoc_train_fit_gap.csv',gaps)
numeric=dict(total_fits=60,numerically_qualified=sum(s['qualified_numerically'] for s in states),
    solver_success=sum(s['optimizer_success'] for s in states),iteration_cap_count=sum(s['n_iter']==1000 for s in states),
    total_iterations=sum(s['n_iter'] for s in states),max_abs_gradient=max(s['max_abs_gradient'] for s in states),
    unqualified=[dict(setting=s['setting'],view=s['view'],inner=s['inner'],n_iter=s['n_iter'],status=s['optimizer_status'],
        max_abs_gradient=s['max_abs_gradient'],message=s['optimizer_message']) for s in states if not s['qualified_numerically']])
io.write(R/'numeric_status.json',numeric)
aggregation=[]
for setting in run.SETTINGS:
    for view in run.VIEWS:
        group=[r for r in gaps if r['setting']==setting and r['view']==view]
        aggregation.append(dict(setting=setting,view=view,
            mean_train_MLP_PCC=np.mean([r['train_MLP_PCC'] for r in group]),
            mean_train_Ridge_PCC=np.mean([r['train_Ridge_PCC'] for r in group]),
            mean_training_MSE_ratio=np.mean([r['train_MLP_MSE']/r['train_Ridge_MSE'] for r in group]),
            qualified_fits=sum(r['numerically_qualified'] for r in group)))
io.write(R/'posthoc_gap_summary.json',dict(posthoc_not_used_for_gate=True,training_only=True,new_fits=0,rows=aggregation))
lines=['# R16：浅层联合MLP没有通过训练内资格','',
       '2026-10-15。固定Core/Full/精确Core初始化的40D输入，单16单元tanh读出；TD24同类控制24D。60个CPU小头完成并独立验收，性能资格与数值资格均未通过，不采用，也不运行四设置新外层评价。', '',
       '| 训练池 | 来源/控制 | OOF ΔPCC（绝对） | MSE相对下降 | 数值合格拟合 |', '|---|---|---:|---:|---:|']
for row in d['records']:
    group=[s for s in states if s['setting']==row['setting'] and s['view']==row['view']]
    lines.append(f"| {row['setting']} | {row['view']} | {row['delta_PCC']:+.6f} | {row['MSE_relative_reduction_pct']:+.4f}% | {sum(s['qualified_numerically'] for s in group)}/5 |")
lines += ['',
    '表中参考是相同内部折、相同冻结投影下固定alpha100/weight1的Ridge，TD控制参考固定alpha100的TD-only；不是用户目标中调参后TD24基线。三个fit池的结果不能冒充四设置测试结果。全部十二组合的PCC方向为负，Core三池均低于−0.005的资格下限。', '',
    f"数值状态：{numeric['numerically_qualified']}/60拟合符合预先固定的solver success且梯度≤1e-4；{numeric['iteration_cap_count']}个达到1000迭代上限，最大末端梯度{numeric['max_abs_gradient']:.6g}，合计{numeric['total_iterations']}个接受更新。所有未收敛记录和参数保留，不提高迭代上限、不更换正则或初始种子。独立验收通过说明产物与计算一致，不表示所有训练已充分收敛。", '',
    '因此，本轮不能证明所有MLP或非线性读出无效，只能说明这个固定的小模型配方不支持继续扩展。它与R3旧160D论文式P13不同，也不构成源模型架构的单因素比较。', '',
    '追加的训练拟合差距描述是在结果已知后计算，只读取已保存参数和相同inner train标签，不增加拟合或改变资格：', '',
    '| 训练池 | Core MLP训练PCC折均值 | Core Ridge训练PCC折均值 | 训练MSE比（MLP/Ridge）折均值 |', '|---|---:|---:|---:|']
for g in aggregation:
    if g['view']=='core_real': lines.append(f"| {g['setting']} | {g['mean_train_MLP_PCC']:.6f} | {g['mean_train_Ridge_PCC']:.6f} | {g['mean_training_MSE_ratio']:.6f} |")
lines += ['',
    '训练拟合与OOF表现须一起看；即使训练误差下降，也不等于有更多可泛化的miRNA迁移信息。完整四视图逐折训练/验证描述保存于posthoc_train_fit_gap.csv，未用于换参数或选测试赢家。', '',
    f"独立审计覆盖60初始向量、60目标/全部解析梯度、180方向有限差分、9196内部OOF行及原PCA/源/划分SHA。最大前向差{v['max_prediction_difference']:.3g}，最大方向导数差{v['max_directional_derivative_difference']:.3g}。参数维度673（来源）/417（TD），不能称容量完全匹配。", '',
    f"CPU控制器墙时{ex[0]['CPU_wall_seconds']:.6f}s；独立验收外层{ve[0]['CPU_wall_seconds']:.6f}s（其中内部{v['CPU_wall_seconds']:.6f}s不重复计）；另有本文件生成与事后训练拟合描述、合成测试、进程启动和开发开销。GPU小时0，新来源前向0，新PCA0，原源训练/导出成本未包含。", '',
    '当前目标保持active且未达5%；本轮没有新的外层性能改进。此前R4预定Core Adaptive平均+2.6812%，R5次要Full Uniform二阶方案+3.1633%尚未确认。重复开发、源keep-overlap/context暴露、单来源42、purged非gene隔离与两Mixset设置非独立等限制持续保留。', '',
    f"科学锁：{io.sha(R/'lock.json')}；诊断：{io.sha(R/'diagnostic.json')}；独立验收：{io.sha(R/'verification.json')}。", '',
    '下一因素尚未登记，不将非凸末端参数当作全局最优，也不由当前负结果自动扩网格。']
report=ROOT/'ROUND16_RESULTS.md'; io.atomic(report,('\n'.join(lines)+'\n').encode('utf-8'))
paths=[R/'lock.json',R/'diagnostic.json',R/'verification.json',R/'numeric_status.json',R/'posthoc_train_fit_gap.csv',R/'posthoc_gap_summary.json',Path(__file__),report]
paths+=list(R.glob('execution_*.json'))+list(R.glob('verification_execution_*.json'))
io.write(R/'closure_receipt.json',dict(closed=True,qualified=False,goal_complete=False,goal_remains_active=True,
    own_live_process=False,new_efficiency_fits=60,new_PCA_fits=0,new_source_forward=0,GPU_hours=0,
    outer_evaluation_started=False,CPU_closure_with_train_gap_seconds=time.monotonic()-started,
    files_sha256={p.relative_to(ROOT).as_posix():io.sha(p) for p in paths}))
io.check_files(ROOT,io.read(R/'closure_receipt.json')['files_sha256'])
io.write(ROOT/'GOAL_TURN_R16_AUDIT.json',dict(previous_goal_turn='progress',current_goal_turn='progress',
    previous_evidence='R15 geometry qualification closed with independent audit and the normalization factor was stopped.',
    current_evidence='60 preregistered CPU neural readouts, full control results and independent numerical audit; fixed metric and numerical gates both false.',
    goal_status_read='active',goal_complete=False,blocker=False,own_live_process=False,
    closure_sha256=io.sha(R/'closure_receipt.json'),CPU_only=True,new_source_forward=0))
with (ROOT/'PROGRESS.md').open('a',encoding='utf-8') as f:
    f.write(f"\n\nR16固定16单元tanh联合小头完成：60新MLP/9196内部OOF，Core在Hu/Mix/Tak相对同alpha100固定Ridge的绝对ΔPCC为−0.143325/−0.042163/−0.083498，MSE增加43.3644%/14.5075%/20.8363%。全部12组合PCC负向；{numeric['numerically_qualified']}/60数值合格、{numeric['iteration_cap_count']}个迭代达限，原末端/日志完整保留，不宣称全部收敛。性能/数值资格均false，不运行新外层、不调整正则/宽度/步数。独立初值/前向/梯度/180方向差分与SHA通过；追加训练拟合差距只作事后描述。控制器CPU{ex[0]['CPU_wall_seconds']:.3f}s，验收外层{ve[0]['CPU_wall_seconds']:.3f}s，GPU/新source forward0。见ROUND16_RESULTS.md及closure。上一轮及本轮progress，无活进程或外部阻断，目标active未达5%。\n")
print(json.dumps(dict(closed=True,qualified=False,numeric=numeric,closure_sha256=io.sha(R/'closure_receipt.json'))))
