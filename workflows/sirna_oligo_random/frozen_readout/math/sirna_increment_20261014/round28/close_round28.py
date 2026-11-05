"""Close verified source-quadratic evidence without new statistical fitting."""
import hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parent


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')


started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'closure_receipt.json').exists()
d=read(R/'diagnostic.json');v=read(R/'verification.json');lock=read(R/'lock.json')
assert v['passed'] and v['all_numerically_qualified'] and not d['qualified'] and not v['qualified']
assert v['lock_sha256']==sha(R/'lock.json') and v['diagnostic_sha256']==sha(R/'diagnostic.json')
for name,h in lock['files'].items():assert sha(R/name)==h,name
for name,h in lock['dependencies'].items():assert sha(Path(name))==h,name
for name,h in d['pilot_files_sha256'].items():assert sha(R/'pilot'/name)==h,name
assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
ex=[(p,read(p)) for p in R.glob('execution_*.json')];vx=[(p,read(p)) for p in R.glob('verification_execution_*.json')]
assert len(ex)==len(vx)==1 and all(x['returncode']==0 and x['error'] is None for p,x in ex+vx)
suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
assert sum(int(s.get('tests',0)) for s in suites)==8
assert not any(int(s.get('errors',0))+int(s.get('failures',0))+int(s.get('skipped',0)) for s in suites)
fit=ex[0][1]['CPU_wall_seconds'];audit=vx[0][1]['CPU_wall_seconds'];tests=sum(float(s.get('time',0)) for s in suites)
cost=dict(CPU_controller_seconds=fit,CPU_audit_outer_seconds=audit,CPU_audit_inner_seconds_included=v['CPU_wall_seconds'],
          CPU_fit_and_audit_seconds=fit+audit,synthetic_seconds_separate=tests,new_maps=45,new_heads=45,
          reused_linear_heads=45,reused_TD_quadratic_heads=15,new_internal_OOF_rows=6897,unique_reused_TD_rows=2299,
          independent_new_head_lstsq=45,independent_reused_TD_normal_equations=15,
          new_source_forward=0,new_PCA=0,GPU_hours=0,
          excluded='Earlier source training/export/PCA/R4/R5 controls, interpreter startup/imports, development and report/closure separate.',
          execution_sha256={p.name:sha(p) for p,x in ex+vx})
save(R/'cost_receipt.json',cost)
lines=['# R28：来源内部二阶响应面没有通过内部资格', '',
       '2026-10-15。45个新176D头及6897行内部OOF完成并通过独立验收。主Core仅在Takayuki改善，Huesken/Mixset下降；performance_gate和source_increment_gate均false，不进入新的四设置测试，也不扩大交互组合/阶数/正则网格。', '',
       '本轮在原TD24+PCA16的40个线性列外，加入来源PC之间的136个平方/交互。乘积只在轻量读出中使用，源编码器保持冻结；不会改变原Raw41的输入通道。它与此前TD×PC、单变量自然样条及原生线性矩阵正则化不同。新增辅助块训练方差约16，总辅助方差增加至约32，容量增加明确披露。', '',
       '|训练池|来源|二阶PCC|原线性PCC|绝对ΔPCC|相对线性MSE下降|TD二阶PCC|', '|---|---|---:|---:|---:|---:|---:|']
for row in d['records']:
    lines.append(f"|{row['setting']}|{row['source']}|{row['metrics']['PCC']:.6f}|{row['reference']['PCC']:.6f}|{row['delta_PCC_vs_R4']:+.6f}|{row['MSE_reduction_vs_R4_pct']:+.4f}%|{row['TD_quadratic_metrics']['PCC']:.6f}|")
lines += ['', '以上是原三个fit池的固定alpha100内部OOF，不能将这些数值当作四设置外层目标或新外部验证。', '',
          '|Core 对原TD24二阶控制|绝对ΔPCC|MSE下降|', '|---|---:|---:|']
for row in d['paired_TD_quadratic']:
    lines.append(f"|{row['setting']}|{row['delta_PCC']:+.6f}|{row['MSE_reduction_pct']:+.4f}%|")
lines += ['', '对同类TD二阶控制，Core在Huesken/Mixset同样更差。Full也主要只在Takayuki改善，初始化结果不统一。因此不能将某队列的正结果推广成来源内部交互普遍有效，也不能事后只保留有利来源/数据集。本结果与额外二阶自由度在部分队列增加估计方差相容，但没有单独证明该原因。', '',
          '复用与计数：45新映射、45新效率头、6897新OOF；45个R4线性头及15个R5 TD二阶头按原SHA复用。TD参考为2299条唯一内部验证出现，在三来源中重复引用。没有重新运行旧来源、PCA、TD拟合或旧指标兼容器。R5原失败/兼容锁与成功验收回执完整保留。', '',
          f"数值：45个新正规方程和独立增广最小二乘、15个旧TD正规方程/预测、全部训练统计/坐标/对称二次型部署/覆盖/SHA通过。最大新正规方程残差{v['max_normal_equation_error']:.3g}，独立系数差{v['max_independent_coefficient_difference']:.3g}，部署/预测重建差{v['max_prediction_difference']:.3g}；TD预测差{v['max_TD_prediction_difference']:.3g}。八项合成测试通过，不能把负结果解释为未收敛。", '',
          f"成本：CPU控制器{fit:.6f}s，独立验收外层{audit:.6f}s（内层{v['CPU_wall_seconds']:.6f}s已包括），合计{fit+audit:.6f}s；纯合成{tests:.6f}s另计。旧训练/导出/PCA/对照、启动/导入、开发与收尾另列，不把缓存读出几秒当作端到端成本。新source forward、PCA和GPU均0。", '',
          '当前四设置主方案仍为R26，平均相对PCC增益+3.2771%，未达5%。上一轮及本轮均progress，当前无活执行句柄或外部阻断，目标保持active。source43范围问题未得到回复，未执行导出。', '',
          '限制继续保留：单source42、验证池反复开发、source keep-overlap/context暴露、purged非gene隔离、Mix两个设置不独立。新监督容量不能替代独立确认。', '',
          '数学背景：[官方多项式特征说明](https://scikit-learn.org/1.7/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)。本轮标准化与缩放为固定实现，并非文献对siRNA效果的保证。', '',
          f"科学锁{sha(R/'lock.json')}；诊断{sha(R/'diagnostic.json')}；独立验收{sha(R/'verification.json')}。", '']
report=ROOT/'ROUND28_RESULTS.md'
with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
paths=[R/'lock.json',R/'diagnostic.json',R/'diagnostic_rows.csv',R/'verification.json',R/'cost_receipt.json',report,Path(__file__)]
paths += [R/n for n in lock['files']]+[p for p,x in ex+vx]
files={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
for name,h in files.items():assert sha(ROOT/name)==h
save(R/'closure_receipt.json',dict(status='closed',qualified=False,goal_complete=False,readback_passed=True,no_live_execution_mutex=True,
    files_sha256=files,all_pilot_files_verified=len(d['pilot_files_sha256']),CPU_close_seconds=time.monotonic()-started))
save(ROOT/'GOAL_TURN_R28_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
    concrete_progress='45 fixed source-quadratic heads failed the internal gates; independent source/TD identities, numerical solves and full negative results closed.',
    new_outer_test_evaluation=False,source43_scope_reply_received=False,no_active_handle=True,genuine_external_blocker=False,
    closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report)))
with (ROOT/'PROGRESS.md').open('a',encoding='utf-8') as f:
    f.write('\n\nR28来源内部二阶响应面闭合：45新176D头/45训练二阶映射/6897新内部OOF，45原线性及15原TD二阶控制按SHA复用；全部45独立lstsq、15TD正规方程/预测、部署/SHA通过。Core对全局线性ΔPCC −.022738/−.026782/+.014647，MSE下降−4.3347%/−4.3216%/+2.2980%；两项gate均false，结束该因素，不做外层/扩阶数或正则网格。四设置主结果仍R26+3.2771%，目标active未达5%。本轮progress，无活进程/阻断；成本见ROUND28_RESULTS.md，GPU/forward/PCA0。\n')
print(json.dumps(dict(closed=True,qualified=False,closure_sha256=sha(R/'closure_receipt.json'),cost=cost)))
