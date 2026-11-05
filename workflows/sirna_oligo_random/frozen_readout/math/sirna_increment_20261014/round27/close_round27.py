"""Record the complete negative hard-router experiment; no new fitting."""
import hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parent


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


start=time.monotonic()
assert not (R/'execution.mutex').exists() and not (R/'closure_receipt.json').exists()
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
assert sum(int(s.get('tests',0)) for s in suites)==10
assert not any(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites)
fit=ex[0][1]['CPU_wall_seconds'];audit=vx[0][1]['CPU_wall_seconds'];tests=sum(float(s.get('time',0)) for s in suites)
cost=dict(CPU_controller_seconds=fit,CPU_audit_outer_seconds=audit,CPU_audit_inner_seconds_included=v['CPU_wall_seconds'],
          CPU_fit_and_audit_seconds=fit+audit,synthetic_seconds_separate=tests,
          fitted_router_estimators=60,new_linear_solves=d['new_linear_solves'],reused_null_heads=60,
          independent_normal_equations=v['candidate_normal_equations'],independent_selected_lstsq=60,
          new_internal_OOF_rows=9196,new_source_forward=0,new_PCA=0,GPU_hours=0,
          excluded='Earlier training/export/PCA/R4 reference, interpreter startup/imports, development and report/closure separate.',
          execution_sha256={p.name:sha(p) for p,x in ex+vx})
save(R/'cost_receipt.json',cost)
lines=['# R27：监督式单层硬路由没有通过内部资格', '',
       '2026-10-15。60个内层路由估计器、2260个新线性候选解及60个原R4空分裂参考已闭合；9196行内部OOF通过独立验收。performance_gate=false、source_increment_gate=false，qualified=false。结束本轮，不做新四设置外层评价、不扩门限/正则网格。', '',
       '方法为一个单层模型树：训练集在TD24坐标的固定分位门限中选择，联合拟合共享基础系数和强收缩的分支系数。每个样本只有一条分支生效，来源网络保持冻结。这不同于原KMeans路由或按OOF分支误差训练分类器，也不等同于重训原五expert MoE。', '',
       '|训练池|来源|硬路由PCC|原全局读出PCC|绝对ΔPCC|MSE下降|', '|---|---|---:|---:|---:|---:|']
for row in d['records']:
    lines.append(f"|{row['setting']}|{row['source']}|{row['metrics']['PCC']:.6f}|{row['reference']['PCC']:.6f}|{row['delta_PCC_vs_R4']:+.6f}|{row['MSE_reduction_vs_R4_pct']:+.4f}%|")
lines += ['', '以上是固定alpha100的三个fit池内部验证，不是原四设置目标指标。不能用它重新计算或替代四设置平均增益。', '',
          '|Core 对同型TD24-only硬路由|绝对ΔPCC|MSE下降|', '|---|---:|---:|']
for row in d['paired_gated_TD']:
    lines.append(f"|{row['setting']}|{row['delta_PCC']:+.6f}|{row['MSE_reduction_pct']:+.4f}%|")
lines += ['', 'Core对同型TD路由在Huesken/Takayuki仍有增量，Mixset反而更差，三池平均相对MSE增量未过门槛。因此不能把简单分段回归自身的改善算成稳定的DuplexFM收益。Full与精确初始化也没有全面正结果，不事后换主来源。', '',
          '|来源|训练池|五内折选择的TD坐标（从0起）|平均训练惩罚目标下降|', '|---|---|---|---:|']
train_diagnostics=[]
for source in ('core_real','full_real','initial_core','TD_only'):
    for setting in ('Huesken','Mixset','Takayuki'):
        states=[read(R/'pilot'/setting/source/f'inner{j}'/'state.json') for j in range(5)]
        chosen=[s['candidates'][s['selected']] for s in states]
        drops=[100*(1-c['objective']/s['candidates'][0]['objective']) for s,c in zip(states,chosen)]
        rec=dict(source=source,setting=setting,features=[c['feature'] for c in chosen],mean_train_objective_reduction_pct=sum(drops)/5)
        train_diagnostics.append(rec)
        lines.append(f"|{source}|{setting}|{rec['features']}|{rec['mean_train_objective_reduction_pct']:+.4f}%|")
lines += ['', '训练惩罚目标有所下降，但未稳定转化为OOF改善，与有限样本下分裂选择方差相容；本试验没有单独证明其因果来源。分裂变量索引变化也不等同于功能路由一定不稳定，因为TD变量可能相关。不能据此断言所有硬MoE无效。', '',
          f"验收：{v['candidate_normal_equations']}个候选完整正规方程、60个选中头的独立增广最小二乘、门限枚举/训练支持/最优选择/one-hot部署/来源身份及全部SHA通过。最大正规方程残差{v['max_normal_equation_error']:.3g}，独立系数差{v['max_independent_coefficient_difference']:.3g}，预测重建差{v['max_prediction_difference']:.3g}。十项纯合成测试通过。数值收敛不是失败原因。", '',
          f"成本：CPU控制器{fit:.6f}s，独立验收外层{audit:.6f}s（内层{v['CPU_wall_seconds']:.6f}s包含在内），合计{fit+audit:.6f}s；合成{tests:.6f}s另计。此前来源训练/导出、PCA、R4参考、解释器启动/导入、开发与收尾另列，不把本次几秒头拟合当成端到端成本。新GPU、source forward、PCA均0。", '',
          '当前四设置主方案仍是R26的+3.2771%，5%目标未达。原验证池反复开发、来源keep-overlap/context暴露、单source42、purged非gene隔离等限制保留；不是新盲外部确认。上一轮和本轮均为progress，无活执行句柄或外部阻断，goal保持active。source43范围问题尚无回复，未导出。', '',
          '背景：[Zeileis/Hothorn/Hornik，Model-based Recursive Partitioning](https://www.zeileis.org/papers/Zeileis%2BHothorn%2BHornik-2008.pdf)。这里只借鉴分段参数思想，不声称实现该文的统计检验/递归算法或获得理论性能保证。', '',
          f"科学锁{sha(R/'lock.json')}；诊断{sha(R/'diagnostic.json')}；独立验收{sha(R/'verification.json')}。", '']
report=ROOT/'ROUND27_RESULTS.md'
with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
save(R/'training_diagnostics.json',dict(records=train_diagnostics,posthoc_descriptive_only=True,no_new_fit=True,no_outer_test=True))
paths=[R/'lock.json',R/'diagnostic.json',R/'diagnostic_rows.csv',R/'verification.json',R/'cost_receipt.json',R/'training_diagnostics.json',report,Path(__file__)]
paths += [R/n for n in lock['files']]+[p for p,x in ex+vx]
files={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
for name,h in files.items():assert sha(ROOT/name)==h
save(R/'closure_receipt.json',dict(status='closed',qualified=False,goal_complete=False,readback_passed=True,
    no_live_execution_mutex=True,files_sha256=files,all_pilot_files_verified=len(d['pilot_files_sha256']),CPU_close_seconds=time.monotonic()-start))
save(ROOT/'GOAL_TURN_R27_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
    concrete_progress='A supervised single-split hard router failed predefined internal performance and source-increment gates; all 2320 candidate certificates passed.',
    new_outer_test_evaluation=False,source43_scope_reply_received=False,no_active_handle=True,genuine_external_blocker=False,
    closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report)))
with (ROOT/'PROGRESS.md').open('a',encoding='utf-8') as f:
    f.write('\n\nR27监督式单层硬路由闭合：60个内层估计器、2260新候选解+60原R4空分裂复用头、9196内部OOF；全部2320正规方程及60独立lstsq、one-hot部署/SHA通过。Core对全局固定读出ΔPCC −.004491/+.001531/+.002650，MSE下降−.8009%/+.2210%/+.4971%；performance与source_increment gate均false，不做外层或扩门限/惩罚网格。四设置现有主结果仍R26+3.2771%，目标active未达5%。本轮progress、无活进程/阻断，CPU成本见ROUND27_RESULTS.md；GPU/forward/PCA0。\n')
print(json.dumps(dict(closed=True,qualified=False,closure_sha256=sha(R/'closure_receipt.json'),cost=cost)))
