"""Bind verified native-matrix pilot and all prior controls; no new statistics."""
import hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
start=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'closure_receipt.json').exists()
d=read(R/'diagnostic.json');v=read(R/'verification.json');lock=read(R/'lock.json')
assert v['passed'] and v['qualified'] and d['qualified'] and v['all_numerically_qualified']
assert v['lock_sha256']==sha(R/'lock.json') and v['diagnostic_sha256']==sha(R/'diagnostic.json')
for name,h in lock['files'].items():assert sha(R/name)==h,name
for name,h in lock['dependencies'].items():assert sha(Path(name))==h,name
for name,h in d['pilot_files_sha256'].items():assert sha(R/'pilot'/name)==h,name
assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
ex=[(p,read(p)) for p in R.glob('execution_*.json')];vx=[(p,read(p)) for p in R.glob('verification_execution_*.json')]
assert len(ex)==len(vx)==1 and all(x['returncode']==0 for p,x in ex+vx)
suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==8
assert not any(int(s.get('failures',0))+int(s.get('errors',0))+int(s.get('skipped',0)) for s in suites)
fit=ex[0][1]['CPU_wall_seconds'];audit=vx[0][1]['CPU_wall_seconds'];tests=sum(float(s.get('time',0)) for s in suites)
cost=dict(CPU_controller_seconds=fit,CPU_audit_outer_seconds=audit,CPU_audit_inner_seconds_included=v['CPU_wall_seconds'],
          CPU_fit_and_audit_seconds=fit+audit,synthetic_seconds_separate=tests,new_scientific_heads=45,
          reused_dense_heads=45,independent_full_joint_certificates=45,new_source_forward=0,new_PCA_fits=0,GPU_hours=0,
          excluded='Earlier training, source export, R21 controls, startup, development and report/closure are separate.',
          execution_sha256={p.name:sha(p) for p,x in ex+vx})
save(R/'cost_receipt.json',cost)
lines=['# R25：原生矩阵系数收缩通过内部资格，尚未评价四设置', '',
       '2026-10-15。45个新头和45个旧dense控制对齐完成，全部独立证书通过。Core三池相对R4固定参考均向好，两个池达到预设ΔPCC≥.005，配对dense门槛也通过。qualified=true只允许另立完整四设置评价，不是5%目标已经达到。', '',
       '模型仍是TD24+source192的单线性头。新约束作用于六个池化区域统计×32共享通道的系数矩阵；不增加乘积、不改变源网络或原池化权重。源mean/max逐列标准化会改变未标准化空间的秩含义，结论限定于固定标准化表示。', '',
       '|训练池|来源|PCC|相对R4绝对ΔPCC|相对R4 MSE下降|匹配dense PCC|', '|---|---|---:|---:|---:|---:|']
for r in d['records']:
    lines.append(f"|{r['setting']}|{r['source']}|{r['metrics']['PCC']:.6f}|{r['delta_PCC_vs_R4']:+.6f}|{r['MSE_reduction_vs_R4_pct']:+.4f}%|{r['dense_metrics']['PCC']:.6f}|")
lines += ['', '|Core配对dense控制|ΔPCC|MSE下降|', '|---|---:|---:|']
for r in d['paired_core_records']:lines.append(f"|{r['setting']}|{r['delta_PCC']:+.6f}|{r['MSE_reduction_pct']:+.4f}%|")
lines += ['', 'Takayuki相对dense并非一定更好；全部来源/配对控制完整保存，不能把相对PCA16参考的收益都归因核范数。初始化在Mixset/Takayuki也改善，当前证据不足以单独证明miRNA监督贡献。', '',
          f"验收：45新头/6897新内部OOF，45旧dense头/6897参考OOF按原SHA复用，未重训控制。最大新预测重建差{v['max_prediction_difference']:.3g}，最大TD正规方程残差{v['max_TD_normal_equation_error']:.3g}，最大相对primal/dual gap {v['max_relative_dual_gap']:.3g}。八项合成验收及全部来源/分区/原生池化源码/输出SHA通过。", '',
          f"CPU控制器{fit:.6f}s，验收外层{audit:.6f}s（内层{v['CPU_wall_seconds']:.6f}s已包括），合计{fit+audit:.6f}s；合成{tests:.6f}s另计，旧来源/控制、启动、开发另计。GPU/新forward/PCA均0。", '',
          '当前无本轮活进程，目标active未完成。上一轮与本轮均progress，无外部阻断。下一步必须先锁同五alpha和原内部选择，再运行四设置；不按本轮或后续测试变更来源、lambda比例、rank或门槛。source43仍未获范围回复，不启动导出。', '',
          '限制：三个fit池反复开发、单source42、source keep-overlap/context暴露、purged_sequence非gene隔离。内部资格不是独立确认或SOTA。', '',
          '参考：[Bach，Trace Norm Minimization](https://www.jmlr.org/papers/v9/bach08a.html)。这里只应用经典矩阵正则化思想，不声称文献保证siRNA性能。', '',
          f"科学锁{sha(R/'lock.json')}；诊断{sha(R/'diagnostic.json')}；独立验收{sha(R/'verification.json')}。", '']
report=ROOT/'ROUND25_RESULTS.md'
with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines))
paths=[R/'lock.json',R/'diagnostic.json',R/'diagnostic_rows.csv',R/'verification.json',R/'cost_receipt.json',report,Path(__file__)]
paths += [R/n for n in lock['files']]+[p for p,x in ex+vx]
files={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
for name,h in files.items():assert sha(ROOT/name)==h
save(R/'closure_receipt.json',dict(status='closed',qualified=True,goal_complete=False,readback_passed=True,no_live_execution_mutex=True,
    files_sha256=files,all_pilot_files_verified=len(d['pilot_files_sha256']),CPU_close_seconds=time.monotonic()-start))
save(ROOT/'GOAL_TURN_R25_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
    concrete_progress='45 native coefficient trace-norm heads passed bounded internal qualification and independent certificates.',
    new_outer_test_evaluation=False,source43_scope_reply_received=False,no_active_handle=True,genuine_external_blocker=False,
    closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report)))
with (ROOT/'PROGRESS.md').open('a',encoding='utf-8') as f:
    f.write('\n\nR25原生6×32系数核范数收缩已闭合：45新头/6897内部OOF，45旧dense控制按SHA复用；全部全局证书通过。Core对R4固定参考ΔPCC +.001441/+.008828/+.027504，MSE下降+.2713%/+1.4932%/+3.4904%，性能与paired gate均true。尚未四设置评价，不能称达到5%。不改池化、不增加乘积、不重训源。CPU成本见ROUND25_RESULTS.md，GPU/forward/PCA0。上一轮及本轮progress，目标active，无活进程/阻断；下一步先锁原五alpha和完整选择协议。\n')
print(json.dumps(dict(closed=True,qualified=True,closure_sha256=sha(R/'closure_receipt.json'),cost=cost)))
