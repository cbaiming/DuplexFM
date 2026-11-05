"""Close the already verified pilot, without new fitting or outer statistics."""
import hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    started=time.monotonic();assert not (R/'closure_receipt.json').exists() and not (R/'execution.mutex').exists()
    d=read(R/'diagnostic.json');v=read(R/'verification.json');lock=read(R/'lock.json')
    assert v['passed'] and v['diagnostic_sha256']==sha(R/'diagnostic.json') and v['lock_sha256']==sha(R/'lock.json')
    assert not d['qualified'] and not d['performance_gate'] and not d['contrastive_gate'] and v['all_numerically_qualified']
    for name,h in lock['files'].items():assert sha(R/name)==h,name
    for name,h in lock['dependencies'].items():assert sha(Path(name))==h,name
    for name,h in d['pilot_files_sha256'].items():assert sha(R/'pilot'/name)==h,name
    executions=sorted(R.glob('execution_*.json'));audits=sorted(R.glob('verification_execution_*.json'))
    assert len(executions)==len(audits)==1
    e,a=read(executions[0]),read(audits[0]);assert e['returncode']==a['returncode']==0
    tests=sum(float(s.get('time',0)) for s in ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    cost=dict(CPU_controller_seconds=e['CPU_wall_seconds'],CPU_audit_outer_seconds=a['CPU_wall_seconds'],
        CPU_audit_inner_seconds_included=v['CPU_wall_seconds'],CPU_fit_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],
        synthetic_seconds_separate=tests,new_heads=45,new_scientific_spectral_decompositions=30,new_projection_standardizations=45,
        reused_reference_heads=45,new_internal_OOF_rows=6897,independent_augmented_LS_solutions=45,
        new_source_forward=0,GPU_hours=0,old_source_training_and_cache_costs_separate=True,
        excluded='Earlier source training/export, old R4 references, process startup/imports, development and report/closure.',
        execution_sha256={p.name:sha(p) for p in executions+audits})
    write(R/'cost_receipt.json',cost)
    lines=['# R34：精确初始化背景的对比PCA结果','',
        '2026-10-15。45个固定CPU读出及独立验收已闭合。主Core-contrast的Huesken/Takayuki小幅退步、Mixset小幅改善；performance=false、contrastive=false，最终qualified=false，不进入新四设置外层测试。不改gamma、16维或alpha，也不按pooled对照的较好结果替换本轮主方案。当前完整四设置主结果仍R26平均相对PCC +3.2771%，5%目标未达到。','',
        '固定方法：从原Core192D训练表示协方差中减去精确初始化的协方差，初始化协方差按trace匹配。取最大的16个正特征方向，再拼TD24并用alpha100的单Ridge。所有协方差、尺度与头均在当前inner train内求取。gamma完全由训练输入决定，无标签选择或超参数搜索。数学背景为[Abid等的对比PCA](https://www.nature.com/articles/s41467-018-04608-8)；此处是固定强度的预测用途变体，原文不保证本任务提升。','',
        'Core-pooled以两协方差相加作控制；initial-contrast用同一对比方向读取初始化。后者的投影仍受trained协方差影响，不是独立的未接触来源任务对照。三者部署各自只需一个编码器，线性读出可折成216维系数与截距；Core部署不需要再次计算初始化表示。','',
        '|训练内部池|条件|原R4对应PCC|新PCC|绝对ΔPCC|相对原R4的MSE下降|',
        '|---|---|---:|---:|---:|---:|']
    for r in d['records']:
        lines.append(f"|{r['setting']}|{r['condition']}|{r['reference']['PCC']:.6f}|{r['metrics']['PCC']:.6f}|{r['delta_PCC_vs_R4']:+.6f}|{r['MSE_reduction_vs_R4_pct']:+.4f}%|")
    lines+=['','以上参考分别是Core或精确初始化的原普通PCA16固定读出；不是原四设置CV汇总，也不是全部测试数据。辅助对照的正值完整保存，但不事后切换主方案。','',
        '|训练内部池|Core-contrast相对同alpha TD24的绝对ΔPCC|MSE下降|相对Core-pooled的MSE下降|',
        '|---|---:|---:|---:|']
    for p in d['pairs']:
        lines.append(f"|{p['setting']}|{p['delta_PCC_vs_TD']:+.6f}|{p['MSE_reduction_vs_TD_pct']:+.4f}%|{p['MSE_reduction_vs_pooled_pct']:+.4f}%|")
    lines+=['',
        '结果说明：强调相对初始化富集的方差方向，没有在三个开发池获得稳定优势。本轮不支持以该规则替换普通压缩；这不等于miRNA监督无效，也不能把方差差异当作生物机制。相同网络的通道在训练中仍可改变含义，协方差差分只是一种统计假设。','',
        f"全部15个对比投影有{v['minimum_contrast_positive_directions']}–{v['maximum_contrast_positive_directions']}个数值合格的正方向，16维门槛通过；因此此次失败不是没有足够正谱或求解失败。最大特征方程误差{v['max_eigen_equation_error']:.3g}，正规方程误差{v['max_normal_equation_error']:.3g}，独立系数差{v['max_coefficient_difference']:.3g}，单向量部署误差{v['max_single_vector_deployment_difference']:.3g}。",'',
        '范围与验收：原outer fold0 fit池1564/319/416，共2299条；30新谱分解、45新投影标准化/效率头、6897内部OOF，每条件每池exact-once；45旧R4头只读核对。8项合成测试通过，30独立谱、45独立增广最小二乘、原PCA与原头、身份/行序/全部SHA均通过。没有新来源forward、来源模型更新、GPU、网络请求训练或外层统计。','',
        f"成本：CPU控制器{e['CPU_wall_seconds']:.6f}秒，独立验收外层{a['CPU_wall_seconds']:.6f}秒（包含其内层{v['CPU_wall_seconds']:.6f}秒，不重复相加），合计{cost['CPU_fit_and_audit_seconds']:.6f}秒；合成{tests:.6f}秒另列。旧来源训练/缓存、启动导入、开发和收尾单列，不能把这个耗时称为端到端成本。",'',
        '统计限制：单source42、多个已反复开发池、源keep-overlap/context暴露、当前purged sequence不是gene隔离、两个Mixset设置相关。新初始化投影没有独立因果含义；未达到5%且没有新鲜盲确认，不能标目标完成或SOTA。source43范围问题未获新回复，没有新导出。','',
        f"科学锁：{sha(R/'lock.json')}。诊断：{sha(R/'diagnostic.json')}。独立验收：{sha(R/'verification.json')}。"]
    report=B/'ROUND34_RESULTS.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    paths=[R/'lock.json',R/'diagnostic.json',R/'diagnostic_rows.csv',R/'verification.json',R/'cost_receipt.json',report,Path(__file__)]+[R/n for n in lock['files']]+executions+audits
    manifest={p.relative_to(B).as_posix():sha(p) for p in paths}
    for name,h in manifest.items():assert sha(B/name)==h
    write(R/'closure_receipt.json',dict(status='closed',qualified=False,goal_complete=False,readback_passed=True,
        no_live_execution_mutex=True,all_pilot_files_verified=len(d['pilot_files_sha256']),files_sha256=manifest,CPU_close_seconds=time.monotonic()-started))
    write(B/'GOAL_TURN_R34_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        concrete_progress='Completed45 paired-initialization contrastive/pooled projection CPU heads;30 spectral decompositions and45 old references independently verified. Fixed qualification failed; stopped this factor without new outer evaluation.',
        new_outer_test_evaluation=False,source43_scope_reply_received=False,no_active_handle=True,genuine_external_blocker=False,
        closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\nR34精确初始化背景对比PCA闭合：45新头/30谱分解/6897内部OOF、45原R4参考只读核验；身份、独立谱/增广lstsq/单向量部署/SHA全部通过。Core主ΔPCC −.000689/+.004100/−.001171，MSE下降−.1076%/+.6693%/−.2525%；performance与contrastive均false、最终gate=false。不改gamma/维度/alpha，不挑pooled对照、不做新外层。四设置主结果仍R26+3.2771%，goal active未达5%；本轮与上一轮progress，无活句柄/外部阻断。CPU拟合+验收{cost["CPU_fit_and_audit_seconds"]:.6f}s，合成{tests:.6f}s另计，GPU/source forward0；详见ROUND34_RESULTS.md。\n')
    print(json.dumps(dict(status='closed',qualified=False,closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report),CPU_fit_and_audit_seconds=cost['CPU_fit_and_audit_seconds'])))


if __name__=='__main__':main()
