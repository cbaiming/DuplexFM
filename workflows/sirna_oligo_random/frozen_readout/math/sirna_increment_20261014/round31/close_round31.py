"""Close already verified internal results; no new fits or outer metrics."""
import hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    started=time.monotonic()
    assert not (R/'closure_receipt.json').exists() and not (R/'execution.mutex').exists()
    d=read(R/'diagnostic.json');v=read(R/'verification.json');lock=read(R/'lock.json')
    assert v['passed'] and v['diagnostic_sha256']==sha(R/'diagnostic.json') and v['lock_sha256']==sha(R/'lock.json')
    assert not d['qualified'] and not d['performance_gate'] and not d['source_increment_gate']
    assert v['all_numerically_qualified'] and d['new_heads']==60 and d['new_internal_OOF_rows']==9196
    assert v['new_scientific_linear_solves']==d['new_linear_solves']==2030
    for name,h in lock['files'].items():assert sha(R/name)==h,name
    for name,h in lock['dependencies'].items():assert sha(Path(name))==h,name
    for name,h in d['pilot_files_sha256'].items():assert sha(R/'pilot'/name)==h,name
    executions=sorted(R.glob('execution_*.json'));audits=sorted(R.glob('verification_execution_*.json'))
    assert len(executions)==len(audits)==1
    e,a=read(executions[0]),read(audits[0]);assert e['returncode']==a['returncode']==0
    tests=sum(float(s.get('time',0)) for s in ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    cost=dict(CPU_controller_seconds=e['CPU_wall_seconds'],CPU_audit_outer_seconds=a['CPU_wall_seconds'],
        CPU_audit_inner_seconds_included=v['CPU_wall_seconds'],
        CPU_fit_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],synthetic_seconds_separate=tests,
        new_heads=60,new_scientific_linear_solves=2030,new_internal_OOF_rows=9196,reused_reference_heads=60,
        independent_final_augmented_lstsq=60,independent_audit_linear_solves=480,
        new_source_forward=0,new_PCA=0,GPU_hours=0,
        excluded='Earlier source training/export, R4 references, startup/imports, development and report/closure are separate.',
        execution_sha256={p.name:sha(p) for p in executions+audits})
    write(R/'cost_receipt.json',cost)
    lines=['# R31：平滑条件系数软门控读出','',
        '2026-10-15。固定软门控候选没有通过内部资格，停止这一配置，不做新的四设置外层评价，不追加正则、步数或初始化网格。当前四设置主结果仍为R26的平均相对PCC增益+3.2771%，5%目标未达到。','',
        '本轮保持TD24与来源PCA16的全局线性路径，仅让来源斜率由TD24的一个25参数tanh gate平滑调节。对给定gate精确求解联合线性系数，再优化gate；单个同时拟合的读出可等价写成两个来源斜率的软MoE，不是两个独立模型预测叠加。数学背景为[可分离非线性最小二乘](https://epubs.siam.org/doi/10.1137/0710036)，不据此保证非凸全局最优或泛化。','',
        '同类型TD-only控制也保留：全局/局部均读取TD24。来源读出56个线性系数、TD-only 48个，两者另有25个gate参数与一个截距，共82/74个名义参数；并非参数量完全相等。固定全局alpha100、局部惩罚400、gate惩罚.01×训练SST，全部用同一预定初值；没有根据验证或测试选择训练终点。','',
        '|训练内池|来源|原固定R4 PCC|新PCC|绝对ΔPCC|相对R4的MSE下降|',
        '|---|---|---:|---:|---:|---:|']
    for r in d['records']:
        lines.append(f"|{r['setting']}|{r['source']}|{r['reference']['PCC']:.6f}|{r['metrics']['PCC']:.6f}|{r['delta_PCC_vs_R4']:+.6f}|{r['MSE_reduction_vs_R4_pct']:+.4f}%|")
    lines+=['','Core在Huesken和Mixset下降，在Takayuki上升，performance=false。Full在Mixset有改善，初始化在Huesken略有改善，均完整保留，不用辅助来源替换主方案。','',
        '|训练内池|Core相对同类型TD-only软门控的绝对ΔPCC|MSE下降|',
        '|---|---:|---:|']
    for p in d['paired_gated_TD']:
        lines.append(f"|{p['setting']}|{p['delta_PCC']:+.6f}|{p['MSE_reduction_pct']:+.4f}%|")
    lines+=['','相对同类型TD-only，Core仍只有Takayuki为正，source_increment=false；全部数值资格通过，但最终qualified=false。不能把数值收敛当成性能合格，也不把单一软路由候选的失败概括为所有MoE无效。','',
        '|Core训练内池|五内折验证gate概率均值|五内折验证gate概率标准差|',
        '|---|---|---|']
    for r in d['records']:
        if r['source']!='core_real':continue
        means='/'.join(f"{x['valid_gate']['mean']:.3f}" for x in r['inner_blocks'])
        stds='/'.join(f"{x['valid_gate']['std']:.3f}" for x in r['inner_blocks'])
        lines.append(f"|{r['setting']}|{means}|{stds}|")
    lines+=['','gate确实随输入变化，没有全部变成一个常数；它调节两条来源斜率，并不是BIO/Inta/开放性的开关，也不是生物学概率。全局来源路径仍保留，不能用此结果证明自动屏蔽专家已经实现。','',
        '以上为三个既定outer fold0 fit池的五折内部OOF，共2299个训练池样本；不是完整四设置外层测试。60个新头产生9196行新OOF，原60个R4固定参考头只读复用。训练共2030次条件线性求解；独立验收另做480次线性求解，其中60个最终增广lstsq、60个初值检查及360个方向差分解。这些不能混算为额外科学模型。','',
        f"验收：八项新增合成测试通过；60个模型全部数值合格，最大归一化gate梯度{v['max_gate_gradient']:.3g}，最大有限差分误差{v['max_finite_difference_error']:.3g}，最大正规方程误差{v['max_normal_equation_error']:.3g}，部署重建误差{v['max_prediction_difference']:.3g}。原PCA、来源和样本身份、初始gate、接受轨迹计数、逐行预测、指标和SHA全部独立核验。非凸优化只核验局部驻点，不声称全局最优。",'',
        f"成本：CPU控制器{e['CPU_wall_seconds']:.6f}秒、独立验收外层{a['CPU_wall_seconds']:.6f}秒，合计{cost['CPU_fit_and_audit_seconds']:.6f}秒；合成测试{tests:.6f}秒另列。GPU、新来源forward和新PCA均0。旧来源训练/导出/参考头、启动导入、开发与收尾成本未包含，不能称端到端耗时。",'',
        '来源keep-overlap/context暴露、反复开发、单source42以及purged sequence不是gene隔离等限制继续保留。初始化控制为精确任务训练前状态，不是所有权重随机。source43范围问题尚未回复，没有导出。当前goal active且未完成；本轮没有活执行句柄或外部阻断。','',
        f"科学锁：{sha(R/'lock.json')}。诊断：{sha(R/'diagnostic.json')}。独立验收：{sha(R/'verification.json')}。"]
    report=B/'ROUND31_RESULTS.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    paths=[R/'lock.json',R/'diagnostic.json',R/'diagnostic_rows.csv',R/'verification.json',R/'cost_receipt.json',report,Path(__file__)]+[R/n for n in lock['files']]+executions+audits
    manifest={p.relative_to(B).as_posix():sha(p) for p in paths}
    for name,h in manifest.items():assert sha(B/name)==h
    write(R/'closure_receipt.json',dict(status='closed',qualified=False,goal_complete=False,readback_passed=True,
        no_live_execution_mutex=True,files_sha256=manifest,all_pilot_files_verified=len(d['pilot_files_sha256']),CPU_close_seconds=time.monotonic()-started))
    write(B/'GOAL_TURN_R31_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        concrete_progress='60 fixed smooth varying-coefficient CPU heads and 60 original controls independently verified; both internal performance gates failed, all positive and negative results retained.',
        new_outer_test_evaluation=False,source43_scope_reply_received=False,no_active_handle=True,genuine_external_blocker=False,
        closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write('\n\nR31单个平滑条件系数软MoE闭合：60新头/2030条件线性解/9196内部OOF、60旧R4参考只读复用；480独立验收线性解及全部数值/来源/PCA/部署/SHA通过。Core相对固定R4的ΔPCC −.012447/−.003922/+.017138，MSE下降−2.2867%/−.6584%/+2.4069%；performance和source_increment均false，不运行新外层或扩大gate/惩罚/种子网格。gate随输入变化但只是两来源斜率的软混合，不能称原专家自动开关。四设置主结果仍R26+3.2771%，goal active未达5%；本轮progress，无活句柄/外部阻断。CPU拟合+独立验收3.782s，GPU/source forward/PCA0，详见ROUND31_RESULTS.md。\n')
    print(json.dumps(dict(status='closed',qualified=False,closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report),CPU_fit_and_audit_seconds=cost['CPU_fit_and_audit_seconds'])))


if __name__=='__main__':main()
