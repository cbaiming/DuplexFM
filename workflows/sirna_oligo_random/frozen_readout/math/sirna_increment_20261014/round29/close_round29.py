"""Close already verified R29 artifacts; no fits or additional test evaluation."""
import hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    start=time.monotonic()
    assert not (R/'closure_receipt.json').exists() and not (R/'execution.mutex').exists()
    d=read(R/'diagnostic.json');v=read(R/'verification.json');lock=read(R/'lock.json')
    assert v['passed'] and v['diagnostic_sha256']==sha(R/'diagnostic.json') and v['lock_sha256']==sha(R/'lock.json')
    assert not d['qualified'] and not d['performance_gate'] and d['source_increment_gate']
    for name,h in lock['files'].items():assert sha(R/name)==h,name
    for name,h in lock['dependencies'].items():assert sha(Path(name))==h,name
    for name,h in d['pilot_files_sha256'].items():assert sha(R/'pilot'/name)==h,name
    executions=sorted(R.glob('execution_*.json'));audits=sorted(R.glob('verification_execution_*.json'))
    assert len(executions)==len(audits)==1
    e,a=read(executions[0]),read(audits[0]);assert e['returncode']==a['returncode']==0
    test_seconds=sum(float(s.get('time',0)) for s in ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    cost=dict(CPU_controller_seconds=e['CPU_wall_seconds'],CPU_audit_outer_seconds=a['CPU_wall_seconds'],
        CPU_audit_inner_seconds_included=v['CPU_wall_seconds'],
        CPU_fit_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],synthetic_seconds_separate=test_seconds,
        new_heads=45,new_internal_OOF_rows=6897,reused_native_heads=45,reused_unique_TD_heads=15,reused_unique_TD_rows=2299,
        independent_full_joint_certificates=45,independent_unique_TD_normal_equations=15,
        new_source_forward=0,new_PCA=0,GPU_hours=0,
        excluded='Earlier source training/export, R4/R21/R25 references, startup/imports, development and report/closure are separate.',
        execution_sha256={p.name:sha(p) for p in executions+audits})
    write(R/'cost_receipt.json',cost)
    lines=['# R29：限制联合读出对TD24系数的扰动','',
        '2026-10-15。固定惩罚未通过训练内性能资格，不推进四设置外层评价。R26主方案的四设置平均相对PCC增益仍为+3.2771%，5%目标未完成。','',
        '本轮在原TD24+source192、6×32来源系数核范数读出中，新增固定100强度的TD系数锚定；参考为同inner train的原alpha100 TD24解。来源、特征、缩放、正则和训练内划分均保持。部署仍是单个216D线性头，不添加TD预测输入或独立残差模型。参考来自同一训练标签，不是独立先验。','',
        '|训练内池|来源|R25原PCC|新PCC|绝对ΔPCC|相对R25的MSE下降|',
        '|---|---|---:|---:|---:|---:|']
    for r in d['records']:
        lines.append(f"|{r['setting']}|{r['source']}|{r['reference']['PCC']:.6f}|{r['metrics']['PCC']:.6f}|{r['delta_PCC_vs_R25']:+.6f}|{r['MSE_reduction_vs_R25_pct']:+.4f}%|")
    lines+=['','Core的三个PCC变化均小于0.001，远未达到预定至少两个池ΔPCC≥0.005的要求；Huesken和Takayuki的MSE略有增加。performance=false、source_increment=true、最终qualified=false。全部45头数值验收通过，负结果不是未收敛造成的；不扩展锚定强度或按Full结果改变主方案。','',
        '|训练内池|Core相对同alpha100 TD的绝对ΔPCC|MSE下降|',
        '|---|---:|---:|']
    for p in d['paired_TD']:
        lines.append(f"|{p['setting']}|{p['delta_PCC']:+.6f}|{p['MSE_reduction_pct']:+.4f}%|")
    lines+=['','这些是原fold0 fit池中的五折内部OOF，不是完整四设置测试结果，不可与R26四设置PCC直接混排。仍呈现Hu/Tak有增量、Mix下降的模式。此结果仅说明本轮固定的系数约束不足以解决负迁移，不能据此证明所有TD系数漂移都无关。','',
        '计数：45个新头、6897条新内部OOF；45个旧R25头和15个唯一旧TD头按SHA复用，2299条TD预测在三个来源中重复引用。没有新来源forward、PCA拟合或GPU用量。','',
        f"验收：八项合成测试通过；45个完整216D目标独立原始/对偶证书、15个旧TD正规方程及全部行序/缩放/预测/部署/SHA通过。最大相对对偶gap {v['max_relative_dual_gap']:.3g}，最大TD梯度误差 {v['max_TD_normal_equation_error']:.3g}，预测重建误差 {v['max_prediction_difference']:.3g}。",'',
        f"成本：CPU控制器{e['CPU_wall_seconds']:.6f}秒，独立验收外层{a['CPU_wall_seconds']:.6f}秒，合计{cost['CPU_fit_and_audit_seconds']:.6f}秒；合成测试{test_seconds:.6f}秒另列。旧模型训练/导出/参考读出、解释器启动、开发与收尾不计入这段数值时间，不能称为端到端成本。",'',
        '三个训练池已反复开发；来源keep-overlap/context暴露、单来源seed42、purged sequence不等于gene隔离等限制保持。没有新增外层测试评价或把训练内小改进包装为独立确认。source43导出范围仍待回复，本轮未执行。','',
        f"科学锁：{sha(R/'lock.json')}。诊断：{sha(R/'diagnostic.json')}。独立验收：{sha(R/'verification.json')}。"]
    report=B/'ROUND29_RESULTS.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    paths=[R/'lock.json',R/'diagnostic.json',R/'diagnostic_rows.csv',R/'verification.json',R/'cost_receipt.json',report,Path(__file__)]+[R/n for n in lock['files']]+executions+audits
    manifest={p.relative_to(B).as_posix():sha(p) for p in paths}
    for name,h in manifest.items():assert sha(B/name)==h
    write(R/'closure_receipt.json',dict(status='closed',qualified=False,goal_complete=False,readback_passed=True,
        no_live_execution_mutex=True,files_sha256=manifest,all_pilot_files_verified=len(d['pilot_files_sha256']),CPU_close_seconds=time.monotonic()-start))
    write(B/'GOAL_TURN_R29_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        concrete_progress='45 fixed TD-anchored native heads failed the internal performance gate; independent global certificates and full negative results closed.',
        new_outer_test_evaluation=False,source43_scope_reply_received=False,no_active_handle=True,genuine_external_blocker=False,
        closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write('\n\nR29单头TD系数锚定闭合：45新头/6897内部OOF、45原生参考和15唯一TD参考只读复用，全部完整凸证书/部署/SHA通过。Core相对R25固定原生读出ΔPCC +.000021/+.000960/+.000367，MSE下降−.0030%/+.1420%/−.2975%；performance=false，最终gate=false，不做外层或扩惩罚网格。四设置主结果仍R26+3.2771%，goal active未达5%；本轮progress，无活句柄/外部阻断。CPU拟合+验收3.593s，GPU/source forward/PCA0，详见ROUND29_RESULTS.md。\n')
    print(json.dumps(dict(status='closed',qualified=False,closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report),CPU_fit_and_audit_seconds=cost['CPU_fit_and_audit_seconds'])))


if __name__=='__main__':main()
