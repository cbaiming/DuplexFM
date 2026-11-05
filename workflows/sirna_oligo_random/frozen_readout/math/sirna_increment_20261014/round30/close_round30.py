"""Close verified internal results only; no extra scientific fits or outer metrics."""
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
    lines=['# R30：原生共享通道组稀疏','',
        '2026-10-15。固定通道组稀疏没有通过训练内性能资格，不推进新四设置外层评价。四设置主结果仍为R26的+3.2771%平均相对PCC增益，5%目标未完成。','',
        '本轮单个TD24+source192读出将来源按32个共享通道分组，每组含六个池化统计块的同一通道系数。联合目标为MSE加alpha100的L2及固定0.1×critical的组范数；没有R29的TD锚定或R25的核范数。它不增加源网络、数据、特征或多个预测，也不等于六任务学习。分组稀疏的数学背景参考[Liu等，UAI2009](https://auai.org/uai2009/papers/UAI2009_0157_72e4f317cc3228a8d51b60ed9ccafc6d.pdf)，本文献不保证本任务的迁移效果。','',
        '|训练内池|来源|R25原PCC|新PCC|绝对ΔPCC|相对R25的MSE下降|五内折保留通道数（共32）|',
        '|---|---|---:|---:|---:|---:|---|']
    for r in d['records']:
        active='/'.join(str(b['active_channels']) for b in r['inner_blocks'])
        assert all(b['active_channels']==b['effective_channels'] for b in r['inner_blocks'])
        lines.append(f"|{r['setting']}|{r['source']}|{r['reference']['PCC']:.6f}|{r['metrics']['PCC']:.6f}|{r['delta_PCC_vs_R25']:+.6f}|{r['MSE_reduction_vs_R25_pct']:+.4f}%|{active}|")
    lines+=['','Core三个训练内池均小幅下降，performance=false；对同alpha100 TD的source_increment=true，但不足以通过联合资格，最终qualified=false。Full/初始化存在零散改善，完整保留，不事后改为主方案。','',
        '实际通道大多没有被完全关闭；Core在Huesken少数内折关闭1–2个通道，其余多为32个保留。因此结果仅拒绝本轮固定强度的候选方法，不能据此断言所有通道稀疏方法无效。也不按观察到的关闭数量追加更强惩罚搜索。','',
        '|训练内池|Core相对同alpha100 TD的绝对ΔPCC|MSE下降|',
        '|---|---:|---:|']
    for p in d['paired_TD']:
        lines.append(f"|{p['setting']}|{p['delta_PCC']:+.6f}|{p['MSE_reduction_pct']:+.4f}%|")
    lines+=['','这些是三个原fold0 fit池的五折OOF，不能与完整四设置测试结果直接混排。Hu/Tak增量和Mix负增量的模式未消除。本轮不计算新的外层测试、不改变TD24分母和选参协议。','',
        '计数：45个新头、6897条内部OOF；45个旧R25头及15个唯一旧TD头/2299行按SHA复用，未重复训练。相同TD预测在三种来源中重复引用不增加样本量。没有新PCA、源forward、GPU或下载。','',
        f"验收：八项合成测试通过；45个完整216D原始/对偶目标证书、15个旧TD正规方程及全部源/输入/缩放/部署/预测/指标/SHA通过。最大相对对偶gap {v['max_relative_dual_gap']:.3g}，最大TD梯度误差 {v['max_TD_normal_equation_error']:.3g}，部署预测重建差 {v['max_prediction_difference']:.3g}。全部数值合格，不能归咎未收敛。",'',
        f"成本：CPU控制器{e['CPU_wall_seconds']:.6f}秒、独立验收外层{a['CPU_wall_seconds']:.6f}秒，合计{cost['CPU_fit_and_audit_seconds']:.6f}秒；合成测试{test_seconds:.6f}秒另列。旧模型训练/导出/参考头、启动导入、开发与收尾不在本段数值成本中，不称端到端时间。",'',
        '限制继续保留：source keep-overlap/context暴露、单来源seed42、反复使用开发池、purged sequence不是gene隔离；没有独立外部或来源多种子确认。source43范围问题未回复，未新导出。目标active未完成，本轮无活进程或外部阻断；完成本轮不等于完成目标。','',
        f"科学锁：{sha(R/'lock.json')}。诊断：{sha(R/'diagnostic.json')}。独立验收：{sha(R/'verification.json')}。"]
    report=B/'ROUND30_RESULTS.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    paths=[R/'lock.json',R/'diagnostic.json',R/'diagnostic_rows.csv',R/'verification.json',R/'cost_receipt.json',report,Path(__file__)]+[R/n for n in lock['files']]+executions+audits
    manifest={p.relative_to(B).as_posix():sha(p) for p in paths}
    for name,h in manifest.items():assert sha(B/name)==h
    write(R/'closure_receipt.json',dict(status='closed',qualified=False,goal_complete=False,readback_passed=True,
        no_live_execution_mutex=True,files_sha256=manifest,all_pilot_files_verified=len(d['pilot_files_sha256']),CPU_close_seconds=time.monotonic()-start))
    write(B/'GOAL_TURN_R30_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        concrete_progress='45 fixed native channel group-sparse heads failed the internal performance gate; independent full certificates, source identities and all negative results closed.',
        new_outer_test_evaluation=False,source43_scope_reply_received=False,no_active_handle=True,genuine_external_blocker=False,
        closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write('\n\nR30原生共享通道组稀疏闭合：45新头/6897内部OOF、45原生与15唯一TD参考只读复用，全部全局证书/源/部署/SHA通过。Core相对R25固定原生头ΔPCC −.000926/−.002203/−.002486，MSE下降−.1715%/−.3659%/−.2715%；performance=false、最终gate=false，不做新外层或扩大惩罚/分组网格。固定强度下大多数32通道仍保留，仅Hu部分关闭1–2，不能概括否定所有组选择。四设置主结果仍R26+3.2771%，goal active未达5%；本轮progress，无活句柄/外部阻断。CPU拟合+验收3.546s，GPU/source forward/PCA0，详见ROUND30_RESULTS.md。\n')
    print(json.dumps(dict(status='closed',qualified=False,closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report),CPU_fit_and_audit_seconds=cost['CPU_fit_and_audit_seconds'])))


if __name__=='__main__':main()
