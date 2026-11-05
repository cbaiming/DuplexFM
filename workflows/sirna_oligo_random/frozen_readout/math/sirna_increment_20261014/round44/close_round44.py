"""Close the fixed channel transform round with full negative/control reporting."""
import sys,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent
sys.path.insert(0,str(R))
from run import io,validate


def main():
    started=time.monotonic();report=B/'ROUND44_RESULTS.md'
    assert not (R/'execution.mutex').exists()
    for p in (R/'cost_receipt.json',R/'closure_receipt.json',report,B/'GOAL_TURN_R44_AUDIT.json'):assert not p.exists(),p
    lock=validate();d=io.read(R/'diagnostic.json');v=io.read(R/'verification.json')
    assert v['passed'] and d['all_numerically_qualified'] and not d['qualified'] and not v['qualified']
    assert not d['performance_gate']
    assert d['lock_sha256']==v['lock_sha256']==io.sha(R/'lock.json') and v['diagnostic_sha256']==io.sha(R/'diagnostic.json')
    assert d['diagnostic_rows_sha256']==io.sha(R/'diagnostic_rows.csv')
    for n,h in d['pilot_files_sha256'].items():assert io.sha(R/'pilot'/n)==h
    assert v['counts']==dict(new=45,old=45,TD=15,rows=6897,transforms=45)
    runs=list(R.glob('execution_*.json'));audits=list(R.glob('verification_execution_*.json'))
    assert len(runs)==len(audits)==1
    e,a=io.read(runs[0]),io.read(audits[0]);assert e['returncode']==a['returncode']==0
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==9
    assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
    synthetic=sum(float(s.get('time',0)) for s in suites)
    cost=dict(CPU_numeric_seconds=e['CPU_wall_seconds'],CPU_independent_audit_outer_seconds=a['CPU_wall_seconds'],
        CPU_independent_audit_inner_seconds_included=v['CPU_wall_seconds'],
        CPU_numeric_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],CPU_synthetic_seconds=synthetic,
        GPU_hours=0,new_source_forward=0,source_training_old_cache_startup_registration_development_and_close_not_included=True,
        execution_sha256=io.sha(runs[0]),verification_execution_sha256=io.sha(audits[0]))
    lines=['# R44：共享通道收缩白化没有稳定改善迁移','',
        '结论：不采用本轮候选，不运行新的外层测试。Core的Huesken/Mixset退步、Takayuki小幅改善；固定性能资格未通过。完整四设置主结果仍为R26的平均相对PCC增益+3.2771%，尚未达到5%。','',
        '本轮保留TD24及来源192维，在原训练标准化之后，使用六区域共享32通道的训练协方差C，固定A=.5I+.5C，W=gamma*A^(-1/2)。gamma保持训练来源总能量，全部维度保留。用同一R25单线性头、alpha100和lambda=.1*profile RHS谱范数；没有新的来源训练、前向、PCA、旧标准化拟合或GPU。','',
        '变换改变正则几何，不能称纯数值加速；lambda公式相同、实际数值随新设计变化。经验协方差经半收缩和能量归一后不等于单位阵。六区域不是六倍独立样本。部署仍可折为216系数+截距，但训练额外估计的协方差/来源网络容量不能忽略。','',
        '下表仅为三个原outer fold0训练池内OOF，Huesken1564/Mixset319/Takayuki416。不是完整四设置结果，不能与R26汇总混算。固定alpha100的TD为本轮控制，不替换目标中的TD24_Ridge_CV。','',
        '|训练池|来源|新PCC|原R25 PCC|ΔPCC|MSE下降 vsR25|固定TD PCC|ΔPCC vsTD|MSE下降 vsTD|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in d['records']:
        deltaTD=r['metrics']['PCC']-r['TD_metrics']['PCC'];redTD=100*(1-r['metrics']['MSE']/r['TD_metrics']['MSE'])
        lines.append(f"|{r['setting']}|{r['source']}|{r['metrics']['PCC']:.6f}|{r['reference']['PCC']:.6f}|{r['delta_PCC_vs_R25']:+.6f}|{r['MSE_reduction_vs_R25_pct']:+.4f}%|{r['TD_metrics']['PCC']:.6f}|{deltaTD:+.6f}|{redTD:+.4f}%|")
    lines+=['',
        f"预定资格：performance={d['performance_gate']}，source_increment={d['source_increment_gate']}，全部数值={d['all_numerically_qualified']}，最终qualified={d['qualified']}。不按Full/初始化的单项表现换主方案，不调整收缩强度、尺度、alpha或门槛。",'',
        '三个Full池均下降；精确初始化在Mix/Tak有所改善。即使局部改善，也不能将它单独解释成miRNA监督增益。这轮仅排除了这一固定相关性处理能稳定提高现有读出的假设，不能断言所有白化或正则化无效，也不能据它解释论文source43的旧增益。','',
        '|训练池|来源|共享C有效秩范围|收缩C条件数范围|能量归一系数范围|',
        '|---|---|---:|---:|---:|']
    for r in d['records']:
        ranges=[]
        for field in ('covariance_effective_rank','shrunk_condition','energy_scale'):
            values=[b[field] for b in r['inner_blocks']];ranges.append(f'{min(values):.4f}–{max(values):.4f}')
        lines.append(f"|{r['setting']}|{r['source']}|"+'|'.join(ranges)+'|')
    lines+=['',
        '有效秩仅描述输入协方差谱，不能直接等同有用信息维度或生物机制。全部45变换都按当前内折训练数据独立估计；验证输入没有参与。','',
        '计数：45新变换、45新头、6897新内部OOF行；45旧R25头和15唯一旧TD头按SHA复用，未训练旧控制。','',
        f"验收：9项纯合成首轮通过；45独立SVD变换和45新/45旧完整凸证书、15TD正规方程、部署、来源/行身份、覆盖及所有指标/门槛/SHA通过。最大变换差{v['max_channel_transform_difference']:.3g}，能量误差{v['max_source_energy_error']:.3g}，相对对偶gap{v['max_relative_dual_gap']:.3g}，部署预测差{v['max_deployment_difference']:.3g}。",'',
        '|训练池|来源|内折|新PCC|新MSE|原R25 PCC|原R25 MSE|TD PCC|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in d['records']:
        for b in r['inner_blocks']:
            lines.append(f"|{r['setting']}|{r['source']}|{b['inner']}|{b['metrics']['PCC']:.6f}|{b['metrics']['MSE']:.6f}|{b['reference']['PCC']:.6f}|{b['reference']['MSE']:.6f}|{b['TD_metrics']['PCC']:.6f}|")
    lines+=['',
        f"成本：CPU数值控制器{e['CPU_wall_seconds']:.6f}秒，独立验收{a['CPU_wall_seconds']:.6f}秒，合计{cost['CPU_numeric_and_audit_seconds']:.6f}秒；合成{synthetic:.6f}秒另列。GPU和来源forward为0。旧训练/缓存、启动、登记、开发及收尾不计入这段数值阶段时间，不能当端到端速度。",'',
        '数学背景：[Kessy等，Optimal whitening and decorrelation](https://arxiv.org/abs/1512.00809)。本轮半收缩能量归一为固定受限实现，不声称其系数是该文证明的最优值。','',
        '限制：当前是反复开发后的训练池诊断，source keep-overlap/context暴露、单来源种子、非gene隔离及两Mixset设置非独立仍保留。源监督/任务分布解释尚需独立验证。不重新解释为SOTA、外部盲确认或已达5%。','',
        f"科学锁：{io.sha(R/'lock.json')}。诊断：{io.sha(R/'diagnostic.json')}。独立验收：{io.sha(R/'verification.json')}。"]
    io.write(R/'cost_receipt.json',cost)
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    files=[R/'lock.json',R/'diagnostic.json',R/'diagnostic_rows.csv',R/'verification.json',R/'cost_receipt.json',report,Path(__file__)]+[R/n for n in lock['files']]+runs+audits
    manifest={p.relative_to(B).as_posix():io.sha(p) for p in files}
    for n,h in manifest.items():assert io.sha(B/n)==h
    io.write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,qualified=False,goal_complete=False,
        no_live_execution_mutex=True,files_sha256=manifest,all_pilot_files_verified=len(d['pilot_files_sha256']),CPU_close_seconds=time.monotonic()-started))
    io.write(B/'GOAL_TURN_R44_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        genuine_external_blocker=False,no_active_handle=True,new_outer_test_evaluation=False,new_efficiency_heads=45,
        concrete_progress='Fixed shared32channel half-shrink covariance transform:45 new heads independently audited; Core Hu/Mix worsen and Tak improves only slightly; qualification false.',
        closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write('\n\nR44共享32通道半收缩协方差变换闭合：45新变换/45新头/6897内部OOF、45旧R25和15唯一TD复用；45独立SVD及完整凸证书/部署/身份/SHA通过。Core相对R25 ΔPCC Hu−.002441/Mix−.002561/Tak+.000857，MSE下降−.4525%/−.4365%/+.4974%；performance=false，最终gate=false。全部192维保留且来源训练总能量保持，不扩收缩/alpha网格、不新外层。'
                f'CPU数值＋验收{cost["CPU_numeric_and_audit_seconds"]:.6f}s，合成{synthetic:.6f}s另计，新GPU/forward/PCA0。完整主结果仍R26+3.2771%，goal active未达5%；上一轮及本轮progress，无活句柄/阻断，详见ROUND44_RESULTS.md。\n')
    print(dict(status='closed',qualified=False,goal_complete=False,closure_sha256=io.sha(R/'closure_receipt.json'),
               report_sha256=io.sha(report),CPU_numeric_and_audit_seconds=cost['CPU_numeric_and_audit_seconds']))


if __name__=='__main__':main()
