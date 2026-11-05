"""Close the controlled readout comparison; preserve all negative results."""
import sys, time, xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent
sys.path.insert(0,str(R))
from run import io,validate


def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists()
    report=B/'ROUND43_RESULTS.md'
    for p in (R/'cost_receipt.json',R/'closure_receipt.json',report,B/'GOAL_TURN_R43_AUDIT.json'):
        assert not p.exists(),p
    lock=validate();d=io.read(R/'diagnostic.json');v=io.read(R/'verification.json')
    assert v['passed'] and not d['qualified'] and not v['qualified']
    assert not d['matched_readout_gate'] and not d['strong_native_gate'] and not d['TD_increment_gate']
    assert d['lock_sha256']==v['lock_sha256']==io.sha(R/'lock.json')
    assert v['diagnostic_sha256']==io.sha(R/'diagnostic.json')
    assert d['diagnostic_rows_sha256']==io.sha(R/'diagnostic_rows.csv')
    for n,h in d['pilot_files_sha256'].items():assert io.sha(R/'pilot'/n)==h
    assert d['new_heads']==v['verified_new_heads']==15
    assert d['new_internal_OOF_rows']==v['new_internal_OOF_rows']==2299
    runs=list(R.glob('execution_*.json'));audits=list(R.glob('verification_execution_*.json'))
    assert len(runs)==len(audits)==1
    e,a=io.read(runs[0]),io.read(audits[0]);assert e['returncode']==a['returncode']==0
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==8
    assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
    synthetic=sum(float(s.get('time',0)) for s in suites)
    cost=dict(CPU_numeric_seconds=e['CPU_wall_seconds'],CPU_independent_audit_outer_seconds=a['CPU_wall_seconds'],
        CPU_independent_audit_inner_seconds_included=v['CPU_wall_seconds'],
        CPU_numeric_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],
        CPU_synthetic_seconds=synthetic,GPU_hours=0,new_source_forward=0,
        source_training_old_cache_startup_registration_development_and_close_not_included=True,
        execution_sha256=io.sha(runs[0]),verification_execution_sha256=io.sha(audits[0]))
    lines=['# R43：统一读出后，旧五专家source42仍未超过当前方案','',
        '结论：本轮不采用，不开启新的外层测试。旧160D来源经全局PCA16后在Mixset略优于同读出的Core192，但Hu/Tak退步；三个训练池都弱于当前R25 Core原生强读出。完整四设置主结果仍R26平均相对PCC增益+3.2771%，尚未达到5%。','',
        '本轮固定TD24＋全局PCA16＋Ridge(alpha100, source权重1)，只给现有发布source42的五专家slot采用同一读出。来源原始160D，投影为16D，最终40D头。旧Core192采用相同scaler、PCA16、PC尺度与Ridge；合成验证192输入时全部数组与原R4完全一致。没有新来源训练、导出、GPU或测试选参。','',
        '下表仅为原fold0 fit池内五折OOF：Huesken1564、Mixset319、Takayuki416条，合计2299。固定alpha100的TD是辅助训练池对照，不替代目标中的TD24_Ridge_CV分母。','',
        '|训练池|新旧架构slot-PCA16 PCC|同读出Core PCC|ΔPCC vsCore|MSE下降 vsCore|原生Core PCC|ΔPCC vs原生Core|MSE下降 vs原生Core|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in d['records']:
        lines.append(f"|{r['setting']}|{r['metrics']['PCC']:.6f}|{r['references']['R4_core']['PCC']:.6f}|{r['delta_PCC_vs_R4_core']:+.6f}|{r['MSE_reduction_vs_R4_core_pct']:+.4f}%|{r['references']['R25_core']['PCC']:.6f}|{r['delta_PCC_vs_R25_core']:+.6f}|{r['MSE_reduction_vs_R25_core_pct']:+.4f}%|")
    lines+=['',
        '同一读出下的来源变化不能单独归因于位置池化或某个expert，因为两种来源的架构与历史训练合同也不同。结果仅说明“旧slot读出方式不同”不能完整解释这组开发池上的差距；没有证明论文source43的增益能否恢复。','',
        '|训练池|固定TD PCC|新方案ΔPCC vsTD|MSE下降 vsTD|旧分组PCA4 PCC|新方案ΔPCC vs分组PCA4|MSE下降 vs分组PCA4|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in d['records']:
        lines.append(f"|{r['setting']}|{r['references']['TD']['PCC']:.6f}|{r['delta_PCC_vs_TD']:+.6f}|{r['MSE_reduction_vs_TD_pct']:+.4f}%|{r['references']['R1_group']['PCC']:.6f}|{r['delta_PCC_vs_R1_group']:+.6f}|{r['MSE_reduction_vs_R1_group_pct']:+.4f}%|")
    lines+=['',
        '相对旧分组PCA4，Hu/Mix小幅改善，Tak退步；与TD相比，Mix/Tak仍负增益，Hu的MSE也未改善。分组PCA4共20维、全局PCA16为16维，且尺度细节不同，因此这个辅助差异不是只改变压缩分配的严格单因素结论。不追加维度或alpha搜索。','',
        f"预定资格：同读出Core比较={d['matched_readout_gate']}，原生强Core比较={d['strong_native_gate']}，TD增量={d['TD_increment_gate']}，数值={d['all_numerically_qualified']}，最终qualified={d['qualified']}。",'',
        '来源明确为发布source42，checkpoint SHA 6c97cd93a1b6089caa62e4ddadde88839ba1dae9fa49eeea18345317d114a8a4。它不是原论文source43。旧160D没有本轮匹配的精确初始化，不能借用新Core192初始化来宣称miRNA监督因果增益。','',
        '计数：15新训练变换包/PCA16/40D头，2299新内部OOF行；15旧同读出Core、15旧原生Core、15旧TD和15旧分组压缩控制均按SHA复用。部署可以折为184系数和截距；40D头的41系数不包含PCA或来源模型容量。','',
        f"验收：8项纯合成首轮通过；15独立SVD投影、15增广lstsq、旧参考正规方程/部署、行身份、覆盖与全部指标/资格/SHA通过。最大平均正规方程误差{v['max_normal_error']:.3g}，独立解差{v['max_independent_solve_difference']:.3g}，原始空间部署差{v['max_deployment_difference']:.3g}。旧R25凸证书沿用原已核验回执，本轮只重新核对其部署，不声称重新训练旧模型。",'',
        '|训练池|内折|新PCC|新MSE|同读出Core PCC|原生Core PCC|TD PCC|分组PCA4 PCC|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in d['records']:
        for b in r['inner_blocks']:
            refs=b['references']
            lines.append(f"|{r['setting']}|{b['inner']}|{b['metrics']['PCC']:.6f}|{b['metrics']['MSE']:.6f}|{refs['R4_core']['PCC']:.6f}|{refs['R25_core']['PCC']:.6f}|{refs['TD']['PCC']:.6f}|{refs['R1_group']['PCC']:.6f}|")
    lines+=['',
        f"成本：数值控制器{e['CPU_wall_seconds']:.6f}秒＋独立验收{a['CPU_wall_seconds']:.6f}秒，合计{cost['CPU_numeric_and_audit_seconds']:.6f}CPU秒；合成{synthetic:.6f}秒另列。GPU/source forward为0。旧来源训练/缓存、进程启动、登记、开发和收尾不在该时间内，不作为端到端效率声明。",'',
        '本轮是在既有开发结果已知后的受限比较，不能称为盲外部确认。源keep-overlap/context暴露、单source种子、非gene隔离、重复开发及两个Mixset设置非独立等限制保持。没有产生新的四设置评价或达到5%的证据。','',
        f"科学锁：{io.sha(R/'lock.json')}。诊断：{io.sha(R/'diagnostic.json')}。独立验收：{io.sha(R/'verification.json')}。"]
    io.write(R/'cost_receipt.json',cost)
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    files=[R/'lock.json',R/'diagnostic.json',R/'diagnostic_rows.csv',R/'verification.json',R/'cost_receipt.json',report,Path(__file__)]+[R/n for n in lock['files']]+runs+audits
    manifest={p.relative_to(B).as_posix():io.sha(p) for p in files}
    for n,h in manifest.items():assert io.sha(B/n)==h
    io.write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,qualified=False,goal_complete=False,
        no_live_execution_mutex=True,files_sha256=manifest,all_pilot_files_verified=len(d['pilot_files_sha256']),CPU_close_seconds=time.monotonic()-started))
    io.write(B/'GOAL_TURN_R43_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        genuine_external_blocker=False,no_active_handle=True,new_outer_test_evaluation=False,new_efficiency_heads=15,
        concrete_progress='Matched global PCA16/Ridge readout of verified released source42 slots: 15 new heads independently audited. Hu/Tak weaker than Core, all three weaker than native Core; all qualification gates false.',
        closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write('\n\nR43已有旧source42五slot160D的统一全局PCA16读出闭合：15新变换/PCA/头、2299内部OOF，15独立SVD/lstsq及60旧参考身份/正规方程/部署/SHA通过。相对同读出Core的ΔPCC Hu−.012511/Mix+.005857/Tak−.011189，相对原生Core−.013952/−.002972/−.038692；matched/strong_native/TD门槛均false。不改来源或压缩/alpha网格，不开新外层；这是source42旧架构，不是论文source43。'
                f'CPU数值＋验收{cost["CPU_numeric_and_audit_seconds"]:.6f}s，合成{synthetic:.6f}s另计，新forward/GPU0。完整主结果仍R26+3.2771%，goal active未达5%；上一轮及本轮progress，无活句柄/外部阻断，详见ROUND43_RESULTS.md。\n')
    print(dict(status='closed',qualified=False,goal_complete=False,closure_sha256=io.sha(R/'closure_receipt.json'),
               report_sha256=io.sha(report),CPU_numeric_and_audit_seconds=cost['CPU_numeric_and_audit_seconds']))


if __name__=='__main__':main()
