"""Close the fixed matched-energy comparison with all controls and costs."""
import hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    start=time.monotonic();assert not (R/'closure_receipt.json').exists() and not (R/'execution.mutex').exists()
    d,v,lock=read(R/'diagnostic.json'),read(R/'verification.json'),read(R/'lock.json')
    assert v['passed'] and v['all_numerically_qualified'] and not v['outer_test_metrics_used']
    assert not d['qualified'] and not d['performance_gate'] and d['source_increment_gate']
    assert v['qualified']==d['qualified'] and v['counts']==dict(new_native=45,new_TD28=15,old_native=45,old_TD24=15,rows=9196)
    assert v['lock_sha256']==sha(R/'lock.json') and v['diagnostic_sha256']==sha(R/'diagnostic.json')
    for n,h in lock['files'].items():assert sha(R/n)==h,n
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    for n,h in d['pilot_files_sha256'].items():assert sha(R/'pilot'/n)==h,n
    assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    runs=list(R.glob('execution_*.json'));audits=list(R.glob('verification_execution_*.json'));assert len(runs)==len(audits)==1
    e,a=read(runs[0]),read(audits[0]);assert e['returncode']==a['returncode']==0
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==9
    assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
    synthetic=sum(float(x.get('time',0)) for x in suites)
    cost=dict(CPU_controller_seconds=e['CPU_wall_seconds'],CPU_audit_outer_seconds=a['CPU_wall_seconds'],CPU_audit_inner_included=v['CPU_wall_seconds'],
        CPU_fit_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],synthetic_seconds_separate=synthetic,
        new_heads=60,new_native_heads=45,new_TD28_heads=15,new_energy_scalers=15,reused_native_heads=45,reused_unique_TD24_heads=15,
        new_internal_OOF_rows=9196,independent_audit_TD28_solves=15,new_PCA=0,new_source_forward=0,GPU_hours=0,
        excluded='Previous source training/export and reference fits, process startup/imports, development and report/closure.',
        execution_sha256={p.name:sha(p) for p in runs+audits})
    write(R/'cost_receipt.json',cost)
    lines=['# R37：当前原生读出的匹配能量补充','',
        '2026-10-15。45个新来源轻量头、15个TD28对照和独立验收全部完成。主性能门槛失败，不采用，不新增外层测试评价。完整四设置主结果仍为R26相对强TD24的PCC平均+3.2771%，5%目标未达到。','',
        '唯一信息变化是将已缓存、已按输入与设计位点核验的三能量和有效flag补回当前读出。TD24只取guide信息；source192在site encoder和energy MLP之前导出，所以当前216D头没有显式读取这四列。这里是旧物理信息在当前头中的受控补充，不是新颖能量特征。旧PCA44D、旧SOTA能量乘积已做过，本轮没有重跑或伪装成首次发现。','',
        '当前模型为TD24+匹配能量4D+source192，220个线性系数和截距；原6×32系数核范数比例0.1、alpha100及source除√12不变。能量尺度仅用各inner train有效行估计；无效行零能量、flag保留；无有效训练能量时禁用整块。三个来源共享同一static28，无新teacher分数、来源前向或预测叠加。','',
        '|内部池|来源|原R25 PCC|补入能量后PCC|绝对ΔPCC|相对原R25 MSE下降|',
        '|---|---|---:|---:|---:|---:|']
    for r in d['records']:
        lines.append(f"|{r['setting']}|{r['source']}|{r['reference']['PCC']:.6f}|{r['metrics']['PCC']:.6f}|{r['delta_PCC_vs_R25']:+.6f}|{r['MSE_reduction_vs_R25_pct']:+.4f}%|")
    lines+=['',
        '主Core在Huesken/Mixset/Takayuki的绝对ΔPCC为−0.001452、−0.004956、+0.005304；MSE下降−0.3578%、−0.8374%、+0.7200%。只有Takayuki达到单池改进条件，未通过至少两池改善的预定主门槛。不得据此只给Takayuki加能量，也不删列、改尺度或调正则寻找正结果。','',
        '|内部池|TD24 PCC|TD28 PCC|绝对ΔPCC|TD28相对TD24 MSE下降|',
        '|---|---:|---:|---:|---:|']
    for r in d['TD28_records']:
        lines.append(f"|{r['setting']}|{r['TD24_metrics']['PCC']:.6f}|{r['metrics']['PCC']:.6f}|{r['delta_PCC']:+.6f}|{r['MSE_reduction_pct']:+.4f}%|")
    lines+=['',
        'TD28物理对照本身没有一致收益。不能把新增能量看作普遍有效的补救，也不能将能量的效果归因为miRNA预训练。','',
        '|内部池|Core+能量相对TD28的绝对ΔPCC|相对TD28 MSE下降|',
        '|---|---:|---:|']
    for r in d['paired_TD28']:
        lines.append(f"|{r['setting']}|{r['delta_PCC']:+.6f}|{r['MSE_reduction_pct']:+.4f}%|")
    lines+=['',
        '相对TD28的辅助来源增量门槛通过，但不能替代相对原方法的主改进门槛；Mixset仍负。Full与精确初始化的所有正负结果均保留，不事后改选来源。','',
        '|内部池|允许fit样本|有效匹配能量|无效匹配能量|',
        '|---|---:|---:|---:|']
    for setting,r in d['energy_coverage'].items():lines.append(f"|{setting}|{r['rows']}|{r['valid']}|{r['invalid']}|")
    lines+=['',
        '范围为原三个outer fold0 fit池2299个唯一输入、原5内部折；60新科学头/9196内部OOF、15新能量尺度，45原R25头和15原TD24头只读复用。新结果未用于外层选参或替换旧主模型，Wang/ALAS1未进入数值范围。','',
        f"验收：9项合成通过；45新与45旧联合凸证书、15新TD28独立增广lstsq及15旧TD24正规方程，所有源/输入/训练尺度/缺失mask/行序/门槛/部署/SHA通过。最大相对对偶gap{v['max_relative_dual_gap']:.3g}，TD28独立系数差{v['max_TD28_lstsq_difference']:.3g}，单向量部署差{v['max_deployment_difference']:.3g}。不是数值未收敛导致主门槛失败。",'',
        f"成本：控制器{e['CPU_wall_seconds']:.6f}秒，独立验收外层{a['CPU_wall_seconds']:.6f}秒（内层{v['CPU_wall_seconds']:.6f}秒已包含），共{cost['CPU_fit_and_audit_seconds']:.6f}秒；合成{synthetic:.6f}秒另列，GPU0。旧来源训练/缓存/参考拟合、启动导入、开发和收尾不含，不称端到端仅需几秒。",'',
        '文献动机来自[Shabalina等，BMC Bioinformatics2006](https://doi.org/10.1186/1471-2105-7-65)对guide及target热力学特征的分析；本轮结果没有验证其所有生物机制，也不能把IntaRNA窗口能量当作细胞内切割速率。','',
        '限制：单来源seed42、重复开发池、源keep-overlap/context暴露、当前purged sequence非gene隔离、两个Mixset设置非独立。此次只否定当前固定加性方案的继续资格，不能声称能量在所有模型中无效或全部增益都来自冗余。','',
        f"科学锁：{sha(R/'lock.json')}。诊断：{sha(R/'diagnostic.json')}。独立验收：{sha(R/'verification.json')}。"]
    report=B/'ROUND37_RESULTS.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    paths=[R/'lock.json',R/'diagnostic.json',R/'verification.json',R/'cost_receipt.json',R/'diagnostic_rows.csv',report,Path(__file__)]+[R/n for n in lock['files']]+runs+audits
    manifest={p.relative_to(B).as_posix():sha(p) for p in paths}
    for n,h in manifest.items():assert sha(B/n)==h
    write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,goal_complete=False,new_model_selected=False,qualified=False,
        no_live_execution_mutex=True,all_pilot_files_verified=len(d['pilot_files_sha256']),files_sha256=manifest,CPU_close_seconds=time.monotonic()-start))
    write(B/'GOAL_TURN_R37_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        concrete_progress='Verified omitted static28 matched-energy information;60 fixed CPU heads and60 original controls audited. Core improves only Takayuki, primary gate fails; no new outer test or parameter search.',
        new_outer_test_evaluation=False,new_model_selected=False,no_active_handle=True,genuine_external_blocker=False,
        closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\nR37当前原生头补匹配能量闭合：45新220D来源头+15新TD28头、9196内部OOF、15能量尺度，45旧R25/15旧TD24参考复用，全部身份/缺失/尺度/联合凸证书/增广lstsq/部署/SHA通过。Core对原R25的ΔPCC −.001452/−.004956/+.005304，MSE下降−.3578%/−.8374%/+.7200%；主performance=false、TD28辅助source_increment=true、最终gate=false，不据Tak单项拼接来源/扩能量或参数网格，不开新外层。CPU拟合+验收{cost["CPU_fit_and_audit_seconds"]:.6f}s，合成{synthetic:.6f}s另计，GPU/source forward/PCA0。四设置主结果仍R26+3.2771%，goal active未达5%；上一轮及本轮progress，无活句柄/外部阻断，详见ROUND37_RESULTS.md。\n')
    print(json.dumps(dict(status='closed',qualified=False,goal_complete=False,closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report),CPU_fit_and_audit_seconds=cost['CPU_fit_and_audit_seconds'])))


if __name__=='__main__':main()
