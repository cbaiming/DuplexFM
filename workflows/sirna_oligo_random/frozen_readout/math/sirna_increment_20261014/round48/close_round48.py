"""Close the fixed input geometry gate pilot without selecting alternate sources."""
import sys,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent
sys.path.insert(0,str(R))
from run import io,validate


def main():
    started=time.monotonic();report=B/'ROUND48_RESULTS.md'
    assert not (R/'execution.mutex').exists()
    for p in (R/'cost_receipt.json',R/'closure_receipt.json',report,B/'GOAL_TURN_R48_AUDIT.json'):assert not p.exists(),p
    lock=validate();d=io.read(R/'diagnostic.json');v=io.read(R/'verification.json')
    assert v['passed'] and d['all_numerically_qualified'] and not d['qualified'] and not v['qualified'] and not d['performance_gate']
    assert d['lock_sha256']==v['lock_sha256']==io.sha(R/'lock.json') and v['diagnostic_sha256']==io.sha(R/'diagnostic.json')
    assert d['diagnostic_rows_sha256']==io.sha(R/'diagnostic_rows.csv')
    for n,h in d['pilot_files_sha256'].items():assert io.sha(R/'pilot'/n)==h,n
    assert v['counts']==dict(new_source=45,new_TD=15,old_source=45,old_TD=15,rows=9196,input_geometries=60)
    assert d['new_source_forward']==d['new_PCA']==d['new_original_scalers']==d['GPU_hours']==0
    runs=list(R.glob('execution_*.json'));audits=list(R.glob('verification_execution_*.json'))
    assert len(runs)==len(audits)==1
    e,a=io.read(runs[0]),io.read(audits[0]);assert e['returncode']==a['returncode']==0
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==11
    assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
    synthetic=sum(float(s.get('time',0)) for s in suites)
    cost=dict(CPU_controller_seconds=e['CPU_wall_seconds'],CPU_independent_audit_outer_seconds=a['CPU_wall_seconds'],
        CPU_independent_audit_inner_included=v['CPU_wall_seconds'],CPU_controller_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],
        CPU_synthetic_seconds=synthetic,new_source_heads=45,new_TD_gated_heads=15,new_input_geometries=60,new_gate_centering_means=60,
        independent_geometry_SVD_replays=60,independent_new_source_full_joint_certificates=45,independent_TD_lstsq_replays=15,
        GPU_hours=0,new_source_forward=0,new_PCA=0,new_original_scalers=0,
        source_training_old_cache_startup_registration_development_and_close_not_included=True,
        execution_sha256=io.sha(runs[0]),verification_execution_sha256=io.sha(audits[0]))
    lines=['# R48：按训练输入范围裁剪来源表示未改善迁移','',
        '固定60个CPU轻量读出及独立数值验收已闭合。Core在三个开发fit池的PCC均小幅下降，MSE均增加；预定performance gate=false、qualified=false。不采用该gate，不扫描分位数/距离正则/中心化或按队列挑选来源，不运行新的外层评价。完整四设置主结果仍为R26相对强TD24的平均PCC增益+3.2771%，未达5%。','',
        '本轮检验输入几何外推约束：对原标准化来源192维，训练输入定义H=SᵀS+100I、h=sᵀH⁻¹s；训练h的90% higher分位数为τ，g=1或sqrt(τ/h)。只缩小超过τ的样本，再减训练集门后均值。保留原TD24输入与单216D核范数读出（alpha100、lambda/critical=.1）。Full/精确初始化使用相同规则；另15个TD-only头接受同一输入几何规则，作为通用处理对照。所有H/τ/门后均值来自对应inner train，不使用验证响应或验证输入拟合gate。','',
        '|开发fit池|来源/控制|原PCC|新PCC|ΔPCC|原MSE|新MSE|MSE下降（负值退步）|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in d['records']:
        m,ref=r['metrics'],r['reference']
        lines.append(f"|{r['setting']}|{r['source']}|{ref['PCC']:.6f}|{m['PCC']:.6f}|{r['delta_PCC_vs_reference']:+.6f}|{ref['MSE']:.6f}|{m['MSE']:.6f}|{r['MSE_reduction_vs_reference_pct']:+.4f}%|")
    lines+=['',
        '来源三条件的“原”是R25同alpha100原生核范数头；TD-only的“原”是相同内折alpha100 Ridge。它们都是开发控制，不替换目标TD24_Ridge_CV。这里的三个fit池1564/319/416、五内折并非新的外部队列或四设置最终成绩。','',
        '初始化对照三个池均小幅改善，而Core均小幅退步；Full仅Hu略改善。TD-gated在Hu/Mix PCC略好、Tak PCC下降，但三池MSE都略降，说明PCC与MSE的响应不必一致。不能将输入处理带来的任何微小收益全部归因于miRNA训练，也不能凭本轮失败认定输入外推不影响误差。本次固定裁剪缺乏采用依据。','',
        '|开发fit池|Core-gated减TD-gated PCC|Core-gated相对TD-gated MSE下降|Core的PCC变化减TD的PCC变化|',
        '|---|---:|---:|---:|']
    for p in d['paired_gated_TD']:
        lines.append(f"|{p['setting']}|{p['delta_PCC']:+.6f}|{p['MSE_reduction_pct']:+.4f}%|{p['delta_gain_above_TD_gate']:+.6f}|")
    lines+=['',
        f"资格：performance={d['performance_gate']}，source_increment_vs_gated_TD={d['source_increment_vs_gated_TD']}，numerical={d['all_numerically_qualified']}，最终qualified={d['qualified']}。预先规定Core至少2池相对R25 ΔPCC>=.005且MSE非劣、所有池ΔPCC>=−.005；新TD-gated增量控制也按原规则检查。主性能门槛失败，不更改门槛或选择Full/初始化作为成功方案。",'',
        '|开发fit池|条件|内折|训练/验证数|τ|训练裁剪数|验证裁剪数|训练g均值|验证g均值/最小值|原验证PCC|新验证PCC|原验证MSE|新验证MSE|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in d['records']:
        for b in r['inner_blocks']:
            tr,va=b['training_geometry'],b['validation_geometry']
            lines.append(f"|{r['setting']}|{r['source']}|{b['inner']}|{tr['n']}/{va['n']}|{b['threshold']:.6f}|{tr['clipped_count']}|{va['clipped_count']}|{tr['weight_mean']:.6f}|{va['weight_mean']:.6f}/{va['weight_min']:.6f}|{b['reference']['PCC']:.6f}|{b['metrics']['PCC']:.6f}|{b['reference']['MSE']:.6f}|{b['metrics']['MSE']:.6f}|")
    lines+=['',
        '完整逐行h/g保存在round48/diagnostic_rows.csv；逐块h的最小/最大/均值在diagnostic.json，不挑选高h子集宣称主效果。半径限制g²h<=τ仅在门后重新中心化之前成立。h不是已校准的预测不确定度，也不是错误概率。TD输入保留但系数联合学习，不能保证g趋近0时等于独立TD预测，亦不能保证非负增益。','',
        '模型容量：新45个来源头仍有216维系数及截距，但部署另需原标准化参数、192×192精度矩阵、τ和192维门后均值；TD-gated需要24×24矩阵及24维均值。不能称217个系数即可独立部署。新60个输入几何和门后均值均如实计为新增训练状态。来源网络冻结，无原五expert的独立路由或生物机制识别。','',
        '验收：11项纯合成首次通过；60个输入几何由独立SVD（包含零空间）重建，分位数由排序重算、逐行g与批独立性通过；45新+45旧核范数头完整凸证书、15新TD增广lstsq及15旧TD正规方程通过。45新来源头/15新TD头共9196新内部OOF行，每条件每池exact-once；45旧来源与15唯一旧TD只读复用。来源/初始化、原标准化、行序、保存状态部署和全部SHA通过。','',
        f"最大精度矩阵差{v['max_precision_difference']:.3g}、TD独立lstsq系数差{v['max_independent_TD_lstsq_difference']:.3g}、全系统正规方程误差{v['max_normal_error']:.3g}、核范数相对对偶间隙{v['max_relative_dual_gap']:.3g}、部署预测差{v['max_deployment_difference']:.3g}。这些验证数值实现正确，不证明总体泛化或因果机制。",'',
        f"成本：CPU控制器{e['CPU_wall_seconds']:.6f}秒＋独立验收{a['CPU_wall_seconds']:.6f}秒＝{cost['CPU_controller_and_audit_seconds']:.6f}秒；合成{synthetic:.6f}秒另列。本轮无失败。新GPU/来源forward/PCA/原scaler拟合均0；独立SVD/lstsq验收成本包含在审计时间。历史来源训练/缓存、启动/登记/开发/收尾不含，不能称端到端速度。",'',
        '公式背景：[Cohen、Musco与Musco的ridge leverage score论文](https://arxiv.org/abs/1511.07263)。文献研究矩阵采样/近似；此处固定输入半径裁剪是本轮假设，不继承其理论保证。','',
        '来源keep-overlap/context暴露、单source42、反复使用开发池、purged非gene隔离和两个Mix设置非独立的限制均保留。未开封新外层测试，目标未达成，保持active。','',
        f"科学锁：{io.sha(R/'lock.json')}。诊断：{io.sha(R/'diagnostic.json')}。独立验收：{io.sha(R/'verification.json')}。"]
    io.write(R/'cost_receipt.json',cost)
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    files=[R/'lock.json',R/'diagnostic.json',R/'diagnostic_rows.csv',R/'verification.json',R/'cost_receipt.json',report,Path(__file__)]+[R/n for n in lock['files']]+runs+audits
    manifest={p.relative_to(B).as_posix():io.sha(p) for p in files}
    for n,h in manifest.items():assert io.sha(B/n)==h
    io.write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,qualified=False,goal_complete=False,
        no_live_execution_mutex=True,files_sha256=manifest,pilot_files_verified=len(d['pilot_files_sha256']),CPU_close_seconds=time.monotonic()-started))
    io.write(B/'GOAL_TURN_R48_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        genuine_external_blocker=False,no_active_handle=True,new_outer_test_evaluation=False,new_source_heads=45,new_TD_gated_heads=15,
        concrete_progress='60 input-only ridge geometry gates/heads, 9196 inner OOF rows, and independent SVD/full-system/lstsq audits completed. Core regresses slightly in all pools; fixed gate false, no outer evaluation or grid.',
        closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write('\n\nR48输入几何半径裁剪闭合：45新来源头+15TD-gated对照、60训练输入几何/门后均值、9196内部OOF；60独立SVD、45新+45旧凸证书、15新TD增广lstsq/15旧TD正规方程、部署/全部SHA通过。Core对原R25 ΔPCC Hu−.001085/Mix−.002891/Tak−.001884，MSE下降−.1822%/−.4787%/−.1255%；performance=false、最终gate=false，不做新外层或分位数/距离/中心化网格，不选择Full/初始化替代主Core。'
                f'CPU控制器+验收{cost["CPU_controller_and_audit_seconds"]:.6f}s，合成{synthetic:.6f}s另列，新GPU/forward/PCA/原scaler0。部署需保存输入相关gate状态，不是独立217coef线性模型或校准置信度。完整主结果仍R26+3.2771%，goal active未达5%；上一轮及本轮progress，无活句柄/阻断，见ROUND48_RESULTS.md。\n')
    print(dict(status='closed',qualified=False,goal_complete=False,closure_sha256=io.sha(R/'closure_receipt.json'),
        report_sha256=io.sha(report),CPU_controller_and_audit_seconds=cost['CPU_controller_and_audit_seconds']))


if __name__=='__main__':main()
