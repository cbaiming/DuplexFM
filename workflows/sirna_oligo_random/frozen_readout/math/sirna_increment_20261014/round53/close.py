"""Close the bounded pilot with all controls and preserved negative results."""
import sys,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent; sys.path.insert(0,str(R)); import run
io=run.io; B=run.B


def main():
    began=time.monotonic(); run.validate(); d=io.read(R/'diagnostic.json'); v=io.read(R/'verification.json')
    assert v['passed'] and v['qualified']==d['qualified'] and not (R/'execution.mutex').exists()
    assert d['lock_sha256']==v['lock_sha256']==io.sha(R/'lock.json') and v['diagnostic_sha256']==io.sha(R/'diagnostic.json')
    io.check_files(R/'pilot',d['pilot_files_sha256']); assert io.sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    assert v['counts']==d['counts']==dict(new_source=39,new_TD=13,identity_reused=8,scalings=45,nonidentity_scalings=39,rows=9196)
    runs=list(R.glob('execution_*.json')); audits=list(R.glob('verification_execution_*.json')); assert len(runs)==len(audits)==1
    e,a=io.read(runs[0]),io.read(audits[0]); assert e['returncode']==a['returncode']==0
    report=B/'ROUND53_RESULTS.md'; assert not report.exists() and not (R/'closure_receipt.json').exists()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==10 and not any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped'))
    synthetic=sum(float(s.get('time',0)) for s in suites)
    cost=dict(CPU_controller_seconds=e['CPU_wall_seconds'],CPU_independent_audit_seconds=a['CPU_wall_seconds'],
        CPU_controller_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],synthetic_seconds_separate=synthetic,
        new_source_heads=39,new_TD_heads=13,reused_identity_heads=8,new_nonidentity_scalings=39,identity_scale_records=6,
        new_source_forward=0,new_PCA=0,new_original_scalers=0,GPU_hours=0,not_end_to_end=True,
        historical_training_cache_startup_development_registration_close_excluded=True,
        execution_sha256=io.sha(runs[0]),verification_execution_sha256=io.sha(audits[0]))
    lines=['# R53：损失几何下的分块保能量对角缩放','',
        f"预登记资格 **{d['qualified']}**；performance={d['performance_gate']}、source_increment={d['source_increment_gate']}、numerical={d['all_numerically_qualified']}。本轮仅为三个既定fit池内部OOF，不是新的四设置外层结果。当前完整主成绩仍为R50相对强TD24平均PCC **+4.190823%**，5%目标未完成。",'',
        '唯一改变是R49投影后的坐标正则尺度：每列按照原训练能量／投影后能量调整，并分别保持TD24和source192块的投影后总能量。无新强度或维数网格；原alpha100、lambda=.1critical、组间gamma=.5、原6×32核范数和源seed42不变。比例固定但critical由新设计重新计算。','',
        '|开发fit池|条件|R49 PCC|新PCC|ΔPCC|R49 MSE|新MSE|MSE下降|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in d['records']:
        m,ref=r['metrics'],r['reference']
        lines.append(f"|{r['setting']}|{r['source']}|{ref['PCC']:.6f}|{m['PCC']:.6f}|{r['delta_PCC_vs_reference']:+.6f}|{ref['MSE']:.6f}|{m['MSE']:.6f}|{r['MSE_reduction_vs_reference_pct']:+.4f}%|")
    lines+=['','|fit池|Core减同处理TD的PCC|Core相对同处理TD的MSE下降|Core改善减TD改善（ΔPCC）|','|---|---:|---:|---:|']
    for p in d['paired_TD']:
        lines.append(f"|{p['setting']}|{p['delta_PCC']:+.6f}|{p['MSE_reduction_pct']:+.4f}%|{p['delta_gain_above_TD']:+.6f}|")
    lines+=['',
        '资格要求Core至少2池ΔPCC≥.005且MSE非劣，所有池ΔPCC≥−.005；对同处理TD至少2池PCC非劣且等权相对MSE非负。Full、精确initial与TD均完整报告，不按条件或队列挑选。',
        ('本轮只取得完整评价资格。后续需单独登记原五alpha选择及四设置协议，不得将本内部结果当作5%完成证据。' if d['qualified'] else '未通过资格，结束当前缩放分支；不扩尺度/alpha/gamma网格、不按Full或初始化替换主方案、不新增四设置评价。'),'',
        '这是真实改变惩罚先验的统计因素，不是原模型数值加速。原来方差缩放并非已证实错误；总能量保持也不保证泛化改善。原组间降权可能丢弃有用差异，输入组不等于gene/assay。部署时将因子折入系数，仍为单216D向量＋截距，不查询验证组或组合其它预测。','',
        '|池|源|内折|恒等|TD因子范围|source因子范围|TD投影能量|source投影能量|',
        '|---|---|---:|---|---:|---:|---:|---:|']
    for r in d['scalings']:
        t,s=r['blocks']
        lines.append(f"|{r['setting']}|{r['source']}|{r['inner']}|{r['identity']}|{t['factor_min']:.6f}–{t['factor_max']:.6f}|{s['factor_min']:.6f}–{s['factor_max']:.6f}|{t['projected_energy']:.6f}|{s['projected_energy']:.6f}|")
    lines+=['',
        '45份因子记录含39非恒等估计和6恒等；TD-only共享Core的前24因子，不依赖来源块。15原训练组及原scaler按SHA复用；新39来源＋13TD拟合、8恒等旧头，共60条件／9196内部OOF行，对应2299fit身份。旧60参考只读使用，不称重新训练。','',
        '|池|条件|内折|R49验证PCC|新验证PCC|R49验证MSE|新验证MSE|恒等复用|',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for r in d['records']:
        for p in r['inner_blocks']:
            lines.append(f"|{r['setting']}|{r['source']}|{p['inner']}|{p['reference']['PCC']:.6f}|{p['metrics']['PCC']:.6f}|{p['reference']['MSE']:.6f}|{p['metrics']['MSE']:.6f}|{p['identity_reused']}|")
    lines+=['',
        f"审计：10项纯合成通过；15独立输入图、45独立列二次型尺度、新旧各45来源完整联合证书／15TD增广lstsq、原输入/身份/原始部署/OOF覆盖及270产物SHA通过。最大缩放差{v['max_scaling_difference']:.3g}，相对能量差{v['max_relative_energy_difference']:.3g}，正规方程误差{v['max_normal_error']:.3g}，相对对偶间隙{v['max_relative_dual_gap']:.3g}，部署差{v['max_deployment_difference']:.3g}。",'',
        f"成本：CPU控制器{e['CPU_wall_seconds']:.6f}s＋独立验收{a['CPU_wall_seconds']:.6f}s＝{cost['CPU_controller_and_audit_seconds']:.6f}s；合成{synthetic:.6f}s另列。新GPU／forward／PCA／原scaler拟合均0。存在新尺度估计和新轻量头；历史训练/缓存、开发/启动/登记/收尾不含，不称端到端耗时。",'',
        '单来源seed42、source keep-overlap/context暴露、purged非gene隔离、反复开发及两个Mix设置不独立继续限制推断。本轮由此前结果引导选择，不能称新独立确认或SOTA。目标active且未完成，无活执行句柄或外部阻断。',
        f"科学锁SHA {io.sha(R/'lock.json')}；诊断SHA {io.sha(R/'diagnostic.json')}；独立验收SHA {io.sha(R/'verification.json')}。"]
    io.write(R/'cost_receipt.json',cost)
    with report.open('x',encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')
    files=[R/'lock.json',R/'diagnostic.json',R/'diagnostic_rows.csv',R/'verification.json',R/'cost_receipt.json',report]+[R/n for n in run.CODE]+runs+audits
    manifest={p.relative_to(B).as_posix():io.sha(p) for p in files}
    io.check_files(B,manifest)
    io.write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,files_sha256=manifest,qualified=d['qualified'],goal_complete=False,
        pilot_files_verified=270,no_live_execution_mutex=True,CPU_close_seconds=time.monotonic()-began))
    io.write(B/'GOAL_TURN_R53_AUDIT.json',dict(previous_turn='progress',previous_turn_evidence='PAPER_GAIN_EXPLANATION_R50_20261015.md',
        this_turn='progress',goal_status='active',objective_achieved=False,genuine_external_blocker=False,no_active_handle=True,
        new_source_heads=39,new_TD_heads=13,new_scaling_records=45,qualification=d['qualified'],current_primary_relative_PCC_gain_pct=4.190823121288767,
        closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)))
    core=[r for r in d['records'] if r['source']=='core_real']
    brief='/'.join(f"{r['delta_PCC_vs_reference']:+.6f}" for r in core)
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write(f"\n\nR53损失几何分块保能量对角缩放闭合：39新来源＋13新TD头／8恒等复用、45尺度／9196内部OOF；15独立输入图、列二次型尺度及新旧凸证书/TD独立解/部署/全部SHA通过。Core对R49 ΔPCC {brief}，资格{d['qualified']}；不调门槛或按控制择优。CPU控制器+验收{cost['CPU_controller_and_audit_seconds']:.6f}s，合成另列，新GPU/forward/PCA0。完整四设置仍R50+4.190823%，goal active未达标；前轮分析与本轮progress，无活句柄或外部阻断。详见ROUND53_RESULTS.md。\n")
    print(dict(status='closed',qualified=d['qualified'],CPU_seconds=cost['CPU_controller_and_audit_seconds'],closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)),flush=True)


if __name__=='__main__': main()
