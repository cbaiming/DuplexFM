"""Close an optimistic diagnostic without converting it into model performance."""
import sys,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent
sys.path.insert(0,str(R))
from run import io,validate


def main():
    started=time.monotonic();report=B/'ROUND45_RESULTS.md'
    assert not (R/'execution.mutex').exists()
    for p in (R/'cost_receipt.json',R/'closure_receipt.json',report,B/'GOAL_TURN_R45_AUDIT.json'):assert not p.exists(),p
    lock=validate();d=io.read(R/'diagnostic.json');v=io.read(R/'verification.json')
    assert v['passed'] and d['all_numerically_qualified'] and not d['qualified'] and not v['qualified']
    assert d['calibration_room_gate'] and not d['paired_ordering_gate']
    assert d['optimistic_diagnostic_only'] and not d['deployable'] and d['new_OOF_rows']==d['new_efficiency_heads']==0
    assert d['lock_sha256']==v['lock_sha256']==io.sha(R/'lock.json') and v['diagnostic_sha256']==io.sha(R/'diagnostic.json')
    assert d['projection_rows_sha256']==io.sha(R/'optimistic_projection_rows.csv')
    for n,h in d['projection_files_sha256'].items():assert io.sha(R/'projection_audit'/n)==h
    assert v['counts']==dict(projections=72,pool=12,inner=60,native_reused=45,TD_reused=15,rows=9196)
    runs=list(R.glob('execution_*.json'));audits=list(R.glob('verification_execution_*.json'))
    assert len(runs)==len(audits)==1
    e,a=io.read(runs[0]),io.read(audits[0]);assert e['returncode']==a['returncode']==0
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==9
    assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
    synthetic=sum(float(s.get('time',0)) for s in suites)
    cost=dict(CPU_diagnostic_seconds=e['CPU_wall_seconds'],CPU_independent_audit_outer_seconds=a['CPU_wall_seconds'],
        CPU_independent_audit_inner_seconds_included=v['CPU_wall_seconds'],
        CPU_diagnostic_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],CPU_synthetic_seconds=synthetic,
        GPU_hours=0,new_source_forward=0,new_OOF_rows=0,new_efficiency_heads=0,new_label_using_projections=72,
        source_training_old_cache_startup_registration_development_and_close_not_included=True,
        execution_sha256=io.sha(runs[0]),verification_execution_sha256=io.sha(audits[0]))
    lines=['# R45：单调刻度有空间，但不足以支持优先推进该方向','',
        '**本报告中的“上界”使用了评价行的已知效值进行保序拟合，是固定分数排序的样本内乐观诊断，不是可部署模型、折外性能或新benchmark成绩。** 原始分数来自已完成的训练池OOF；新的投影使用这些行的标签，不能继续称OOF。','',
        '结论：刻度空间资格通过，Core与同样处理TD的配对排序资格未通过；最终qualified=false。不启动后续校准器、不改变阈值/排序或挑选Full。完整四设置主结果仍为R26平均相对PCC +3.2771%，目标5%未达到。','',
        '固定三个原outer fold0 fit池Hu1564/Mix319/Tak416，原5内折；Core/Full/精确初始化采用旧R25单头分数，TD为旧固定alpha100头。只拟合非递减一维响应。对每个原分数，若改变的是单调刻度而不产生新排序，可得到条件于这些标签的经验PCC上界。','',
        '|训练池|原Core PCC|Core乐观PCC上界|原TD PCC|TD乐观PCC上界|上界差 Core−TD|',
        '|---|---:|---:|---:|---:|---:|']
    for p in d['paired_ceilings']:
        cm=next(r['metrics'] for r in d['records'] if r['setting']==p['setting'] and r['condition']=='core_real')
        tm=next(r['metrics'] for r in d['records'] if r['setting']==p['setting'] and r['condition']=='TD_only')
        lines.append(f"|{p['setting']}|{cm['raw_PCC']:.6f}|{p['Core_upper_PCC']:.6f}|{tm['raw_PCC']:.6f}|{p['TD_upper_PCC']:.6f}|{p['delta_upper_PCC']:+.6f}|")
    lines+=['',
        'Core的三个池均存在PCC刻度空间，但TD也存在。Hu/Tak的配对上界更高，Mix低约0.0130，超过预设非劣边界−0.005。因此没有足够理由把“只改当前分数刻度”作为优先优化路径。这不证明任何校准方法一定失败，也不证明整个5%目标不可能；上界只针对当前固定分数/样本，不约束重新训练改变排序后的模型。','',
        '|训练池|条件|原PCC|乐观上界|最多绝对PCC空间|原MSE|最佳非负仿射MSE|乐观单调MSE|相对最佳仿射MSE下降|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in d['records']:
        m=r['metrics'];assert m['raw_PCC'] is not None and m['upper_attained']
        lines.append(f"|{r['setting']}|{r['condition']}|{m['raw_PCC']:.6f}|{m['upper_PCC']:.6f}|{m['absolute_PCC_room']:+.6f}|{m['raw_MSE']:.6f}|{m['positive_affine_MSE']:.6f}|{m['optimistic_projection_MSE']:.6f}|{m['optimistic_MSE_reduction_vs_affine_pct']:+.4f}%|")
    lines+=['',
        f"资格：calibration_room={d['calibration_room_gate']}，paired_ordering={d['paired_ordering_gate']}，数值={d['all_numerically_qualified']}，最终qualified={d['qualified']}。主Core固定，不将Full或初始化替换为主模型。",'',
        '正仿射映射改变截距/斜率而不改变PCC；将单调投影与最佳非负仿射相比，可以区分非线性间距调整与单纯尺度校准。即使单调调整能增加PCC，也没有产生新的排序信息，更不能单独归因miRNA监督。所有控制都在同样的乐观条件下处理。','',
        '上界依据：中心化保序锥投影q_c满足yc=q_c+e，e属于极锥且与q_c正交。任意非递减非恒定h的相关性不超过||q_c||/||yc||；q非恒定时可达到。恒定投影仅报告0上界和未达到/不可定义标记，不把其PCC伪设为0。本轮真实72个投影均通过数值核验，不能将经验上界外推为总体极限。','',
        '|训练池|条件|旧验证内折|行数|原PCC|该块乐观上界|该块PCC空间|',
        '|---|---|---:|---:|---:|---:|---:|']
    def fmt(value):return '未定义' if value is None else f'{value:.6f}'
    for r in d['records']:
        for b in r['inner_blocks']:
            m=b['metrics'];lines.append(f"|{r['setting']}|{r['condition']}|{b['inner']}|{b['n']}|{fmt(m['raw_PCC'])}|{m['upper_PCC']:.6f}|{fmt(m['absolute_PCC_room'])}|")
    lines+=['',
        '每块小样本保序上界通常更乐观，分块表只用于披露，不挑选最佳块或据它调门槛。原OOF训练样本有重叠，不将行或折当独立重复，不计算p值或置信区间。','',
        '计数：72次新标签使用保序投影（12全池＋60分块），9196行旧OOF分数各保存全池/分块两种乐观投影。45旧来源效率头、15唯一TD头按SHA复用。新增来源forward/GPU/可部署效率头/真实OOF均0；保序诊断本身是有监督拟合，不能称未使用标签。','',
        f"验收：9项合成首轮通过；72次独立SciPy投影和原始行KKT/Moreau/上界算术、全部来源/划分/覆盖/资格/SHA通过。最大对偶gap绝对值{v['max_absolute_duality_gap']:.3g}，独立投影差{v['max_projection_difference']:.3g}，驻点误差{v['max_stationarity_error']:.3g}。",'',
        f"成本：CPU诊断{e['CPU_wall_seconds']:.6f}秒＋独立验收{a['CPU_wall_seconds']:.6f}秒，合计{cost['CPU_diagnostic_and_audit_seconds']:.6f}秒；合成{synthetic:.6f}秒另列。GPU/source forward0。旧来源训练/缓存、启动、登记、开发和收尾不计入该阶段时间，不声称端到端速度。",'',
        '数学背景：[Kakade等，NeurIPS 2011单指标模型与保序回归](https://papers.neurips.cc/paper_files/paper/2011/hash/30bb3825e8f631cc6075c0f87bb4978c-Abstract.html)。本轮不是SLIsotron实现，也不借用其泛化保证；固定排序的经验投影上界单独证明和验收。','',
        '后续方向必须能够改变从已有特征学到的排序，或解决来源/协议一致性；不得把这次乐观投影用于提交结果。source keep-overlap/context暴露、单来源种子、重复开发、非gene隔离及两个Mixset设置非独立等限制保留。无新的外层评价、未达5%、目标仍active。','',
        f"科学锁：{io.sha(R/'lock.json')}。诊断：{io.sha(R/'diagnostic.json')}。独立验收：{io.sha(R/'verification.json')}。"]
    io.write(R/'cost_receipt.json',cost)
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    files=[R/'lock.json',R/'diagnostic.json',R/'optimistic_projection_rows.csv',R/'verification.json',R/'cost_receipt.json',report,Path(__file__)]+[R/n for n in lock['files']]+runs+audits
    manifest={p.relative_to(B).as_posix():io.sha(p) for p in files}
    for n,h in manifest.items():assert io.sha(B/n)==h
    io.write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,qualified=False,goal_complete=False,
        no_live_execution_mutex=True,files_sha256=manifest,projection_files_verified=len(d['projection_files_sha256']),CPU_close_seconds=time.monotonic()-started))
    io.write(B/'GOAL_TURN_R45_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        genuine_external_blocker=False,no_active_handle=True,new_outer_test_evaluation=False,new_efficiency_heads=0,new_label_using_projections=72,
        concrete_progress='72 fixed-order optimistic isotonic projections independently audited. Calibration room exists, but Core vs TD ordering ceiling fails Mix noninferiority; do not train downstream calibrator.',
        closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write('\n\nR45固定排序单调响应上界闭合：72新标签使用投影（12池+60块）、9196旧OOF分数，72独立SciPy/KKT/Moreau/覆盖/SHA通过；不是新OOF性能或可部署模型。Core乐观上界减TD乐观上界Hu+.012613/Mix−.013002/Tak+.020997，calibration_room=true、paired_ordering=false、最终资格false。不训练后续校准器、不改门槛/排序、不新外层。'
                f'CPU诊断＋验收{cost["CPU_diagnostic_and_audit_seconds"]:.6f}s，合成{synthetic:.6f}s另计，新GPU/forward/效率头0。完整主结果仍R26+3.2771%，goal active未达5%；上一轮及本轮progress，无活句柄/阻断，见ROUND45_RESULTS.md。\n')
    print(dict(status='closed',qualified=False,goal_complete=False,closure_sha256=io.sha(R/'closure_receipt.json'),
               report_sha256=io.sha(report),CPU_diagnostic_and_audit_seconds=cost['CPU_diagnostic_and_audit_seconds']))


if __name__=='__main__':main()
