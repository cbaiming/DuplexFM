"""Full positive/negative report for the fixed target random-intercept pilot."""
import sys,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import run
io=run.io;B=run.B


def main():
    began=time.monotonic();run.validate();d=io.read(R/'diagnostic.json');v=io.read(R/'verification.json')
    assert v['passed'] and v['qualified']==d['qualified'] and not (R/'execution.mutex').exists()
    assert d['lock_sha256']==v['lock_sha256']==io.sha(R/'lock.json') and v['diagnostic_sha256']==io.sha(R/'diagnostic.json')
    io.check_files(R/'pilot',d['pilot_files_sha256']);assert io.sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    assert v['counts']==d['counts']==dict(new_source=39,new_TD=13,equal_reused=8,geometries=15,rows=9196)
    executions=list(R.glob('execution_*.json'));audits=list(R.glob('verification_execution_*.json'));assert len(executions)==len(audits)==1
    e,a=io.read(executions[0]),io.read(audits[0]);assert e['returncode']==a['returncode']==0
    report=B/'ROUND54_RESULTS.md';assert not report.exists() and not (R/'closure_receipt.json').exists()
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==10 and not any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped'))
    synthetic=sum(float(s.get('time',0)) for s in suites)
    cost=dict(CPU_controller_seconds=e['CPU_wall_seconds'],CPU_independent_audit_seconds=a['CPU_wall_seconds'],
        CPU_controller_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],synthetic_seconds_separate=synthetic,
        new_source_heads=39,new_TD_heads=13,equal_reference_reused=8,new_group_geometry_records=15,
        new_source_forward=0,new_PCA=0,new_original_scalers=0,GPU_hours=0,not_end_to_end=True,
        historical_training_cache_startup_development_registration_close_excluded=True,
        execution_sha256=io.sha(executions[0]),verification_execution_sha256=io.sha(audits[0]))
    lines=['# R54：按训练组大小收缩共同靶区域截距','',
        f"预登记资格 **{d['qualified']}**；performance={d['performance_gate']}、source_increment={d['source_increment_gate']}、numerical={d['all_numerically_qualified']}。本轮仅为三个固定fit池内部OOF，当前完整四设置主成绩仍为R50相对强TD24平均PCC **+4.190823%**，5%目标尚未完成。",'',
        '唯一因素是把R49各组截距惩罚n_g改为公共rho=n/K。每组的profile残差权重为rho/(rho+n_g)，全局截距按相应权重求解，并保留原n−1中心化loss迹。原始特征/scaler、alpha100、核范数比例.1与求解器不变。大组偏移吸收较强，小组收缩较强；该规则是固定先验，不是估计真实批次方差。','',
        '|开发fit池|条件|R49 PCC|新PCC|ΔPCC|R49 MSE|新MSE|MSE下降|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in d['records']:
        m,rr=r['metrics'],r['reference']
        lines.append(f"|{r['setting']}|{r['source']}|{rr['PCC']:.6f}|{m['PCC']:.6f}|{r['delta_PCC_vs_reference']:+.6f}|{rr['MSE']:.6f}|{m['MSE']:.6f}|{r['MSE_reduction_vs_reference_pct']:+.4f}%|")
    lines+=['','|fit池|Core减同处理TD的PCC|Core相对同处理TD的MSE下降|Core改善减TD改善（ΔPCC）|','|---|---:|---:|---:|']
    for p in d['paired_TD']:lines.append(f"|{p['setting']}|{p['delta_PCC']:+.6f}|{p['MSE_reduction_pct']:+.4f}%|{p['delta_gain_above_TD']:+.6f}|")
    lines+=['',
        '预定门槛为Core至少2池ΔPCC≥.005且MSE非劣，所有池ΔPCC≥−.005；对同处理TD至少2池PCC非劣且等权相对MSE非负。Full、初始化、TD完整报告，不用控制组的有利结果替换Core。',
        ('只取得下一阶段资格，需另立原五alpha与完整四设置评价；内部结果不是达到5%的证据。' if d['qualified'] else '未通过资格，结束这个固定公共组惩罚分支；不调整rho/分组或扩大网格，不开展新的四设置外层评价。'),'',
        '|池|内折|训练数|组数|公共rho|组间权重最小/最大|loss迹归一系数|与R49等价|',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for g in d['geometries']:
        lines.append(f"|{g['setting']}|{g['inner']}|{g['n']}|{g['K']}|{g['common_group_penalty']:.6f}|{g['gamma_min']:.6f}/{g['gamma_max']:.6f}|{g['normalizer']:.6f}|{g['reference_equal']}|")
    lines+=['',
        '39新来源＋13新TD头、8等价旧头，共60条件／9196内部OOF行，来自2299个fit身份。15训练分组按SHA复用，仅新增15份组大小规则记录。不重建源表示、不读取新外层标签；旧60参考只读复用。每个训练组的隐式截距不进入部署，最终仍是单216D系数＋截距；额外训练容量必须披露。','',
        '|池|条件|内折|R49验证PCC|新验证PCC|R49验证MSE|新验证MSE|等价复用|',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for r in d['records']:
        for p in r['inner_blocks']:
            lines.append(f"|{r['setting']}|{r['source']}|{p['inner']}|{p['reference']['PCC']:.6f}|{p['metrics']['PCC']:.6f}|{p['reference']['MSE']:.6f}|{p['metrics']['MSE']:.6f}|{p['equal_reused']}|")
    lines+=['',
        f"验收：10项纯合成通过；15独立输入图与profile矩阵、45来源完整联合凸证书／15显式组截距增广lstsq、新旧参考、原始空间部署、逐行覆盖与210产物SHA通过。最大正规方程误差{v['max_normal_error']:.3g}，相对对偶间隙{v['max_relative_dual_gap']:.3g}，TD独立系数差{v['max_TD_lstsq_difference']:.3g}，全局截距方程误差{v['max_intercept_normal_error']:.3g}，部署差{v['max_deployment_difference']:.3g}。",'',
        f"成本：CPU控制器{e['CPU_wall_seconds']:.6f}s＋独立验收{a['CPU_wall_seconds']:.6f}s＝{cost['CPU_controller_and_audit_seconds']:.6f}s；合成{synthetic:.6f}s另列。新GPU/forward/PCA/原scaler拟合0；历史训练缓存、启动/开发/登记/收尾不含，不称端到端时间。",'',
        '动机来自混合效应的惩罚最小二乘，参见[Bates等，JSS 2015](https://www.jstatsoft.org/article/view/v067i01)。这里未作REML或方差估计，输入连接组也不是已确认的gene/assay。不能把共同偏移吸收解释为证明生物学机制或保证跨队列稳健。',
        'source keep-overlap/context暴露、单source42、purged非gene隔离、反复开发与两个Mix设置不独立继续限制推断。目标active、未达5%，无活执行句柄或外部阻断，不宣称SOTA或新独立确认。',
        f"科学锁SHA {io.sha(R/'lock.json')}；诊断SHA {io.sha(R/'diagnostic.json')}；独立验收SHA {io.sha(R/'verification.json')}。"]
    io.write(R/'cost_receipt.json',cost)
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    files=[R/'lock.json',R/'diagnostic.json',R/'diagnostic_rows.csv',R/'verification.json',R/'cost_receipt.json',report]+[R/n for n in run.CODE]+executions+audits
    manifest={p.relative_to(B).as_posix():io.sha(p) for p in files};io.check_files(B,manifest)
    io.write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,files_sha256=manifest,qualified=d['qualified'],goal_complete=False,
        pilot_files_verified=210,no_live_execution_mutex=True,CPU_close_seconds=time.monotonic()-began))
    io.write(B/'GOAL_TURN_R54_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        genuine_external_blocker=False,no_active_handle=True,new_source_heads=39,new_TD_heads=13,new_group_geometry_records=15,
        qualification=d['qualified'],current_primary_relative_PCC_gain_pct=4.190823121288767,
        closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)))
    brief='/'.join(f"{r['delta_PCC_vs_reference']:+.6f}" for r in d['records'] if r['source']=='core_real')
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write(f"\n\nR54公共靶区域截距惩罚n/K闭合：39新来源＋13TD／8等价复用、15规则记录／9196内部OOF；独立指示矩阵profile、全部凸证书/显式组TD增广lstsq/原始部署/覆盖/SHA通过。Core对R49 ΔPCC {brief}，资格{d['qualified']}；不调rho/分组或按控制择优。CPU控制器+验收{cost['CPU_controller_and_audit_seconds']:.6f}s，合成另列，新GPU/forward/PCA0。完整四设置仍R50+4.190823%，goal active未完成；前轮/本轮progress，无活句柄或外部阻断。详见ROUND54_RESULTS.md。\n")
    print(dict(status='closed',qualified=d['qualified'],CPU_seconds=cost['CPU_controller_and_audit_seconds'],closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)),flush=True)


if __name__=='__main__':main()
