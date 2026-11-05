"""Close qualified partial target-group pilot; outer confirmation remains pending."""
import sys,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent
sys.path.insert(0,str(R))
from run import io,validate


def main():
    started=time.monotonic();report=B/'ROUND49_RESULTS.md'
    assert not (R/'execution.mutex').exists()
    for p in (R/'cost_receipt.json',R/'closure_receipt.json',report,B/'GOAL_TURN_R49_AUDIT.json'):assert not p.exists(),p
    lock=validate();d=io.read(R/'diagnostic.json');v=io.read(R/'verification.json')
    assert v['passed'] and d['qualified'] and v['qualified'] and d['performance_gate'] and d['source_increment_gate'] and d['all_numerically_qualified']
    assert d['lock_sha256']==v['lock_sha256']==io.sha(R/'lock.json') and v['diagnostic_sha256']==io.sha(R/'diagnostic.json')
    assert d['diagnostic_rows_sha256']==io.sha(R/'diagnostic_rows.csv')
    for n,h in d['pilot_files_sha256'].items():assert io.sha(R/'pilot'/n)==h,n
    assert v['counts']==dict(new_source=39,new_TD=13,identity_reused=8,old_source=45,old_TD=15,rows=9196,partitions=15)
    runs=list(R.glob('execution_*.json'));audits=list(R.glob('verification_execution_*.json'));assert len(runs)==len(audits)==1
    e,a=io.read(runs[0]),io.read(audits[0]);assert e['returncode']==a['returncode']==0
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==11
    assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'));synthetic=sum(float(s.get('time',0)) for s in suites)
    cost=dict(CPU_controller_seconds=e['CPU_wall_seconds'],CPU_independent_audit_outer_seconds=a['CPU_wall_seconds'],
        CPU_independent_audit_inner_included=v['CPU_wall_seconds'],CPU_controller_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],
        CPU_synthetic_seconds=synthetic,new_source_heads=39,new_TD_heads=13,identity_reused_condition_heads=8,new_training_partitions=15,
        independent_input_graphs=15,independent_condition_source_certificates=45,independent_TD_lstsq=15,
        GPU_hours=0,new_source_forward=0,new_PCA=0,new_original_scalers=0,
        old_training_cache_startup_registration_development_and_close_not_included=True,
        execution_sha256=io.sha(runs[0]),verification_execution_sha256=io.sha(audits[0]))
    lines=['# R49：靶区域组间偏移部分降权通过开发资格','',
        'Core三个固定开发池的PCC与MSE均改善，预定performance/source_increment/numerical gate全部通过。该方法取得进一步四设置评价资格，但当前完整主成绩仍为R26平均相对PCC增益+3.2771%；本轮不能称达到5%或独立确认。','',
        '借用早期gene组内训练的思路，本轮只使用当前inner train靶序列共享canonical25-mer的连通分量，不能称已确认gene/assay。对组间残差固定乘.5、组内残差保留，并按中心化空间trace归一。来源192/TD24、原尺度、alpha100/核范数比例.1/来源seed42均不变。预测端仍是单线性头＋全局截距，不查询组或恢复组偏移。','',
        '|开发fit池|条件|原PCC|新PCC|ΔPCC|原MSE|新MSE|MSE下降|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in d['records']:
        m,ref=r['metrics'],r['reference']
        lines.append(f"|{r['setting']}|{r['source']}|{ref['PCC']:.6f}|{m['PCC']:.6f}|{r['delta_PCC_vs_reference']:+.6f}|{ref['MSE']:.6f}|{m['MSE']:.6f}|{r['MSE_reduction_vs_reference_pct']:+.4f}%|")
    lines+=['',
        '所有来源和TD对照均改善，因此收益很大一部分属于训练目标的一般性改进。Core相对同处理TD在Hu/Tak仍有正增量，Mix仍为负；不能把全部改善归于miRNA监督或宣称自动消除了生物学混杂。精确初始化也改善，进一步支持保留来源对照。这里的9196条件行来自2299个固定fit池样本的内部OOF，并非新的外层性能。','',
        '|开发fit池|Core减同处理TD的PCC|Core相对同处理TD的MSE下降|Core改善减TD改善（ΔPCC）|',
        '|---|---:|---:|---:|']
    for p in d['paired_TD']:lines.append(f"|{p['setting']}|{p['delta_PCC']:+.6f}|{p['MSE_reduction_pct']:+.4f}%|{p['delta_gain_above_TD']:+.6f}|")
    lines+=['',
        '固定资格要求Core至少2池ΔPCC>=.005且MSE非劣，全部ΔPCC>=−.005；Mix/Tak满足幅度门槛，Hu小幅改善。相对同处理TD至少2池PCC非劣、等权MSE改善非负也通过。没有修改门槛或按来源挑选。下一阶段需要固定原五alpha和同选参规则，完整四设置同时报告三个来源与同处理TD；不扩大gamma/25-mer长度网格。','',
        '|池|内折|训练数|组数|最大组比例|非单例组/样本数|迹归一系数|与原目标恒等|',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for p in d['partitions']:
        lines.append(f"|{p['setting']}|{p['inner']}|{p['n']}|{p['K']}|{p['largest_fraction']:.4f}|{p['nonsingleton_groups']}/{p['nonsingleton_rows']}|{p['normalizer']:.6f}|{p['identity_on_centered_space']}|")
    lines+=['',
        'Hu各内折46–138组，Mix104–166组，Tak1–2组。各内折重新用训练输入构图，删除部分训练样本会断开序列连接，因此组数可以明显变化。Mix有不少单例，降权也改变单例与重复区域之间的相对影响，不能仅解释为移除实验批次。Tak两内折为单组，四条件共8头直接引用原结果；其余39来源+13TD新拟合。','',
        '|池|条件|内折|原验证PCC|新验证PCC|原验证MSE|新验证MSE|原始训练MSE|加权训练SSE|复用恒等头|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in d['records']:
        for p in r['inner_blocks']:
            lines.append(f"|{r['setting']}|{r['source']}|{p['inner']}|{p['reference']['PCC']:.6f}|{p['metrics']['PCC']:.6f}|{p['reference']['MSE']:.6f}|{p['metrics']['MSE']:.6f}|{p['training_residual']['raw_training_MSE']:.6f}|{p['training_residual']['transformed_residual_SSE']:.6f}|{p['identity_reused']}|")
    lines+=['',
        '训练增加K−1隐式组对比，部署217系数并不代表总监督容量相同。保留组内信号的同时，可能降权真实组间生物学差异；结果不能作因果解释。','',
        f"核验：11项新合成首次通过，15输入分组由独立稀疏关联图重建；60条件头含52新拟合与8恒等复用，45来源完整凸证书/15TD增广lstsq、45旧来源及15旧TD参考、组投影/原尺度/响应/OOF覆盖/部署/SHA全部通过。最大正规方程误差{v['max_normal_error']:.3g}、相对对偶间隙{v['max_relative_dual_gap']:.3g}、独立TD系数差{v['max_independent_TD_lstsq_difference']:.3g}、部署差{v['max_deployment_difference']:.3g}。",'',
        f"成本：CPU控制器{e['CPU_wall_seconds']:.6f}秒＋独立验收{a['CPU_wall_seconds']:.6f}秒＝{cost['CPU_controller_and_audit_seconds']:.6f}秒；合成{synthetic:.6f}秒另列。本轮无数值运行失败。一次只读文件命令的自动审批超时未启动科学计算，重试读取成功，不计作训练成本。新GPU/forward/PCA/原scaler拟合均0；历史源训练/缓存、启动/登记/开发/收尾不含，不能称端到端耗时。",'',
        '方法背景：[Rothenhäusler等，JRSSB 2021](https://academic.oup.com/jrsssb/article/83/2/215/7056043)的组投影损失分解。此处输入定义组不是外生anchor、gamma固定小于1，不继承分布稳健或因果保证。','',
        'source keep-overlap/context暴露、反复开发、单source42、purged非gene隔离及两个Mix设置不独立仍保留。新四设置尚未计算，目标active未完成。','',
        f"科学锁：{io.sha(R/'lock.json')}。诊断：{io.sha(R/'diagnostic.json')}。独立验收：{io.sha(R/'verification.json')}。"]
    io.write(R/'cost_receipt.json',cost)
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    files=[R/'lock.json',R/'diagnostic.json',R/'diagnostic_rows.csv',R/'verification.json',R/'cost_receipt.json',report,Path(__file__)]+[R/n for n in lock['files']]+runs+audits
    manifest={p.relative_to(B).as_posix():io.sha(p) for p in files}
    for n,h in manifest.items():assert io.sha(B/n)==h
    io.write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,qualified=True,goal_complete=False,
        no_live_execution_mutex=True,files_sha256=manifest,pilot_files_verified=len(d['pilot_files_sha256']),CPU_close_seconds=time.monotonic()-started))
    io.write(B/'GOAL_TURN_R49_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        genuine_external_blocker=False,no_active_handle=True,new_outer_test_evaluation=False,new_source_heads=39,new_TD_heads=13,
        concrete_progress='Fixed half-weight target-component nuisance pilot and independent graph/projection/certificate audit qualified. Full four-setting comparison must retain same-treatment TD to separate generic loss benefit from source information.',
        closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write('\n\nR49靶区域组间偏移部分降权闭合并通过开发资格：15训练输入分组，39新来源/13新TD头与8恒等旧头，9196内部OOF；独立图/投影/全部证书/TD增广lstsq/部署/SHA通过。Core对R25 ΔPCC Hu+.004035/Mix+.019295/Tak+.011218，MSE下降+.6689%/+3.1962%/+1.6702%；performance/source_increment/numerical均true。TD同处理也改善，Core相对其Mix仍负，不把收益全归于来源监督。'
                f'CPU数值+审计{cost["CPU_controller_and_audit_seconds"]:.6f}s，合成{synthetic:.6f}s另列，新GPU/forward/PCA0。当前四设置仍R26+3.2771%，目标active；下一阶段固定原五alpha、同时保留TD同处理对照，不扩分组/长度/gamma。详见ROUND49_RESULTS.md。\n')
    print(dict(status='closed',qualified=True,goal_complete=False,closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report),CPU_seconds=cost['CPU_controller_and_audit_seconds']))


if __name__=='__main__':main()
