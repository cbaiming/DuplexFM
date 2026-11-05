"""Close fixed conditional spectral-refit pilot with all controls and negatives."""
import sys,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent
sys.path.insert(0,str(R))
from run import io,validate


def main():
    started=time.monotonic();report=B/'ROUND47_RESULTS.md'
    assert not (R/'execution.mutex').exists()
    for p in (R/'cost_receipt.json',R/'closure_receipt.json',report,B/'GOAL_TURN_R47_AUDIT.json'):assert not p.exists(),p
    lock=validate();d=io.read(R/'diagnostic.json');v=io.read(R/'verification.json')
    assert v['passed'] and d['all_numerically_qualified'] and not d['qualified'] and not v['qualified']
    assert not d['performance_gate']
    assert d['lock_sha256']==v['lock_sha256']==io.sha(R/'lock.json') and v['diagnostic_sha256']==io.sha(R/'diagnostic.json')
    assert d['diagnostic_rows_sha256']==io.sha(R/'diagnostic_rows.csv')
    for n,h in d['pilot_files_sha256'].items():assert io.sha(R/'pilot'/n)==h,n
    assert v['counts']==dict(new=45,old=45,TD=15,rows=6897,supervised_direction_decompositions=45,active_system_solves=2707)
    assert d['new_source_forward']==d['new_PCA']==d['new_original_scalers']==d['GPU_hours']==0
    runs=list(R.glob('execution_*.json'));audits=list(R.glob('verification_execution_*.json'))
    assert len(runs)==len(audits)==1
    e,a=io.read(runs[0]),io.read(audits[0]);assert e['returncode']==a['returncode']==0
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==10
    assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
    synthetic=sum(float(s.get('time',0)) for s in suites)
    cost=dict(CPU_controller_seconds=e['CPU_wall_seconds'],CPU_independent_audit_outer_seconds=a['CPU_wall_seconds'],
        CPU_independent_audit_inner_included=v['CPU_wall_seconds'],
        CPU_controller_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],CPU_synthetic_seconds=synthetic,
        new_heads=45,new_profile_factors=45,new_active_system_solves=2707,independent_BVLS_replays=45,
        GPU_hours=0,new_source_forward=0,new_PCA=0,new_original_scalers=0,
        source_training_old_cache_startup_registration_development_and_close_not_included=True,
        execution_sha256=io.sha(runs[0]),verification_execution_sha256=io.sha(audits[0]))
    lines=['# R47：固定方向重估强度未改善迁移，保留原谱收缩','',
        '45个新CPU轻量读出及独立增广BVLS验收完成。Core在Huesken/Mixset内层验证退步，Takayuki变化很小；预定performance gate=false、qualified=false，不推进四设置外层评价，不调rank/lambda/L2或非负约束，也不事后挑选Full。当前完整四设置主结果仍为R26相对强TD24的平均PCC增益+3.2771%，未达到5%。','',
        '本轮是固定方向后再估计强度：复用旧R25训练系数的SVD，按原s>1e-8保留方向；在同一训练集上联合重估TD24系数与非负奇异方向幅度，保留L2 alpha100、去掉核范数的线性收缩项。原192维来源、池化、标准化、数据和内折不变。方向只由该内折训练标签决定；预测部署仍编译为216D系数+截距。没有新模型预测作为输入，不是源模型或预测集成。','',
        '|开发训练池|来源|原R25 PCC|新PCC|ΔPCC|原MSE|新MSE|MSE下降（负数为退步）|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in d['records']:
        m,ref=r['metrics'],r['reference']
        lines.append(f"|{r['setting']}|{r['source']}|{ref['PCC']:.6f}|{m['PCC']:.6f}|{r['delta_PCC_vs_R25']:+.6f}|{ref['MSE']:.6f}|{m['MSE']:.6f}|{r['MSE_reduction_vs_R25_pct']:+.4f}%|")
    lines+=['',
        '这里是三个固定开发fit池（1564/319/416）的内部OOF，不是新外层benchmark成绩。Core、Full、初始化在Hu/Mix都退步，Tak变化方向均小幅正向，不支持将收益或失败单独归因于miRNA监督。当前证据更支持保留既有收缩，不能概括所有减轻收缩的方法都无效。','',
        '|开发训练池|Core相对固定alpha100 TD的ΔPCC|Core相对该TD的MSE下降|',
        '|---|---:|---:|']
    for p in d['paired_TD']:lines.append(f"|{p['setting']}|{p['delta_PCC']:+.6f}|{p['MSE_reduction_pct']:+.4f}%|")
    lines+=['',
        f"固定资格：performance={d['performance_gate']}，source_increment={d['source_increment_gate']}，数值={d['all_numerically_qualified']}，最终qualified={d['qualified']}。原Core至少2池ΔPCC>=.005且MSE不差、所有池ΔPCC>=−.005要求未满足；没有降低门槛。内部TD alpha100控制不是完整目标的TD24_Ridge_CV分母，后者保持不变。",'',
        '|开发训练池|来源|内折|验证数|保留方向数|非零幅度数|幅度变化范数|训练L2目标下降|原验证PCC|新验证PCC|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in d['records']:
        for b in r['inner_blocks']:
            lines.append(f"|{r['setting']}|{r['source']}|{b['inner']}|{b['n']}|{b['selected_rank']}|{b['active_amplitudes']}|{b['amplitude_change_norm']:.6f}|{b['objective_reduction']:.6f}|{b['reference']['PCC']:.6f}|{b['metrics']['PCC']:.6f}|")
    lines+=['',
        '训练L2目标下降是给定方向后重新求条件最优的预期结果，不能替代验证泛化。45旧头中41个矩阵rank6、4个rank5，核范数在这里主要提供谱收缩，不能说它普遍找到了极低秩结构。新阶段仅25+r个标量求解变量，但监督方向来自原216D训练，不能把这个数当作全部监督容量；部署仍为217个系数。','',
        '所有原训练/验证行序与SHA核验通过；每条件每池exact-once，共6897新内层OOF行。45新方向分解/幅度头、45次TD Schur因子分解、2707个活动集合线性子系统；45旧R25参考和15唯一旧TD参考只读复用。另45次独立BVLS重放属于验收成本，不冒充额外随机种子。','',
        f"10项纯合成首次通过；45个独立增广BVLS与条件KKT、45旧头完整凸证书、15TD正规方程、来源/初始化/原始部署/选择/覆盖/SHA全部通过。最大条件KKT误差{v['max_conditional_KKT_error']:.3g}，独立系数差{v['max_independent_BVLS_difference']:.3g}，原始部署差{v['max_deployment_difference']:.3g}。条件凸证书不证明整个两阶段估计器无偏或具有总体最优泛化。",'',
        f"成本：CPU控制器{e['CPU_wall_seconds']:.6f}秒＋独立验收{a['CPU_wall_seconds']:.6f}秒，合计{cost['CPU_controller_and_audit_seconds']:.6f}秒；合成{synthetic:.6f}秒另列。本轮无运行失败。新GPU/source forward/PCA/scaler拟合均0。旧来源训练/缓存、启动/登记/开发与收尾不包含，不能称端到端耗时。",'',
        '依据：[Mazumder、Hastie与Tibshirani，JMLR 2010，第7节公式25](https://www.jmlr.org/papers/volume11/mazumder10a/mazumder10a.pdf)提出固定奇异向量后重新估计强度；本轮移用该思想到矩阵系数监督回归，并保留L2。文献不提供本数据的效果保证。','',
        '来源keep-overlap/context暴露、同开发池反复迭代、单source42、非gene隔离及两个Mix设置非独立的限制保持。未重新开封外层测试、未达5%、目标保持active。','',
        f"科学锁：{io.sha(R/'lock.json')}。诊断：{io.sha(R/'diagnostic.json')}。独立验收：{io.sha(R/'verification.json')}。"]
    io.write(R/'cost_receipt.json',cost)
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    files=[R/'lock.json',R/'diagnostic.json',R/'diagnostic_rows.csv',R/'verification.json',R/'cost_receipt.json',report,Path(__file__)]+[R/n for n in lock['files']]+runs+audits
    manifest={p.relative_to(B).as_posix():io.sha(p) for p in files}
    for n,h in manifest.items():assert io.sha(B/n)==h
    io.write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,qualified=False,goal_complete=False,
        no_live_execution_mutex=True,files_sha256=manifest,pilot_files_verified=len(d['pilot_files_sha256']),CPU_close_seconds=time.monotonic()-started))
    io.write(B/'GOAL_TURN_R47_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        genuine_external_blocker=False,no_active_handle=True,new_outer_test_evaluation=False,new_efficiency_heads=45,
        concrete_progress='45 fixed-direction nonnegative spectral refits and independent BVLS/KKT audits completed. Core Hu/Mix regress, Tak tiny gain; gate false, no new outer evaluation or grid.',
        closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write('\n\nR47固定奇异方向强度重估闭合：45新头/45监督方向分解/2707活动子系统/6897内部OOF，45独立BVLS及全部新条件KKT/旧凸证书/15TD/部署/SHA通过。Core对原R25 ΔPCC Hu−.003424/Mix−.008758/Tak+.001163，MSE下降−.7483%/−1.5751%/+.1186%；performance=false、最终gate=false，不做新外层或扩大rank/lambda/alpha/非负约束网格。训练L2目标下降但验证无稳定改善，保留原收缩。'
                f'CPU控制器+验收{cost["CPU_controller_and_audit_seconds"]:.6f}s，合成{synthetic:.6f}s另列，新GPU/forward/PCA/scaler0。完整主结果仍R26+3.2771%，goal active未达5%；上一轮及本轮progress，无活句柄/阻断，见ROUND47_RESULTS.md。\n')
    print(dict(status='closed',qualified=False,goal_complete=False,closure_sha256=io.sha(R/'closure_receipt.json'),
        report_sha256=io.sha(report),CPU_controller_and_audit_seconds=cost['CPU_controller_and_audit_seconds']))


if __name__=='__main__':main()
