"""Close the complete four-setting partial-group result with all controls."""
import sys,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent
sys.path.insert(0,str(R))
from run import io,validate


def main():
    started=time.monotonic();report=B/'ROUND50_RESULTS.md'
    assert not (R/'execution.mutex').exists()
    for p in (R/'cost_receipt.json',R/'closure_receipt.json',report,B/'GOAL_TURN_R50_AUDIT.json'):assert not p.exists(),p
    lock=validate();c=io.read(R/'results/complete.json');v=io.read(R/'verification.json');assert v['passed']
    assert c['lock_sha256']==v['lock_sha256']==io.sha(R/'lock.json') and v['complete_sha256']==io.sha(R/'results/complete.json')
    io.check_files(R/'results',c['files_sha256'])
    expected=dict(inner_new=1428,inner_R49=60,inner_R26_identity=84,inner_TD_identity=28,final_new=57,final_R26_identity=5,final_TD_identity=2)
    assert c['counts']==v['counts']==expected and v['source_full_joint_certificates']==1248 and v['TD_independent_augmented_lstsq']==416
    assert v['independent_training_graphs']==96 and v['OOF_rows']==15192
    runs=list(R.glob('execution_*.json'));audits=list(R.glob('verification_execution_*.json'));assert len(runs)==len(audits)==1
    e,a=io.read(runs[0]),io.read(audits[0]);assert e['returncode']==a['returncode']==0
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==8
    assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'));synthetic=sum(float(s.get('time',0)) for s in suites)
    summary=io.table(R/'results/summary.csv');goals=io.read(R/'results/goal_summary.json');pairs=io.table(R/'results/paired_R26.csv');sels=io.table(R/'results/selection.csv')
    primary=next(g for g in goals if g['primary']);assert primary['condition']=='core_real_partial_group' and not primary['reaches_development_5pct']
    previous=next(r for r in io.read(B/'round26/results/goal_summary.json') if r['primary'])
    delta=primary['mean_relative_PCC_gain_pct']-previous['mean_relative_PCC_gain_pct']
    cost=dict(CPU_controller_seconds=e['CPU_wall_seconds'],CPU_independent_audit_outer_seconds=a['CPU_wall_seconds'],
        CPU_independent_audit_inner_included=v['CPU_wall_seconds'],CPU_controller_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],
        CPU_synthetic_seconds=synthetic,new_scientific_heads=1485,reused_condition_heads=179,counts=expected,
        new_training_partitions=81,reused_training_partitions=15,independent_training_graphs=96,independent_source_certificates=1248,independent_TD_lstsq=416,
        GPU_hours=0,new_source_forward=0,new_PCA=0,new_original_scalers=0,
        prior_R49_training_old_caches_startup_registration_development_and_close_not_included=True,
        execution_sha256=io.sha(runs[0]),verification_execution_sha256=io.sha(audits[0]))
    lines=['# R50：靶区域组间偏移部分降权，四设置增益达到4.19%','',
        f"预先固定的Core主方案相对原强TD24_Ridge_CV的四设置PCC相对增益等权均值为**{primary['mean_relative_PCC_gain_pct']:+.4f}%**，此前R26为{previous['mean_relative_PCC_gain_pct']:+.4f}%，增加{delta:.4f}个百分点。Mixset转为正增益，Huesken→Mixset仍略为负；**未达到5%，目标未完成**。这是完整开发评价，尚无新盲外部或来源种子确认，不能称SOTA。",'',
        '本轮沿用R49已通过的唯一训练因素：只用当前训练靶序列共享canonical25-mer形成输入组，将组间残差权重固定降至.5、组内保留，并归一中心化空间总权重。原source192+TD24、6×32系数核范数、原五alpha和内OOF PCC选择不变。预测仍为单线性向量+全局截距，不使用测试组、不恢复组偏移、不融合其它模型预测。','',
        '|设置|原强TD24 PCC|Core PCC|相对PCC增益|相对强TD MSE下降|同处理TD PCC|Core相对同处理TD PCC增益|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['source']=='core_real':
            lines.append(f"|{r['setting']}|{float(r['TD24_PCC']):.6f}|{float(r['PCC']):.6f}|{float(r['PCC_relative_gain_pct']):+.4f}%|{float(r['MSE_relative_reduction_pct']):+.4f}%|{float(r['same_treatment_TD_PCC']):.6f}|{float(r['PCC_relative_gain_vs_same_TD_pct']):+.4f}%|")
    lines+=['',
        f"Core相对同处理、同选参的TD-only平均PCC增益为{primary['mean_relative_gain_vs_same_TD_pct']:+.4f}%。同处理TD相对原TD平均−0.1176%，因此本轮总体增益不能仅由TD训练目标的通用改进解释。但Mixset初始化也从0.520388提高至0.528216，接近训练Core的0.529757；不能把Mixset改善全部归于miRNA监督。",'',
        '|来源/控制|四设置相对原TD平均PCC增益|相对同处理TD平均PCC增益|原TD负增益设置|',
        '|---|---:|---:|---|']
    for g in goals:
        lines.append(f"|{g['condition']}|{g['mean_relative_PCC_gain_pct']:+.4f}%|{g['mean_relative_gain_vs_same_TD_pct']:+.4f}%|{', '.join(g['negative_settings']) or '/'}|")
    lines+=['',
        'Full平均2.3330%，没有跟随Core一起稳定提高：Takayuki Full相对旧R26 PCC下降0.012463。精确初始化平均接近零；同处理下训练Core四设置PCC均高于其匹配初始化，但当前只是一份来源seed42，不能据此证明初始化稳定性或普遍迁移。没有按设置切换Core/Full、选择最好来源或平均种子预测。','',
        '|设置|条件|PCC|Spearman|MSE|MAE|AP|F1|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append('|'+r['setting']+'|'+r['source']+'|'+ '|'.join(f"{float(r[k]):.6f}" for k in ('PCC','Spearman','MSE','MAE','AP','F1'))+'|')
    lines+=['',
        'AP使用连续预测分数及原二值标签契约，F1沿用0.5阈值；未按测试重选阈值。完整AUROC及所有指标见summary.csv。两个Mix设置共享测试队列，不能当作两个独立外部验证。','',
        '|设置|来源|原R26 PCC|新PCC|绝对ΔPCC|相对R26 MSE下降|',
        '|---|---|---:|---:|---:|---:|']
    for r in pairs:
        lines.append(f"|{r['setting']}|{r['source']}|{float(r['previous_PCC']):.6f}|{float(r['new_PCC']):.6f}|{float(r['delta_PCC']):+.6f}|{float(r['MSE_reduction_pct']):+.4f}%|")
    lines+=['',
        'Core相对原R26四设置PCC/MSE均改善，但幅度不均；最大变化在Mixset，Tak几乎不变。此处变化为已反复开发的样本上的描述性差值，未提供新显著性结论，不据此扩gamma/连接长度或拼接方法。组间降权也可能丢弃真实生物差异；输入连通组不是已确认的基因、实验批次或外生因果anchor。','',
        '|设置|外折|条件|训练内选择alpha|所选内OOF PCC|',
        '|---|---:|---|---:|---:|']
    for r in sels:lines.append(f"|{r['setting']}|{r['fold']}|{r['source']}|{r['alpha']}|{float(r['inner_PCC']):.6f}|")
    lines+=['',
        '选参只来自原outer fit内部五个purged折：PCC最大，平局取MSE更低、再取更强正则。所有最终模型和预测有models_fixed.json哈希回执，早于数值测试汇总。源模型、原划分/标签预算/标准化、响应、X/N排除等契约保留；outer validation未使用。','',
        '|设置|外折|最终fit样本数|最终组数|最大组比例|隐式组对比数|',
        '|---|---:|---:|---:|---:|---:|']
    for p in v['partition_metadata']:
        if p['stage']=='final':lines.append(f"|{p['setting']}|{p['fold']}|{p['n']}|{p['K']}|{p['largest_fraction']:.6f}|{p['implicit_group_contrasts']}|")
    lines+=['',
        '96份训练输入分组由四条件共享：81份新增、15份R49复用；训练增加K−1隐式组对比，部署217系数（TD-only25）不代表相同的总监督容量。Tak部分训练组为单组、目标恒等，直接复用对应旧头，避免二次计算。','',
        '计算计数：1600内条件头+64最终头。新增1428内头+57最终头＝1485；复用R49内头60、恒等R26内头84、恒等TD内头28、恒等R26最终头5、恒等TD最终头2，共179复用。15192预测出现=四条件各3798；独立输入3329，两个Mix设置不独立。旧R26控制只按SHA使用，未重训。','',
        f"验收：8项新合成首次通过，全部1248来源完整联合凸证书、416TD独立增广lstsq、96独立输入图/投影、原尺度/初始化/选择/部署/覆盖/SHA通过。最大正规方程误差{v['max_normal_error']:.3g}、相对对偶间隙{v['max_relative_dual_gap']:.3g}、独立TD系数差{v['max_independent_TD_coef_difference']:.3g}、部署预测差{v['max_deployment_difference']:.3g}。没有放宽solver迭代/数值容差或静默修改锁。",'',
        f"成本：CPU控制器{e['CPU_wall_seconds']:.6f}秒＋独立验收{a['CPU_wall_seconds']:.6f}秒＝{cost['CPU_controller_and_audit_seconds']:.6f}秒；合成{synthetic:.6f}秒另列。本轮无运行失败；新GPU/来源forward/PCA/原scaler拟合均0。R49、历史源训练/缓存、启动/登记/开发/收尾不含，不能称端到端速度。",'',
        '主线意义：训练中降低输入组共同偏移影响，使冻结Core表示在同一轻量读出里更有用，且推理仍简单。但改善不是普遍适用于Full或TD，跨队列负增益未消失。保留R49/本轮完整正负结果，不用增加架构复杂度代替来源增量证据。','',
        'source keep-overlap/context暴露、单来源seed42、反复开发、purged非gene隔离仍限制结论。数学动机见R49协议及其[JRSSB原始文献](https://academic.oup.com/jrsssb/article/83/2/215/7056043)，本数据不满足据此宣称因果的条件。目标尚差约0.81个相对增益百分点且仍需稳定性/确认，不标完成，不改门槛或基线。','',
        f"科学锁：{io.sha(R/'lock.json')}。完成清单：{io.sha(R/'results/complete.json')}。独立验收：{io.sha(R/'verification.json')}。"]
    io.write(R/'cost_receipt.json',cost)
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    files=[R/'lock.json',R/'results/complete.json',R/'verification.json',R/'cost_receipt.json',report,Path(__file__)]+[R/n for n in lock['files']]+runs+audits
    manifest={p.relative_to(B).as_posix():io.sha(p) for p in files}
    for n,h in manifest.items():assert io.sha(B/n)==h
    io.write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,goal_complete=False,
        current_primary_relative_PCC_gain_pct=primary['mean_relative_PCC_gain_pct'],no_live_execution_mutex=True,
        files_sha256=manifest,result_files_verified=len(c['files_sha256']),CPU_close_seconds=time.monotonic()-started))
    io.write(B/'GOAL_TURN_R50_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        genuine_external_blocker=False,no_active_handle=True,new_outer_test_evaluation=True,new_scientific_heads=1485,
        concrete_progress='Qualified R49 factor evaluated once across all four locked settings with original alpha selection and matched TD control. Core mean relative PCC gain 4.190823%, all four improve over R26 but cross-cohort remains negative; no 5% or confirmation claim.',
        closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write('\n\nR50四设置评价闭合：R49固定组间.5降权+原五alpha/同内层PCC选择，1485新头+179复用、15192行；1248来源凸证书/416TD增广lstsq/96训练输入图及全部选择/部署/SHA通过。预定Core相对原强TD24各设置+3.906680%/+1.324744%/+11.782799%/−.250931%，均值+4.190823%，相对R26增加.913677个百分点，仍未达5%。同处理TD均值−.117609%；Core相对同处理TD均值+4.311925%。Full+2.332959%、initial−.023496%，不按设置换来源或声称普遍改善。'
                f'CPU数值+审计{cost["CPU_controller_and_audit_seconds"]:.6f}s，合成{synthetic:.6f}s另列，新GPU/forward/PCA0。R49及R50均完整保存，下一轮不能重跑/扩大gamma或25-mer网格；本轮/上一轮progress，无活句柄/外部阻断，goal active；见ROUND50_RESULTS.md。\n')
    print(dict(status='closed',goal_complete=False,mean_relative_PCC_gain_pct=primary['mean_relative_PCC_gain_pct'],
        closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report),CPU_seconds=cost['CPU_controller_and_audit_seconds']))


if __name__=='__main__':main()
