"""Close a descriptive training-size study without choosing a new method."""
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
    assert v['passed'] and v['all_numerically_qualified'] and not v['model_selection'] and not v['outer_test_metrics_used']
    assert v['lock_sha256']==sha(R/'lock.json') and v['diagnostic_sha256']==sha(R/'diagnostic.json')
    for n,h in lock['files'].items():assert sha(R/n)==h,n
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    for n,h in d['pilot_files_sha256'].items():assert sha(R/'pilot'/n)==h,n
    for n,h in d['output_sha256'].items():assert sha(R/n)==h,n
    runs=list(R.glob('execution_*.json'));audits=list(R.glob('verification_execution_*.json'));assert len(runs)==len(audits)==1
    e,a=read(runs[0]),read(audits[0]);assert e['returncode']==a['returncode']==0
    synthetic=sum(float(x.get('time',0)) for x in ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    cost=dict(CPU_controller_seconds=e['CPU_wall_seconds'],CPU_audit_outer_seconds=a['CPU_wall_seconds'],CPU_audit_inner_included=v['CPU_wall_seconds'],
        CPU_fit_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],synthetic_seconds_separate=synthetic,
        new_heads=360,new_native_heads=270,new_TD_heads=90,reused_reference_heads=60,new_scalers=270,new_PCA=0,
        new_internal_OOF_rows=55176,reused_internal_OOF_rows=9196,new_source_forward=0,GPU_hours=0,
        excluded='Previous source training/export and reference fits, process startup/imports, development, plotting and report/closure.',
        execution_sha256={p.name:sha(p) for p in runs+audits})
    write(R/'cost_receipt.json',cost)
    agg={(r['setting'],r['condition'],r['fraction']):r for r in d['aggregate']};findings=[]
    for setting in ('Huesken','Mixset','Takayuki'):
        core=[agg[setting,'core_real',f] for f in (.5,.75,1.)]
        row=dict(setting=setting,Core_PCC_increment_increases_over_observed_sizes=all(core[i+1]['PCC_delta_mean']>=core[i]['PCC_delta_mean'] for i in range(2)),
            Core_absolute_PCC_monotone=all(core[i+1]['PCC_mean']>=core[i]['PCC_mean'] for i in range(2)),
            Core_minus_TD_PCC_at_full=core[-1]['PCC_delta_mean'],Core_minus_TD_MSE_at_full=core[-1]['MSE_mean']-agg[setting,'TD_only',1.]['MSE_mean'],
            perturbation_variance_ratios=[agg[setting,'core_real',f]['subset_perturbation_variance']/agg[setting,'TD_only',f]['subset_perturbation_variance'] for f in (.5,.75)])
        findings.append(row)
    write(R/'descriptive_findings.json',dict(findings=findings,model_selection=False,no_extrapolation=True,goal_complete=False,
        interpretation='Finite subset sensitivity supports estimation variance as a partial issue. It does not isolate irreducible noise, causality, or guarantee that more data eliminates negative transfer.'))
    lines=['# R35：样本量与来源读出波动诊断','',
        '2026-10-15，360新CPU轻量头、60复用完整样本头及独立验收均完成。没有选择新方法、重训来源或评价外层测试。当前完整四设置主结果仍R26平均相对PCC +3.2771%，5%目标未达到。','',
        '本轮沿用当前native216D单模型读出及6×32系数核范数收缩，固定100%时alpha100与惩罚比例.1。50%/75%仅改变inner train的样本子集，三个hash抽样重复共享ID，按n_full/n_sub统一加权保持平均损失意义下的正则强度。验证池始终相同，不据其选参或采样。','',
        '![训练样本量诊断](round35/learning_curve.png)','',
        '图中误差条仅为3个子集抽样的配对PCC差异SD；不是来源训练种子的SD或置信区间。100%是旧单个完整结果，没有伪造重复估计。','',
        '|内部验证池|50% Core相对TD的绝对ΔPCC|75%|100%|100% Core相对TD的MSE下降|',
        '|---|---:|---:|---:|---:|']
    for setting in ('Huesken','Mixset','Takayuki'):
        cc=[agg[setting,'core_real',f] for f in (.5,.75,1.)]
        lines.append(f"|{setting}|{cc[0]['PCC_delta_mean']:+.6f}|{cc[1]['PCC_delta_mean']:+.6f}|{cc[2]['PCC_delta_mean']:+.6f}|{cc[2]['MSE_reduction_vs_TD_pct_mean']:+.4f}%|")
    lines+=['',
        '三个池的相对TD24增量在这些观测点均随样本增加而提高，说明样本量/估计稳定性值得关注。但Mixset完整训练池仍明显弱于TD24，不能推断多收一点数据一定转正。Takayuki绝对PCC并不单调：Core从50%的0.406848到75%的0.391106再到100%的0.402167；相对收益增加不等于绝对精度单调提高。','',
        '|内部池|50% Core/TD子集扰动方差比|75%方差比|',
        '|---|---:|---:|']
    for r in findings:lines.append(f"|{r['setting']}|{r['perturbation_variance_ratios'][0]:.3f}|{r['perturbation_variance_ratios'][1]:.3f}|")
    lines+=['',
        'Core对训练子集变化的预测方差均高于TD24；这提示来源分量带来额外估计不稳定性。该量条件于现有池和3个固定子集，不能当作总体噪声上限、不可约误差或因果解释。没有把三个预测平均后作为新模型评分/部署，也没有利用样本量曲线外推达到5%需要多少新样本。','',
        '下一步可考虑一个与当前结果一致的单因素候选：仅在训练目标中惩罚来源池化块对抽样/缺失的敏感性，部署保持单线性头。它尚未登记或运行；不能把这句话当成新的成功实验。原五expert与六个区域mean/max池化块不是同一概念，不将块级正则称为自动关闭BIO/Inta等expert。','',
        '|内部池|条件|训练比例|PCC均值|PCC SD|MSE均值|相对同预算TD的MSE下降|',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in d['aggregate']:
        sd=f"{r['PCC_SD']:.6f}" if r['PCC_SD'] is not None else '未估计'
        lines.append(f"|{r['setting']}|{r['condition']}|{100*r['fraction']:.0f}%|{r['PCC_mean']:.6f}|{sd}|{r['MSE_mean']:.6f}|{r['MSE_reduction_vs_TD_pct_mean']:+.4f}%|")
    lines+=['',
        'Full、精确初始化及TD控制的正负结果全部列出；单draw详细记录见diagnostic.json/diagnostic_rows.csv。原来源模型仍只有seed42，42/43/44只是训练子集身份抽样，未用它们挑最佳结果。原100%参考为固定alpha100内部OOF，不是当前四设置选alpha后的主成绩。','',
        '范围：Huesken/Mixset/Takayuki原outer fold0 fit池1564/319/416，共2299 unique行。新增270核范数头+90TD头、270训练scaler、55176内部OOF；复用45核范数+15TD完整头及9196旧行。没有PCA或来源前向。验证标签仅用于诊断，但已经读取，因此本轮不能冒充总标签预算50%/75%的低标签benchmark。','',
        f"验收：7项合成通过；270新完整联合凸证书、90独立TD增广lstsq、45旧凸证书/15旧正规方程，全部子集/嵌套/尺度/行序/指标/方差/SHA通过。最大相对dual gap {v['max_relative_dual_gap']:.3g}，预测重建差{v['max_prediction_difference']:.3g}，单向量部署差{v['max_deployment_difference']:.3g}。",'',
        f"成本：控制器{e['CPU_wall_seconds']:.6f}秒，独立验收外层{a['CPU_wall_seconds']:.6f}秒（已含内层），共{cost['CPU_fit_and_audit_seconds']:.6f}秒；合成{synthetic:.6f}秒另列，新增GPU0。旧来源训练/缓存、启动导入、开发绘图和收尾另计，不能称端到端仅需这些秒数。",'',
        '学习曲线解释参考[Viering与Loog，TPAMI2023](https://doi.org/10.1109/TPAMI.2022.3220744)；有限观测点不支持固定外推曲线或保证数据越多越好。来源keep-overlap/context暴露、多个反复开发池、当前purged非gene隔离以及两个Mixset设置不独立的限制仍在。','',
        f"科学锁：{sha(R/'lock.json')}。诊断：{sha(R/'diagnostic.json')}。独立验收：{sha(R/'verification.json')}。"]
    report=B/'ROUND35_RESULTS.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    paths=[R/'lock.json',R/'diagnostic.json',R/'verification.json',R/'cost_receipt.json',R/'descriptive_findings.json',report,Path(__file__),R/'plot_curve.py',R/'learning_curve.png',R/'learning_curve.svg']+[R/n for n in lock['files']]+[R/n for n in d['output_sha256']]+runs+audits
    manifest={p.relative_to(B).as_posix():sha(p) for p in paths}
    for n,h in manifest.items():assert sha(B/n)==h
    write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,goal_complete=False,model_selection=False,
        no_live_execution_mutex=True,all_pilot_files_verified=len(d['pilot_files_sha256']),files_sha256=manifest,CPU_close_seconds=time.monotonic()-start))
    write(B/'GOAL_TURN_R35_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        concrete_progress='360 fixed CPU subset heads and60 original controls verified. Source increment increases with observed sample size, but Mixset remains negative; extra source subset variance quantified without model selection or extrapolation.',
        new_outer_test_evaluation=False,new_model_selected=False,no_active_handle=True,genuine_external_blocker=False,
        closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\nR35固定样本量诊断闭合：50/75%×3hash抽样，360新头/55176内部OOF、100%60旧头/9196旧行，所有身份/尺度/凸证书/TD独立解/部署/方差/SHA通过。Core相对TD的ΔPCC从50%到100%为Hu .006547→.015211、Mix −.031784→−.019671、Tak .024445→.033227；各内部池相对增量改善但Mix仍负，Tak绝对PCC不单调。Core子集扰动方差均高于TD；不是不可约噪声或来源种子不确定性，不作样本量外推/新模型选择，不开新外层。CPU拟合+验收{cost["CPU_fit_and_audit_seconds"]:.6f}s，合成{synthetic:.6f}s另计，GPU/source forward/PCA0。四设置主结果仍R26+3.2771%，goal active未达5%；前轮和本轮progress，无活句柄/外部阻断，见ROUND35_RESULTS.md。\n')
    print(json.dumps(dict(status='closed',goal_complete=False,closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report),CPU_fit_and_audit_seconds=cost['CPU_fit_and_audit_seconds'])))


if __name__=='__main__':main()
