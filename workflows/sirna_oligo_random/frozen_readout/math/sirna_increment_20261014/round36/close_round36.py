"""Archive all fixed-dropout results without opening a new outer evaluation."""
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
    assert v['qualified']==d['qualified'] and v['counts']==dict(new=45,old=45,TD=15,rows=6897)
    assert v['lock_sha256']==sha(R/'lock.json') and v['diagnostic_sha256']==sha(R/'diagnostic.json')
    for n,h in lock['files'].items():assert sha(R/n)==h,n
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    for n,h in d['pilot_files_sha256'].items():assert sha(R/'pilot'/n)==h,n
    assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    runs=list(R.glob('execution_*.json'));audits=list(R.glob('verification_execution_*.json'));assert len(runs)==len(audits)==1
    e,a=read(runs[0]),read(audits[0]);assert e['returncode']==a['returncode']==0
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==8
    assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
    synthetic=sum(float(x.get('time',0)) for x in suites)
    cost=dict(CPU_controller_seconds=e['CPU_wall_seconds'],CPU_audit_outer_seconds=a['CPU_wall_seconds'],CPU_audit_inner_included=v['CPU_wall_seconds'],
        CPU_fit_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],synthetic_seconds_separate=synthetic,
        new_heads=45,reused_native_heads=45,reused_unique_TD_heads=15,new_internal_OOF_rows=6897,
        exact_mask_loss_evaluations=2880,mask_evaluations_are_not_fits=True,new_PCA=0,new_source_forward=0,GPU_hours=0,
        excluded='Previous source training/export and reference fits, process startup/imports, development and report/closure.',
        execution_sha256={p.name:sha(p) for p in runs+audits})
    write(R/'cost_receipt.json',cost)
    lines=['# R36：来源池化块的精确dropout期望正则','',
        '2026-10-15。45个新CPU轻量读出及独立验收已经完成，数值全部合格，但预定性能资格失败，不采用、不进入新的四设置外层评价。当前完整主结果仍为R26平均相对PCC +3.2771%，尚未达到5%目标。','',
        '依据R35的子集敏感性诊断，仅新增一个固定因素：source192六个32D池化块独立保留概率0.9，训练目标精确积分所有mask。TD24与截距不屏蔽。六块分别对应guide三个区域的mean/max池化，不能称为BIO、Inta、开放性等expert的自动屏蔽。','',
        '平方损失增加项为((1-p)/p)×各块训练输出平方和。核范数比例0.1、alpha100、旧scaler及source除√12均保持。部署无dropout，仍为216个系数加截距的一次线性预测。训练块缺失与样本子集扰动并不等价，本轮只是一个稳定化干预。','',
        '|内部池|来源|参考PCC|本轮PCC|绝对ΔPCC|相对参考MSE下降|',
        '|---|---|---:|---:|---:|---:|']
    for r in d['records']:
        lines.append(f"|{r['setting']}|{r['source']}|{r['reference']['PCC']:.6f}|{r['metrics']['PCC']:.6f}|{r['delta_PCC_vs_R25']:+.6f}|{r['MSE_reduction_vs_R25_pct']:+.4f}%|")
    lines+=['',
        '主Core在Huesken/Mixset/Takayuki相对R25的绝对ΔPCC分别为−0.000013、+0.000929、−0.000402；只有Mixset略好，没有两个池达到ΔPCC≥0.005且MSE不恶化的预定门槛。主性能gate=false；不调整保留概率、惩罚比例、alpha或池化分组寻找正结果。Full和初始化全部列出，不改选辅助来源。','',
        '|内部池|Core相对同alpha TD24的绝对ΔPCC|相对TD24 MSE下降|',
        '|---|---:|---:|']
    for r in d['paired_TD']:
        lines.append(f"|{r['setting']}|{r['delta_PCC']:+.6f}|{r['MSE_reduction_pct']:+.4f}%|")
    lines+=['',
        '相对TD24的辅助增量gate=true，但它不能替代主方法改进gate；Mixset仍低于TD24。这个负结果只否定当前固定强度方案的继续资格，不能证明所有dropout或所有expert门控无效。','',
        '范围：原Huesken/Mixset/Takayuki outer fold0 fit池1564/319/416，三个来源各五内部折。45新头、6897新OOF行；45原R25头和15唯一TD24头按原SHA只读复用，无重新来源前向、PCA、GPU或新的外层测试。','',
        f"验收：8项合成通过；45新与45旧完整联合凸证书、15TD正规方程、数据身份/源/部署/OOF覆盖/SHA全部通过。每个新头枚举64个mask，总2880次期望损失计算，不是新增2880次拟合。最大期望SSE误差{v['max_expected_SSE_identity_error']:.3g}，最大相对对偶gap{v['max_relative_dual_gap']:.3g}，单向量部署差{v['max_deployment_difference']:.3g}。",'',
        f"成本：控制器{e['CPU_wall_seconds']:.6f}秒，独立验收外层{a['CPU_wall_seconds']:.6f}秒（包含内层），共{cost['CPU_fit_and_audit_seconds']:.6f}秒；合成{synthetic:.6f}秒另列。新增GPU0。旧来源训练、缓存、参考拟合以及进程启动/导入、开发和收尾未包含，不能称这些秒数为端到端成本。",'',
        '仍为单source42、重复开发池、源keep-overlap/context暴露，当前purged sequence非gene隔离；两个Mixset设置并非独立队列。本轮不提供新的盲确认。论文旧模型、划分与强TD基线的桥接尚未完成，不能依据本轮将历史增益下降归因于单一因素。','',
        f"科学锁：{sha(R/'lock.json')}。诊断：{sha(R/'diagnostic.json')}。独立验收：{sha(R/'verification.json')}。"]
    report=B/'ROUND36_RESULTS.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    paths=[R/'lock.json',R/'diagnostic.json',R/'verification.json',R/'cost_receipt.json',R/'diagnostic_rows.csv',report,Path(__file__)]+[R/n for n in lock['files']]+runs+audits
    manifest={p.relative_to(B).as_posix():sha(p) for p in paths}
    for n,h in manifest.items():assert sha(B/n)==h
    write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,goal_complete=False,new_model_selected=False,
        qualified=False,no_live_execution_mutex=True,all_pilot_files_verified=len(d['pilot_files_sha256']),files_sha256=manifest,CPU_close_seconds=time.monotonic()-start))
    write(B/'GOAL_TURN_R36_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        concrete_progress='45 fixed CPU source-pool expected-dropout heads and60 original controls independently verified; main gate failed, no new outer evaluation or grid expansion.',
        user_question='Reconciled paper and current gains using documented baseline, split, source and sample changes; no unsupported single-factor attribution.',
        new_outer_test_evaluation=False,new_model_selected=False,no_active_handle=True,genuine_external_blocker=False,
        closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\nR36六来源池化块精确dropout期望正则闭合：45新头/6897内部OOF、45原生与15唯一TD参考只读核验，45×64 mask期望恒等式和全部凸证书/部署/SHA通过。Core相对R25固定头ΔPCC −.000013/+.000929/−.000402，MSE下降+.0208%/+.1698%/−.0257%；performance=false、辅助source_increment=true、最终gate=false。不扩p/正则/分组，不做新外层。CPU拟合+验收{cost["CPU_fit_and_audit_seconds"]:.6f}s，合成{synthetic:.6f}s另计，新GPU/forward/PCA0。四设置主结果仍R26+3.2771%，goal active未达5%；本轮progress，无活句柄/外部阻断，见ROUND36_RESULTS.md。\n')
    print(json.dumps(dict(status='closed',qualified=False,goal_complete=False,closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report),CPU_fit_and_audit_seconds=cost['CPU_fit_and_audit_seconds'])))


if __name__=='__main__':main()
