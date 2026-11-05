"""Close the fixed four-setting experiment only after independent acceptance."""
import sys,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import run
io=run.io;B=run.B


def main():
    began=time.monotonic();run.validate();v=io.read(R/'verification.json');root=R/'results';d=io.read(root/'complete.json')
    assert v['passed'] and not (R/'execution.mutex').exists()
    assert v['lock_sha256']==d['lock_sha256']==io.sha(R/'lock.json') and v['complete_sha256']==io.sha(root/'complete.json')
    io.check_files(root,d['files_sha256']);assert v['counts']==d['counts']
    executions=list(R.glob('execution_*.json'));audits=list(R.glob('verification_execution_*.json'))
    assert len(executions)==len(audits)==1;e,a=io.read(executions[0]),io.read(audits[0]);assert e['returncode']==a['returncode']==0
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    assert sum(int(s.get('tests',0)) for s in suites)==8 and not any(int(s.get(k,0)) for s in suites for k in ('failures','errors','skipped'))
    cost=dict(CPU_controller_seconds=e['CPU_wall_seconds'],CPU_independent_audit_seconds=a['CPU_wall_seconds'],
        CPU_controller_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],
        synthetic_seconds_separate=sum(float(s.get('time',0)) for s in suites),counts=d['counts'],
        new_source_forward=0,new_PCA=0,new_original_scalers=0,GPU_hours=0,not_end_to_end=True,
        historical_training_cache_startup_development_registration_close_excluded=True,
        execution_sha256=io.sha(executions[0]),verification_execution_sha256=io.sha(audits[0]))
    records=io.table(root/'summary.csv');goals=io.read(root/'goal_summary.json');pairs=io.table(root/'paired_R50.csv')
    primary=next(r for r in goals if r['primary']);gain=primary['mean_relative_PCC_gain_pct'];reaches=primary['reaches_development_5pct']
    report=B/'ROUND55_RESULTS.md';assert not report.exists() and not (R/'closure_receipt.json').exists()
    lines=['# R55：共同靶区域截距惩罚的四设置评价','',
        f"预定主方案Core相对原强TD24的四设置等权平均相对PCC增益 **{gain:+.6f}%**；开发5%门槛 **{reaches}**。全部模型先由训练内数据选择并固定，再汇总外层指标。目标尚不能标记完成：重复开发、单来源种子和确认条件仍需单独审计。",'',
        '本轮仅把R54固定rho=n/K共同组截距目标扩展到原五alpha/原16外折，使用原内层PCC选择。全局截距按profile权重估计；损失中心化迹归一n−1。冻结source42/原TD24及192D表示、原尺度、6×32核范数比例.1不变，预测时不读取组或加回组偏移。','',
        '|设置|条件|PCC|Spearman|MSE|MAE|相对原TD24 PCC|相对同处理TD PCC|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"|{r['setting']}|{r['source']}|{float(r['PCC']):.6f}|{float(r['Spearman']):.6f}|{float(r['MSE']):.6f}|{float(r['MAE']):.6f}|{float(r['PCC_relative_gain_pct']):+.4f}%|{float(r['PCC_relative_gain_vs_same_TD_pct']):+.4f}%|")
    lines+=['','|条件|四设置相对原TD24平均增益|相对同处理TD平均增益|原TD24负增益设置|','|---|---:|---:|---|']
    for r in goals:
        lines.append(f"|{r['condition']}|{r['mean_relative_PCC_gain_pct']:+.6f}%|{r['mean_relative_gain_vs_same_TD_pct']:+.6f}%|{', '.join(r['negative_settings']) or '无'}|")
    lines+=['','|设置|来源|R50 PCC|R55 PCC|ΔPCC|MSE下降|','|---|---|---:|---:|---:|---:|']
    for r in pairs:
        lines.append(f"|{r['setting']}|{r['source']}|{float(r['previous_PCC']):.6f}|{float(r['new_PCC']):.6f}|{float(r['delta_PCC']):+.6f}|{float(r['MSE_reduction_pct']):+.4f}%|")
    lines+=['',
        '四设置主分母仍为原TD24_Ridge_CV，未因同处理TD变化而更换。完整Full、初始化、同处理TD均列出，不挑最优来源或每个设置换模型；通用目标的收益不能全部称为miRNA监督收益。AP/F1仍按既定连续分数和0.5阈值，完整数值保存在summary.csv。',
        f"本轮计数 {d['counts']}。1600内条件头+64最终头，15192预测出现，每条件3798（3329唯一输入）。96训练分组原SHA复用，15规则来自R54、81规则新算；复用头不是重新训练。",
        f"独立验收：96训练图、1248来源完整联合凸证书、416含组截距TD增广lstsq及所有尺度、候选选择、覆盖、初始化/来源身份、原尺度部署和{len(d['files_sha256'])}产物SHA通过。最大正规方程误差{v['max_normal_error']:.3g}、相对对偶间隙{v['max_relative_dual_gap']:.3g}、TD系数差{v['max_independent_TD_coef_difference']:.3g}、部署差{v['max_deployment_difference']:.3g}。",
        f"成本：CPU控制器{e['CPU_wall_seconds']:.6f}s＋独立验收{a['CPU_wall_seconds']:.6f}s＝{cost['CPU_controller_and_audit_seconds']:.6f}s；8项合成{cost['synthetic_seconds_separate']:.6f}s另列。新GPU/来源forward/PCA/原scaler拟合0。历史训练与缓存、开发/启动/登记/收尾不含，不能称端到端时间。",'',
        '源keep-overlap和context暴露、单source42、反复开发，以及当前purged划分非gene/完整连通组隔离继续限制推断。两个Mix设置共享测试样本。训练输入关联组不是确认的gene或assay；隐式K−1组对比增加训练容量，部署217系数不代表全部容量相同。',
        'R51已确定Takayuki在既定关联定义下只有一个闭包，不能把逐条bootstrap当作新的跨独立组确认。若本轮达到数值门槛，仍需固定预测稳定性和来源初始化/独立数据确认，不直接宣称SOTA。若未达到，不改目标/分母/rho或扩大网格。',
        f"科学锁SHA {io.sha(R/'lock.json')}；结果complete SHA {io.sha(root/'complete.json')}；独立验收SHA {io.sha(R/'verification.json')}。"]
    io.write(R/'cost_receipt.json',cost)
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    files=[R/'lock.json',R/'plan.json',root/'complete.json',R/'verification.json',R/'cost_receipt.json',report]+[R/n for n in run.FILES]+executions+audits
    manifest={p.relative_to(B).as_posix():io.sha(p) for p in files};io.check_files(B,manifest)
    io.write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,files_sha256=manifest,
        development_threshold_met=reaches,goal_complete=False,primary_relative_PCC_gain_pct=gain,
        result_files_verified=len(d['files_sha256']),no_live_execution_mutex=True,CPU_close_seconds=time.monotonic()-began))
    io.write(B/'GOAL_TURN_R55_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        genuine_external_blocker=False,no_active_handle=True,development_threshold_met=reaches,
        primary_relative_PCC_gain_pct=gain,closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write(f"\n\nR55公共rho=n/K四设置评价闭合：{d['counts']}、15192行，96输入图/1248来源凸证书/416TD独立增广解/选择/部署/SHA全部通过。Core相对原强TD24平均PCC增益{gain:+.6f}%，开发门槛{reaches}；不按来源择优或修改分母。CPU控制器+验收{cost['CPU_controller_and_audit_seconds']:.6f}s，新GPU/forward/PCA0。确认条件仍未完成，goal active；上一轮和本轮progress，无活句柄或外部阻断。详见ROUND55_RESULTS.md。\n")
    print(dict(status='closed',primary_gain=gain,development_threshold_met=reaches,goal_complete=False,closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)),flush=True)


if __name__=='__main__':main()
