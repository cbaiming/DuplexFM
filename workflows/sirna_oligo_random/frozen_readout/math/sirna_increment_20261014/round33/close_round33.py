"""Close the fixed kernel pilot without refitting or opening outer metrics."""
import hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    started=time.monotonic();assert not (R/'closure_receipt.json').exists() and not (R/'execution.mutex').exists()
    d=read(R/'diagnostic.json');v=read(R/'verification.json');lock=read(R/'lock.json')
    assert v['passed'] and v['diagnostic_sha256']==sha(R/'diagnostic.json') and v['lock_sha256']==sha(R/'lock.json')
    assert not d['qualified'] and not d['performance_gate'] and d['source_increment_gate'] and v['all_numerically_qualified']
    for name,h in lock['files'].items():assert sha(R/name)==h,name
    for name,h in lock['dependencies'].items():assert sha(Path(name))==h,name
    for name,h in d['pilot_files_sha256'].items():assert sha(R/'pilot'/name)==h,name
    executions=sorted(R.glob('execution_*.json'));audits=sorted(R.glob('verification_execution_*.json'))
    assert len(executions)==len(audits)==1
    e,a=read(executions[0]),read(audits[0]);assert e['returncode']==a['returncode']==0
    tests=sum(float(s.get('time',0)) for s in ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    models=list((R/'pilot').rglob('model.npz'));assert len(models)==60
    cost=dict(CPU_controller_seconds=e['CPU_wall_seconds'],CPU_audit_outer_seconds=a['CPU_wall_seconds'],
        CPU_audit_inner_seconds_included=v['CPU_wall_seconds'],CPU_fit_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],
        synthetic_seconds_separate=tests,new_heads=60,new_linear_solves=60,reused_reference_heads=60,
        new_internal_OOF_rows=9196,independent_LU_solutions=60,positive_Cholesky_systems=60,
        stored_training_row_occurrences=d['stored_training_row_occurrences'],saved_model_bytes=sum(p.stat().st_size for p in models),
        largest_saved_model_bytes=max(p.stat().st_size for p in models),new_source_forward=0,new_PCA=0,GPU_hours=0,
        excluded='Earlier source training/export, R4 references, startup/imports, development and report/closure are separate.',
        execution_sha256={p.name:sha(p) for p in executions+audits})
    write(R/'cost_receipt.json',cost)
    lines=['# R33：固定一阶ReLU核轻量读出','',
        '2026-10-15。60个固定CPU核读出及独立验收已闭合。Core内部PCC三个池均上升，但只有Mixset达到预定改善幅度，Takayuki MSE略退步；performance=false、source_increment=true、最终qualified=false。不运行新的四设置测试，不扩大核深度/阶数/bias/正则网格。当前四设置主结果仍R26+3.2771%，5%目标未达到。','',
        '本轮采用单层一阶arc-cosine核，保留TD24线性部分，与非线性辅助核联合求解一个alpha100的核Ridge。输入为原TD24与原训练PCA16，辅助平均迹固定16；只有训练分区拟合中心与尺度。没有训练神经网络隐藏权重、生成随机特征或叠加其它工具预测。数学背景见[Cho与Saul，NeurIPS2009](https://proceedings.neurips.cc/paper_files/paper/2009/file/5751ec3e9a4feab575962e78e006250d-Paper.pdf)，不将其应用结果推断为本任务保证。','',
        '与R7同时改变了RBF核函数并去掉64基点Nyström近似，因此不能把差异单独归因核函数。它保留样本范数，不是R15的行归一化；也没有执行R32未通过的边际变换。','',
        '|训练内池|来源|原R4固定PCC|新PCC|绝对ΔPCC|相对R4的MSE下降|',
        '|---|---|---:|---:|---:|---:|']
    for r in d['records']:
        lines.append(f"|{r['setting']}|{r['source']}|{r['reference']['PCC']:.6f}|{r['metrics']['PCC']:.6f}|{r['delta_PCC_vs_R4']:+.6f}|{r['MSE_reduction_vs_R4_pct']:+.4f}%|")
    lines+=['','Core在Mixset的改善较明确，但Hu/Tak绝对PCC改善均低于本轮预设.005，且Tak的MSE下降为负。不能因PCC三个方向均正就改门槛。Full、初始化及TD-only全部保留，不能按辅助来源的更好结果替换Core主方法。','',
        '|训练内池|Core相对同类型TD-only核的绝对ΔPCC|MSE下降|',
        '|---|---:|---:|']
    for p in d['paired_kernel_TD']:
        lines.append(f"|{p['setting']}|{p['delta_PCC']:+.6f}|{p['MSE_reduction_pct']:+.4f}%|")
    lines+=['','相对同类型TD-only，Core在Hu/Tak较好，在Mixset仍较差；这个比较避免把一般核化收益全部归于来源监督。初始化在Hu/Mix也改善，TD-only在Hu的改善较大，进一步说明“读出改善”与“miRNA监督增量”必须区分。','',
        '范围是三个原outer fold0 fit池（1564/319/416条）的五折内部OOF，共2299条；不是完整四设置外层测试。60个新头/60个科学线性解/9196新OOF，60个原固定参考头按原SHA复用。35,268个训练参考行出现跨头重复，不是新增独立样本；无新PCA、来源forward或GPU。','',
        f"验收：九项合成测试通过；60个独立LU解与正Cholesky系统、旧60头正规方程、PCA/核中心/目标/截距/部署/全部行序和SHA通过。最大正规方程误差{v['max_normal_equation_error']:.3g}，独立dual差{v['max_dual_coefficient_difference']:.3g}，预测差{v['max_prediction_difference']:.3g}。数值全部通过，但数值正确不代表性能合格。",'',
        f"成本：CPU控制器{e['CPU_wall_seconds']:.6f}秒、独立验收外层{a['CPU_wall_seconds']:.6f}秒，合计{cost['CPU_fit_and_audit_seconds']:.6f}秒；合成{tests:.6f}秒另列。60个model.npz合计{cost['saved_model_bytes']:,}B，最大单头{cost['largest_saved_model_bytes']:,}B，包含训练参考向量/dual及内部验证预测。",'',
        '精确核需要保存训练向量、每次预测计算其核相似度，部署成本高于固定维线性头；没有保存训练Gram，也不能称只有41/217个部署参数。CPU时间不含旧来源训练/导出、启动导入、开发和收尾，不能称端到端时间。','',
        '源keep-overlap/context暴露、反复开发、单source42、purged sequence不是gene隔离和两个Mixset设置非独立等限制继续保留。未做新的外层测试或独立确认；source43范围问题未回复，没有导出。上一轮与本轮均为progress，无活执行句柄或外部阻断，goal active且未完成。','',
        f"科学锁：{sha(R/'lock.json')}。诊断：{sha(R/'diagnostic.json')}。独立验收：{sha(R/'verification.json')}。"]
    report=B/'ROUND33_RESULTS.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    paths=[R/'lock.json',R/'diagnostic.json',R/'diagnostic_rows.csv',R/'verification.json',R/'cost_receipt.json',report,Path(__file__)]+[R/n for n in lock['files']]+executions+audits
    manifest={p.relative_to(B).as_posix():sha(p) for p in paths}
    for name,h in manifest.items():assert sha(B/name)==h
    write(R/'closure_receipt.json',dict(status='closed',qualified=False,goal_complete=False,readback_passed=True,
        no_live_execution_mutex=True,files_sha256=manifest,all_pilot_files_verified=len(d['pilot_files_sha256']),CPU_close_seconds=time.monotonic()-started))
    write(B/'GOAL_TURN_R33_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        concrete_progress='60 exact ReLU kernel CPU heads and 60 original controls independently verified. Core PCC directions positive, but fixed performance gate failed; no new outer evaluation or expanded grid.',
        new_outer_test_evaluation=False,source43_scope_reply_received=False,no_active_handle=True,genuine_external_blocker=False,
        closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write('\n\nR33固定一阶ReLU核闭合：60新头/9196内部OOF、60原R4参考只读复用；独立核公式/LU/正系统/部署/SHA全部通过。Core相对固定R4 ΔPCC +.001363/+.012988/+.003966，MSE下降+.2271%/+2.1373%/−.0378%；performance=false、source_increment=true、最终gate=false。不改门槛或扩核/正则/源网格，不做新外层。四设置主结果仍R26+3.2771%，goal active未达5%；本轮progress，无活句柄/阻断。CPU拟合+验收10.797s，合成.376s另计，新GPU/forward/PCA0，详见ROUND33_RESULTS.md。\n')
    print(json.dumps(dict(status='closed',qualified=False,closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report),CPU_fit_and_audit_seconds=cost['CPU_fit_and_audit_seconds'])))


if __name__=='__main__':main()
