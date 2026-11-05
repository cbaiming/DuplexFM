"""Close the verified fixed loss pilot; preserve failed prelock test evidence."""
import csv,hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    start=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'closure_receipt.json').exists()
    lock=read(R/'lock.json');d=read(R/'diagnostic.json');v=read(R/'verification.json')
    assert v['passed'] and v['all_numerically_qualified'] and not d['qualified'] and not v['qualified']
    assert v['lock_sha256']==d['lock_sha256']==sha(R/'lock.json') and v['diagnostic_sha256']==sha(R/'diagnostic.json')
    for n,h in lock['files'].items():assert sha(R/n)==h,n
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    for n,h in d['pilot_files_sha256'].items():assert sha(R/'pilot'/n)==h,n
    assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    runs=list(R.glob('execution_*.json'));audits=list(R.glob('verification_execution_*.json'));assert len(runs)==len(audits)==1
    e,a=read(runs[0]),read(audits[0]);assert e['returncode']==a['returncode']==0
    current=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'));failed=list(ET.parse(R/'prelock_test_v1/pytest.xml').getroot().iter('testsuite'))
    assert sum(int(t.get('tests','0')) for t in current)==8 and sum(int(t.get('failures','0')) for t in current)==0
    assert sum(int(t.get('failures','0')) for t in failed)==1
    synthetic=sum(float(t.get('time','0')) for t in current+failed)
    cost=dict(CPU_numeric_seconds=e['CPU_wall_seconds'],CPU_independent_audit_outer_seconds=a['CPU_wall_seconds'],
        CPU_independent_audit_inner_seconds_included=v['CPU_wall_seconds'],
        CPU_numeric_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],CPU_synthetic_both_attempts_seconds=synthetic,
        GPU_hours=0,new_source_forward=0,new_PCA=0,source_training_and_original_cache_not_included=True,
        process_startup_imports_development_and_close_not_included=True,execution_sha256=sha(runs[0]),verification_execution_sha256=sha(audits[0]))
    write(R/'cost_receipt.json',cost)
    lines=['# R39：平方ε损失没有通过资格','',
        '结论：不采用，不开启新的外层评价，不改变ε、核、alpha或阈值。当前四设置主结果仍R26 Core相对强TD24的等权平均相对PCC增益+3.2771%，未达到5%。','',
        '本轮只将R33平方损失换为平方ε不敏感损失，ε=当前inner train标签总体SD的0.1；核、原训练尺度/PCA、alpha100、三个来源和划分完全一致。训练自由截距，单个联合核预测，没有TD预测残差叠加。ε不是测量噪声估计，不能据结果谈生物噪声或因果机制。','',
        '|训练池|来源|R33 PCC|新PCC|ΔPCC vs R33|MSE下降 vs R33|ΔPCC vs原R4|MSE下降 vs原R4|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in d['records']:
        lines.append(f"|{r['setting']}|{r['source']}|{r['R33_reference']['PCC']:.6f}|{r['metrics']['PCC']:.6f}|{r['delta_PCC_vs_R33']:+.6f}|{r['MSE_reduction_vs_R33_pct']:+.4f}%|{r['delta_PCC_vs_R4']:+.6f}|{r['MSE_reduction_vs_R4_pct']:+.4f}%|")
    lines+=['','主Core只在Mixset比同核平方损失的PCC更好；Takayuki MSE稍好但PCC下降，符合两个指标并不等价。Full和初始化也没有稳定受益，不能挑个别格或按数据集拼接不同来源。','',
        f"预登记主性能gate={d['performance_gate']}，损失因素配对gate={d['loss_factor_gate']}，同损失TD对照辅助gate={d['source_increment_gate']}，最终qualified={d['qualified']}。辅助门槛通过不覆盖主门槛失败。",'',
        '|训练池|Core相对同损失TD的ΔPCC|MSE下降|','|---|---:|---:|']
    for r in d['paired_kernel_TD']:lines.append(f"|{r['setting']}|{r['delta_PCC']:+.6f}|{r['MSE_reduction_pct']:+.4f}%|")
    lines+=['','范围：Huesken1564/Mixset319/Takayuki416，2299唯一训练池样本；60新头、9440近端更新、9196内部OOF行；60旧R33和60旧R4固定读出只读复用。训练参考行出现35268次，不是35268个独立样本。无新来源forward/PCA/GPU，未计算新的外层测试指标。','',
        '首次纯合成7/8通过，ε0系数误差暴露gap停止条件不足。原代码/测试/失败已保存，未读真实标签；追加更严格KKT条件后8/8通过，随后登记。这不是改真实结果或放宽容差。','',
        f"独立验收通过：60新原始/对偶全局证书、60旧R33和60旧R4正规方程、原PCA/身份、atan2核重建、独立截距根、部署、逐行覆盖、统计和SHA。最大相对gap{v['max_relative_primal_dual_gap']:.3g}，KKT差{v['max_loss_gradient_difference']:.3g}（各训练SD的固定容差均通过），零和误差{v['max_zero_sum_error']:.3g}，部署差{v['max_deployment_difference']:.3g}。",'',
        '|训练池|来源|内折|PCC|MSE|ε|零损失比例|更新数|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in d['records']:
        for b in r['inner_blocks']:
            lines.append(f"|{r['setting']}|{r['source']}|{b['inner']}|{b['metrics']['PCC']:.6f}|{b['metrics']['MSE']:.6f}|{b['epsilon']:.6f}|{100*b['inactive_fraction']:.2f}%|{b['iterations']}|")
    lines+=['',
        f"成本：控制器{e['CPU_wall_seconds']:.6f}秒、独立验收外层{a['CPU_wall_seconds']:.6f}秒（内层{v['CPU_wall_seconds']:.6f}秒已包含），合计{cost['CPU_numeric_and_audit_seconds']:.6f}CPU秒；两次合成{synthetic:.6f}秒另列，GPU0。未包含旧来源训练/缓存、进程启动导入、开发和关闭，不作为端到端时间。",'',
        '方法参考：[Smola与Schölkopf2004，支持向量回归教程](https://alex.smola.org/papers/2004/SmoSch04.pdf)第3.3节支持凸损失扩展；本轮平方ε实现有独立数值证书，文献不提供siRNA增益保证。','',
        '反复开发后的一次固定候选不是盲确认；单source42、source keep-overlap/context暴露、purged非gene隔离、两个Mixset设置依赖仍须披露。此结果不能给论文增益减弱作单因素因果解释；旧来源权重/划分/基线桥接仍欠缺。','',
        f"科学锁：{sha(R/'lock.json')}。诊断：{sha(R/'diagnostic.json')}。独立验收：{sha(R/'verification.json')}。"]
    report=B/'ROUND39_RESULTS.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    paths=[R/'lock.json',R/'diagnostic.json',R/'verification.json',R/'cost_receipt.json',R/'diagnostic_rows.csv',report,Path(__file__)]+[R/n for n in lock['files']]+runs+audits+list((R/'prelock_test_v1').iterdir())
    manifest={p.relative_to(B).as_posix():sha(p) for p in paths if p.is_file()}
    for n,h in manifest.items():assert sha(B/n)==h
    write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,goal_complete=False,qualified=False,
        no_live_execution_mutex=True,all_pilot_files_verified=len(d['pilot_files_sha256']),files_sha256=manifest,CPU_close_seconds=time.monotonic()-start))
    write(B/'GOAL_TURN_R39_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        concrete_progress='Fixed squared-epsilon loss pilot and independent primal-dual audit completed. Main and paired loss gates false; no new outer tests or parameter grid.',
        new_outer_test_evaluation=False,new_heads=60,new_proximal_updates=9440,no_active_handle=True,genuine_external_blocker=False,
        closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\nR39固定ReLU核平方ε损失闭合：60新头/9440近端更新/9196内部OOF，60旧R33/60旧R4参考只读复用；全60数值证书/身份/PCA/独立核/部署/SHA通过。Core对同核平方损失ΔPCC −.000237/+.000717/−.003611，MSE下降−.0312%/+.3284%/+.2179%；主performance=false、loss_factor=false、辅助source_increment=true，最终gate=false。不扩ε/核/alpha、不做新外层。登记前7/8合成失败已保留并收紧KKT后8/8通过。CPU数值+验收{cost["CPU_numeric_and_audit_seconds"]:.6f}s，两次合成{synthetic:.6f}s另计，GPU/source forward/PCA0。完整四设置主结果仍R26+3.2771%，goal active未达5%；上一轮和本轮progress，无活句柄/外部阻断，见ROUND39_RESULTS.md。\n')
    print(json.dumps(dict(status='closed',qualified=False,goal_complete=False,closure_sha256=sha(R/'closure_receipt.json'),
        report_sha256=sha(report),CPU_numeric_and_audit_seconds=cost['CPU_numeric_and_audit_seconds'])))


if __name__=='__main__':main()
