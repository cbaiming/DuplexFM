"""Close the native-kernel comparison without promoting failed qualification."""
import hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
def write(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    started=time.monotonic();assert not (R/'execution.mutex').exists() and not (R/'closure_receipt.json').exists()
    lock=read(R/'lock.json');d=read(R/'diagnostic.json');v=read(R/'verification.json')
    assert v['passed'] and v['all_numerically_qualified'] and not d['qualified'] and not v['qualified']
    assert d['lock_sha256']==v['lock_sha256']==sha(R/'lock.json') and v['diagnostic_sha256']==sha(R/'diagnostic.json')
    for n,h in lock['files'].items():assert sha(R/n)==h,n
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    for n,h in d['pilot_files_sha256'].items():assert sha(R/'pilot'/n)==h,n
    assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    runs=list(R.glob('execution_*.json'));audits=list(R.glob('verification_execution_*.json'));assert len(runs)==len(audits)==1
    e,a=read(runs[0]),read(audits[0]);assert e['returncode']==a['returncode']==0
    tests=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'));assert sum(int(t.get('tests',0)) for t in tests)==8
    assert not any(int(t.get(k,0)) for t in tests for k in ('errors','failures','skipped'));synthetic=sum(float(t.get('time',0)) for t in tests)
    cost=dict(CPU_numeric_seconds=e['CPU_wall_seconds'],CPU_independent_audit_outer_seconds=a['CPU_wall_seconds'],
        CPU_independent_audit_inner_seconds_included=v['CPU_wall_seconds'],CPU_numeric_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],
        CPU_synthetic_seconds=synthetic,GPU_hours=0,new_source_forward=0,new_PCA=0,new_scaler_fits=0,
        source_training_original_cache_startup_imports_development_and_close_not_included=True,
        execution_sha256=sha(runs[0]),verification_execution_sha256=sha(audits[0]))
    write(R/'cost_receipt.json',cost)
    lines=['# R40：完整表示改善核读出，但未超过当前原生方案的保留门槛','',
        '结论：本轮不采用，不开启新的外层测试。完整四设置主结果仍R26 Core平均相对PCC增益+3.2771%，未达到5%。','',
        '先做代数审计：线性训练PCC的最优方向与OLS相同；一种含二次方差惩罚的相关性替代目标与Ridge方向相同。不把等价改写当新算法，也不将此结论推广到所有PCC损失或非线性模型。两项纯合成已验收，未作真实数据相关性目标重训。','',
        '实际实验只改变固定ReLU核的来源输入：PCA16白化坐标换成完整逐坐标标准化192维，按块除sqrt维数。原训练尺度、核阶数/深度/中心化、辅助均对角16、TD线性核、alpha100和平方损失不变。既去除截断也去除PC白化，不能单独归因于维数；没有新来源训练或其它模型预测堆叠。','',
        '|训练池|来源|新PCC|ΔPCC vs PCA16核R33|MSE下降 vs R33|ΔPCC vs原生R25|MSE下降 vs R25|ΔPCC vs R4|MSE下降 vs R4|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in d['records']:
        lines.append(f"|{r['setting']}|{r['source']}|{r['metrics']['PCC']:.6f}|{r['delta_PCC_vs_R33']:+.6f}|{r['MSE_reduction_vs_R33_pct']:+.4f}%|{r['delta_PCC_vs_R25']:+.6f}|{r['MSE_reduction_vs_R25_pct']:+.4f}%|{r['delta_PCC_vs_R4']:+.6f}|{r['MSE_reduction_vs_R4_pct']:+.4f}%|")
    lines+=['',
        '主Core在三个池都超过同核PCA16，表明当前压缩/白化处理会影响这个读出所利用的信息。但相比当前较强原生R25，Takayuki PCC下降0.012730、MSE增加2.4287%；Full与初始化在该池也弱于原生对照。不能只保留Hu/Mix或在各队列拼接不同方法。','',
        f"预定gate：主性能={d['performance_gate']}，同核压缩配对={d['compression_pair_gate']}，原生强对照={d['native_reference_gate']}，同类TD增量={d['source_increment_gate']}，数值={d['all_numerically_qualified']}，最终qualified={d['qualified']}。",'',
        '|训练池|旧TD-only核PCC（复用）|Core相对该TD的ΔPCC|MSE下降|','|---|---:|---:|---:|']
    for p in d['paired_kernel_TD']:
        t=next(t for t in d['TD_records'] if t['setting']==p['setting'])
        lines.append(f"|{p['setting']}|{t['metrics']['PCC']:.6f}|{p['delta_PCC']:+.6f}|{p['MSE_reduction_pct']:+.4f}%|")
    lines+=['','Mixset即便超过旧来源读出，仍未超过相同核的TD-only。这说明“表达更多来源信息”和“超过强TD基准”是不同要求。','',
        '计数：2299唯一原fold0训练池样本，45新核头/6897新OOF，15旧TD头/2299旧OOF，合计9196行；45旧R33来源核、45旧R4来源线性、45旧R25原生头均按SHA复用。新训练参考出现26451次不是独立样本数。原生kernel需保存训练reference和dual，部署成本高于217系数线性头。','',
        f"验收：8项新合成通过；45新独立LU重解、原尺度/旧PCA、不同atan2核、45旧R33和45旧R4正规方程、45旧R25部署、15旧TD复用/覆盖/门槛/SHA全部通过。旧R25凸证书引用原已核验回执，本轮未重新求解旧头。最大正规方程误差{v['max_normal_error']:.3g}，独立dual差{v['max_dual_difference']:.3g}，部署差{v['max_deployment_difference']:.3g}。",'',
        '|训练池|来源|内折|新PCC|PCA16核PCC|原生R25 PCC|新MSE|','|---|---|---:|---:|---:|---:|---:|']
    for r in d['records']:
        for x in r['inner_blocks']:
            lines.append(f"|{r['setting']}|{r['source']}|{x['inner']}|{x['metrics']['PCC']:.6f}|{x['R33_reference']['PCC']:.6f}|{x['R25_reference']['PCC']:.6f}|{x['metrics']['MSE']:.6f}|")
    lines+=['',
        f"成本：数值控制器{e['CPU_wall_seconds']:.6f}秒，独立验收外层{a['CPU_wall_seconds']:.6f}秒（内层{v['CPU_wall_seconds']:.6f}秒已包含），合计{cost['CPU_numeric_and_audit_seconds']:.6f}CPU秒；合成{synthetic:.6f}秒另列。无GPU/新source forward/PCA/scaler拟合。旧训练、缓存、进程启动导入、开发和收尾不含，不能当端到端用时。",'',
        '原始核参考：[Cho与Saul2009](https://proceedings.neurips.cc/paper_files/paper/2009/file/5751ec3e9a4feab575962e78e006250d-Paper.pdf)。它支持所用核的形式，不保证siRNA迁移增益。','',
        '这次结果在已知R33/R25表现后提出，只是自适应开发证据。单source42、keep-overlap/context暴露、purged非gene隔离、多个Mixset设置依赖与缺少新鲜盲外部的限制不变。没有新的四设置结果或5%完成证据。','',
        f"科学锁：{sha(R/'lock.json')}。诊断：{sha(R/'diagnostic.json')}。独立验收：{sha(R/'verification.json')}。"]
    report=B/'ROUND40_RESULTS.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    files=[R/'lock.json',R/'diagnostic.json',R/'diagnostic_rows.csv',R/'verification.json',R/'cost_receipt.json',report,Path(__file__)]+[R/n for n in lock['files']]+runs+audits
    manifest={p.relative_to(B).as_posix():sha(p) for p in files}
    for n,h in manifest.items():assert sha(B/n)==h
    write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,goal_complete=False,qualified=False,no_live_execution_mutex=True,
        all_pilot_files_verified=len(d['pilot_files_sha256']),files_sha256=manifest,CPU_close_seconds=time.monotonic()-started))
    write(B/'GOAL_TURN_R40_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        concrete_progress='PCC surrogate equivalence checked; 45 uncompressed exact-kernel heads and independent LU audit completed. Improved all Core pools versus PCA kernel but lost Takayuki versus strong native head; qualification false.',
        new_outer_test_evaluation=False,new_heads=45,no_active_handle=True,genuine_external_blocker=False,
        closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\nR40固定ReLU核保留原生192输入闭合：45新头/6897新OOF、15旧TD头/2299旧行；45独立LU/原尺度/不同核公式/原生部署/SHA通过。Core相对PCA16核ΔPCC +.005752/+.007037/+.010808，但相对较强R25为+.005673/+.011197/−.012730，Tak MSE增加2.4287%；主性能/压缩配对/TD增量true，native_reference=false，最终gate=false，不新外层或扩核/维度/alpha。相关性替代目标与Ridge方向等价的纯合成/代数已审计，不另拟合等价方法。CPU数值+验收{cost["CPU_numeric_and_audit_seconds"]:.6f}s，合成{synthetic:.6f}s另计，GPU/source forward/PCA/scaler fits0。完整主结果仍R26+3.2771%，goal active未达5%；上一轮与本轮progress，无活句柄/阻断，见ROUND40_RESULTS.md。\n')
    print(json.dumps(dict(status='closed',qualified=False,goal_complete=False,closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report),CPU_numeric_and_audit_seconds=cost['CPU_numeric_and_audit_seconds'])))


if __name__=='__main__':main()
