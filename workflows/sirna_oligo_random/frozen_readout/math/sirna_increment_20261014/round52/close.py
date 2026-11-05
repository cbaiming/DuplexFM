"""Complete positive and negative qualification report, without model promotion."""
import sys,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import run as io
def fmt(x):return '/' if x is None else f'{x:+.5f}'
def main():
    began=time.monotonic();io.validate();d=io.read(R/'diagnostic.json');v=io.read(R/'verification.json')
    assert v['passed'] and v['qualified']==d['qualified'] and not (R/'execution.mutex').exists()
    runs=list(R.glob('execution_*.json'));audits=list(R.glob('verification_execution_*.json'));assert len(runs)==len(audits)==1
    e,a=io.read(runs[0]),io.read(audits[0]);assert e['returncode']==a['returncode']==0
    report=io.B/'ROUND52_RESULTS.md';assert not report.exists() and not (R/'closure_receipt.json').exists()
    synthetic=sum(float(x.get('time',0)) for x in ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    cost=dict(CPU_controller_seconds=e['CPU_wall_seconds'],CPU_independent_audit_seconds=a['CPU_wall_seconds'],
        CPU_controller_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],synthetic_seconds_separate=synthetic,
        new_efficiency_fits=0,new_source_forward=0,new_PCA=0,GPU_hours=0,not_end_to_end=True)
    lines=['# R52：组内残差方差的跨半复现资格','',
        f"预登记资格结果：**{d['qualified']}**。本轮仅检查异方差加权前提，没有训练新的效率模型或新增外层结果。当前完整四设置预定主成绩仍为R50相对强TD24平均PCC **+4.190823%**，目标5%尚未达到。",'',
        '复用Huesken1564/Mixset319/Takayuki416原fit池的R49固定alpha100 OOF，四条件共9196行。每个n≥8的输入序列组按ID哈希分两半，两个布局均固定。校准半估计向全局收缩的组方差，验证半评价组内尺度；两方向都报告，不按布局挑结果。正的分数改善表示方差预测优于只用全局方差，不是PCC或效率增益。','',
        '|训练池|条件|布局|可估计组数|覆盖样本/比例|两半方差Spearman|A→B分数改善|B→A分数改善|平均改善|资格|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in d['records']:
        lines.append(f"|{r['setting']}|{r['source']}|{r['layout']}|{r['eligible_groups']}|{r['covered_samples']}/{100*r['coverage']:.1f}%|{fmt(r['variance_spearman'])}|{fmt(r['A_to_B']['score_improvement'])}|{fmt(r['B_to_A']['score_improvement'])}|{fmt(r['mean_score_improvement'])}|{r['qualified']}|")
    lines+=['',
        f"主Core三个池资格：{d['primary_pool_qualifications']}。要求每池≥8个合格组、覆盖≥50%，两个布局各自rho≥.3且平均分数改善≥.01 nats，至少两池通过。不采用Full/初始化/TD控制中的有利结果替换主Core，也不改变门槛。",'',
        ('资格通过只允许另立训练内权重协议；本轮不是新模型性能，不可以直接使用当前含内验证响应的全fit权重。' if d['qualified'] else '资格未通过，关闭当前组方差加权分支：不训练加权效率头、不做新的四设置评价、不调整先验自由度/分组/布局或阈值。'),'',
        '这里的组不是实验重复。残差包括未建模信号、标签误差和不同OOF训练模型的差异；组间方差相关即便为正，也不能证明估计到了测量噪声。样本少、覆盖不足或布局不稳定都可能使权重不可靠。两布局共享数据和原模型，不能当两个独立实验。',
        '与R9极端误差诊断、R24最坏群体风险不同，本轮检查的是组残差尺度的可复现性，不重新包装已失败的Huber或group-DRO；没有新增gate、专家开关、source或超参数网格。数学动机采用[NIST WLS说明](https://itl.nist.gov/div898/handbook/pmd/section1/pmd143.htm)关于误差方差与可靠权重的条件，不宣称该条件已在本数据得到证明。','',
        f"审计：8项纯合成首次通过；3个输入图独立重建，{v['independent_layout_group_splits']}组布局切分、24条件布局、60旧OOF块/9196行的身份/原预测/样本方差/秩相关/分数/资格/SHA全部通过。最大独立数值差{v['maximum_absolute_difference']:.3g}。",'',
        f"成本：CPU控制器{cost['CPU_controller_seconds']:.6f}s＋独立验收{cost['CPU_independent_audit_seconds']:.6f}s＝{cost['CPU_controller_and_audit_seconds']:.6f}s；合成{synthetic:.6f}s另列。无新效率拟合/forward/PCA/GPU。方差参数仅为本轮诊断估计，不能称没有任何数值估计；历史源训练/缓存、启动/登记/开发/收尾不含，不称端到端耗时。",'',
        '当前目标active、未达5%；本轮提供新的训练内证据，上一轮和本轮均progress，无活动句柄或外部阻断。source keep-overlap/context暴露、单source42、purged非gene隔离、反复开发和两Mix设置不独立继续限制推断，不宣称SOTA。',
        f"科学锁SHA {io.sha(R/'lock.json')}；诊断SHA {io.sha(R/'diagnostic.json')}；独立验收SHA {io.sha(R/'verification.json')}。"]
    io.write(R/'cost_receipt.json',cost)
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    files=[R/'lock.json',R/'diagnostic.json',R/'summary.csv',R/'halves_fixed.json',R/'verification.json',R/'cost_receipt.json',report]+[R/n for n in io.CODE]+runs+audits
    manifest={p.relative_to(io.B).as_posix():io.sha(p) for p in files}
    for name,h in manifest.items():assert io.sha(io.B/name)==h,name
    io.write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,files_sha256=manifest,qualified=d['qualified'],goal_complete=False,
        no_live_execution_mutex=True,CPU_close_seconds=time.monotonic()-began))
    io.write(io.B/'GOAL_TURN_R52_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        genuine_external_blocker=False,no_active_handle=True,new_efficiency_heads=0,new_train_pool_variance_evidence=True,
        qualification=d['qualified'],current_primary_relative_PCC_gain_pct=4.190823121288767,
        closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)))
    with (io.B/'PROGRESS.md').open('a',encoding='utf-8') as f:f.write(f"\n\nR52训练内组残差方差资格闭合：3池2299身份/四条件9196旧OOF，24条件×布局，3独立图及{v['independent_layout_group_splits']}布局组分半/方差/校准-验证隔离/全部SHA通过。主Core池资格{d['primary_pool_qualifications']}，总体{d['qualified']}；不改先验/分组/布局/门槛，不按控制择优。新效率头/forward/PCA/GPU0，CPU诊断+验收{cost['CPU_controller_and_audit_seconds']:.6f}s，合成另计。完整主成绩仍R50+4.190823%，goal active未完成；前轮与本轮progress，无活句柄/外部阻断。详见ROUND52_RESULTS.md。\n")
    print(dict(status='closed',qualified=d['qualified'],CPU_seconds=cost['CPU_controller_and_audit_seconds'],closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)),flush=True)

if __name__=='__main__':main()
