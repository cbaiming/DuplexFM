"""Archive verified label-free evidence; never fit a prediction model."""
import hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    start=time.monotonic();assert not (R/'closure_receipt.json').exists() and not (R/'execution.mutex').exists()
    d=read(R/'diagnostic.json');v=read(R/'verification.json');lock=read(R/'lock.json')
    assert v['passed'] and not d['qualified'] and not v['qualified'] and v['core_pool_qualification']==[False,True,False]
    assert v['diagnostic_sha256']==sha(R/'diagnostic.json') and v['lock_sha256']==sha(R/'lock.json')
    for name,h in lock['files'].items():assert sha(R/name)==h,name
    for name,h in lock['dependencies'].items():assert sha(Path(name))==h,name
    executions=sorted(R.glob('execution_*.json'));audits=sorted(R.glob('verification_execution_*.json'))
    assert len(executions)==len(audits)==1
    e,a=read(executions[0]),read(audits[0]);assert e['returncode']==a['returncode']==0
    tests=sum(float(s.get('time',0)) for s in ET.parse(R/'pytest.xml').getroot().iter('testsuite'))
    cost=dict(CPU_controller_seconds=e['CPU_wall_seconds'],CPU_audit_outer_seconds=a['CPU_wall_seconds'],
        CPU_audit_inner_seconds_included=v['CPU_wall_seconds'],CPU_numeric_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],
        synthetic_seconds_separate=tests,source_blocks=9,unique_TD_blocks=3,scalar_coordinate_checks=1800,
        unique_input_rows=2299,source_row_occurrences=6897,new_efficiency_heads=0,new_source_forward=0,new_PCA=0,GPU_hours=0,
        excluded='Source training/export, original readouts, process startup/imports, development and report/closure are separate.',
        execution_sha256={p.name:sha(p) for p in executions+audits})
    write(R/'cost_receipt.json',cost)
    lines=['# R32：来源表示的逐坐标尾部集中度','',
        '2026-10-15。无标签训练输入诊断已经闭合并独立验收。Core只有Mixset满足预定尾部资格，未达到至少2/3池的规则，因此不启动边际变换或新的监督拟合，不改变阈值。当前四设置主结果仍为R26平均相对PCC增益+3.2771%，5%目标未完成。','',
        '本轮检查每个来源坐标的训练分布，而不是R15中的整条样本L2长度。一个坐标须同时满足：约1%最极端训练样本贡献至少20%的总平方偏差，最大绝对z至少6，且不是原尺度下限之下的近恒定坐标。每池至少20/192个坐标满足才合格，主Core至少两个池合格才使后续稳健边际变换值得另立协议。该规则是工程筛查，不是显著性检验。','',
        '|训练内池|来源|训练行数|合格尾部坐标/有效坐标|全部坐标的尾部份额中位数|最大尾部份额|最大绝对z|池资格|',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for r in d['records']:
        s=r['statistics']
        lines.append(f"|{r['setting']}|{r['source']}|{s['n']}|{s['flagged_columns']}/{s['eligible_columns']}|{100*s['median_tail_share']:.2f}%|{100*s['maximum_tail_share']:.2f}%|{s['maximum_abs_z']:.3f}|{'是' if s['pool_qualified'] else '否'}|")
    lines+=['','Core在Mixset有22个坐标达到规则，Huesken为11个、Takayuki为6个；这提示Mixset的来源激活分布较特殊，但不能据此断言它导致了负迁移。这里没有计算输入尾部与效率误差的关系，更没有把测试表现用于筛选这些坐标。','',
        '精确初始化控制的三个池均为0个合格坐标。它说明当前已训练表示与初始化的边际形态不同；极端激活也可能编码真实而少见的模式，不能直接称为过拟合、错误值或应删除的信息。','',
        '|训练内池|TD24合格尾部坐标/24|尾部份额中位数|最大尾部份额|最大绝对z|',
        '|---|---:|---:|---:|---:|']
    for r in d['TD_controls']:
        s=r['statistics']
        lines.append(f"|{r['setting']}|{s['flagged_columns']}/24|{100*s['median_tail_share']:.2f}%|{100*s['maximum_tail_share']:.2f}%|{s['maximum_abs_z']:.3f}|")
    lines+=['','TD24控制仅作描述，没有进入来源资格；基线未改。所有逐列统计保存在diagnostic.json，不只报告正面坐标。相关坐标不等于独立观察，平方偏差份额也不是回归hat矩阵杠杆或因果影响。','',
        '动作：结束本轮统一尾部变换候选的资格分支，不降低“两个池/20个坐标”的要求，不按Mixset单项结果立刻修改模型，不实际执行quantile/Gaussian化、截断、行归一化或新参数网格。失败不证明所有单调变换必定无效；它只说明预先要求的普遍尾部前提未成立。','',
        '参考[Liu等，The Nonparanormal，JMLR2009](https://www.jmlr.org/papers/v10/liu09a.html)的单调边际变换思路，仅用于说明候选方向。该文研究图模型，不保证本任务的迁移效果，本轮没有复现其图模型。','',
        '范围：三个原outer fold0 fit池共2299条，三来源合计6897来源行出现；9个来源统计块与3个唯一TD统计块，共1800坐标。标签字符串仅随输入文件身份核验读取，未数值化；无效率拟合、新PCA、来源forward、GPU或外层新指标。没有重新运行旧模型或已完成统计。','',
        f"验收：七项纯合成测试通过；独立math.fsum/逐列排序复算1800坐标与全部资格、来源/输入/fit IDs/SHA。最大绝对统计差{v['max_absolute_statistic_difference']:.3g}。控制器与独立验收均rc0。",'',
        f"成本：CPU数值控制器{e['CPU_wall_seconds']:.6f}秒、独立验收外层{a['CPU_wall_seconds']:.6f}秒，合计{cost['CPU_numeric_and_audit_seconds']:.6f}秒；合成测试{tests:.6f}秒另列。源训练/导出、原读出、启动导入、开发和收尾成本另列，不能将这几秒称端到端模型时间。",'',
        '限制继续保留：单source42、来源keep-overlap/context暴露、反复开发、purged sequence不是gene隔离、Mixset两设置非独立。本轮输入资格不能当作新性能或独立外部确认。source43范围仍未得到新回复，没有导出或改目标。上一轮和本轮均产出新证据，属于progress；无活进程、无外部阻断，goal保持active未完成。','',
        f"科学锁：{sha(R/'lock.json')}。输入诊断：{sha(R/'diagnostic.json')}。独立验收：{sha(R/'verification.json')}。"]
    report=B/'ROUND32_RESULTS.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    paths=[R/'lock.json',R/'diagnostic.json',R/'verification.json',R/'cost_receipt.json',report,Path(__file__)]+[R/n for n in lock['files']]+executions+audits
    manifest={p.relative_to(B).as_posix():sha(p) for p in paths}
    for name,h in manifest.items():assert sha(B/name)==h
    write(R/'closure_receipt.json',dict(status='closed',qualified=False,goal_complete=False,readback_passed=True,
        no_live_execution_mutex=True,files_sha256=manifest,CPU_close_seconds=time.monotonic()-start))
    write(B/'GOAL_TURN_R32_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        concrete_progress='Preregistered label-free coordinate tail audit independently verified 1800 coordinates. Only Mixset met the pool criterion; global eligibility failed, so no new transform or predictor is launched.',
        new_outer_test_evaluation=False,source43_scope_reply_received=False,no_active_handle=True,genuine_external_blocker=False,
        closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write('\n\nR32逐坐标尾部资格闭合：三个原fold0训练池2299条，三来源9块/6897来源行及3唯一TD块，共1800坐标独立标量复算通过。Core合格尾部坐标Hu11/Mix22/Tak6（各需20），仅Mix通过，整体gate=false；不按单池改阈值，不启动边际变换或监督拟合。标签不数值化，新头/forward/PCA/GPU0。CPU数值+验收1.125s、合成.134s另计；详见ROUND32_RESULTS.md。四设置主结果仍R26+3.2771%，goal active未达5%；本轮progress、无活句柄/外部阻断。\n')
    print(json.dumps(dict(status='closed',qualified=False,closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report),CPU_numeric_and_audit_seconds=cost['CPU_numeric_and_audit_seconds'])))


if __name__=='__main__':main()
