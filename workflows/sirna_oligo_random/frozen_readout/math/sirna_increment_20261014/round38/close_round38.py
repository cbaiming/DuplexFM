"""Close label-free paired-background qualification without an efficacy fit."""
import hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    started=time.monotonic();assert not (R/'closure_receipt.json').exists() and not (R/'execution.mutex').exists()
    d,v,lock=read(R/'diagnostic.json'),read(R/'verification.json'),read(R/'lock.json')
    assert v['passed'] and not v['qualified'] and not d['qualified'] and not v['numeric_labels_read']
    assert v['counts']==dict(maps=45,rows=6897,unique=2299) and v['new_efficiency_heads']==v['GPU_hours']==v['new_source_forward']==0
    assert v['lock_sha256']==sha(R/'lock.json') and v['diagnostic_sha256']==sha(R/'diagnostic.json')
    for n,h in lock['files'].items():assert sha(R/n)==h,n
    for p,h in lock['dependencies'].items():assert sha(Path(p))==h,p
    for n,h in d['map_files_sha256'].items():assert sha(R/'maps'/n)==h,n
    assert sha(R/'diagnostic_rows.csv')==d['diagnostic_rows_sha256']
    runs=list(R.glob('execution_*.json'));audits=list(R.glob('verification_execution_*.json'));assert len(runs)==len(audits)==1
    e,a=read(runs[0]),read(audits[0]);assert e['returncode']==a['returncode']==0
    suites=list(ET.parse(R/'pytest.xml').getroot().iter('testsuite'));assert sum(int(s.get('tests',0)) for s in suites)==9
    assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
    synthetic=sum(float(s.get('time',0)) for s in suites)
    cost=dict(CPU_controller_seconds=e['CPU_wall_seconds'],CPU_audit_outer_seconds=a['CPU_wall_seconds'],CPU_audit_inner_included=v['CPU_wall_seconds'],
        CPU_numeric_and_audit_seconds=e['CPU_wall_seconds']+a['CPU_wall_seconds'],synthetic_seconds_separate=synthetic,
        new_multivariate_input_maps=45,independent_augmented_SVD_solves=45,vector_OOF_rows=6897,vector_width=192,unique_fit_rows=2299,
        new_efficiency_heads=0,new_source_forward=0,new_PCA=0,GPU_hours=0,numeric_labels_read=False,
        excluded='Previous source training/export and reference fits, process startup/imports, development and report/closure.',
        execution_sha256={p.name:sha(p) for p in runs+audits})
    write(R/'cost_receipt.json',cost)
    lines=['# R38：配对初始化背景的留出预测资格','',
        '2026-10-15。45个无效率标签的多输出映射与独立验收已完成，预定资格未通过。不进入后续背景惩罚效率头，不调alpha、尺度或资格阈值。当前完整四设置主结果仍R26平均相对PCC+3.2771%，5%目标未达到。','',
        '本轮只问：精确初始化的192D表示，能否在未参与该映射拟合的输入上预测Core192变化？它与R34的边际协方差对比PCA不同，使用相同RNA输入的配对关系。训练只用原三个fold0 fit池中的inner train，未数值化沉默效率/分类标签，没有外层指标。','',
        '|内部输入池|配对初始化→Core R²|打乱初始化→Core R²|TD24→Core R²|配对比TD24的R²差|通过资格|',
        '|---|---:|---:|---:|---:|---|']
    for r in d['records']:
        c=r['conditions'];q=next(x for x in d['pool_qualification'] if x['setting']==r['setting'])
        lines.append(f"|{r['setting']}|{c['paired_initial']['R2']:.6f}|{c['shuffled_initial']['R2']:.6f}|{c['TD24']['R2']:.6f}|{q['delta_vs_TD24']:+.6f}|{'是' if q['qualified'] else '否'}|")
    lines+=['',
        'R²按每个内部折的预测SSE和训练均值null SSE先累加再取比；不是siRNA效率PCC。正确配对在三个池都优于打乱配对，但相比TD24只在Takayuki略高，差约0.0183，未达到预设至少两个池同时R²≥0.10、比TD24及打乱对照均高0.05的条件。三池都未通过单池资格。','',
        '这意味着当前固定alpha100和尺度下，尚无充分工程依据将“初始化可预测的Core方向”作为新的抑制目标。不能把未解释比例称为miRNA独有信息、不可约差异或生物机制；未解释部分也可能受线性模型、正则强度、训练样本量及表示坐标变化影响。本轮失败后不据结果更换这些设置。初始化共享部分仍可能有用，不等于噪声。','',
        'Core的initial SHA与初始缓存checkpoint一致，同variant/source receipt/输入接口全部核验。Full未进入这项配对诊断，因为协议只指定了Core的精确初始化对照，而不是看过结果后排除Full。三个控制使用同Core目标、同inner划分/训练均值与alpha；192D初始化和24D TD控制容量不同，本轮不据R²差作因果归因。','',
        '|输入池|内部折|配对R²|打乱R²|TD24 R²|打乱训练固定点数|',
        '|---|---:|---:|---:|---:|---:|']
    for r in d['records']:
        for b in r['inner_blocks']:
            st=read(R/'maps'/r['setting']/'shuffled_initial'/f"inner{b['inner']}"/'state.json');c=b['conditions']
            lines.append(f"|{r['setting']}|{b['inner']}|{c['paired_initial']['R2']:.6f}|{c['shuffled_initial']['R2']:.6f}|{c['TD24']['R2']:.6f}|{st['shuffled_fixed_points']}|")
    lines+=['',
        '每折六个mean/max池化块的SSE、null和R²保存在diagnostic.json；所有逐条192D预测及训练预测Gram在maps。固定一次训练行置换可能包含少数固定点，以上完整披露，没有为了更差的对照反复置换。','',
        '范围：Huesken1564/Mixset319/Takayuki416，共2299个唯一训练池输入；45个多输出输入映射、6897条向量OOF，每条192维。不是45个siRNA效率头，也不是6897个独立样本。没有PCA、新来源forward、GPU或新的效率指标。','',
        f"验收：9项合成通过；45个独立增广SVD least-squares、输入身份、旧train scaler、置换、正规方程、Gram独立公式、逐行覆盖、统计及SHA全部通过。最大系数差{v['max_coefficient_difference']:.3g}，Gram差{v['max_Gram_difference']:.3g}，正规方程误差{v['max_normal_error']:.3g}。",'',
        f"成本：数值控制器{e['CPU_wall_seconds']:.6f}秒，独立验收外层{a['CPU_wall_seconds']:.6f}秒（内层{v['CPU_wall_seconds']:.6f}秒已包含），共{cost['CPU_numeric_and_audit_seconds']:.6f}秒；合成{synthetic:.6f}秒另列。旧来源训练/缓存、进程启动导入、开发与收尾不包含，不能称端到端成本。",'',
        '与边际对比PCA的区别可参见[Abid等2018](https://www.nature.com/articles/s41467-018-04608-8)；本文的配对多输出回归不是该文算法。反复开发、单source42、源keep-overlap/context暴露、purged非gene隔离和两个Mixset设置非独立等限制不变，不称新的盲确认。','',
        f"科学锁：{sha(R/'lock.json')}。诊断：{sha(R/'diagnostic.json')}。独立验收：{sha(R/'verification.json')}。"]
    report=B/'ROUND38_RESULTS.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    paths=[R/'lock.json',R/'diagnostic.json',R/'verification.json',R/'cost_receipt.json',R/'diagnostic_rows.csv',report,Path(__file__)]+[R/n for n in lock['files']]+runs+audits
    manifest={p.relative_to(B).as_posix():sha(p) for p in paths}
    for n,h in manifest.items():assert sha(B/n)==h
    write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,goal_complete=False,qualified=False,new_efficiency_model_selected=False,
        no_live_execution_mutex=True,all_map_files_verified=len(d['map_files_sha256']),files_sha256=manifest,CPU_close_seconds=time.monotonic()-started))
    write(B/'GOAL_TURN_R38_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        concrete_progress='45 label-free paired/shuffled/TD maps and independent SVD audits complete. Paired initial lacks the predeclared improvement over TD24, so proposed background-penalty efficacy experiment is not started.',
        numeric_labels_read=False,new_outer_test_evaluation=False,new_efficiency_heads=0,no_active_handle=True,genuine_external_blocker=False,
        closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write(f'\n\nR38配对初始化背景资格闭合：45无效率标签多输出映射/6897向量OOF/2299unique输入，45独立增广SVD、原尺度/身份/置换/Gram/覆盖/SHA通过。配对初始化→Core留出R²为.136028/.066539/.092310，TD24→Core为.151449/.102081/.074029，三池均未过预定资格，不开始后续背景惩罚效率头，不调alpha/置换/尺度。未解释比例不称miRNA独有信号。CPU数值+验收{cost["CPU_numeric_and_audit_seconds"]:.6f}s、合成{synthetic:.6f}s另计，新效率拟合/forward/PCA/GPU0。完整四设置主结果仍R26+3.2771%，goal active未达5%；上一轮及本轮progress，无活句柄/外部阻断，详见ROUND38_RESULTS.md。\n')
    print(json.dumps(dict(status='closed',qualified=False,goal_complete=False,closure_sha256=sha(R/'closure_receipt.json'),report_sha256=sha(report),CPU_numeric_and_audit_seconds=cost['CPU_numeric_and_audit_seconds'])))


if __name__=='__main__':main()
