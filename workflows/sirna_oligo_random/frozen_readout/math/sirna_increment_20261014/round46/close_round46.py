"""Preserve complete negative qualification including the interface-audit failure."""
import sys,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent;B=R.parent
sys.path.insert(0,str(R))
from run import io,validate


def main():
    started=time.monotonic();report=B/'ROUND46_RESULTS.md'
    assert not (R/'execution.mutex').exists()
    for p in (R/'cost_receipt.json',R/'closure_receipt.json',report,B/'GOAL_TURN_R46_AUDIT.json'):assert not p.exists(),p
    lock=validate();d=io.read(R/'diagnostic.json');v=io.read(R/'verification.json');compat=io.read(R/'verification_execution_lock.json')
    assert v['passed'] and not d['qualified'] and not v['qualified']
    assert d['lock_sha256']==v['lock_sha256']==io.sha(R/'lock.json') and v['diagnostic_sha256']==io.sha(R/'diagnostic.json')
    assert d['rows_sha256']==io.sha(R/'diagnostic_rows.csv')
    assert compat['original_science_lock_sha256']==io.sha(R/'lock.json') and compat['diagnostic_sha256']==io.sha(R/'diagnostic.json')
    for n,h in compat['files_sha256'].items():assert io.sha(R/n)==h,n
    for n,h in d['graph_files_sha256'].items():assert io.sha(R/'graph_audit'/n)==h,n
    assert v['counts']==dict(graphs=60,masks=15,rows=9196,TD_models_checked=15,source_scales_checked=45)
    assert d['new_efficiency_heads']==d['new_OOF_rows']==d['new_source_forward']==d['new_PCA_fits']==d['GPU_hours']==0
    runs=list(R.glob('execution_*.json'))
    audits=sorted(p for p in R.glob('verification_execution_*.json') if p.stem.removeprefix('verification_execution_').isdigit())
    assert len(runs)==1 and len(audits)==2
    e=io.read(runs[0]);av=[io.read(p) for p in audits]
    assert e['returncode']==0 and [a['returncode'] for a in av]==[1,0]
    assert io.sha(audits[0])==compat['failed_execution_sha256'] and 'IndexError: too many indices' in av[0]['error']
    tests={}
    for filename,n in (('pytest.xml',10),('pytest_compat.xml',3)):
        suites=list(ET.parse(R/filename).getroot().iter('testsuite'))
        assert sum(int(s.get('tests',0)) for s in suites)==n
        assert not any(int(s.get(k,0)) for s in suites for k in ('errors','failures','skipped'))
        tests[filename]=sum(float(s.get('time',0)) for s in suites)
    cost=dict(CPU_diagnostic_seconds=e['CPU_wall_seconds'],CPU_failed_audit_seconds=av[0]['CPU_wall_seconds'],
        CPU_successful_audit_outer_seconds=av[1]['CPU_wall_seconds'],CPU_successful_audit_inner_included=v['CPU_wall_seconds'],
        CPU_diagnostic_and_all_audits_seconds=e['CPU_wall_seconds']+sum(a['CPU_wall_seconds'] for a in av),
        CPU_synthetic_seconds_by_report=tests,GPU_hours=0,new_source_forward=0,new_efficiency_heads=0,new_OOF_rows=0,
        new_graphs=60,eligibility_masks=15,old_source_training_cache_startup_registration_development_close_not_included=True,
        execution_sha256=io.sha(runs[0]),audit_execution_sha256=[io.sha(p) for p in audits])
    lines=['# R46：表示邻域有弱信号，但未达到图约束的推进门槛','',
        '只用既有冻结缓存与原训练池，60个图和15个直接序列排除矩阵已完成独立验收。本轮为有标签的训练池资格诊断，不是新的siRNA测试成绩；未新增效率模型、来源forward或GPU计算。主Core三个池均未通过预定单池条件，整体qualified=false，不实施后续图平滑读出、不调k或门槛。','',
        '固定Huesken1564/Mixset319/Takayuki416与原五个内折。图只在同一个inner validation块中构造，每个端点残差都由同一个未见该整块标签的旧TD alpha100模型产生。近邻使用输入，不使用效率；排除guide Hamming<=2或任意规范target25共享的配对及自身。每行最多10邻居，距离ties按sample_id。','',
        'R是近邻残差差平方能量相对所有允许配对的比值，越低表示平滑性越强；R=1是匹配基准。它不是MSE下降百分比或PCC增益，不代表模型能获得同等提升。五块先合并能量再取比值。','',
        '|开发训练池|Core残差比值R|TD几何残差比值R|Core−TD|Core低于1的块数|单池资格|',
        '|---|---:|---:|---:|---:|---|']
    for p in d['per_pool']:
        r=next(x for x in d['records'] if x['setting']==p['setting'] and x['condition']=='core_real')
        lines.append(f"|{p['setting']}|{p['Core_ratio']:.6f}|{p['TD_ratio']:.6f}|{p['Core_minus_TD']:+.6f}|{r['aggregate']['blocks_below_null']}/5|{p['passed']}|")
    lines+=['',
        '预定门槛为Core至少两池R<=.90、比TD几何低>=.05、至少4/5块R<1，并满足所有块90%邻居覆盖及90%节点至少5邻居。数据覆盖通过，主要信号强度和相对TD优势不足。不能据此证明所有图方法无效；它只否定本轮固定几何/邻居数下优先增加这类约束的依据。','',
        '|开发训练池|输入几何|残差R|原效率R（描述）|残差近邻能量|残差全部允许能量|低于1的块数|覆盖合格|',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for r in d['records']:
        a=r['aggregate'];x=a['residual'];y=a['response']
        lines.append(f"|{r['setting']}|{r['condition']}|{x['ratio']:.6f}|{y['ratio']:.6f}|{x['knn_mean']:.6f}|{x['null_mean']:.6f}|{a['blocks_below_null']}/5|{a['coverage_qualified']}|")
    lines+=['',
        'Core在Hu/Tak的残差邻域比初始化更平滑，Mix则初始化更平滑；TD几何自身也解释部分邻域结构。不能将原效率的邻域平滑直接归为miRNA监督的独有信息，也不能把残差全部称为不可约噪声。此结果与当前迁移收益的队列差异相容，但没有单独证明旧论文与当前成绩差异的因果原因。','',
        '|开发训练池|条件|原内折|节点数|至少一个允许邻居比例|至少5邻居比例|残差R|原效率R|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return '未定义' if x is None else f'{x:.6f}'
    for r in d['records']:
        for b in r['inner_blocks']:
            x=b['residual'];y=b['response']
            lines.append(f"|{r['setting']}|{r['condition']}|{b['inner']}|{b['n']}|{x['coverage']:.4f}|{x['coverage_ge5']:.4f}|{fmt(x['ratio'])}|{fmt(y['ratio'])}|")
    lines+=['',
        '计数：60个输入图、15个共享资格矩阵、9196条件行，复用15个唯一TD头；45个来源尺度读取及独立算术核对。没有新的PCA/标准化模型拟合、来源效率头、外层预测或测试评价。不把这些图称为部署模型，不依据Full/初始化选择新主方案。','',
        '验收首次发现旧TD截距是长度5的一维数组，原脚本误加候选索引导致IndexError。原科学代码/锁/诊断/失败均保留。显式verify_compat仅在内存替换一个精确读取语句，并断言shape==(5,)；独立执行锁绑定有效代码SHA。原10项合成及3项新增兼容合成通过。修复后60图的直接序列排除、独立逐行距离排序、所有逐行/块/池能量、阈值、身份/划分/SHA及15旧TD正规方程均通过；没有重跑诊断。','',
        f"最大TD正规方程误差{v['max_TD_normal_equation_error']:.3g}，预测重建误差{v['max_TD_prediction_error']:.3g}。",'',
        f"成本：CPU诊断{e['CPU_wall_seconds']:.6f}秒、首次失败验收{av[0]['CPU_wall_seconds']:.6f}秒、成功验收{av[1]['CPU_wall_seconds']:.6f}秒，合计{cost['CPU_diagnostic_and_all_audits_seconds']:.6f}秒。合成原测试{tests['pytest.xml']:.6f}秒、兼容{tests['pytest_compat.xml']:.6f}秒另列。新GPU/source forward0；旧来源训练/缓存、启动/登记/开发/收尾不包含，不能称端到端耗时。",'',
        '文献背景：[Belkin等，Manifold Regularization，JMLR 2006](https://www.jmlr.org/papers/v7/belkin06a.html)。本轮仅检验使用该类正则的一个必要经验动机，没有实现其半监督模型或移用泛化保证。','',
        '当前完整四设置主结果仍是R26平均相对PCC +3.2771%，未达5%。源keep-overlap/context暴露、同池反复开发、单来源seed42、非gene隔离、两个Mixset设置非独立的限制保持。此次资格失败不改变基线，不隐去负结果，不宣称目标已达成。','',
        f"科学锁：{io.sha(R/'lock.json')}。执行兼容锁：{io.sha(R/'verification_execution_lock.json')}。诊断：{io.sha(R/'diagnostic.json')}。独立验收：{io.sha(R/'verification.json')}。"]
    io.write(R/'cost_receipt.json',cost)
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    files=[R/'lock.json',R/'diagnostic.json',R/'diagnostic_rows.csv',R/'verification.json',R/'cost_receipt.json',R/'verification_execution_lock.json',report,Path(__file__)]+[R/n for n in lock['files']]+[R/n for n in compat['files_sha256']]+runs+audits
    manifest={p.relative_to(B).as_posix():io.sha(p) for p in files}
    for n,h in manifest.items():assert io.sha(B/n)==h
    io.write(R/'closure_receipt.json',dict(status='closed',readback_passed=True,qualified=False,goal_complete=False,
        no_live_execution_mutex=True,files_sha256=manifest,graph_files_verified=len(d['graph_files_sha256']),CPU_close_seconds=time.monotonic()-started))
    io.write(B/'GOAL_TURN_R46_AUDIT.json',dict(previous_turn='progress',this_turn='progress',goal_status='active',objective_achieved=False,
        genuine_external_blocker=False,no_active_handle=True,new_outer_test_evaluation=False,new_efficiency_heads=0,new_diagnostic_graphs=60,
        concrete_progress='60 fixed within-block input graphs, sequence exclusions and TD-residual energies independently audited. All Core pools fail fixed signal-strength qualification; do not implement graph readout.',
        closure_sha256=io.sha(R/'closure_receipt.json'),report_sha256=io.sha(report)))
    with (B/'PROGRESS.md').open('a',encoding='utf-8') as f:
        f.write('\n\nR46样本邻域资格闭合：60图/15直接序列资格矩阵/9196条件行，15旧TD头与45来源尺度/所有邻居及能量独立核验通过。Core残差R Hu.921792/Mix.978749/Tak.934694，TD几何.943763/1.002863/.938053；三个Core池均未过强度/配对门槛，qualified=false。不实现图效率头、不改k/距离/阈值、不新外层。首次验收旧截距1D/2D接口失败保留，原锁未改，显式单语句兼容和3新合成后验收通过，原10合成通过。'
                f'CPU诊断+失败与成功验收{cost["CPU_diagnostic_and_all_audits_seconds"]:.6f}s，合成另列，新GPU/forward/效率头/PCA0。完整主结果仍R26+3.2771%，goal active未达5%；上一轮和本轮progress，无活句柄/外部阻断，见ROUND46_RESULTS.md。\n')
    print(dict(status='closed',qualified=False,goal_complete=False,closure_sha256=io.sha(R/'closure_receipt.json'),
        report_sha256=io.sha(report),CPU_diagnostic_and_all_audits_seconds=cost['CPU_diagnostic_and_all_audits_seconds']))


if __name__=='__main__':main()
