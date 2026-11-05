"""Render the already independently verified results; no fits or selection."""
import csv,hashlib,json
from pathlib import Path

R=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    audit=read(R/'local_verification.json')
    assert audit['passed'] and audit['results_complete_sha256']==sha(R/'results/complete.json')
    rows=table(R/'results/summary.csv');sels=table(R/'results/selection.csv')
    settings=['Huesken','Mixset','Takayuki','Inter_Hu_to_Mix','wang2009','alas1_patent']
    labels=dict(zip(settings,['Huesken','Mixset','Takayuki','Huesken→Mixset','Wang','ALAS1']))
    vals={(r['condition'],r['setting']):r for r in rows}
    conditions=['TD24_Ridge','Expert5_Ridge','Expert5_GroupSparse','ExpertsOnly_Ridge',
                'ExpertsOnly_GroupSparse','SelectedExperts_Ridge','SelectedExperts_TD24_Ridge']
    names=['TD24','原全专家+TD24 Ridge','原同时选择+TD24','无TD24全专家Ridge',
           '无TD24稀疏选择','固定开关、无TD24 Ridge','固定开关、最后加TD24']
    lines=['# 先单独拟合DuplexFM，再加入TD24：六设置结果', '',
           '2026-10-14。全部在本地CPU单线程完成，GPU新增使用0。复用已核验的原五专家seed42特征，来源编码器/LoRA/开放性没有更新。本报告中的retrain指siRNA专家选择和效率头重训，不能称为来源模型联合微调。', '',
           '**主要结果：全关专家从前轮8/26折降至0/26折，但24/26折选择全部五专家。保留更多专家没有带来普遍改善；最终结果在四个设置与原全专家Ridge相同，其余两个略差。**', '',
           '## 实际训练流程', '',
           '第一阶段只使用五专家160D及fit/validation的siRNA标签，选择分组稀疏头和五个开关。开关及SHA先落盘。第二阶段固定这些开关，分别用保留专家、保留专家+TD24训练普通Ridge；不输入第一阶段预测，不做预测叠加，来源参数冻结。', '',
           '**第二阶段重新拟合效率系数，继承的是开关，而不是第一阶段的效率系数或新编码表示。** 当五专家全部开启时，它与原184D全专家Ridge具有相同的输入、目标、alpha网格和划分，因而会回到相同解。它并没有实现通过来源微调改变表示的另一种两阶段方法。', '',
           '## PCC完整比较', '',
           '|设置|'+'|'.join(names)+'|', '|---|'+'---:|'*len(names)]
    for setting in settings:
        lines.append('|'+labels[setting]+'|'+'|'.join(f"{float(vals[c,setting]['PCC']):.4f}" if vals[c,setting]['PCC'] else '/' for c in conditions)+'|')
    lines += ['', '## 最后加入TD24后的MSE变化', '',
              '以下为MSE变化率=(新MSE/对照MSE−1)×100%；负值是改善，正值是恶化。', '',
              '|设置|相对TD24|相对原全专家Ridge|相对原同时选择|相对同开关无TD24|', '|---|---:|---:|---:|---:|']
    for setting in settings:
        new=float(vals['SelectedExperts_TD24_Ridge',setting]['MSE'])
        changes=[100*(new/float(vals[c,setting]['MSE'])-1) for c in ['TD24_Ridge','Expert5_Ridge','Expert5_GroupSparse','SelectedExperts_Ridge']]
        changes=[0. if abs(v)<1e-8 else v for v in changes]
        lines.append('|'+labels[setting]+'|'+'|'.join(f'{v:+.2f}%' for v in changes)+'|')
    lines += ['', '## 专家保留情况', '', '|设置|各折保留专家数|关闭的专家|', '|---|---|---|']
    experts=['accessibility','utr_cross_attention','score_matrix_attention','biological_priors','intarna_energy_geometry']
    for setting in settings:
        selected=[v for v in sels if v['setting']==setting and v['condition']=='ExpertsOnly_GroupSparse']
        off=[f"fold{v['fold']}: "+','.join(k for k in experts if v[k]=='0') for v in selected if any(v[k]=='0' for k in experts)]
        lines.append('|'+labels[setting]+'|'+'、'.join(v['active_groups'] for v in selected)+'|'+('; '.join(off) if off else '无')+'|')
    lines += ['', '## 解读', '',
        '- TD24确实改变了选择问题：前轮选择的是专家在TD24之外的条件增量；这一轮选择的是专家单独解释siRNA效率的能力。二者不同，专家独立有用不保证在TD24之外仍有增量。',
        '- Huesken与Huesken→Mixset：全部专家保留，最终回到原全专家Ridge；比原同时稀疏选择更差。之前的收缩/关闭并非完全有害。',
        '- Mixset：从五折全部退回TD24变成每折保留4–5个专家，但最终PCC从0.5163降至0.4522，MSE比TD24增加12.81%。本轮不支持“强迫保留来源信息就能改善迁移”。',
        '- Takayuki与Wang：相对原同时选择，最终MSE分别下降3.06%与3.28%，保留源专家帮助避免前轮部分抑制；但只恢复到原全专家Ridge附近，没有超过其表现。',
        '- ALAS1：专家独立Ridge PCC约0.3955，加入TD24后降到0.2905；同开关下MSE增加37.31%，比TD24增加75.05%。增加信息并不保证有限样本下验证选参可靠。没有因这一结果改alpha或追加网格。',
        '- 不能把改善/退步全部归因于出现顺序：最终还去掉了组惩罚，并重新拟合效率头。无TD24稀疏、同开关普通Ridge、同开关加TD24这三个中间结果均完整保留。', '',
        '## 下一步建议与限制', '',
        '这轮优先回答了“是否被TD24挤掉”这一窄问题。结果显示先独立选择能改变开关，却没有稳定的最终增量。当前更值得解决的是小样本模型选择和“独立有用”与“条件增量”不一致的问题，而不是单纯要求更多专家开启。', '',
        '如果下一步仍保留冻结来源，可预先固定训练池内分组交叉验证来选择开关及Ridge，而不是继续根据已开封测试集调稀疏系数。若要让第一阶段真正改变来源表示，则是另外的encoder/adapter微调实验；本次没有实施、不能声称已验证。', '',
        '只有原来源seed42。既有数据反复开发、来源keep-overlap及context暴露、purged sequence不是gene隔离等限制保留。六设置含Mixset重复应用，不能当六个独立数据集，不声称统计显著或SOTA。不用历史Raw41/Core+Access数值替代当前原五专家同条件对照。', '',
        '## 验收与成本', '',
        '- 10项合成测试通过。全部26折、104个新头、1040个候选评价、16628条新OOF预测通过独立重算；旧12471条预测按原SHA复用。1040包含解析端点和截距候选，不等同1040次迭代训练。',
        f"- 独立验收最大组KKT残差 {audit['max_group_KKT']:.3e}（逐候选按预定相对容差通过）；Ridge正规条件残差 {audit['max_Ridge_normal_residual']:.3e}。全部选择、开关、逐行预测、42行汇总指标通过；旧输入和结果字节未变。",
        f"- 新运行控制器墙时 {audit['execution_wall_seconds_including_failures']:.3f} 秒（包含加载/校验/拟合/保存）；外部命令墙时8.378秒包含Python启动。独立验收内部耗时 {audit['wall_seconds']:.3f} 秒，外部命令2.848秒。",
        '- 注册前一次隔离Python导入失败约0.949秒，未开始真实拟合；原文件和失败记录保留。合成测试内部1.09秒。来源特征直接复用，原提取耗时529.11秒属于前轮，不计成本为本轮新增，也不能隐去为“从原始序列只需7秒”。',
        '- 本地CPU、CUDA不可见，没有GPU训练/推理，没有重跑来源缓存。结果留在本地；本次SSH握手失败后未向5090上传新结果。', '',
        '## 文件', '',
        '- `PROTOCOL.md`：运行前固定协议；`experiment_lock.json`：代码/输入/旧结果/环境SHA。',
        '- `results/summary.csv`：七条件全部指标；`results/selection.csv`：新四条件逐折开关。',
        '- `results/folds/*/fold*/stage1.json`、`selection_lock.json`、`stage2.json`：全部候选、标准化和最终系数。',
        '- `local_verification.json`：独立数值验收；`CODE_AUDIT.md`：代码审计。', '']
    report=R/'RESULTS.md'
    if report.exists():raise FileExistsError('Do not overwrite a published report')
    report.write_text('\n'.join(lines),encoding='utf-8')
    done=dict(complete=True,scientific_scope='frozen five-expert siRNA readout retraining, then TD24',
              report_sha256=sha(report),local_verification_sha256=sha(R/'local_verification.json'),
              experiment_lock_sha256=sha(R/'experiment_lock.json'),results_complete_sha256=sha(R/'results/complete.json'),
              source_updates=0,new_GPU_hours=0,device='local CPU')
    with (R/'COMPLETED.json').open('x',encoding='utf-8') as f:json.dump(done,f,indent=2);f.write('\n')
    print(report)

if __name__=='__main__':main()
