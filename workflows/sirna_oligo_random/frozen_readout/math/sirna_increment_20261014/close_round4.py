"""Describe completed R4 only; no fitting, model selection, or source inference."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R4 = ROOT / 'round4'
RESULTS = R4 / 'results'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(text)


verification = read(R4 / 'verification.json')
assert verification['passed']
assert verification['complete_sha256'] == sha(RESULTS / 'complete.json')
complete = read(RESULTS / 'complete.json')
for name, digest in complete['files_sha256'].items():
    assert sha(RESULTS / name) == digest, name
executions = [read(p) for p in R4.glob('execution_*.json')]
assert len(executions) == 1 and executions[0]['returncode'] == 0
assert executions[0]['GPU_hours'] == 0
summary = table(RESULTS / 'summary.csv')
selections = table(RESULTS / 'selection.csv')
predictions = table(RESULTS / 'oof_predictions.csv')
goals = read(RESULTS / 'goal_summary.json')
assert len(summary) == 24 and len(selections) == 96 and len(predictions) == 22788
settings = ('Huesken', 'Mixset', 'Takayuki', 'Inter_Hu_to_Mix')
lookup = {(r['setting'], r['condition']): r for r in summary}
primary = 'core_real_Adaptive40'
diagnostics = []
for setting in settings:
    real = lookup[setting, primary]
    init = lookup[setting, 'initial_core_Adaptive40']
    selected = [r for r in selections if r['setting'] == setting and r['condition'] == primary]
    zero_folds = {int(r['fold']) for r in selected if float(r['weight']) == 0.}
    zero_rows = sum(r['setting'] == setting and r['condition'] == primary and int(r['fold']) in zero_folds for r in predictions)
    diagnostics.append(dict(setting=setting,
        primary_vs_initial_PCC_delta=float(real['PCC'])-float(init['PCC']),
        primary_vs_initial_relative_PCC_pct=100*(float(real['PCC'])/float(init['PCC'])-1),
        source_weights=dict(Counter(r['weight'] for r in selected)),
        TD_only_folds=len(zero_folds), total_folds=len(selected),
        TD_only_rows=zero_rows, n=int(real['n'])))
mean_mse = sum(float(lookup[s, primary]['MSE_relative_reduction_pct']) for s in settings)/4
report = ['# R4：局部来源表示重新获得部分增益，但未达到5%目标', '',
    '2026-10-15。预先指定主方案为Core设计靶位点表示＋TD24，经训练内PCA16与两组独立正则形成一个40D线性头。三种来源、两种收缩方式均完整报告；没有组合来源预测、占用GPU或新训练来源网络。', '',
    '本轮与R1–R3更换了来源架构和表示，不能把变化单独归因于靶位点池化、Core屏蔽或gate。Core/Full均是既有seed42、8轮32728更新的机制实验终点，不是旧发布source43。', '',
    '| 设置 | 固定TD24 PCC | Core＋TD24 PCC | PCC相对变化 | MSE相对下降 |',
    '|---|---:|---:|---:|---:|']
for setting in settings:
    r = lookup[setting, primary]
    report.append(f"| {setting} | {float(r['TD24_PCC']):.6f} | {float(r['PCC']):.6f} | {float(r['PCC_relative_gain_pct']):+.4f}% | {float(r['MSE_relative_reduction_pct']):+.4f}% |")
report += ['', f'四设置PCC相对增益的等权均值为 **+2.6812%**，低于固定5%目标；MSE相对下降等权均值为{mean_mse:+.4f}%。Mixset及Huesken→Mixset仍下降，没有达到普遍非劣。', '',
    '| 来源 | 相同正则Uniform40 | 独立收缩Adaptive40 |', '|---|---:|---:|']
for source in ('core_real', 'full_real', 'initial_core'):
    g = {r['condition']: r for r in goals}
    report.append(f"| {source} | {g[source+'_Uniform40']['mean_relative_PCC_gain_pct']:+.4f}% | {g[source+'_Adaptive40']['mean_relative_PCC_gain_pct']:+.4f}% |")
report += ['', '以上均相对同一TD24基线。initial_core是Core精确更新前初始化，但已包含公共RNA基础模型，不能称完全随机模型。', '',
    '| 设置 | 真实miRNA监督相对Core初始化的PCC绝对差 | 主方案退回TD24的外折 | 对应测试行 |', '|---|---:|---:|---:|']
for d in diagnostics:
    report.append(f"| {d['setting']} | {d['primary_vs_initial_PCC_delta']:+.6f} | {d['TD_only_folds']}/{d['total_folds']} | {d['TD_only_rows']}/{d['n']} |")
report += ['',
    'Core训练后的来源在Huesken和Takayuki优于精确初始化，但在Mixset及跨队列未显示增量。因此可以说存在部分与miRNA监督相关的可迁移信息，不能说这种信息已跨队列稳定成立。这里只是单个来源种子、已反复开发数据上的诊断，不是生物因果证据或盲外部确认。', '',
    '主方案仅Mixset两个外折选择TD-only，其他设置均保留来源。故目前瓶颈不能只解释为gate保守。相反，Full的自适应收缩在Takayuki由Uniform的+6.6746%降至−0.2655%，提示训练内最优选择未必可靠外推；这不是继续扩大搜索网格的证据。选择机制的方差/分布偏移需进一步受控诊断。', '',
    '来源信息有效与否、如何缩减冗余和如何选择使用强度需要分开。下一步应首先利用已保存的训练内OOF残差检查补充信息的稳定性，再决定是否值得引入更复杂读出；不能根据每个测试集事后选择Core、Full或TD24来拼一个“达标模型”。', '',
    '验收：6000个内层正规方程、96个最终头、22788行逐行预测全部独立核对。TD-only五个候选与原TD24基线的最大逐行差8.33e−15；所有标准化/PCA仅当前训练分区拟合，原外层validation未使用。最大归一正规方程残差1.07e−16。源码/数据/来源导出/实际checkpoint与精确initial均绑定SHA。', '',
    f"CPU控制器墙时{executions[0]['CPU_wall_seconds']:.3f}s，独立验收{verification['CPU_wall_seconds']:.3f}s，另有进程启动开销；新GPU耗时0、来源前向0。以上不含既有来源训练和特征导出的成本，不能称端到端训练仅需几秒。", '',
    f"科学锁：`{sha(R4/'lock.json')}`。", f"完成回执：`{sha(RESULTS/'complete.json')}`。", f"独立验收：`{sha(R4/'verification.json')}`。", '',
    '全部六条件逐设置指标在round4/results/summary.csv；所有候选和选择、逐行预测、source/initial身份在同目录及plan.json。失败目标和所有负结果保留。', '',
    '限制：原来源keep-overlap、context暴露及四设置反复开发；当前purged sequence不是gene隔离，两个Mixset设置并非独立队列；单source seed42，没有利用当前结果修改划分或基线。目标继续active，未达标。', '']
save(ROOT / 'ROUND4_RESULTS.md', '\n'.join(report))
derived = dict(kind='post_result_description_only', primary=primary,
    source_result_sha256=sha(RESULTS/'complete.json'), verification_sha256=sha(R4/'verification.json'),
    primary_mean_MSE_relative_reduction_pct=mean_mse, primary_vs_initial_and_routing=diagnostics,
    no_training=True, no_new_model_selection=True, GPU_hours=0)
save(R4 / 'derived_description.json', json.dumps(derived, ensure_ascii=False, indent=2)+'\n')

workspace = ROOT.parents[1]
manifest_path = workspace / 'DuplexFM-main/results/siRNA/oligo_p13_heads/feature_manifest.json'
checkpoint_path = workspace / 'DuplexFM-main/checkpoints/source/seed43/best_feature_level_moe.pt'
manifest = read(manifest_path)
actual_sha = sha(checkpoint_path)
assert actual_sha == manifest['source_provenance']['source_checkpoint_sha256']
source_audit = dict(date='2026-10-15', kind='source_identity_and_bounded_inventory_note',
    source43_checkpoint=str(checkpoint_path), actual_checkpoint_sha256=actual_sha,
    published_feature_manifest=str(manifest_path), manifest_sha256=sha(manifest_path),
    original_feature_dimension=160, new_source_forward=0, GPU_hours=0,
    reusable_four_setting_source43_160D_cache='not located in bounded local release/research inventory; absence not established globally',
    source42_slot_cache='expert_selection_20261014/results_download/verified/features; different source identity',
    r4_sources='verified core_six_settings_20261011 exports; different source architecture, not source43 reproduction',
    next_requirements=['If source43 cache found, verify checkpoint hash and all input identities before reuse.',
        'If CPU export needed, qualify the original native exporter and dependencies first; no unreviewed path/checkpoint substitution.'],
    no_training=True, no_download=True)
save(ROOT / 'SOURCE_ASSET_AUDIT_20261015.json', json.dumps(source_audit, ensure_ascii=False, indent=2)+'\n')
save(ROOT / 'SOURCE_ASSET_AUDIT_20261015.md', '\n'.join([
    '# 旧source43资产可比性核查', '',
    '2026-10-15。原发布source43实际权重已按特征manifest核验，SHA为`'+actual_sha+'`。', '',
    '在本轮有界的本地release/research盘点中，尚未找到能直接用于当前四设置的、具有完整核验记录的source43 160D缓存。这不是对全盘/所有归档的不存在证明，也不是模型文件缺失。', '',
    '现有五专家160D缓存属于source42；本轮R4使用的是Raw41/Core机制实验的设计靶位点192D缓存，来源架构和训练合同都不同。不能把它们替代source43后称为论文严格复现。', '',
    '旧native CPU导出入口依赖原RNA编码器、物理输入、配置与路径，未经资格审计不能仅替换checkpoint直接运行。当前未做新导出、下载、来源训练或GPU计算。具体文件身份见同名JSON。', '']))
print(json.dumps(dict(report='ROUND4_RESULTS.md',primary_mean_gain_pct=g[primary]['mean_relative_PCC_gain_pct'],
    target_met=False,source43_checkpoint_verified=True),ensure_ascii=False))
