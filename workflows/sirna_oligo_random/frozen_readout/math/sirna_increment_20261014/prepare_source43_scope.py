"""Read known assets only; this is not an exporter or a scope change."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
prior=read(ROOT/'SOURCE_ASSET_AUDIT_20261015.json')
checkpoint=Path(prior['source43_checkpoint']);manifest=Path(prior['published_feature_manifest'])
assert checkpoint.is_file() and sha(checkpoint)==prior['actual_checkpoint_sha256']
assert manifest.is_file() and sha(manifest)==prior['manifest_sha256']
m=read(manifest);assert m['source_provenance']['source_checkpoint_sha256']==sha(checkpoint)
assert m['concat_feature_dim_before_td24']==160 and m['td24_feature_dim']==24 and m['concat_feature_dim']==184
config=checkpoint.parent/'run_config.json';runtime=BASE/'.tools/python5060/python.exe'
result=dict(status='metadata_prepared_scope_question_pending',source_export_launched=False,scope_changed=False,
    new_source_forward=0,GPU_hours=0,checkpoint=dict(path=str(checkpoint),bytes=checkpoint.stat().st_size,sha256=sha(checkpoint)),
    manifest=dict(path=str(manifest),sha256=sha(manifest)),native_feature_dimension=160,with_TD24_dimension=184,
    original_expert_names=m['expert_names'],runtime_file_exists=runtime.is_file(),runtime_import_verified=False,
    original_config=dict(path=str(config),exists=config.is_file(),sha256=sha(config) if config.is_file() else None,
        expected_sha256=m['source_provenance']['source_run_config_sha256']),
    current_goal_constraints='CPU, existing verified source42 caches; requested expansion remains unanswered',
    required_before_export=['explicit scope answer','native source43 CPU export interface and dependencies audit',
        'input IDs and guide19/target57-to-native-contract alignment','reuse physical features only after full input/coordinate identity checks',
        'freeze protocol, response, split, TD baseline and feature extraction point before fitting',
        'CPU-only bounded parity/throughput qualification before full export'])
with (ROOT/'SOURCE43_SCOPE_PREPARATION_20261015.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
config_status='存在，尚需与原清单核对后接入' if config.is_file() else '当前已知路径未找到，需定位原配置'
text=f'''# 原论文source43桥接：范围确认前的资产准备

2026-10-15。只读复核已有16,425,027字节checkpoint和原feature manifest，未加载模型、推理、训练、下载或修改当前目标。扩展到source43的CPU导出范围已向用户询问，尚未获得回复；不会因为时间经过而视为同意。

原权重SHA：{sha(checkpoint)}。原特征清单SHA：{sha(manifest)}。二者仍与先前来源审计及原表来源一致。原生表示是五个32D专家适配隐藏输出，共160维，加TD24后184维；它不同于当前Core/Full的192D位点表示。

原配置状态：{config_status}。本地Torch环境的Python入口存在性为{runtime.is_file()}，本次没有导入Torch或声称完整环境/基础模型依赖已经通过验收。完整四设置source43缓存先前有界盘点未定位；这不是全盘不存在证明，不重复广泛搜索或下载。

若用户同意，先审计原生导出器、foundation/accessibility/物理输入及坐标合同，再以CPU小样本验证形状、确定性、特征身份和吞吐。所有可复用输入要逐行身份一致，不能仅更换checkpoint路径就沿用不兼容缓存。实际完整导出前另登记资源预算和可恢复分块协议。

桥接比较固定当前2196/469/664输入与purged划分、标签预算和TD24_Ridge_CV分母。保持旧模型原生160D接口，先与相同架构source42作同读出比较；新192D Core另列，不能把架构/位点池化变化当成来源种子变化。原Table4行级划分仅用于后续历史解释，不能用它替代当前5%目标。

本文件是待确认方案，不是新的性能结果或训练授权。R20已完整闭合；目标未完成，当前没有本任务活进程。
'''
with (ROOT/'SOURCE43_SCOPE_PREPARATION_20261015.md').open('x',encoding='utf-8') as f:f.write(text)
print(json.dumps(dict(metadata_ready=True,new_forward=0,config_exists=config.is_file(),config_hash_matches=(sha(config)==m['source_provenance']['source_run_config_sha256']) if config.is_file() else None)))
