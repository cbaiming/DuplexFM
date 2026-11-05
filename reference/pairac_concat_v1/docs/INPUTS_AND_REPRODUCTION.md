# 输入与复现边界

本包只提供代码，以下真实资产必须另行准备，不会自动下载或生成替代物：

- `assets/mrnabert/`、`assets/rnafm/`：公共编码器及其tokenizer、配置和实现。`foundation_reference.json`仅保留原文件哈希，无文件内容。
- `assets/mirbench/manifest.json`、`data/*.npy`、`accessibility.npy`：来源样本映射、标签、划分和冻结反应性缓存。
- 完成来源训练的注册记录及initial/terminal checkpoint；本包不声称能从seed重建字节相同的历史initial。
- 四设置的benchmark表、原内外层划分和训练target25组定义、固定TD24、source192导出、标签变换契约及基线引用。

`PairStore`预期token 0/1/2/3对应ACGU，PAD=4；target/guide分别按唯一序列存储，并用target_ids/guide_ids映射样本。
guide_lengths及可选target_lengths指定真实长度。存储中真实序列左对齐，collate时target居中补至60。原accessibility按真实target坐标存储。
必须提供SHA清单。没有真实反应性时不能填零后声称复现了Full；没有真实划分时不能用随机划分冒充已报道结果。

siRNA预处理保留历史activity/1.341等响应契约，R55入口读取准备好的连续label及二分类y，不在拟合时再次归一。
guide19 target57及19:38设计核心契约固定；未知碱基样本排除。四设置为Huesken、Mixset、Takayuki、Huesken→Mixset。

## 路径配置

来源程序使用显式`--root`、`--assets`；导出参考程序额外支持：

|变量|用途|
|---|---|
|`DUPLEXFM_ASSETS`|公共编码器和准备好的来源缓存|
|`DUPLEXFM_FULL_ROOT`|含原源码、注册记录和checkpoint的Full运行根|
|`DUPLEXFM_ABLATION_ROOT`|消融运行根|
|`DUPLEXFM_DATA_ROOT`|benchmark和冻结输入缓存的父目录|

`sirna_transfer/full/readout.py`等保留历史依赖布局，实际benchmark、plan、输入哈希与参考结果缺失时会拒绝运行；它们是完整执行逻辑的审阅入口，不是自动生成新协议的通用CLI。
`reference_full.json`中的`${PAIRAC_FULL_REFERENCE_ROOT}`是审阅占位符；原注册锁不随包分发，不要将这份副本拿来恢复已有AFS运行。

## 已知运行修复

补充消融曾在并发修改状态文件时触发备份`Unstable backup input`。
`execution_reference/backup_compat.py`及`parallel_compat_run.py`保留实际使用的备份兼容层，原科学文件不变。
兼容层要求原锁和ready证据；这些属于特定运行状态，不包含在代码包。
因此代码包不替代正式AFS作业发行版。未来新的训练部署应重新注册并验收运行设置，不能伪造历史锁以通过检查。
