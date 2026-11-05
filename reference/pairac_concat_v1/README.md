# DuplexFM PairAC — code review snapshot

这是供 CS 合作者审查的 **PairAC 三分支直接拼接版本**，日期 2026-10-22。
选定模型为可学习有方向碱基对向量 + Cross + 冻结的逐碱基结构反应性代理。
**不是 MoE 版本，不含 BIO/Inta 分支或手工乘积通道。**

包内仅有源码、合成测试、依赖说明、非敏感配置/哈希清单和文档。
没有真实序列、样本标签、划分 ID、预测结果、模型参数、数据缓存、SSH/Dev Tunnel 配置或凭据。
本包未执行上传、提交 Git 或创建 GitHub 仓库。

## 建议阅读顺序

1. `source_full/code/duplexfm_mechanism/model.py`：PairACFusion、在线编码和 source192。
2. `source_full/code/duplexfm_joint/backbones.py`：公共编码器、fresh LoRA、Q/K投影。
3. `source_full/code/duplexfm_joint/data.py`：真实长度、居中PAD、坐标和输入完整性。
4. `source_full/code/duplexfm_joint/model.py`：共享CellNorm、局部卷积、反平行索引。
5. `source_full/code/duplexfm_mechanism/protocol.py`、`training_hourly.py`：固定预算与断点。
6. `ablations/*/code/duplexfm_mechanism/model.py`：同形状置零消融。
7. `sirna_transfer/full/readout.py`：固定R55嵌套选择、拟合、评价和独立验收。
8. `sirna_transfer/runtime/sirna_increment_20261014/round55/head.py`：实际迁移目标与求解器。

详见 [架构说明](docs/ARCHITECTURE.md)、[审查重点](docs/REVIEW_GUIDE.md)、[输入与复现边界](docs/INPUTS_AND_REPRODUCTION.md)。

## 离线 CPU 审查

Python 3.10 或 3.11；完整来源训练历史环境为 torch 2.6.0，审查包的合成测试在本机 torch 2.7.1 CPU路径验收。
以下只运行生成于测试函数中的合成样本，不下载权重，不连接服务器，不启动真实训练。

```bash
python -m pip install -r requirements-review.txt
python scripts/verify_package.py
python scripts/run_review_tests.py
```

PyTorch 可自行按机器选择相应 CPU wheel；这里不自动安装环境。
原有测试全部保留。两个 `test_fixed_counts_and_scope` 依赖本次不包含的真实划分清单，审查入口明确将其 deselect，不伪造划分通过。
完整编码器、H100/BF16和真实数据性能不由这些合成测试重新认证。

## 目录

|目录|用途|
|---|---|
|`source_full/`|原 Full seed42/43/44 训练和四个miRNA测试集评价代码|
|`ablations/seed42/`|原三个专家置零消融的seed42配置和测试|
|`ablations/seeds43_44/`|原六个补充消融配置和测试|
|`execution_reference/`|补充消融实际使用的稳定备份兼容层及回归测试；原运行锁不分发|
|`sirna_transfer/full/`|原三种子训练后/初始化表示导出、固定R55读出和合成测试|
|`sirna_transfer/ablations/`|剩余seed43/44消融表示导出/读出调度参考|
|`sirna_transfer/runtime/`|R55实际导入的13个数值模块；保留历史目录名以免改写数学代码|
|`scripts/`|包完整性核验、CPU审查入口|

这些来源快照单独导入同名模块，因此测试必须在独立进程中运行，不要把三个训练目录同时加入PYTHONPATH。
保留多份快照是为了可追溯，并不表示需要串行运行所有旧实验。

## 版本与使用范围

`SOURCE_PROVENANCE.json`逐文件记录原相对位置、原SHA及打包SHA。模型、训练算法、数值求解器未重写；仅统一UTF-8/LF、去除私有位置元数据，并把导出器路径改为环境变量/相对路径。
本包**不是原AFS运行目录的副本**。不包含真实运行锁、实验回执或checkpoint；旧 `start_h100_two.sh` 和恢复兼容入口保留作审阅参考，不能把缺少真实输入/锁的本包当成可直接恢复旧H100作业的环境。

`requirements-source.txt`与`requirements-source-torch.txt`保留历史来源环境依赖；不是本次重新联网安装验证的环境锁。
沿用项目已有的 [许可说明](LICENSE-NOTICE.md)，第三方模型、软件和数据的许可不因此改变。

审查优先关注实现正确性、输入/标签隔离、迁移数学目标和消融可比性。模型性能需结合另行提供的实验报告，不能由源码包本身证明。
