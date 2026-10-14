# 发布结果

| 文件 | 内容 |
|---|---|
| `cross_validation_summary.json` | 5-fold episode 分组交叉验证完整结果（逐折 loss、指标、吞吐、GPU/软件版本；路径已脱敏） |
| `fold_assignments.json` | group（`dataset_id×10^6 + episode_index`）→ fold 的固定划分，用同 seed 可精确复现 |

说明：

- 运行规模：959,718 帧、1,186 个 episode group、393 维特征、5 折。
- 模型与 baseline 的对照、含义边界见仓库根 [README](../README.md) 第 9 节与 [docs/validation.md](../docs/validation.md)。
- 该结果对应的数据与 checkpoint 体积约 1 GB，不随仓库分发；按 README 第 11 节命令可在本地重建。
