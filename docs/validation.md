# 验证与交叉验证协议

## 数据验证（只读）

```bash
python scripts/validate_dataset.py --data data/physical_semantics --check-source-hash
```

验证顺序：manifest/schema → NPZ checksum →（可选）上游 Parquet checksum → shape/dtype 一致性 →
NaN/Inf 扫描 → NOCS ∈ [0,1] → 接触 mask 二值 / 冲量非负 / 支撑顶点数界内 → 稳定性标签二值 →
episode 边界 `termination_reason` 非 0 → 同 episode 内 next-state 对齐、边界保持末帧。

退出码：`0` 通过；`2` schema/文件缺失；`3` checksum 不匹配；`4` 内容违规。
报告写入 `validation_report.json`（可用 `--report` 指定路径）。

仓库自带 120 帧样本同样可验证（无需完整数据集）：

```bash
python scripts/validate_dataset.py --data tests/fixtures/mini_dataset
```

## 训练/测试交叉验证

- 任务：由当前增强观测和物理语义预测下一帧 14 维双臂动作。
- 分组：`dataset_id × 10^6 + episode_index`；固定 seed 打乱 group 后轮转分配 5 折（不跨集泄漏）。
- 归一化：每折只用训练 groups 的均值/标准差；标准化后裁剪到 [-8, 8]。
- 训练：GPU（BF16 AMP）MLP，4 epochs、batch 4096；保存每折 checkpoint、loss 历史与吞吐。
- 对照：persistence baseline（`prediction = augmented_state`）。
- 指标：MSE、MAE、R²，逐折 + mean±std；结果写 `experiments/<run>/summary.json`。

## 随仓库发布的参考结果

- `reports/cross_validation_summary.json` — 本次发布引用的运行（脱敏路径）
- `reports/fold_assignments.json` — 可复现的 group → fold 划分

| 指标 | 模型 mean±std | baseline mean±std |
|---|---:|---:|
| MSE | 0.02886 ± 0.00684 | 0.06934 ± 0.00721 |
| MAE | 0.06630 ± 0.00648 rad | 0.11768 ± 0.00860 rad |
| R² | 0.84134 ± 0.03479 | 0.61796 ± 0.03245 |

## 解释限制

- "84%" 是回归拟合水平（解释的动作方差比例），不是分类准确率、平衡成功率或 sim-to-real 指标。
- 稳定标签在 Dex3/DiverseManip 中接近全 1（这些数据集只记录双臂，下肢沿用 `stand` 基准）；
  WBT 含完整下肢/基座，区分度更高（约 83%）。
- 若模型不优于 persistence baseline，应如实报告；训练 loss 下降不等于控制性能提升。
