# 测试样本数据集（mini fixture）

`mini_dataset/` 是一个 120 帧（4 秒 @30 Hz）的演示数据集，取自 `G1_WBT` 的第 0 个 episode，
用于让单元测试与验证器在**没有完整 G1_Action 数据**的机器上也能离线运行。

## 内容

```text
mini_dataset/
├── index.json
└── G1_Sample/
    ├── manifest.json      # 脱敏 manifest：相对/占位路径 + 数组 schema、单位、profile、checksum
    └── samples.npz        # 43 个数组，约 163 KB
```

数组结构与完整产物完全一致：`observation_state/action/next_*`、`qpos[36]`、`qvel[35]`、
`imu[12]`、`part_world_pos/part_nocs[14,3]`、接触 8 槽、支撑多边形、CoM/ZMP、稳定性标签、
扰动标量（质量/摩擦缩放、延迟、外力、profile id）与 episode 索引字段。

## 与完整 episode 的差异

- 仅为原 episode 的前 120 帧；末帧被显式标记为 episode 边界（`termination_reason=1`），
  且 `next_observation_state / next_action` 在边界上保持自身，以符合验证器的边界规则。
- manifest 中的绝对路径已替换为占位文本，`source_files` 为空列表（跳过源文件哈希校验）。

## 快速校验

```bash
python scripts/validate_dataset.py --data tests/fixtures/mini_dataset
python -m unittest discover -s tests -v
```

## 可选：重新生成同内容样本

生成器使用固定 seed（同一 episode → 同一扰动 profile），因此在相同数据根目录下重放
前 120 帧可得到相同语义内容（末尾三帧边界改写除外）：

```bash
python scripts/generate_physical_semantics.py \
  --config configs/g1_physical_semantics.yaml \
  --dataset G1_WBT --max-frames-per-dataset 120 \
  --output data/smoke
```
