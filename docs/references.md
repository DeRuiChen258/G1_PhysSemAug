# 参考论文与实现映射

本文档只记录论文中实际读到、且本项目确实使用的表示思想。论文算法与结论没有被包装成 G1 动力学的已验证事实。
本仓库不分发论文 PDF，请通过 arXiv 获取原文。

## [R1] NOCS — *Normalized Object Coordinate Space for Category-Level 6D Object Pose and Size Estimation*（CVPR 2019，arXiv:1901.02970）

- 问题（第 2 页，Sec. 1–2）：从 RGB/RGB-D 估计类别级物体的规范坐标、实例 mask、6D pose 和尺寸，以支持未见实例泛化。
- 核心表示（第 3 页，Sec. 2–3）：NOCS 是包含在单位立方体中的共享规范空间，`(x,y,z)∈[0,1]^3`；
  通过统一朝向与尺度建立跨实例对应关系。
- 与本项目的映射：`part_nocs[N,14,3]` 将 G1 14 个双臂刚体 body 原点在 torso frame 的局部位置
  按固定 G1 部件包围范围映射到 `[0,1]^3`。它是物理语义坐标字段，不是 RGB NOCS 网络输出。
- 不直接迁移：G1 身体不是论文中的类别级桌面物体；本实现不从相机像素预测 NOCS，
  也不声称仅靠 NOCS 恢复 G1 6D pose。

## [R2] ANCSH — *Category-Level Articulated Object Pose Estimation*（CVPR 2020，arXiv:1912.11913）

- 问题（第 1–2 页，Sec. 1）：为可动对象构造层级规范空间，预测部件分割/坐标、部件 pose/scale、
  关节参数和关节状态。
- 核心表示（第 2 页图示；第 5 页 Sec. 4.1）：根级 NAOCS 规范化对象朝向/尺度/关节状态，
  叶级 NPCS 规范化部件 pose/尺度；关节约束把部件联系起来。
- 与本项目的映射：`part_nocs` 对应部件级规范坐标；`part_parent_offset` 对应父子部件层级几何；
  `joint_axis_torso`、`joint_angle`、`joint_limit_fraction` 对应可动部件的关节语义。
- 不直接迁移：G1 是已知 MJCF 的机器人本体，不是未知类别的单深度点云估计；本实现不训练 PointNet++，
  不把人体关节当作普通物体类别 6D pose。

## [R3] P-learning — *Inverting the Bellman Equation: From Q-Values to World Models*（arXiv:2606.21173）

- 问题（第 1–2 页，Sec. 1）：从足够丰富的 goal-conditioned Q / 策略 / 奖励中反演与 Bellman 方程
  一致的隐式 transition / world model。
- 方法与边界（第 4–6 页，Sec. 3–4）：离散情形涉及矩阵秩/列可分性与 Moore–Penrose 伪逆；
  连续情形依赖目标覆盖条件；Q 近似误差会传递到 world-model 误差。
- 与本项目的映射：生成数据保留 `observation_state`、`next_observation_state`、`action`、`reward`、
  `termination_reason`，可作为后续 transition/world-model 实验输入；本仓库只做监督动作预测与物理一致性检查。
- 不直接迁移：没有构造多目标 Q/策略集合，也没有实现 P-learning 反演；不能据此宣称
  G1 连续高维动力学被唯一恢复或 sim-to-real 已证明。

## 依赖关系

```text
G1 MJCF + recorded joints
          │
          ├── MuJoCo forward → body/contact/CoM/ZMP
          ├── R1-inspired   → part_nocs
          └── R2-inspired   → part_parent_offset + joint semantics

recorded transition fields ───────────────→ future P-learning work (R3)
```
