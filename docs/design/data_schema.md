# 物理语义数据字典

每个 `data/physical_semantics/<dataset>/samples.npz` 是同一 dataset 的固定首维数组；`manifest.json` 记录 dtype、shape、单位、源文件 hash、模型 hash 和 profile。

## 坐标和单位

- MuJoCo world frame：右手系，长度 m，速度 m/s，力 N，冲量 N·s，时间 s。
- torso frame：`torso_link` 的 `xpos/xmat`；`part_parent_offset` 和 `joint_axis_torso` 在该层级下表达。
- 四元数：G1_Action 的 base pose 为 `(qx,qy,qz,qw)`，写入 MuJoCo qpos 时转换为 `(qw,qx,qy,qz)`。
- `part_nocs`：由 torso frame 局部 body 原点按模型部件包围范围映射到 `[0,1]^3`；这是 R1/R2-inspired 规范坐标，不是相机像素标签。

## 数组

| 字段 | shape | dtype | 含义 |
|---|---:|---|---|
| `observation_state` / `augmented_state` | `[N,14]` | float32 | 原始/扰动后的双臂关节角 rad |
| `action` / `augmented_action` | `[N,14]` | float32 | 原始/扰动动作 rad |
| `next_observation_state` / `next_action` | `[N,14]` | float32 | 同 episode 下一帧；末帧保持自身 |
| `joint_angle`, `joint_velocity` | `[N,14]` | float32 | MuJoCo qpos 和差分 rad/s |
| `joint_limit_fraction` | `[N,14]` | float32 | 按 MJCF limit 归一化到 `[0,1]` |
| `qpos`, `qvel` | `[N,36]`, `[N,35]` | float32 | 完整 MuJoCo 状态；qpos 含 free-base quaternion，qvel 含关节速度 |
| `imu` | `[N,12]` | float32 | torso/pelvis angular velocity + linear acceleration 传感器值 |
| `part_world_pos` | `[N,14,3]` | float32 | 14 个 arm body 的 world 坐标 m |
| `part_nocs` | `[N,14,3]` | float32 | 部件原点规范坐标 |
| `part_parent_offset` | `[N,14,3]` | float32 | 父 body frame 中的部件偏移 m |
| `joint_axis_torso` | `[N,14,3]` | float32 | 关节轴在 torso frame 的单位向量 |
| `contact_points`, `contact_normals` | `[N,8,3]` | float32 | 最多 8 个接触点/法向 |
| `contact_impulses` | `[N,8]` | float32 | MuJoCo 法向力×dt，N·s |
| `contact_mask`, `contact_body_ids` | `[N,8]`, `[N,8,2]` | uint8/int32 | 接触有效位和两侧 body id |
| `support_polygon` | `[N,8,2]` | float32 | 地面接触点 XY 凸包，m；零填充 |
| `support_count` | `[N]` | int16 | 凸包顶点数 |
| `com`, `com_velocity`, `zmp` | `[N,3]` | float32 | CoM、CoM 速度、接触力加权 ZMP |
| `zmp_valid` | `[N]` | uint8 | 有效地面接触且法向冲量非零 |
| `external_force` | `[N,3]` | float32 | 当前 frame torso 外力 N |
| `mass_scale`, `friction_scale` | `[N]` | float32 | episode profile 缩放 |
| `delay_steps`, `sensor_occlusion` | `[N]` | int64 | 观测延迟和遮挡元数据 |
| `stability`, `future_stability` | `[N]` | uint8 | CoM 投影位于支撑凸包且高度有效；未来窗口全稳定 |
| `reward` | `[N]` | float32 | `stability - 0.5||CoM_xy-ZMP_xy|| - 0.002||dq||` |
| `termination_reason` | `[N]` | int8 | 0 normal，1 episode end，2 当前 frame unstable |
| `validity_flags` | `[N]` | int64 | 有效性位（当前实现：CoM 有限时写入 `0b111`，否则 `0`） |
| `dataset_id`, `episode_index`, `frame_index`, `task_index`, `profile_id` | `[N]` | int64 | 可追溯索引 |

## 版本策略

主版本变更表示数组语义或坐标约定不兼容；次版本新增可选数组；补丁版本只修正文档或校验。读取脚本要求 `schema_version == 1.0.0`，未知版本直接失败。
