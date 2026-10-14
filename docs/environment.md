# 环境与依赖

实测时间：2026-10（Ubuntu 26.04 LTS）。生成与训练均在本地离线完成，未连接真机。

## 实测环境快照

| 项目 | 实测值 |
|---|---|
| OS | Ubuntu 26.04 LTS (x86_64) |
| CPU 工具链 | GCC 15.2 |
| Python | 3.12 |
| MuJoCo | 3.13.0（pip） |
| PyTorch（训练） | 2.14.0+cu130，`torch.cuda.is_available() == True` |
| CUDA | toolkit 13.2（nvcc 13.2.86） |
| GPU | NVIDIA GeForce RTX 5070 Laptop，8 GB 显存 |
| Parquet reader | pyarrow 25.x |
| G1 模型 | `mujoco_menagerie/unitree_g1/scene.xml`（`nq=36`，29 actuators，keyframe `stand`） |

环境自检：

```bash
nvidia-smi
nvcc --version
python -c "import mujoco, torch, pyarrow; print(mujoco.__version__, torch.__version__, torch.cuda.is_available(), pyarrow.__version__)"
```

## 依赖矩阵

| 依赖 | 用途 | 版本下限 | 说明 |
|---|---|---|---|
| numpy | 数组、NPZ、RNG | 1.26 | 生成 + 训练 |
| mujoco | `mj_forward`、接触力、CoM | 3.3 | 仅生成端 |
| pyarrow | 读取 LeRobot Parquet | 15.0 | 仅生成端 |
| PyYAML | 配置解析 | 6.0 | 生成 + 训练 |
| torch | GPU 训练 | 2.6（CUDA wheel） | 仅训练端 |

> 生成端与训练端可以使用不同的解释器：生成需要 `pyarrow + mujoco`，训练需要 CUDA 版 `torch`。
> 所有脚本都通过命令行显式指定解释器运行，不假设任何固定环境名。

## CPU / GPU 分工

- CPU：Parquet 解码、episode 顺序整理、MuJoCo `mj_forward` / 接触语义、NPZ 压缩写盘。
- GPU：特征批量归一化、BF16 AMP 训练、5-fold 测试推理。
- GPU-only 门禁：训练脚本发现 CUDA 不可用立即退出，不提供无 CUDA 的 trainer 降级。
- 视频流默认不读取（G1_Action 主体为 35 GB AV1 视频），避免无关解码开销。

## 边界

- G1_Action 不含可直接对齐的 MuJoCo 原始力传感器真值；接触冲量与 ZMP 是给定 qpos/扰动下的前向推导。
- 生成的 manifest 会记录运行机器的绝对路径（命令、源文件位置），用于本机复现；
  跨机器复现时按 README 第 11 节重新生成即可。
