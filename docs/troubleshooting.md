# 故障排查

## 生成时报缺少 pyarrow / mujoco

生成端需要 `pyarrow` 与 `mujoco` 同时可用：

```bash
python -c "import pyarrow, mujoco; print(pyarrow.__version__, mujoco.__version__)"
```

若缺依赖：`python -m pip install -r requirements.txt`（生成端不需要 CUDA 版 torch）。

## 配置路径报“仍包含未解析的环境变量”

生成器要求 `robot.xml` 与 `data.root` 能解析为真实路径：

```bash
export G1_MJCF=/path/to/mujoco_menagerie/unitree_g1/scene.xml
export G1_ACTION_ROOT=/path/to/G1_Action
```

或直接把配置里的 `${G1_MJCF}` / `${G1_ACTION_ROOT}` 换成绝对路径。

## GPU 训练直接退出

训练是 GPU-only：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

若输出 `False`，先修复驱动/CUDA 环境；脚本不会静默切换到 CPU。

## MuJoCo 模型加载失败

确认 `scene.xml` 与其 `assets/` 目录保持官方仓库的相对结构；生成器在首个 forward 前就会因
缺失 body / joint / keyframe 直接报错，并打印缺失名称。

## 输出目录已存在

生成器默认拒绝覆盖已有 `samples.npz`；确认目标目录后显式加 `--force`。验证器只读，不会修改数据。
