# 项目简介

Unitree G1 MuJoCo 物理语义数据增强工具：将三套 G1_Action 数据统一映射为带关节、接触、NOCS/ANCSH、CoM/ZMP 和扰动标签的序列数据，并用 GPU 进行 5-fold episode 分组交叉验证。下一动作预测模型整体拟合水平约为 84%（R²=0.841），该指标不等于控制成功率或真机准确率。
