# Go2 深度感知运动控制

基于 Unitree Go2、Unitree RL Lab 与 Isaac Lab 的 Teacher–Student 深度感知运动控制项目。本仓库包含本项目使用的训练任务配置、策略网络、蒸馏与评测脚本，以及 Go2 机器人描述资源。

Teacher 在仿真中使用特权地形信息训练；Student 使用部署阶段可获得的深度图和本体状态，通过 LSTM 编码本体历史、交叉注意力融合深度与当前状态，并以 Teacher 的隐变量和动作作为蒸馏监督。

## Student 策略结构

- 深度输入：`32 × 48` 几何深度图。
- 深度编码：划分为 `4 × 4` patch，每个 patch 展平为 16 维 token，共 96 个 token；当前本体状态作为 Query，深度 token 作为 Key/Value。
- 时序编码：LSTM 编码 21 帧本体历史，hidden size 为 128。
- 特征融合：融合时序特征与交叉注意力特征，再与当前本体状态共同输入 Actor MLP，输出 12 维关节动作。
- 蒸馏目标：Teacher 冻结，Student 使用隐表示 L2 loss 与动作 L2 loss；当前蒸馏流程不包含 Student PPO 更新。

核心实现位于 [`depth_student.py`](source/unitree_rl_lab/unitree_rl_lab/models/depth_student.py)、[`train_distill.py`](scripts/rsl_rl/train_distill.py) 和 Go2 任务配置目录 `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/`。

## 仿真结果

最终 10,000 iteration 的 Student 训练结果在 Isaac Lab 中评测 192 个 episode：存活率为 98.4%，平均平面速度误差为 0.117 m/s。评测使用程序生成的几何深度，未模拟渲染相机噪声。该结果仅代表仿真表现，不代表真机或 Sim2Real 结果。评测协议及地形分项记录见 [`doc/UPDATE_2026-10-12.md`](doc/UPDATE_2026-10-12.md)。

## 目录结构

```text
assets/go2_description/   Go2 URDF、xacro、网格与机器人描述文件
doc/                      项目说明、训练记录与评测协议
 scripts/                 训练、蒸馏、回放和评测脚本
 source/unitree_rl_lab/   Isaac Lab 任务、配置与策略模型
```

训练权重、TensorBoard 日志和原始评测输出未纳入仓库。个别机器人资源路径可能需要按本机目录调整；运行前请检查 `source/unitree_rl_lab/unitree_rl_lab/assets/robots/` 下的配置。

## 环境与复现

已记录的训练环境为 Isaac Sim 5.1、Isaac Lab 2.3.1 和 `rsl_rl` 5.4.2。Isaac Sim、Isaac Lab 及其依赖需要单独安装。请先按 [Unitree RL Lab 上游安装说明](https://github.com/unitreerobotics/unitree_rl_lab#installation) 配置环境，再按本机路径修改并运行脚本：

```bash
# 训练 Teacher（请按本机环境修改 Python 路径、任务名及训练参数）
PYTHONEXE=~/venvs/locomotion_rl/bin/python ~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/train.py \
  --task Unitree-Go2-Velocity-Teacher --headless --max_iterations 20000

# 训练 LSTM + Cross-Attention Student
bash scripts/rsl_rl/run_lstm_ca_long.sh

# 运行三种 Student 结构的评测（需提供脚本要求的 checkpoint）
~/IsaacLab/_isaac_sim/python.sh scripts/rsl_rl/eval_three_way.py
```

脚本中的 Isaac Lab 安装路径和 Python 环境路径是示例值，需要按实际安装位置调整。Student 蒸馏依赖 Teacher checkpoint，本仓库不提供该权重。由于仿真器与权重未随仓库分发，完整复现还需要准备相同版本的依赖、训练资源和 checkpoint。

## 上游项目与许可

本项目基于 [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab)。仓库中的上游代码及相关文件遵循其 Apache License 2.0，许可证文本见 [`LICENCE`](LICENCE)；请保留源文件中已有的版权与归属声明。第三方依赖、资源和本项目新增内容可能具有各自的许可或使用条件，使用前请核对对应声明。

