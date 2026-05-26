# Unitree G1 - Eye-to-Hand 手眼标定工具包

本项目用于 Unitree G1 机器人的眼在外（Eye-to-Hand）手眼标定。通过使用 RealSense 相机捕捉固定在机械臂末端的 AprilTag，并结合机械臂通过正运动学（Forward Kinematics）实时获取的末端 TCP 位姿，来解算相机在机器人基座（pelvis）坐标系下的相对位姿转移矩阵。

除此之外，本项目还提供了一些实用的机械臂控制和调试脚本，例如重力补偿模式（拖拽示教）切换工具。

## 环境依赖与配置

本项目统一使用名为 `hand_eye_calib` 的 Conda 环境，且提供了快速唤起的 Shell 脚本。

**激活环境的方法：**
```bash
# 方法一：直接 source 激活脚本
source activate_handeye_env.sh

# 方法二：使用 conda 命令
conda activate hand_eye_calib
```

如果在未激活该环境的终端中仅仅想执行某个 Python 脚本，可以使用 `run_pin_arm.sh` 包装执行：
```bash
./run_pin_arm.sh right_arm_mode.py
```

## 核心功能与使用说明

### 1. 机械臂模式切换 (`right_arm_mode.py`)
在进行手眼标定数据采集时，通常需要手动拖动机械臂变换不同的姿态以采集丰富的样本。本项目提供了右臂模式切换脚本，支持进入低阻尼重力补偿模式。

**使用方法：**
```bash
conda activate hand_eye_calib
python right_arm_mode.py
```
进入交互模式后，可以输入以下命令进行切换：
- `free`：右臂进入“可手拖”模式，进行重力补偿并保留低阻尼。可以在标定过程中手动调整右臂。若觉得太松或太紧，可增加 `--free-kd` 和 `--free-wrist-kd` 参数进行调整。
- `lock`：读取当前右臂关节角，按当前位置锁住。
- `status`：查看当前状态。
- `quit`：退出。

*单命令行启动方式：*
```bash
python right_arm_mode.py --mode free
python right_arm_mode.py --mode lock
```

### 2. 相机与 AprilTag 姿态识别 (`realsense_apriltag_pose.py`)
用于单元测试，测试 RealSense 相机能否正确识别 AprilTag 并解算出相对相机坐标系的位姿矩阵。

```bash
python realsense_apriltag_pose.py --help
```

### 3. 【标定第一步】数据采集 (`collect_data.py`)
实时读取 RealSense 画面检测 AprilTag，并同步获取 G1 机器人右臂末端相对于基座的位姿，存储齐次变换矩阵。

**功能说明：**
- 打开 RealSense 相机读取 RGB 及 SDK 内参。
- 实时检测 AprilTag (Tag36h11)。
- 连接 G1 机器人，实时读取右臂末端 TCP 相对 Pelvis 的 4x4 位姿齐次变换矩阵。
- 按下 `s` 键录制保存当前帧的数据（图片、Camera-Tag 位姿矩阵、Base-TCP 位姿矩阵、相机内参）。

**使用方法：**
```bash
# 启动真机采集
python collect_data.py

# 指定 tag 尺寸 (米)，默认 0.08
python collect_data.py --tag-size 0.08

# 指定特定 Id 进行跟踪
python collect_data.py --tag-id 0
```
数据将自动保存在项目目录下的 `hand_eye_data/<timestamp>/` 目录内。

### 4. 【标定第二步】标定计算 (`compute_to_hand.py`)
使用前一步采集到的位姿数据，同时运用 OpenCV 的 5 种标定方法（Tsai / Park / Horaud / Andreff / Daniilidis）进行 `Eye-to-Hand` 解算（$T_{base\_cam} \times T_{cam\_tag} = T_{base\_ee} \times T_{ee\_tag}$）。该脚本会自动计算误差（通过 $T_{ee\_tag}$ 衡量一致性），输出标定精度并选择最优的方法结果。

**使用方法：**
```bash
# 自动查找并使用最新采集的数据集进行标定求解
python compute_to_hand.py

# 指定特定的历史数据集目录
python compute_to_hand.py hand_eye_data/20260409_165000

# 强制使用指定算法
python compute_to_hand.py --method TSAI
```
标定计算的结果将输出并保存为 dataset 下的 `calibration_result.npz`。

## 文件与目录结构

- `activate_handeye_env.sh`: Conda 环境激活快捷脚本。
- `run_pin_arm.sh`: 在正确环境与系统依赖下一次性执行 Python 脚本的包装脚本。
- `right_arm_mode.py`: 切换机械臂（右臂）运动模式：拖拽/锁死。
- `read_right_arm_joints.py`: 测试脚本，读取右臂关节角度。
- `point_control_demo.py`: 位置控制 Demo 测试脚本。
- `realsense_apriltag_pose.py`: 相机 AprilTag 识别核心检测工具。
- `collect_data.py`: 手眼标定流程核心脚本（1）：采集并记录标定位姿帧。
- `compute_to_hand.py`: 手眼标定流程核心脚本（2）：核心解算工具，输出最终标定矩阵。
- `logging_mp.py`: 多进程可用的日志管理工具。
- `assets/`: 包含了各种机器人模型及末端抓手的 URDF、XML 以及 Mesh 文件资源（G1、H1、H1_2、BrainCo 手、Inspire 手、Unitree Dex3 灵巧手等）。
- `robot_control/`: 基于 Pinocchio 的机器人控制底层封装，包含逆运动学 (IK) 和关节读取工具。
- `utils/`: 包含数学兼容处理以及滤波相关的通用代码。
