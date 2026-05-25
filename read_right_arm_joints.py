"""
读取 G1_29 右臂 7 个关节角度 (度) 并计算当前右手 TCP 位姿。

用法:
    python read_right_arm_joints.py          # 真机
    python read_right_arm_joints.py --sim    # 仿真
"""
import argparse
import numpy as np
import os, sys

# 将本文件所在目录加入 sys.path，确保 import logging_mp / robot_control / utils 可用
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

from utils.pinocchio_compat import import_pinocchio

pin = import_pinocchio()

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from robot_control.robot_arm import G1_29_ArmController, G1_29_JointArmIndex
from robot_control.robot_arm_ik import G1_29_ArmIK

# 右臂关节名称（与 G1_29_JointArmIndex 枚举顺序对应的后 7 个）
RIGHT_ARM_JOINT_NAMES = [
    "RightShoulderPitch",   # index 22
    "RightShoulderRoll",    # index 23
    "RightShoulderYaw",     # index 24
    "RightElbow",           # index 25
    "RightWristRoll",       # index 26
    "RightWristPitch",      # index 27
    "RightWristYaw",        # index 28
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', action='store_true', help='Enable simulation mode')
    parser.add_argument('--hz', type=float, default=5.0, help='Print frequency (default 5 Hz)')
    parser.add_argument('--network-interface', type=str, default=None)
    args = parser.parse_args()

    domain_id = 1 if args.sim else 0
    ChannelFactoryInitialize(domain_id, networkInterface=args.network_interface)

    # 这里 motion_mode=False 只订阅状态，不会接管机器人运动
    arm_ctrl = G1_29_ArmController(motion_mode=False, simulation_mode=args.sim)

    # 加载 IK 模型（内含 Pinocchio 运动学模型，用于正解 FK）
    arm_ik = G1_29_ArmIK()

    # ========== 1. 读取关节角度 ==========
    dual_arm_q_rad = arm_ctrl.get_current_dual_arm_q()   # 14维: 前7左臂, 后7右臂
    right_arm_q_rad = dual_arm_q_rad[7:]                 # 取后 7 个 (右臂)
    right_arm_q_deg = np.degrees(right_arm_q_rad)        # 弧度 → 角度

    print("\n" + "═" * 60)
    print("  右臂关节角度")
    print("─" * 60)
    for name, deg, rad in zip(RIGHT_ARM_JOINT_NAMES, right_arm_q_deg, right_arm_q_rad):
        print(f"  {name:>22s}:  {deg:+8.2f}°  ({rad:+7.4f} rad)")

    # ========== 2. 计算右手 TCP 位姿 (基于 pelvis 坐标系) ==========
    model = arm_ik.reduced_robot.model
    data  = arm_ik.reduced_robot.data
    pin.framesForwardKinematics(model, data, dual_arm_q_rad)

    r_ee_id = model.getFrameId("R_ee")
    r_pose  = data.oMf[r_ee_id]       # SE3 对象

    # 位置 (米)
    pos = r_pose.translation
    # 旋转 → 欧拉角 RPY (度)
    rpy_deg = np.degrees(pin.rpy.matrixToRpy(r_pose.rotation))

    print("\n" + "─" * 60)
    print("  右手 TCP 位姿 (相对于 pelvis)")
    print("─" * 60)
    print(f"  位置 X: {pos[0]:+.5f} m")
    print(f"  位置 Y: {pos[1]:+.5f} m")
    print(f"  位置 Z: {pos[2]:+.5f} m")
    print(f"  Roll  : {rpy_deg[0]:+.2f}°")
    print(f"  Pitch : {rpy_deg[1]:+.2f}°")
    print(f"  Yaw   : {rpy_deg[2]:+.2f}°")

    print("\n  4x4 齐次变换矩阵:")
    print(r_pose.homogeneous)
    print("═" * 60)
