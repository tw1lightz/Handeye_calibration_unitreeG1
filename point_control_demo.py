import time
import os
import sys
import argparse
from multiprocessing import Array, Lock

import numpy as np

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
from robot_control.robot_arm import G1_29_ArmController
from robot_control.robot_arm_ik import G1_29_ArmIK

# =========================================================================
# ⚙️ 配置区：在这里填入你的右手目标位姿 (相对于 pelvis 中心)
#    左手不参与控制，会自动保持在启动时的当前位置
# =========================================================================

# 默认模式：真机 (False=物理域 domain 0)
SIM_MODE = False

# 右手 6 个参数：[X, Y, Z, Roll, Pitch, Yaw] (单位：米 和 度)
R_EE_POSE = [ 0.31763, -0.14558,  0.2389,  -84.46,  -27.66,  12.74 ]

# 真机模式下关节速度限幅（rad/s）。数值越小，到达目标越慢更柔和。
ARM_VELOCITY_LIMIT = 8.0

# =========================================================================

def create_target_pose(x, y, z, roll, pitch, yaw):
    """辅助函数：将 [x, y, z, roll, pitch, yaw] 转换为 4x4 齐次变换矩阵"""
    pose = np.eye(4)
    pose[0, 3], pose[1, 3], pose[2, 3] = x, y, z
    pose[:3, :3] = pin.rpy.rpyToMatrix(np.radians(np.array([roll, pitch, yaw], dtype=np.float64)))
    return pose

def get_current_left_tcp_pose(arm_ik, current_dual_q):
    """通过 FK 正解获取左手 TCP 当前位姿，用于让左手保持原位不动"""
    model = arm_ik.reduced_robot.model
    data  = arm_ik.reduced_robot.data
    pin.framesForwardKinematics(model, data, current_dual_q)
    l_ee_id = model.getFrameId("L_ee")
    return data.oMf[l_ee_id].homogeneous.copy()

# 站立姿态下双臂关节角度 (rad)，退出时平滑回到这个姿态
STANDING_ARM_Q = np.array([
    +0.2620, +0.2871, -0.0881, +0.8216, -0.0020, +0.0021, +0.0014,   # 左臂
    +0.2644, -0.2881, +0.0927, +0.7943, -0.0118, +0.0130, -0.0001,   # 右臂
])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="G1 双臂点位控制 Demo")
    parser.add_argument("--sim", action="store_true", default=SIM_MODE, help="仿真模式 (domain=1，默认关闭)")
    parser.add_argument("--real", action="store_true", help="真机模式 (domain=0，默认模式，优先级高于 --sim)")
    parser.add_argument("--network-interface", type=str, default=None,
                        help="DDS 网卡名，仿真通常留空，真机可指定如 eth0/eth1")
    parser.add_argument("--arm-velocity-limit", type=float, default=ARM_VELOCITY_LIMIT,
                        help="真机关节速度限幅(rad/s)，默认更慢更稳")
    args = parser.parse_args()

    sim_mode = args.sim and not args.real

    # 1. 初始化 DDS (domain 1 为仿真, 0 为物理)
    domain_id = 1 if sim_mode else 0
    ChannelFactoryInitialize(domain_id, networkInterface=args.network_interface)

    # 2. 实例化控制组件
    logger_mp.info(f"Initializing for {'SIMULATION' if sim_mode else 'REAL ROBOT'}...")
    logger_mp.info(f"DDS domain_id={domain_id}, network_interface={args.network_interface}")
    arm_ik = G1_29_ArmIK(Visualization=False)
    # 使用 motion_mode=True 确保退出程序时腿部依然保持站立控制，不倒下
    arm_ctrl = G1_29_ArmController(motion_mode=True, simulation_mode=sim_mode)
    if not sim_mode:
        arm_ctrl.arm_velocity_limit = args.arm_velocity_limit
        logger_mp.info(f"Real mode arm_velocity_limit={args.arm_velocity_limit} rad/s")

    # 3. 灵巧手控制器（可选）
    # 如果环境缺少 dex_retargeting，跳过手部控制，不影响双臂点位控制。
    hand_ctrl = None
    try:
        from robot_control.robot_hand_unitree import Dex3_1_Controller

        left_hand_pos_array = Array('d', 75, lock=True)
        right_hand_pos_array = Array('d', 75, lock=True)
        with left_hand_pos_array.get_lock():
            left_hand_pos_array[:] = np.zeros(75).tolist()
        with right_hand_pos_array.get_lock():
            right_hand_pos_array[:] = np.zeros(75).tolist()
        hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, Lock(),
                                      Array('d', 14), Array('d', 14), simulation_mode=sim_mode)
    except ModuleNotFoundError as e:
        if e.name == "dex_retargeting":
            logger_mp.warning("未检测到 dex_retargeting，已跳过手部控制，仅运行双臂点位控制。")
            logger_mp.warning("如需启用手部控制，请安装: conda run -n hand_eye_calib python -m pip install dex-retargeting")
        else:
            raise

    # 平滑启动（保留较慢限幅，不再自动升到最大速度）
    logger_mp.info("---------------------🚀 Start Tracking 🚀-------------------------")

    try:
        # 4. 左手保持当前位姿不动，右手移动到配置区指定的目标点
        current_q = arm_ctrl.get_current_dual_arm_q()
        left_target_pose  = get_current_left_tcp_pose(arm_ik, current_q)
        right_target_pose = create_target_pose(*R_EE_POSE)
        logger_mp.info(f"左手: 保持当前位姿")
        logger_mp.info(f"右手目标: {R_EE_POSE}")

        while True:
            loop_start = time.time()
            
            # 获取当前角度作为解算初始值
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # 5. 调用 IK 求逆解并下发
            sol_q, sol_tauff = arm_ik.solve_ik(left_target_pose, right_target_pose, current_lr_arm_q, current_lr_arm_dq)
            arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            # 控制循环频率 (~30Hz)
            time.sleep(max(0, (1.0/30.0) - (time.time() - loop_start)))

    except KeyboardInterrupt:
        # ===================== 第一阶段 Ctrl+C: 平滑回到站立姿态 =====================
        logger_mp.info("⚠️ 收到退出信号，手臂开始缓慢回到站立姿态...")
        
        # 停止 IK 目标更新，将当前位置作为插值起点
        start_q = arm_ctrl.get_current_dual_arm_q()
        target_q = STANDING_ARM_Q

        # 平滑插值：从当前位置 → 站立姿态 (约3秒, 750步@250Hz 控制周期)
        ctrl_dt = 1.0 / 250.0  # 底层控制频率
        num_interp_steps = 750
        
        for step in range(num_interp_steps + 1):
            alpha = step / num_interp_steps
            interp_q = start_q + alpha * (target_q - start_q)
            
            # 直接下发关节位角度，前馈力矩置零
            arm_ctrl.ctrl_dual_arm(interp_q, np.zeros(14))
            time.sleep(ctrl_dt)

        # 到位后再保持一小段时间确保稳定
        for _ in range(250):  # ~1秒
            arm_ctrl.ctrl_dual_arm(target_q, np.zeros(14))
            time.sleep(ctrl_dt)

        # 平滑释放 motion mode 权重 (1→0)
        arm_ctrl.smooth_exit()
        logger_mp.info("✅ 手臂已回到站立姿态，控制权已安全归还！")
        
    finally:
        logger_mp.info("程序已安全关闭。")
