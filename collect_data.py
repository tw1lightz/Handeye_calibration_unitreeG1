#!/usr/bin/env python3
"""
Eye-to-Hand 手眼标定数据采集脚本。

功能:
  1. 打开 RealSense 相机 (1280x720)，自动读取 SDK 内参
  2. 实时检测 AprilTag (tag36h11)，计算 tag 在相机坐标系下的位姿
  3. 连接 G1_29 机器人，实时读取右臂末端 TCP 位姿 (相对 pelvis)
  4. 按 's' 保存当前帧数据:
       - 彩色图片 (.png)
       - tag 在相机坐标系下的 4x4 齐次变换矩阵
       - 机械臂末端 TCP 的 4x4 齐次变换矩阵
       - 相机内参
  5. 数据保存在 hand_eye_data/<timestamp>/ 目录下

用法:
    python collect_data.py                           # 真机
    python collect_data.py --sim                     # 仿真模式
    python collect_data.py --tag-size 0.08           # 指定 tag 边长 (米)
    python collect_data.py --tag-id 0                # 只跟踪指定 id
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# ---------------------------------------------------------------------------
# 复用 realsense_apriltag_pose 中的工具
# ---------------------------------------------------------------------------
from realsense_apriltag_pose import (
    configure_opencv_qt_fonts,
    DetectorWrapper,
    CameraIntrinsics,
    PoseResult,
    pick_device_info,
    start_color_pipeline_with_fallback,
    get_color_intrinsics,
    tag_object_points,
    solve_tag_pose,
    draw_detection_overlay,
)

configure_opencv_qt_fonts()
import cv2
import pyrealsense2 as rs

# ---------------------------------------------------------------------------
# 机器人 SDK
# ---------------------------------------------------------------------------
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)

from utils.pinocchio_compat import import_pinocchio
pin = import_pinocchio()

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from robot_control.robot_arm import G1_29_ArmController
from robot_control.robot_arm_ik import G1_29_ArmIK


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def tag_pose_to_homogeneous(pose: PoseResult) -> np.ndarray:
    """将 PoseResult 的旋转矩阵和平移向量合成 4x4 齐次变换矩阵。"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = pose.rotation_matrix
    T[:3, 3] = pose.tvec.reshape(3)
    return T


def get_tcp_homogeneous(arm_ik: G1_29_ArmIK, dual_arm_q_rad: np.ndarray) -> np.ndarray:
    """通过正运动学计算右手 TCP 相对 pelvis 的 4x4 齐次变换矩阵。"""
    model = arm_ik.reduced_robot.model
    data = arm_ik.reduced_robot.data
    pin.framesForwardKinematics(model, data, dual_arm_q_rad)

    r_ee_id = model.getFrameId("R_ee")
    r_pose = data.oMf[r_ee_id]  # pinocchio SE3
    return r_pose.homogeneous.copy()


def save_sample(
    save_dir: Path,
    index: int,
    color_image: np.ndarray,
    tag_T: np.ndarray,
    tcp_T: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> Path:
    """
    保存单次采样数据。

    目录结构:
        <save_dir>/
            image_000.png
            tag_pose/
                tag_pose_000.npy    # tag 在相机坐标系下 4x4
            tcp_pose/
                tcp_pose_000.npy    # 机械臂末端 4x4 (pelvis 坐标系)
            intrinsics.npz          # 相机内参 (只写一次)
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    tag_dir = save_dir / "tag_pose"
    tcp_dir = save_dir / "tcp_pose"
    tag_dir.mkdir(exist_ok=True)
    tcp_dir.mkdir(exist_ok=True)

    suffix = f"{index:03d}"
    img_path = save_dir / f"image_{suffix}.png"
    tag_path = tag_dir / f"tag_pose_{suffix}.npy"
    tcp_path = tcp_dir / f"tcp_pose_{suffix}.npy"

    cv2.imwrite(str(img_path), color_image)
    np.save(str(tag_path), tag_T)
    np.save(str(tcp_path), tcp_T)

    # 内参只需保存一次
    intr_path = save_dir / "intrinsics.npz"
    if not intr_path.exists():
        np.savez(
            str(intr_path),
            camera_matrix=intrinsics.camera_matrix,
            dist_coeffs=intrinsics.dist_coeffs,
            width=intrinsics.width,
            height=intrinsics.height,
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.cx,
            cy=intrinsics.cy,
        )

    return img_path


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Eye-to-Hand 手眼标定数据采集",
    )
    # 相机参数
    parser.add_argument("--width", type=int, default=1280, help="Color stream width (default 1280)")
    parser.add_argument("--height", type=int, default=720, help="Color stream height (default 720)")
    parser.add_argument("--fps", type=int, default=30, help="Color stream FPS")
    parser.add_argument("--serial", type=str, default=None, help="RealSense serial number")
    # AprilTag 参数
    parser.add_argument("--tag-size", type=float, default=0.08, help="AprilTag 边长 (米)")
    parser.add_argument("--tag-id", type=int, default=None, help="只跟踪指定 tag id")
    parser.add_argument("--axis-length", type=float, default=None, help="坐标轴显示长度 (米)")
    # 机器人参数
    parser.add_argument("--sim", action="store_true", help="仿真模式")
    parser.add_argument("--network-interface", type=str, default=None)
    args = parser.parse_args()

    axis_length_m = args.axis_length if args.axis_length else args.tag_size * 0.25

    # ---- 创建保存目录 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(current_dir) / "hand_eye_data" / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    sample_index = 0

    # ---- 初始化机器人 ----
    print("=" * 60)
    print("  初始化机器人 SDK ...")
    domain_id = 1 if args.sim else 0
    ChannelFactoryInitialize(domain_id, networkInterface=args.network_interface)
    arm_ctrl = G1_29_ArmController(motion_mode=False, simulation_mode=args.sim)
    arm_ik = G1_29_ArmIK()
    print("  机器人 SDK 初始化完成")

    # ---- 初始化相机 ----
    print("  初始化 RealSense ...")
    detector = DetectorWrapper("tag36h11")
    object_points = tag_object_points(args.tag_size)
    device_info = pick_device_info(args.serial)

    pipeline = rs.pipeline()
    profile, active_mode = start_color_pipeline_with_fallback(
        pipeline, device_info, args.width, args.height, args.fps,
    )

    try:
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = get_color_intrinsics(color_profile)

        print(f"  设备: {device_info.name} (serial={device_info.serial})")
        print(f"  分辨率: {active_mode[0]}x{active_mode[1]}@{active_mode[2]}")
        print(f"  内参: fx={intrinsics.fx:.1f}  fy={intrinsics.fy:.1f}  "
              f"cx={intrinsics.cx:.1f}  cy={intrinsics.cy:.1f}")
        print(f"  检测器: {detector.backend}")
        print(f"  数据保存到: {save_dir}")
        print("=" * 60)
        print("  操作: [s] 采集数据  |  [q/ESC] 退出")
        print("=" * 60)
        sys.stdout.flush()

        window_name = "Eye-to-Hand Data Collection"

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            # ---- 检测 AprilTag ----
            detections = detector.detect(gray)
            poses = []
            for detection in detections:
                tag_id = int(getattr(detection, "tag_id", -1))
                if args.tag_id is not None and tag_id != args.tag_id:
                    continue
                pose = solve_tag_pose(detection, intrinsics, object_points)
                if pose is not None:
                    poses.append(pose)

            poses.sort(key=lambda p: p.tvec[2, 0])

            # ---- 绘制检测结果 ----
            display = color_image.copy()
            for pose in poses:
                draw_detection_overlay(display, pose, axis_length_m, intrinsics)

            # ---- 状态栏 ----
            status = f"Tags: {len(poses)} | Saved: {sample_index}"
            cv2.putText(display, status, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)

            # 如果有检测到 tag，显示最近 tag 的距离
            if poses:
                t = poses[0].tvec.reshape(3)
                dist_text = f"Dist: {np.linalg.norm(t):.3f}m"
                cv2.putText(display, dist_text, (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2)

            cv2.imshow(window_name, display)

            # ---- 按键处理 ----
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break

            if key == ord('s'):
                if not poses:
                    print("  [!] 未检测到 AprilTag，跳过采集")
                    sys.stdout.flush()
                    continue

                # 取最近的 tag
                best_pose = poses[0]
                tag_T = tag_pose_to_homogeneous(best_pose)

                # 读取机械臂末端位姿
                dual_arm_q_rad = arm_ctrl.get_current_dual_arm_q()
                tcp_T = get_tcp_homogeneous(arm_ik, dual_arm_q_rad)

                # 保存
                img_path = save_sample(
                    save_dir, sample_index,
                    color_image, tag_T, tcp_T, intrinsics,
                )
                sample_index += 1

                # 打印信息
                tag_t = best_pose.tvec.reshape(3)
                tcp_pos = tcp_T[:3, 3]
                print(f"\n  ✓ 样本 #{sample_index - 1}")
                print(f"    Tag id={best_pose.tag_id}  "
                      f"位置(cam): [{tag_t[0]:.4f}, {tag_t[1]:.4f}, {tag_t[2]:.4f}]")
                print(f"    TCP 位置(pelvis): [{tcp_pos[0]:.4f}, {tcp_pos[1]:.4f}, {tcp_pos[2]:.4f}]")
                print(f"    保存: {img_path.name}")
                sys.stdout.flush()

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\n  共采集 {sample_index} 个样本 → {save_dir}")


if __name__ == "__main__":
    main()
