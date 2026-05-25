#!/usr/bin/env python3
"""
Eye-to-Hand 手眼标定计算脚本。

从 collect_data.py 采集的数据集计算相机在机器人基座 (pelvis) 坐标系下的位姿。

原理:
  Eye-to-hand 场景: 相机固定在外部，AprilTag 固定在机械臂末端。
  关系式: T_base_cam * T_cam_tag = T_base_ee * T_ee_tag
  使用 OpenCV calibrateHandEye (传入 TCP 逆矩阵) 求解 T_cam2base。

流程:
  1. 加载数据集 (tag_pose/*.npy + tcp_pose/*.npy)
  2. 验证旋转矩阵有效性 (det≈1, R^T R≈I)
  3. 同时使用 5 种算法 (Tsai/Park/Horaud/Andreff/Daniilidis)
  4. 用 T_ee_tag 一致性衡量标定精度，自动选择最优方法
  5. 保存 calibration_result.npz

用法:
    python compute_to_hand.py                                 # 自动使用最新数据集
    python compute_to_hand.py hand_eye_data/20260409_165000   # 指定数据集
    python compute_to_hand.py --method TSAI                   # 指定算法
"""

import argparse
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# 脚本所在目录 (用于定位 hand_eye_data/)
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# 可用标定算法
# ---------------------------------------------------------------------------
METHODS = {
    "TSAI":      cv2.CALIB_HAND_EYE_TSAI,
    "PARK":      cv2.CALIB_HAND_EYE_PARK,
    "HORAUD":    cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF":   cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


# ---------------------------------------------------------------------------
# 自动查找最新数据集
# ---------------------------------------------------------------------------

def find_latest_dataset() -> Path:
    """在 hand_eye_data/ 中按文件夹名排序，返回最新的时间戳目录。"""
    data_root = SCRIPT_DIR / "hand_eye_data"
    if not data_root.exists():
        raise FileNotFoundError(f"数据根目录不存在: {data_root}")

    # 时间戳文件夹格式 YYYYMMDD_HHMMSS，字典序即时间序
    subdirs = sorted(
        [d for d in data_root.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    if not subdirs:
        raise FileNotFoundError(f"hand_eye_data/ 下没有数据集文件夹")

    return subdirs[0]


# ---------------------------------------------------------------------------
# 数据加载与验证
# ---------------------------------------------------------------------------

def load_dataset(data_dir: Path):
    """加载 tag 和 tcp 位姿数据。"""
    tag_dir = data_dir / "tag_pose"
    tcp_dir = data_dir / "tcp_pose"

    if not tag_dir.exists() or not tcp_dir.exists():
        raise FileNotFoundError(f"缺少 tag_pose/ 或 tcp_pose/ 子目录: {data_dir}")

    tag_files = sorted(tag_dir.glob("tag_pose_*.npy"))
    tcp_files = sorted(tcp_dir.glob("tcp_pose_*.npy"))

    if len(tag_files) != len(tcp_files):
        raise ValueError(
            f"数据不匹配: {len(tag_files)} 个 tag_pose vs {len(tcp_files)} 个 tcp_pose"
        )
    if len(tag_files) < 3:
        raise ValueError(f"样本数不足 (需要 ≥ 3，实际 {len(tag_files)})")

    tag_poses = [np.load(str(f)) for f in tag_files]
    tcp_poses = [np.load(str(f)) for f in tcp_files]
    return tag_poses, tcp_poses, len(tag_files)


def is_valid_rotation(R: np.ndarray, tol: float = 1e-3) -> bool:
    """判断 R 是否为有效旋转矩阵 (det≈+1, R^T R≈I)。"""
    if R.shape != (3, 3):
        return False
    if abs(np.linalg.det(R) - 1.0) > tol:
        return False
    if np.linalg.norm(R.T @ R - np.eye(3)) > tol:
        return False
    return True


def validate_poses(poses: list, name: str) -> list:
    """验证所有位姿的旋转矩阵，返回有效索引。"""
    valid = []
    for i, T in enumerate(poses):
        R = T[:3, :3]
        if not is_valid_rotation(R):
            det = np.linalg.det(R)
            print(f"  [!] {name}[{i}] 旋转矩阵无效 (det={det:.6f})")
        else:
            valid.append(i)
    return valid


# ---------------------------------------------------------------------------
# 标定核心
# ---------------------------------------------------------------------------

def calibrate_eye_to_hand(
    tag_poses: list,
    tcp_poses: list,
    method: int,
) -> np.ndarray:
    """
    使用 cv2.calibrateHandEye 求解 eye-to-hand 标定。

    技巧: 对于 eye-to-hand，将 TCP 位姿取逆后传入 "R_gripper2base"。
    输出的 X 即为 T_cam2base (相机相对于 pelvis 的位姿)。

    Args:
        tag_poses: T_cam_tag 列表 (tag 在相机坐标系下的 4x4 齐次矩阵)
        tcp_poses: T_base_ee 列表 (末端在 pelvis 坐标系下的 4x4 齐次矩阵)
        method:    cv2.CALIB_HAND_EYE_* 方法枚举

    Returns:
        T_cam2base: 4x4 齐次矩阵 (相机在 pelvis 坐标系下的位姿)
    """
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for i in range(len(tag_poses)):
        # eye-to-hand: 传入 TCP 逆矩阵
        T_inv = np.linalg.inv(tcp_poses[i])
        R_gripper2base.append(T_inv[:3, :3])
        t_gripper2base.append(T_inv[:3, 3].reshape(3, 1))

        # tag 位姿照常传入
        R_target2cam.append(tag_poses[i][:3, :3])
        t_target2cam.append(tag_poses[i][:3, 3].reshape(3, 1))

    R_result, t_result = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=method,
    )

    T_cam2base = np.eye(4, dtype=np.float64)
    T_cam2base[:3, :3] = R_result.reshape(3, 3)
    T_cam2base[:3, 3] = t_result.reshape(3)
    return T_cam2base


# ---------------------------------------------------------------------------
# 精度评估
# ---------------------------------------------------------------------------

def compute_consistency_error(
    T_cam2base: np.ndarray,
    tag_poses: list,
    tcp_poses: list,
) -> dict:
    """
    评估标定精度。

    若标定完美，则对所有样本:
        T_ee_tag = inv(T_base_ee) * T_base_cam * T_cam_tag
    应保持恒定 (tag 固定在末端)。

    用 T_ee_tag 的离散程度衡量标定质量。
    """
    T_base_cam = T_cam2base

    ee_tag_list = []
    for i in range(len(tag_poses)):
        T_ee_tag_i = np.linalg.inv(tcp_poses[i]) @ T_base_cam @ tag_poses[i]
        ee_tag_list.append(T_ee_tag_i)

    # 平移分散度
    translations = np.array([T[:3, 3] for T in ee_tag_list])
    t_mean = translations.mean(axis=0)
    t_errors = np.linalg.norm(translations - t_mean, axis=1)

    # 旋转分散度 (与均值的角度差)
    R_ref = ee_tag_list[0][:3, :3]
    r_errors = []
    for T in ee_tag_list:
        R_diff = R_ref.T @ T[:3, :3]
        cos_val = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_val))
        r_errors.append(angle_deg)
    r_errors = np.array(r_errors)

    return {
        "t_mean_error_mm": float(t_errors.mean() * 1000),
        "t_max_error_mm":  float(t_errors.max() * 1000),
        "t_std_mm":        translations.std(axis=0) * 1000,
        "r_mean_error_deg": float(r_errors.mean()),
        "r_max_error_deg":  float(r_errors.max()),
    }


# ---------------------------------------------------------------------------
# 显示工具
# ---------------------------------------------------------------------------

def rotation_to_euler_xyz_deg(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 欧拉角 (XYZ 内旋，角度制)。"""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees(np.array([x, y, z]))


def print_transform(name: str, T: np.ndarray) -> None:
    """打印 4x4 齐次变换矩阵。"""
    pos = T[:3, 3]
    euler = rotation_to_euler_xyz_deg(T[:3, :3])
    print(f"\n  {name}:")
    print(f"    位置  : X={pos[0]:+.5f}  Y={pos[1]:+.5f}  Z={pos[2]:+.5f}  (m)")
    print(f"    欧拉角: Roll={euler[0]:+.2f}°  Pitch={euler[1]:+.2f}°  Yaw={euler[2]:+.2f}°")
    print(f"    4x4 齐次矩阵:")
    for row in T:
        print(f"      [{row[0]:+10.6f}  {row[1]:+10.6f}  {row[2]:+10.6f}  {row[3]:+10.6f}]")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Eye-to-Hand 手眼标定计算")
    parser.add_argument(
        "data_dir", type=str, nargs="?", default=None,
        help="数据目录 (默认自动使用 hand_eye_data/ 下最新的时间戳文件夹)",
    )
    parser.add_argument(
        "--method", type=str, default="all",
        choices=list(METHODS.keys()) + ["all"],
        help="标定算法 (默认 all: 尝试全部并选最优)",
    )
    args = parser.parse_args()

    # 自动查找最新数据集
    if args.data_dir is None:
        data_dir = find_latest_dataset()
        print(f"  自动选择最新数据集: {data_dir.name}")
    else:
        data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"  [错误] 目录不存在: {data_dir}")
        sys.exit(1)

    # ── 1. 加载数据 ──────────────────────────────────────────
    print("=" * 60)
    print("  Eye-to-Hand 手眼标定计算")
    print("=" * 60)

    tag_poses, tcp_poses, n_samples = load_dataset(data_dir)
    print(f"\n  加载了 {n_samples} 组样本  ({data_dir})")

    # ── 2. 验证数据 ──────────────────────────────────────────
    print("\n  验证旋转矩阵 ...")
    valid_tag = validate_poses(tag_poses, "tag_pose")
    valid_tcp = validate_poses(tcp_poses, "tcp_pose")
    valid_idx = sorted(set(valid_tag) & set(valid_tcp))

    if len(valid_idx) < 3:
        print(f"  [错误] 有效样本不足 (需 ≥ 3, 实际 {len(valid_idx)})")
        sys.exit(1)

    if len(valid_idx) < n_samples:
        print(f"  移除 {n_samples - len(valid_idx)} 个无效样本")
        tag_poses = [tag_poses[i] for i in valid_idx]
        tcp_poses = [tcp_poses[i] for i in valid_idx]

    print(f"  有效样本: {len(tag_poses)} 组")

    # ── 3. 标定 (多方法对比) ─────────────────────────────────
    methods_to_try = METHODS if args.method == "all" else {args.method: METHODS[args.method]}

    results = {}
    for name, method_id in methods_to_try.items():
        try:
            T = calibrate_eye_to_hand(tag_poses, tcp_poses, method_id)
            if not is_valid_rotation(T[:3, :3], tol=0.01):
                print(f"  [{name}] ⚠ 结果旋转矩阵异常，跳过")
                continue
            err = compute_consistency_error(T, tag_poses, tcp_poses)
            results[name] = {"T_cam2base": T, "error": err}
        except Exception as e:
            print(f"  [{name}] ✗ 失败: {e}")

    if not results:
        print("\n  [错误] 所有方法均失败，请检查数据质量")
        sys.exit(1)

    # ── 4. 结果对比 ──────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  方法对比  (T_ee_tag 一致性)")
    print("─" * 60)
    header = f"  {'方法':<12s}  {'平移误差均值':>12s}  {'旋转误差均值':>12s}  {'平移误差最大':>12s}"
    print(header)
    print("  " + "─" * 54)

    best_name = None
    best_score = float("inf")

    for name, res in results.items():
        e = res["error"]
        # 加权得分: 平移 (mm) + 旋转 (°) * 10
        score = e["t_mean_error_mm"] + e["r_mean_error_deg"] * 10
        if score < best_score:
            best_score = score
            best_name = name
        print(
            f"  {name:<12s}"
            f"  {e['t_mean_error_mm']:>9.3f} mm"
            f"  {e['r_mean_error_deg']:>9.3f} °"
            f"  {e['t_max_error_mm']:>9.3f} mm"
        )

    # ── 5. 输出最优结果 ──────────────────────────────────────
    best = results[best_name]
    T_cam2base = best["T_cam2base"]
    T_base2cam = np.linalg.inv(T_cam2base)

    print(f"\n  ★ 最优方法: {best_name}")
    print_transform("T_cam2base  (相机 → pelvis)", T_cam2base)
    print_transform("T_base2cam  (pelvis → 相机)", T_base2cam)

    # ── 6. 保存 ─────────────────────────────────────────────
    calib_dir = SCRIPT_DIR / "calib_result"
    calib_dir.mkdir(exist_ok=True)
    output_path = calib_dir / f"{data_dir.name}_calib.npz"
    np.savez(
        str(output_path),
        T_cam2base=T_cam2base,
        T_base2cam=T_base2cam,
        method=best_name,
        t_mean_error_mm=best["error"]["t_mean_error_mm"],
        r_mean_error_deg=best["error"]["r_mean_error_deg"],
        n_samples=len(tag_poses),
    )
    print(f"\n  结果已保存 → {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
