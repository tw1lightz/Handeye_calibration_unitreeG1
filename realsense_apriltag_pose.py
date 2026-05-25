#!/usr/bin/env python3
import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np


def configure_opencv_qt_fonts(force: bool = False) -> None:
    current = os.environ.get("QT_QPA_FONTDIR")
    if current and Path(current).is_dir() and not force:
        return

    cv2_dir = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/cv2"
    bundled_font_dir = cv2_dir / "qt/fonts"
    if bundled_font_dir.is_dir():
        os.environ["QT_QPA_FONTDIR"] = str(bundled_font_dir)
        return

    for candidate in (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation2"),
        Path("/usr/share/fonts/truetype"),
    ):
        if candidate.is_dir():
            os.environ["QT_QPA_FONTDIR"] = str(candidate)
            return


configure_opencv_qt_fonts()

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: opencv-python. Install it in your runtime env, for example:\n"
        "  conda run -n hand_eye_calib python -m pip install opencv-python"
    ) from exc

configure_opencv_qt_fonts(force=True)

try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pyrealsense2. Install it in your runtime env, for example:\n"
        "  conda run -n hand_eye_calib python -m pip install pyrealsense2"
    ) from exc


class DetectorWrapper:
    def __init__(self, family: str) -> None:
        self.backend = None
        self.detector = None
        self.family = family

        try:
            from pupil_apriltags import Detector

            self.backend = "pupil_apriltags"
            self.detector = Detector(
                families=family,
                nthreads=1,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0,
            )
            return
        except ImportError:
            pass

        try:
            import apriltag

            self.backend = "apriltag"
            options = apriltag.DetectorOptions(families=family)
            self.detector = apriltag.Detector(options)
            return
        except ImportError as exc:
            raise SystemExit(
                "Missing AprilTag dependency. Install one of:\n"
                "  conda run -n hand_eye_calib python -m pip install pupil-apriltags\n"
                "  conda run -n hand_eye_calib python -m pip install apriltag"
            ) from exc

    def detect(self, gray: np.ndarray) -> List[Any]:
        if self.backend == "pupil_apriltags":
            return list(self.detector.detect(gray, estimate_tag_pose=False))
        return list(self.detector.detect(gray))


@dataclass
class CameraIntrinsics:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class PoseResult:
    tag_id: int
    family: str
    corners: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    rotation_matrix: np.ndarray
    euler_xyz_deg: np.ndarray


@dataclass
class DeviceInfo:
    serial: str
    name: str


def list_supported_color_modes(device_serial: str) -> List[Tuple[int, int, int]]:
    context = rs.context()
    devices = context.query_devices()
    for device in devices:
        if device.get_info(rs.camera_info.serial_number) != device_serial:
            continue

        modes = set()
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                try:
                    video_profile = profile.as_video_stream_profile()
                except RuntimeError:
                    continue
                if video_profile.stream_type() != rs.stream.color:
                    continue
                if video_profile.format() != rs.format.bgr8:
                    continue
                modes.add((video_profile.width(), video_profile.height(), video_profile.fps()))
        return sorted(modes)
    return []


def start_color_pipeline_with_fallback(
    pipeline: rs.pipeline,
    device_info: DeviceInfo,
    width: int,
    height: int,
    fps: int,
) -> Tuple[rs.pipeline_profile, Tuple[int, int, int]]:
    requested_mode = (width, height, fps)
    supported_modes = list_supported_color_modes(device_info.serial)

    candidates: List[Tuple[int, int, int]] = [requested_mode]
    if supported_modes:
        sorted_modes = sorted(
            supported_modes,
            key=lambda mode: (
                abs(mode[0] - width) + abs(mode[1] - height),
                abs(mode[2] - fps),
                -(mode[0] * mode[1]),
                -mode[2],
            ),
        )
        for mode in sorted_modes:
            if mode not in candidates:
                candidates.append(mode)

    last_error: Optional[Exception] = None
    for mode in candidates:
        config = rs.config()
        config.enable_device(device_info.serial)
        config.enable_stream(rs.stream.color, mode[0], mode[1], rs.format.bgr8, mode[2])
        try:
            return pipeline.start(config), mode
        except RuntimeError as exc:
            last_error = exc

    supported_text = ", ".join(f"{w}x{h}@{f}" for w, h, f in supported_modes[:12])
    raise SystemExit(
        "Failed to start RealSense color stream. "
        f"Requested {requested_mode[0]}x{requested_mode[1]}@{requested_mode[2]} bgr8. "
        f"Sample supported modes: {supported_text if supported_text else 'unknown'}. "
        f"Last error: {last_error}"
    )


def pick_device_info(serial_hint: Optional[str]) -> DeviceInfo:
    context = rs.context()
    devices = context.query_devices()
    if len(devices) == 0:
        raise SystemExit("No RealSense device detected.")

    matched: Optional[DeviceInfo] = None
    fallback: Optional[DeviceInfo] = None
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        info = DeviceInfo(serial=serial, name=name)

        if fallback is None:
            fallback = info

        if serial_hint and serial == serial_hint:
            return info

        lower_name = name.lower()
        if "d435i" in lower_name or "d435" in lower_name:
            matched = info
            if "d435i" in lower_name:
                return info

    if serial_hint:
        raise SystemExit(f"Requested RealSense serial not found: {serial_hint}")
    if matched is not None:
        return matched
    if fallback is not None:
        return fallback
    raise SystemExit("No usable RealSense device found.")


def get_color_intrinsics(profile: rs.video_stream_profile) -> CameraIntrinsics:
    intr = profile.get_intrinsics()
    camera_matrix = np.array(
        [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    coeffs = list(intr.coeffs)
    if intr.model in (rs.distortion.brown_conrady, rs.distortion.inverse_brown_conrady):
        dist_coeffs = np.array(coeffs[:5], dtype=np.float64)
    else:
        dist_coeffs = np.zeros((5,), dtype=np.float64)

    return CameraIntrinsics(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        width=intr.width,
        height=intr.height,
        fx=intr.fx,
        fy=intr.fy,
        cx=intr.ppx,
        cy=intr.ppy,
    )


def tag_object_points(tag_size_m: float) -> np.ndarray:
    half = tag_size_m / 2.0
    return np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=np.float64,
    )


def corners_from_detection(detection: Any) -> np.ndarray:
    corners = np.asarray(detection.corners, dtype=np.float64)
    if corners.shape != (4, 2):
        raise ValueError(f"Unexpected corners shape: {corners.shape}")
    return corners


def order_corners_tl_tr_br_bl(corners: np.ndarray) -> np.ndarray:
    center = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    ordered = corners[np.argsort(angles)]
    start_idx = int(np.argmin(np.sum(ordered, axis=1)))
    ordered = np.roll(ordered, -start_idx, axis=0)
    return ordered


def solve_tag_pose(
    detection: Any,
    intrinsics: CameraIntrinsics,
    object_points: np.ndarray,
) -> Optional[PoseResult]:
    image_points = order_corners_tl_tr_br_bl(corners_from_detection(detection))
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            return None

    if float(tvec[2, 0]) <= 0.0:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    euler_xyz_deg = rotation_matrix_to_euler_xyz(rotation_matrix)

    tag_family = getattr(detection, "tag_family", b"tag36h11")
    if isinstance(tag_family, bytes):
        tag_family = tag_family.decode("utf-8", errors="ignore")

    return PoseResult(
        tag_id=int(getattr(detection, "tag_id", -1)),
        family=str(tag_family),
        corners=image_points,
        rvec=rvec.reshape(3, 1),
        tvec=tvec.reshape(3, 1),
        rotation_matrix=rotation_matrix,
        euler_xyz_deg=euler_xyz_deg,
    )


def rotation_matrix_to_euler_xyz(rotation_matrix: np.ndarray) -> np.ndarray:
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0.0

    return np.degrees(np.array([x, y, z], dtype=np.float64))


def draw_pose_axes(
    image: np.ndarray,
    intrinsics: CameraIntrinsics,
    rvec: np.ndarray,
    tvec: np.ndarray,
    axis_length_m: float,
) -> None:
    axis_points = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length_m, 0.0, 0.0],
            [0.0, axis_length_m, 0.0],
            [0.0, 0.0, -axis_length_m],
        ],
        dtype=np.float64,
    )
    image_points, _ = cv2.projectPoints(
        axis_points,
        rvec,
        tvec,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
    )
    image_points = image_points.reshape(-1, 2)
    if not np.isfinite(image_points).all():
        return

    h, w = image.shape[:2]
    rect = (0, 0, w, h)
    origin = tuple(np.round(image_points[0]).astype(int))
    endpoints = [
        (tuple(np.round(image_points[1]).astype(int)), (0, 0, 255)),
        (tuple(np.round(image_points[2]).astype(int)), (0, 255, 0)),
        (tuple(np.round(image_points[3]).astype(int)), (255, 0, 0)),
    ]

    for endpoint, color in endpoints:
        ok, p1, p2 = cv2.clipLine(rect, origin, endpoint)
        if ok:
            cv2.line(image, p1, p2, color, 2)


def draw_detection_overlay(
    image: np.ndarray,
    pose: PoseResult,
    axis_length_m: float,
    intrinsics: CameraIntrinsics,
) -> None:
    corners = pose.corners.astype(int)
    for idx in range(4):
        pt1 = tuple(corners[idx])
        pt2 = tuple(corners[(idx + 1) % 4])
        cv2.line(image, pt1, pt2, (0, 255, 255), 2)

    center = tuple(np.mean(corners, axis=0).astype(int))
    cv2.circle(image, center, 4, (255, 255, 0), -1)
    label = f"id={pose.tag_id} {pose.family}"
    cv2.putText(
        image,
        label,
        (center[0] + 10, center[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    draw_pose_axes(image, intrinsics, pose.rvec, pose.tvec, axis_length_m)


def print_pose(pose: PoseResult) -> None:
    translation = pose.tvec.reshape(3)
    print(f"tag_id: {pose.tag_id}")
    print(f"family: {pose.family}")
    print(
        "translation_m: "
        f"[{translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}]"
    )
    print(
        "euler_xyz_deg: "
        f"[{pose.euler_xyz_deg[0]:.3f}, {pose.euler_xyz_deg[1]:.3f}, {pose.euler_xyz_deg[2]:.3f}]"
    )
    print("rotation_matrix:")
    print(np.array2string(pose.rotation_matrix, precision=6, suppress_small=True))
    print()
    sys.stdout.flush()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open a D435i with pyrealsense2, detect tag36h11 AprilTags, draw axes, and print pose on keypress.",
    )
    parser.add_argument("--width", type=int, default=640, help="Color stream width")
    parser.add_argument("--height", type=int, default=480, help="Color stream height")
    parser.add_argument("--fps", type=int, default=30, help="Color stream FPS")
    parser.add_argument("--tag-size", type=float, default=0.08, help="AprilTag side length in meters")
    parser.add_argument("--tag-id", type=int, default=None, help="If set, only track this tag id")
    parser.add_argument("--serial", type=str, default=None, help="Optional RealSense serial number")
    parser.add_argument(
        "--axis-length",
        type=float,
        default=None,
        help="Axis length in meters. Defaults to 25%% of the tag size (0.02m for an 8cm tag).",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="D435i AprilTag Pose",
        help="OpenCV window title",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    axis_length_m = args.axis_length if args.axis_length is not None else args.tag_size * 0.25

    detector = DetectorWrapper("tag36h11")
    object_points = tag_object_points(args.tag_size)
    device_info = pick_device_info(args.serial)

    pipeline = rs.pipeline()
    profile, active_mode = start_color_pipeline_with_fallback(
        pipeline,
        device_info,
        args.width,
        args.height,
        args.fps,
    )
    try:
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = get_color_intrinsics(color_profile)

        print(f"Using RealSense device: {device_info.name} (serial={device_info.serial})")
        print("RealSense color intrinsics:")
        print(
            f"  width={intrinsics.width}, height={intrinsics.height}, "
            f"fx={intrinsics.fx:.3f}, fy={intrinsics.fy:.3f}, cx={intrinsics.cx:.3f}, cy={intrinsics.cy:.3f}"
        )
        if active_mode != (args.width, args.height, args.fps):
            print(
                "Requested stream not supported, fallback to "
                f"{active_mode[0]}x{active_mode[1]}@{active_mode[2]} bgr8"
            )
        print(f"Detector backend: {detector.backend}")
        print("Controls: press 's' to print pose once, 'q' or ESC to quit.")
        print()
        sys.stdout.flush()

        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

            detections = detector.detect(gray)
            poses: List[PoseResult] = []
            for detection in detections:
                tag_id = int(getattr(detection, "tag_id", -1))
                if args.tag_id is not None and tag_id != args.tag_id:
                    continue
                pose = solve_tag_pose(detection, intrinsics, object_points)
                if pose is not None:
                    poses.append(pose)

            poses.sort(key=lambda item: item.tvec[2, 0])
            display = color_image.copy()
            for pose in poses:
                draw_detection_overlay(display, pose, axis_length_m, intrinsics)

            status_text = f"detections: {len(poses)}"
            cv2.putText(display, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)
            cv2.imshow(args.window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('s'):
                if poses:
                    print_pose(poses[0])
                else:
                    print("No matching AprilTag detected.")
                    print()
                    sys.stdout.flush()
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
