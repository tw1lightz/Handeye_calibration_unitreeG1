"""
Terminal tool for switching the G1_29 right arm between:
1. free-drag mode with gravity compensation
2. lock mode that holds the current joint pose

Examples:
    python right_arm_mode.py
    python right_arm_mode.py --mode free
    python right_arm_mode.py --sim
"""

import argparse
import os
import subprocess
import sys
import threading
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import numpy as np
from utils.pinocchio_compat import import_pinocchio

pin = import_pinocchio()

import logging_mp

logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from robot_control.robot_arm import G1_29_ArmController, G1_29_JointArmIndex
from robot_control.robot_arm_ik import G1_29_ArmIK


SIM_MODE = False
ARM_VELOCITY_LIMIT = 8.0
CONTROL_HZ = 250.0

LEFT_ARM_JOINTS = [joint for joint in G1_29_JointArmIndex if joint.name.startswith("kLeft")]
RIGHT_ARM_JOINTS = [joint for joint in G1_29_JointArmIndex if joint.name.startswith("kRight")]
RIGHT_WRIST_JOINTS = {
    G1_29_JointArmIndex.kRightWristRoll,
    G1_29_JointArmIndex.kRightWristPitch,
    G1_29_JointArmIndex.kRightWristYaw,
}
EXPECTED_MODE_MACHINES = {2, 5, 9}
CONFLICT_KEYWORDS = (
    "g1_controller_commands.py",
    "teleop_hand_and_arm.py",
    "free_arm_demo.py",
    "point_control_demo.py",
    "right_arm_mode.py",
)


class RightArmModeSession:
    def __init__(
        self,
        sim_mode=False,
        network_interface=None,
        arm_velocity_limit=ARM_VELOCITY_LIMIT,
        control_hz=CONTROL_HZ,
        free_kp=0.0,
        free_kd=1.0,
        free_wrist_kp=0.0,
        free_wrist_kd=0.3,
        stop_conflicts=False,
    ):
        self.sim_mode = sim_mode
        self.control_dt = 1.0 / control_hz
        self.free_kp = free_kp
        self.free_kd = free_kd
        self.free_wrist_kp = free_wrist_kp
        self.free_wrist_kd = free_wrist_kd
        self.stop_conflicts = stop_conflicts

        domain_id = 1 if sim_mode else 0
        ChannelFactoryInitialize(domain_id, networkInterface=network_interface)

        self.arm_ik = G1_29_ArmIK(Visualization=False)
        self.arm_ctrl = G1_29_ArmController(motion_mode=True, simulation_mode=sim_mode)
        if not sim_mode:
            self.arm_ctrl.arm_velocity_limit = arm_velocity_limit

        current_q = self.arm_ctrl.get_current_dual_arm_q()
        self.left_dof = len(LEFT_ARM_JOINTS)
        self.right_dof = len(RIGHT_ARM_JOINTS)
        self.arm_dof = current_q.shape[0]

        if self.arm_dof == self.right_dof:
            self.left_dof = 0
        elif self.arm_dof != (self.left_dof + self.right_dof):
            raise ValueError(
                f"Unexpected arm dof={self.arm_dof}, expected right-only {self.right_dof} or dual-arm {self.left_dof + self.right_dof}"
            )

        self.left_lock_q = self._left_slice(current_q).copy()
        self.right_lock_q = self._right_slice(current_q).copy()
        self.mode = "lock"

        self._mode_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._started = False

        self._zero_v = np.zeros(self.arm_ik.reduced_robot.model.nv)

    def _left_slice(self, q):
        if self.left_dof == 0:
            return q[:0]
        return q[: self.left_dof]

    def _right_slice(self, q):
        return q[-self.right_dof :]

    def start(self):
        self._apply_right_arm_gains("lock")
        self._control_thread.start()
        self._started = True
        self._warn_about_conflicting_processes(stop_conflicts=self.stop_conflicts)
        if self.arm_ctrl.motion_mode:
            if self.arm_ctrl.wait_for_motion_mode_ready(timeout=3.0):
                logger_mp.info(
                    "arm_sdk ready | mode_machine=%s | mode_pr=%s | weight=%.2f",
                    self.arm_ctrl.get_mode_machine(),
                    self.arm_ctrl.get_mode_pr(),
                    self.arm_ctrl.get_motion_mode_weight(),
                )
            else:
                logger_mp.warning("arm_sdk ramp-up did not finish within 3s; commands may not take effect yet.")

        mode_machine = self.arm_ctrl.get_mode_machine()
        if mode_machine not in EXPECTED_MODE_MACHINES:
            logger_mp.warning(
                "Current mode_machine=%s is unusual for G1_29 arm control (expected one of %s).",
                mode_machine,
                sorted(EXPECTED_MODE_MACHINES),
            )

        if not self.sim_mode:
            logger_mp.info(
                "Real robot note: rt/arm_sdk only takes effect when G1 is in Regular control mode (R1+X); "
                "Running mode (R2+A) will ignore these commands."
            )

    def _list_conflicting_processes(self):
        conflicts = []
        try:
            my_pid = os.getpid()
            result = subprocess.run(
                ["ps", "-eo", "pid,user,cmd"],
                check=False,
                text=True,
                capture_output=True,
            )
            if result.returncode != 0:
                return conflicts

            for line in result.stdout.splitlines()[1:]:
                parts = line.strip().split(None, 2)
                if len(parts) < 3:
                    continue
                pid_str, user, cmd = parts
                try:
                    pid = int(pid_str)
                except ValueError:
                    continue
                if pid == my_pid:
                    continue
                if "python" not in cmd:
                    continue
                if any(keyword in cmd for keyword in CONFLICT_KEYWORDS):
                    conflicts.append((pid, user, cmd))
        except Exception as exc:
            logger_mp.debug("Process conflict check skipped: %s", exc)
        return conflicts

    def _warn_about_conflicting_processes(self, stop_conflicts=False):
        conflicts = self._list_conflicting_processes()
        if not conflicts:
            return

        logger_mp.warning(
            "Detected potential control-process conflicts (these may override rt/arm_sdk commands):"
        )
        for pid, user, cmd in conflicts[:8]:
            logger_mp.warning("  pid=%s user=%s cmd=%s", pid, user, cmd)

        if not stop_conflicts:
            logger_mp.warning(
                "Please stop other arm-control scripts before testing free/lock. "
                "You can also relaunch with --stop-conflicts."
            )
            return

        target_pids = [str(pid) for pid, _user, _cmd in conflicts]
        logger_mp.warning("Trying to stop %d conflicting process(es) via sudo -n kill ...", len(target_pids))
        result = subprocess.run(
            ["sudo", "-n", "kill", *target_pids],
            check=False,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            stderr_text = (result.stderr or "").strip()
            logger_mp.warning(
                "Failed to stop conflicting processes automatically. stderr=%s",
                stderr_text if stderr_text else "<empty>",
            )
            return

        time.sleep(0.3)
        remaining = self._list_conflicting_processes()
        if remaining:
            logger_mp.warning("Some conflicting processes are still running after auto-stop:")
            for pid, user, cmd in remaining[:8]:
                logger_mp.warning("  pid=%s user=%s cmd=%s", pid, user, cmd)
        else:
            logger_mp.info("Conflicting control processes were stopped successfully.")

    def _apply_right_arm_gains(self, mode):
        for joint_id in RIGHT_ARM_JOINTS:
            if mode == "free":
                if joint_id in RIGHT_WRIST_JOINTS:
                    kp, kd = self.free_wrist_kp, self.free_wrist_kd
                else:
                    kp, kd = self.free_kp, self.free_kd
            else:
                if joint_id in RIGHT_WRIST_JOINTS:
                    kp, kd = self.arm_ctrl.kp_wrist, self.arm_ctrl.kd_wrist
                else:
                    kp, kd = self.arm_ctrl.kp_low, self.arm_ctrl.kd_low

            self.arm_ctrl.msg.motor_cmd[joint_id].mode = 1
            self.arm_ctrl.msg.motor_cmd[joint_id].kp = kp
            self.arm_ctrl.msg.motor_cmd[joint_id].kd = kd

    def _compute_gravity_torque(self, current_q):
        return pin.rnea(
            self.arm_ik.reduced_robot.model,
            self.arm_ik.reduced_robot.data,
            current_q,
            self._zero_v,
            self._zero_v,
        )

    def _control_loop(self):
        while not self._stop_event.is_set():
            loop_start = time.time()
            current_q = self.arm_ctrl.get_current_dual_arm_q()
            tauff = self._compute_gravity_torque(current_q)

            with self._mode_lock:
                mode = self.mode
                left_lock_q = self.left_lock_q.copy()
                right_lock_q = self.right_lock_q.copy()

            q_target = current_q.copy()
            if self.left_dof > 0:
                q_target[: self.left_dof] = left_lock_q
            if mode == "lock":
                q_target[-self.right_dof :] = right_lock_q
            else:
                q_target[-self.right_dof :] = self._right_slice(current_q)

            self.arm_ctrl.ctrl_dual_arm(q_target, tauff)

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, self.control_dt - elapsed))

    def set_mode(self, mode):
        if mode not in {"free", "lock"}:
            raise ValueError(f"Unsupported mode: {mode}")

        current_q = self.arm_ctrl.get_current_dual_arm_q()
        with self._mode_lock:
            self.left_lock_q = self._left_slice(current_q).copy()
            if mode == "lock":
                self.right_lock_q = self._right_slice(current_q).copy()
            self.mode = mode

        self._apply_right_arm_gains(mode)
        right_q_deg = np.array2string(np.degrees(self._right_slice(current_q)), precision=1, suppress_small=True)
        right_kp = [self.arm_ctrl.msg.motor_cmd[joint_id].kp for joint_id in RIGHT_ARM_JOINTS]
        right_kd = [self.arm_ctrl.msg.motor_cmd[joint_id].kd for joint_id in RIGHT_ARM_JOINTS]
        mode_machine = self.arm_ctrl.get_mode_machine()
        mode_pr = self.arm_ctrl.get_mode_pr()
        motion_weight = self.arm_ctrl.get_motion_mode_weight()
        logger_mp.info(
            "Right arm mode -> %s | right arm q(deg)=%s | kp=%s | kd=%s | mode_machine=%s | mode_pr=%s | weight=%.2f",
            mode,
            right_q_deg,
            np.array2string(np.array(right_kp), precision=2, suppress_small=True),
            np.array2string(np.array(right_kd), precision=2, suppress_small=True),
            mode_machine,
            mode_pr,
            motion_weight,
        )

    def print_status(self):
        current_q = self.arm_ctrl.get_current_dual_arm_q()
        with self._mode_lock:
            mode = self.mode
            right_lock_q = self.right_lock_q.copy()

        print("")
        print(f"mode: {mode}")
        print(f"dds mode_machine: {self.arm_ctrl.get_mode_machine()}")
        print(f"dds mode_pr     : {self.arm_ctrl.get_mode_pr()}")
        print(f"motion weight   : {self.arm_ctrl.get_motion_mode_weight():.2f}")
        print(f"motion ready    : {self.arm_ctrl.is_motion_mode_ready()}")
        print("right q current (deg):", np.array2string(np.degrees(self._right_slice(current_q)), precision=2, suppress_small=True))
        print("right q lock    (deg):", np.array2string(np.degrees(right_lock_q), precision=2, suppress_small=True))
        if not self.sim_mode:
            print("hint: if weight=1.00 but the arm still does not respond, check the R3 remote is in Regular control mode (R1+X).")
        print("")

    def shutdown(self):
        current_q = self.arm_ctrl.get_current_dual_arm_q()
        with self._mode_lock:
            self.left_lock_q = self._left_slice(current_q).copy()
            self.right_lock_q = self._right_slice(current_q).copy()
            self.mode = "lock"
        self._apply_right_arm_gains("lock")
        self._stop_event.set()
        if self._started:
            self._control_thread.join(timeout=1.0)

        q_target = np.concatenate([self.left_lock_q, self.right_lock_q])
        hold_steps = max(1, int(0.5 / self.control_dt))
        for _ in range(hold_steps):
            current_q = self.arm_ctrl.get_current_dual_arm_q()
            tauff = self._compute_gravity_torque(current_q)
            self.arm_ctrl.ctrl_dual_arm(q_target, tauff)
            time.sleep(self.control_dt)

        self.arm_ctrl.smooth_exit()


def build_argparser():
    parser = argparse.ArgumentParser(description="Switch the G1_29 right arm between free and lock modes.")
    parser.add_argument(
        "--mode",
        choices=["interactive", "free", "lock"],
        default="interactive",
        help="interactive: switch in the terminal; free: start in free-drag mode; lock: start in lock mode",
    )
    parser.add_argument("--sim", action="store_true", default=SIM_MODE, help="Simulation mode (DDS domain 1)")
    parser.add_argument("--real", action="store_true", help="Real robot mode (DDS domain 0), overrides --sim")
    parser.add_argument("--network-interface", type=str, default=None, help="DDS network interface name")
    parser.add_argument(
        "--arm-velocity-limit",
        type=float,
        default=ARM_VELOCITY_LIMIT,
        help="Real robot arm velocity limit in rad/s",
    )
    parser.add_argument("--control-hz", type=float, default=CONTROL_HZ, help="Free/lock update frequency")
    parser.add_argument("--free-kp", type=float, default=0.0, help="Right shoulder/elbow kp in free mode")
    parser.add_argument("--free-kd", type=float, default=1.0, help="Right shoulder/elbow kd in free mode")
    parser.add_argument("--free-wrist-kp", type=float, default=0.0, help="Right wrist kp in free mode")
    parser.add_argument("--free-wrist-kd", type=float, default=0.3, help="Right wrist kd in free mode")
    parser.add_argument(
        "--stop-conflicts",
        action="store_true",
        help="Try to stop conflicting control scripts automatically via sudo -n kill",
    )
    return parser


def run_interactive(session):
    session.start()
    session.set_mode("lock")

    print("")
    print("Commands: free | lock | status | quit")
    print("Ctrl+C also exits safely.")
    print("")

    while True:
        try:
            command = input("right-arm> ").strip().lower()
        except EOFError:
            break

        if command in {"free", "f"}:
            session.set_mode("free")
        elif command in {"lock", "l"}:
            session.set_mode("lock")
        elif command in {"status", "s"}:
            session.print_status()
        elif command in {"quit", "q", "exit"}:
            break
        elif command == "":
            continue
        else:
            print("Unknown command. Use: free | lock | status | quit")


def run_fixed_mode(session, mode):
    session.start()
    session.set_mode(mode)
    print("")
    print(f"Right arm is now in {mode} mode. Press Ctrl+C to exit.")
    print("")
    while True:
        time.sleep(1.0)


def main():
    args = build_argparser().parse_args()
    sim_mode = args.sim and not args.real

    session = RightArmModeSession(
        sim_mode=sim_mode,
        network_interface=args.network_interface,
        arm_velocity_limit=args.arm_velocity_limit,
        control_hz=args.control_hz,
        free_kp=args.free_kp,
        free_kd=args.free_kd,
        free_wrist_kp=args.free_wrist_kp,
        free_wrist_kd=args.free_wrist_kd,
        stop_conflicts=args.stop_conflicts,
    )

    try:
        if args.mode == "interactive":
            run_interactive(session)
        else:
            run_fixed_mode(session, args.mode)
    except KeyboardInterrupt:
        logger_mp.info("Received exit signal, switching to safe shutdown.")
    finally:
        session.shutdown()
        logger_mp.info("Program closed.")


if __name__ == "__main__":
    main()
