#!/usr/bin/env python3
"""Keyboard board control with synchronized camera/IMU angle logging.

The current powered pose is the count-space origin. Camera and IMU level zeros
are loaded from ``calib/board_zero.json`` and ``calib/imu_zero.json``; press
``z`` only when the board is physically level to replace both together.

Keys:
    A / D or left / right   alpha (roll) -/+
    S / W or down / up      beta (pitch) -/+
    [ / ]                   decrease/increase step size
    0                       command the captured level counts
    z                       jointly zero camera + IMU at current level
    q / Esc                 quit and release both servos

Each run writes CSV samples, JSONL command events, and a summary under
``artifacts/keyboard_compare/<timestamp>/``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import select
import sys
import termios
import time
import tty
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contract.servo_contract import ServoContract  # noqa: E402
from tag_vision.control.directional_calibration import (  # noqa: E402
    DirectionalMotorCalibration,
    DirectionalMotorOrigin,
)
from tag_vision.control.fused_trim import FusedStaticTrim  # noqa: E402
from tag_vision.core.angle_fusion import CameraImuFusion  # noqa: E402
from tag_vision.core.board_geometry import BoardGeometry  # noqa: E402
from tag_vision.core.board_pose import BoardPoseEstimator  # noqa: E402
from tag_vision.hardware.imu import BNO086Stream  # noqa: E402
from tag_vision.hardware.sts3215 import Mode, Register, STS3215Bus  # noqa: E402
from tools.camera_imu_check import (  # noqa: E402
    ImuReader,
    camera_source,
    capture_joint_zero,
    relative_angles_deg,
)

ROOT = Path(__file__).resolve().parents[1]
SERVO_IDS = (1, 2)
CSV_FIELDS = [
    "host_time", "elapsed_s", "frame", "pose_ok", "tag_count",
    "reprojection_px", "camera_alpha_deg", "camera_beta_deg",
    "imu_alpha_deg", "imu_beta_deg", "delta_alpha_deg", "delta_beta_deg",
    "fused_alpha_deg", "fused_beta_deg", "fusion_camera_alpha_used",
    "fusion_camera_beta_used",
    "imu_sd_alpha_deg", "imu_sd_beta_deg", "imu_accuracy",
    "target_alpha_deg", "target_beta_deg", "servo1_counts", "servo2_counts",
    "goal_servo1_counts", "goal_servo2_counts", "trim_error_alpha_deg",
    "trim_error_beta_deg", "trim_iteration", "trim_active",
    "detection_mode", "settled",
]


def clamp(value: float, low: float, high: float) -> float:
    return min(max(float(value), low), high)


def key_action(key: int) -> str | None:
    """Normalize ASCII and common OpenCV/Linux arrow key codes."""
    mapping = {
        ord("a"): "alpha_down", ord("A"): "alpha_down",
        ord("d"): "alpha_up", ord("D"): "alpha_up",
        ord("s"): "beta_down", ord("S"): "beta_down",
        ord("w"): "beta_up", ord("W"): "beta_up",
        ord("["): "step_down", ord("]"): "step_up",
        ord("0"): "level", ord("z"): "zero", ord("Z"): "zero",
        ord("q"): "quit", ord("Q"): "quit", 27: "quit",
        81: "alpha_down", 2424832: "alpha_down",
        83: "alpha_up", 2555904: "alpha_up",
        84: "beta_down", 2621440: "beta_down",
        82: "beta_up", 2490368: "beta_up",
        65361: "alpha_down", 65363: "alpha_up",
        65364: "beta_down", 65362: "beta_up",
    }
    return mapping.get(key)


class TerminalKeys:
    """Non-blocking single-key input alongside the OpenCV window."""

    def __init__(self) -> None:
        self.fd = sys.stdin.fileno() if sys.stdin.isatty() else None
        self.previous = None

    def enable(self) -> None:
        if self.fd is not None:
            self.previous = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)

    def restore(self) -> None:
        if self.fd is not None and self.previous is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.previous)
            self.previous = None

    def poll(self) -> int:
        if self.fd is None:
            return -1
        ready, _, _ = select.select([self.fd], [], [], 0.0)
        if not ready:
            return -1
        data = os.read(self.fd, 1)
        return data[0] if data else -1


class MotorOrigin:
    """Measured angle deltas relative to the encoders at joint zero."""

    def __init__(self, contract: ServoContract, counts: dict[int, int]) -> None:
        self.contract = contract
        self.counts = dict(counts)

    def targets(self, alpha_deg: float, beta_deg: float) -> dict[int, int]:
        result: dict[int, int] = {}
        for axis, angle_deg in zip(
            self.contract.axes, (alpha_deg, beta_deg)
        ):
            delta = (axis.sign * axis.counts_per_rad
                     * math.radians(float(angle_deg)))
            raw = int(round(self.counts[axis.servo_id] + delta))
            result[axis.servo_id] = min(max(raw, axis.min_counts), axis.max_counts)
        return result


def servo_positions(bus: STS3215Bus) -> dict[int, int]:
    return {
        servo_id: bus.read_word(servo_id, Register.PRESENT_POSITION)
        for servo_id in SERVO_IDS
    }


def hold_current(bus: STS3215Bus) -> dict[int, int]:
    """Enable torque without jumping to a stale goal register."""
    positions = servo_positions(bus)
    for servo_id in SERVO_IDS:
        bus.apply_config(servo_id)
        bus.set_goal_position(servo_id, positions[servo_id])
        bus.torque_enable(servo_id)
    return positions


def release(bus: STS3215Bus) -> None:
    for servo_id in SERVO_IDS:
        try:
            bus.torque_disable(servo_id)
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", default="auto")
    parser.add_argument("--imu-port", default=None)
    parser.add_argument("--servo-port", default=None)
    parser.add_argument("--calib", type=Path,
                        default=ROOT / "calib/camera_calib.json")
    parser.add_argument("--board", type=Path,
                        default=ROOT / "calib/board_tags.json")
    parser.add_argument("--camera-zero", type=Path,
                        default=ROOT / "calib/board_zero.json")
    parser.add_argument("--imu-zero", type=Path,
                        default=ROOT / "calib/imu_zero.json")
    parser.add_argument("--servo-calibration", type=Path,
                        default=ROOT / "calib/servo_calibration.json")
    parser.add_argument("--directional-calibration", type=Path, default=None,
                        help="candidate direction-dependent fused lookup table")
    parser.add_argument("--fused-trim", action="store_true",
                        help="apply bounded settled fused-angle corrections")
    parser.add_argument("--trim-tolerance-deg", type=float, default=0.10)
    parser.add_argument("--trim-delay", type=float, default=0.8)
    parser.add_argument("--trim-hold", type=float, default=0.4,
                        help="continuous in-tolerance time before convergence")
    parser.add_argument("--trim-gain", type=float, default=0.7)
    parser.add_argument("--trim-max-step-counts", type=int, default=80)
    parser.add_argument("--trim-max-iterations", type=int, default=6)
    parser.add_argument("--step-deg", type=float, default=0.5)
    parser.add_argument("--max-angle-deg", type=float, default=4.0)
    parser.add_argument("--settle-window", type=float, default=0.25)
    parser.add_argument("--settle-sd", type=float, default=0.05)
    parser.add_argument("--fusion-time-constant", type=float, default=0.5)
    parser.add_argument("--fusion-camera-gate", type=float, default=2.0)
    parser.add_argument("--max-reprojection", type=float, default=4.0)
    parser.add_argument("--max-load", type=int, default=350)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.step_deg <= 0 or args.max_angle_deg <= 0:
        parser.error("--step-deg and --max-angle-deg must be positive")
    if args.imu_port is not None and args.imu_port == args.servo_port:
        parser.error("IMU and servo ports must differ")
    if args.fused_trim and args.directional_calibration is None:
        parser.error("--fused-trim requires --directional-calibration")
    for path in (args.camera_zero, args.imu_zero):
        if not path.is_file():
            parser.error(f"missing joint zero file: {path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / "artifacts/keyboard_compare" / stamp
    output.mkdir(parents=True, exist_ok=False)

    geometry = BoardGeometry.load(args.board)
    estimator = BoardPoseEstimator(args.calib, geometry, min_tags=4)
    estimator.load_zero(args.camera_zero)
    contract = ServoContract.from_json(args.servo_calibration)
    if not contract.measured:
        print("Servo calibration is not measured; refusing motor commands.")
        return 1

    try:
        imu = BNO086Stream(port=args.imu_port)
        imu.load_zero(args.imu_zero)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not open/load IMU: {exc}")
        return 1
    reader = ImuReader(imu)
    reader.start()

    source = camera_source(args.camera)
    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    if not capture.isOpened():
        reader.stop(); imu.close()
        print(f"Could not open camera {source!r}")
        return 1
    actual = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    expected = estimator.image_size
    if actual[0] * expected[1] != actual[1] * expected[0]:
        capture.release(); reader.stop(); imu.close()
        print(f"Camera aspect {actual} does not match calibration {expected}")
        return 1

    try:
        bus = STS3215Bus(port=args.servo_port)
        for servo_id in SERVO_IDS:
            if not bus.ping(servo_id):
                raise RuntimeError(f"servo {servo_id} did not answer")
            mode = bus.read_byte(servo_id, Register.MODE)
            if mode != Mode.POSITION:
                raise RuntimeError(
                    f"servo {servo_id} is in {Mode(mode).name}, not POSITION")
        base_counts = hold_current(bus)
    except Exception as exc:  # noqa: BLE001
        capture.release(); reader.stop(); imu.close()
        print(f"Could not initialize servos: {exc}")
        return 1

    directional = (DirectionalMotorCalibration.from_json(
        args.directional_calibration) if args.directional_calibration else None)
    origin = (DirectionalMotorOrigin(directional) if directional is not None
              else MotorOrigin(contract, base_counts))
    trim = (FusedStaticTrim(
        directional, tolerance_deg=args.trim_tolerance_deg,
        settle_delay_s=args.trim_delay, convergence_hold_s=args.trim_hold,
        gain=args.trim_gain,
        max_step_counts=args.trim_max_step_counts,
        max_iterations=args.trim_max_iterations)
        if args.fused_trim and directional is not None else None)
    fusion = CameraImuFusion(
        args.fusion_time_constant, args.fusion_camera_gate)
    target_alpha = target_beta = 0.0
    step_deg = args.step_deg
    commanded_counts = dict(base_counts)
    goal_counts = dict(base_counts)
    camera_rotations: deque[np.ndarray] = deque(maxlen=30)
    stable_pairs: list[tuple[float, float, float, float]] = []
    commands = 0
    frames = poses = 0
    frame_index = 0
    last_state_read = 0.0
    load_faults = 0
    trim_events = 0
    start_monotonic = time.monotonic()
    events_path = output / "commands.jsonl"

    manifest = {
        "created": datetime.now().isoformat(),
        "camera": source,
        "camera_size": actual,
        "step_deg": step_deg,
        "max_angle_deg": args.max_angle_deg,
        "base_counts": base_counts,
        "camera_zero": str(args.camera_zero.resolve()),
        "imu_zero": str(args.imu_zero.resolve()),
        "servo_calibration": str(args.servo_calibration.resolve()),
        "directional_calibration": (str(args.directional_calibration.resolve())
                                    if args.directional_calibration else None),
        "fusion": {
            "camera_time_constant_s": args.fusion_time_constant,
            "max_camera_residual_deg": args.fusion_camera_gate,
        },
        "fused_trim": ({
            "enabled": True,
            "tolerance_deg": args.trim_tolerance_deg,
            "delay_s": args.trim_delay,
            "convergence_hold_s": args.trim_hold,
            "gain": args.trim_gain,
            "max_step_counts": args.trim_max_step_counts,
            "max_iterations": args.trim_max_iterations,
            "jacobian_deg_per_count": directional.jacobian_deg_per_count,
        } if trim is not None else {"enabled": False}),
        "keys": "A/D alpha, S/W beta, [/] step, 0 level, z joint zero, q quit",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Holding current level counts: {base_counts}")
    print(f"Logging to {output}")
    print("A/D alpha  S/W beta  [/] step  0 level  z joint zero  q quit")
    terminal_keys = TerminalKeys()
    terminal_keys.enable()

    def command(action: str) -> None:
        nonlocal target_alpha, target_beta, step_deg, commanded_counts, commands
        nonlocal origin, base_counts, goal_counts
        if action == "alpha_down": target_alpha -= step_deg
        elif action == "alpha_up": target_alpha += step_deg
        elif action == "beta_down": target_beta -= step_deg
        elif action == "beta_up": target_beta += step_deg
        elif action == "level": target_alpha = target_beta = 0.0
        elif action == "step_down":
            step_deg = max(0.1, step_deg / 2.0)
            return
        elif action == "step_up":
            step_deg = min(2.0, step_deg * 2.0)
            return
        target_alpha = clamp(
            target_alpha, -args.max_angle_deg, args.max_angle_deg)
        target_beta = clamp(
            target_beta, -args.max_angle_deg, args.max_angle_deg)
        goal_counts = origin.targets(target_alpha, target_beta)
        commanded_counts = dict(goal_counts)
        bus.sync_write_positions(goal_counts)
        if trim is not None:
            trim.arm((target_alpha, target_beta), goal_counts, time.monotonic())
        commands += 1
        event = {
            "host_time": time.time(), "action": action,
            "target_alpha_deg": target_alpha,
            "target_beta_deg": target_beta,
            "counts": goal_counts,
        }
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    try:
        with (output / "samples.csv").open(
            "w", newline="", encoding="utf-8") as csv_handle:
            writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            while True:
                ok, full_frame = capture.read()
                if not ok:
                    print("\nCamera stopped returning frames")
                    break
                host_time = time.time()
                native_gray = cv2.cvtColor(full_frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(
                    full_frame, expected, interpolation=cv2.INTER_AREA)
                pose = estimator.estimate(native_gray)
                pose_ok = bool(
                    pose is not None
                    and pose.reprojection_px <= args.max_reprojection)
                frames += 1
                if pose_ok:
                    poses += 1
                    camera_rotations.append(pose.rotation.copy())

                rows = reader.recent(args.settle_window)
                imu_mean = imu_sd = np.array([math.nan, math.nan])
                accuracy = -1
                if rows:
                    imu_angles = relative_angles_deg(
                        rows, imu.zero_rotation, imu.mount_rotation)
                    imu_mean = np.mean(imu_angles, axis=0)
                    imu_sd = np.std(imu_angles, axis=0)
                    accuracy = min(row[2] for row in rows)
                settled = bool(
                    len(rows) >= 20 and np.all(np.isfinite(imu_sd))
                    and float(np.max(imu_sd)) <= args.settle_sd)
                camera_angles = np.array([
                    pose.alpha_deg, pose.beta_deg]) if pose_ok else np.array([
                        math.nan, math.nan])
                delta = camera_angles - imu_mean
                latest = reader.latest()
                latest_imu = None
                latest_imu_time = None
                if latest is not None:
                    latest_imu = relative_angles_deg(
                        [latest], imu.zero_rotation, imu.mount_rotation)[0]
                    latest_imu_time = latest[0]
                fused = fusion.update(
                    camera_angles if pose_ok else None, latest_imu,
                    timestamp=host_time, imu_timestamp=latest_imu_time)
                fused_angles = (fused.array if fused is not None else
                                np.array([math.nan, math.nan]))
                trim_error = np.array([
                    target_alpha, target_beta], dtype=np.float64) - fused_angles
                if pose_ok and settled:
                    stable_pairs.append((
                        float(camera_angles[0]), float(camera_angles[1]),
                        float(imu_mean[0]), float(imu_mean[1])))

                now = time.monotonic()
                trim_update = (trim.update(
                    fused_angles, timestamp=now, settled=settled)
                    if trim is not None else None)
                if trim_update is not None:
                    goal_counts = dict(trim_update.counts)
                    if any(trim_update.delta_counts):
                        bus.sync_write_positions(goal_counts)
                        trim_events += 1
                    event = {
                        "host_time": host_time,
                        "action": "fused_trim",
                        "target_alpha_deg": target_alpha,
                        "target_beta_deg": target_beta,
                        "error_deg": trim_update.error_deg,
                        "delta_counts_alpha_beta": trim_update.delta_counts,
                        "counts": goal_counts,
                        "iteration": trim.iterations,
                        "converged": trim_update.converged,
                        "exhausted": trim_update.exhausted,
                    }
                    with events_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(event) + "\n")
                if now - last_state_read >= 0.2:
                    last_state_read = now
                    states = {i: bus.read_state(i) for i in SERVO_IDS}
                    commanded_counts = {
                        i: states[i].position for i in SERVO_IDS}
                    high_and_still = any(
                        abs(state.load) > args.max_load and not state.moving
                        for state in states.values())
                    load_faults = load_faults + 1 if high_and_still else 0
                    if load_faults >= 3:
                        print("\nLOAD FAULT: releasing both servos")
                        break

                writer.writerow({
                    "host_time": host_time,
                    "elapsed_s": now - start_monotonic,
                    "frame": frame_index,
                    "pose_ok": int(pose_ok),
                    "tag_count": len(estimator.last_found),
                    "reprojection_px": pose.reprojection_px if pose_ok else "",
                    "camera_alpha_deg": camera_angles[0],
                    "camera_beta_deg": camera_angles[1],
                    "imu_alpha_deg": imu_mean[0],
                    "imu_beta_deg": imu_mean[1],
                    "delta_alpha_deg": delta[0],
                    "delta_beta_deg": delta[1],
                    "fused_alpha_deg": fused_angles[0],
                    "fused_beta_deg": fused_angles[1],
                    "fusion_camera_alpha_used": int(
                        fused.camera_used[0]) if fused else 0,
                    "fusion_camera_beta_used": int(
                        fused.camera_used[1]) if fused else 0,
                    "imu_sd_alpha_deg": imu_sd[0],
                    "imu_sd_beta_deg": imu_sd[1],
                    "imu_accuracy": accuracy,
                    "target_alpha_deg": target_alpha,
                    "target_beta_deg": target_beta,
                    "servo1_counts": commanded_counts[1],
                    "servo2_counts": commanded_counts[2],
                    "goal_servo1_counts": goal_counts[1],
                    "goal_servo2_counts": goal_counts[2],
                    "trim_error_alpha_deg": trim_error[0],
                    "trim_error_beta_deg": trim_error[1],
                    "trim_iteration": trim.iterations if trim else 0,
                    "trim_active": int(trim.active) if trim else 0,
                    "detection_mode": estimator.last_detection_mode,
                    "settled": int(settled),
                })
                frame_index += 1

                display = estimator.undistort_frame(frame)
                for tag_id, points in estimator.last_found.items():
                    shown = np.rint(estimator.undistort(points)).astype(np.int32)
                    cv2.polylines(display, [shown], True, (0, 255, 0), 2)
                    centre = tuple(shown.mean(axis=0).astype(int))
                    cv2.putText(display, str(tag_id), centre,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
                lines = [
                    (f"TARGET  alpha {target_alpha:+.2f}  beta {target_beta:+.2f} "
                     f"step {step_deg:.2f} deg"),
                    (f"CAM     alpha {camera_angles[0]:+.3f}  "
                     f"beta {camera_angles[1]:+.3f}"),
                    (f"IMU     alpha {imu_mean[0]:+.3f}  beta {imu_mean[1]:+.3f}  "
                     f"sd {float(np.max(imu_sd)):.3f}"),
                    (f"FUSED   alpha {fused_angles[0]:+.3f}  "
                     f"beta {fused_angles[1]:+.3f}"),
                    (f"DELTA   alpha {delta[0]:+.3f}  beta {delta[1]:+.3f}  "
                     f"{'SETTLED' if settled else 'MOVING'}"),
                    (f"TRIM    error {trim_error[0]:+.3f} / "
                     f"{trim_error[1]:+.3f}  iter "
                     f"{trim.iterations if trim else 0}  "
                     f"{'ACTIVE' if trim and trim.active else 'IDLE'}"),
                    (f"COUNTS  {commanded_counts[1]} / {commanded_counts[2]}  "
                     "A/D roll  S/W pitch  0 level  z zero  q quit"),
                ]
                for index, text in enumerate(lines):
                    cv2.putText(display, text, (10, 28 + 27 * index),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                                (0, 255, 0) if index < 6 else (0, 255, 255), 2)
                cv2.imshow("keyboard motor | camera vs IMU", display)
                window_key = cv2.waitKeyEx(1)
                terminal_key = terminal_keys.poll()
                action = key_action(
                    terminal_key if terminal_key >= 0 else window_key)
                if action == "quit":
                    break
                if action == "zero":
                    if directional is not None:
                        print("\nCannot re-zero with a directional candidate "
                              "loaded; regenerate it from the new zero.")
                        continue
                    zero_counts = servo_positions(bus)
                    if capture_joint_zero(
                        estimator, imu, reader, camera_rotations,
                        args.camera_zero, args.imu_zero,
                        imu_extra={
                            "servo_counts_at_zero": {
                                str(key): value
                                for key, value in zero_counts.items()
                            },
                        }):
                        base_counts = servo_positions(bus)
                        origin = MotorOrigin(contract, base_counts)
                        target_alpha = target_beta = 0.0
                        commanded_counts = dict(base_counts)
                        goal_counts = dict(base_counts)
                        fusion.reset([0.0, 0.0], [0.0, 0.0],
                                     timestamp=time.time(),
                                     imu_timestamp=(latest[0]
                                                    if latest else None))
                elif action is not None:
                    command(action)
    except KeyboardInterrupt:
        pass
    finally:
        terminal_keys.restore()
        release(bus)
        bus.close()
        capture.release()
        reader.stop()
        imu.close()
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass

    summary = {
        "frames": frames,
        "poses": poses,
        "pose_rate": poses / max(frames, 1),
        "commands": commands,
        "trim_events": trim_events,
        "stable_comparisons": len(stable_pairs),
        "imu_dropped": imu.dropped,
        "imu_crc_errors": imu.crc_errors,
        "load_fault": load_faults >= 3,
    }
    if stable_pairs:
        values = np.asarray(stable_pairs)
        error = values[:, :2] - values[:, 2:]
        summary["settled_bias_deg"] = np.mean(error, axis=0).tolist()
        summary["settled_rmse_deg"] = np.sqrt(
            np.mean(error ** 2, axis=0)).tolist()
        summary["imu_span_deg"] = np.ptp(values[:, 2:], axis=0).tolist()
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Torque released. Logs: {output}")
    return 1 if summary["load_fault"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
