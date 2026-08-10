#!/usr/bin/env python3
"""Real-hardware marble viewer, model-data logger, and adaptive reset brake.

Safe sequence:
  1. Collect modest manual motion (default is observation-only):
       python3 tools/real_ball_control.py --mode manual --execute
  2. Fit it:
       python3 tools/fit_ball_dynamics.py artifacts/ball_control/.../samples.csv
  3. Preview reset commands without moving motors:
       python3 tools/real_ball_control.py --mode reset
  4. Enable reset braking only after the preview signs look correct:
       python3 tools/real_ball_control.py --mode reset --execute

Manual keys: A/D alpha, S/W beta, 0 level, Q/Esc quit.
Identify mode holds level while positioning; press Space to begin excitation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tag_vision.control.ball_dynamics import BallDynamicsModel  # noqa: E402
from tag_vision.control.ball_state import BallStateFilter  # noqa: E402
from tag_vision.control.directional_calibration import (  # noqa: E402
    DirectionalMotorCalibration, DirectionalMotorOrigin)
from tag_vision.control.reset_brake import AdaptiveResetBrake  # noqa: E402
from tag_vision.control.fixed_reset_brake import FixedResetBrake  # noqa: E402
from tag_vision.core.angle_fusion import CameraImuFusion  # noqa: E402
from tag_vision.core.ball_detection import BlueBallDetector  # noqa: E402
from tag_vision.core.board_geometry import BoardGeometry  # noqa: E402
from tag_vision.core.board_pose import BoardPoseEstimator  # noqa: E402
from tag_vision.hardware.imu import BNO086Stream  # noqa: E402
from tag_vision.hardware.sts3215 import Mode, Register, STS3215Bus  # noqa: E402
from tools.camera_imu_check import (  # noqa: E402
    ImuReader, camera_source, relative_angles_deg)
from tools.keyboard_motor_compare import hold_current, release  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("time_s", "ball_visible", "x_mm", "y_mm", "raw_x_mm", "raw_y_mm",
          "ball_confidence", "vx_mm_s", "vy_mm_s",
          "speed_mm_s", "camera_alpha_deg", "camera_beta_deg",
          "imu_alpha_deg", "imu_beta_deg", "fused_alpha_deg",
          "fused_beta_deg", "target_alpha_deg", "target_beta_deg", "phase")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=(
        "observe", "manual", "identify", "continuous-id", "hard-reset",
        "reset"),
                        default="observe")
    parser.add_argument("--execute", action="store_true",
                        help="actually enable and command the servos")
    parser.add_argument("--camera", default="auto")
    parser.add_argument("--servo-port", default=None)
    parser.add_argument("--imu-port", default=None)
    parser.add_argument("--calib", type=Path,
                        default=ROOT / "calib/camera_calib.json")
    parser.add_argument("--board", type=Path,
                        default=ROOT / "calib/board_tags.json")
    parser.add_argument("--camera-zero", type=Path,
                        default=ROOT / "calib/board_zero.json")
    parser.add_argument("--imu-zero", type=Path,
                        default=ROOT / "calib/imu_zero.json")
    parser.add_argument("--motor-calibration", type=Path,
                        default=ROOT / "calib/directional_motor.json")
    parser.add_argument("--model", type=Path,
                        default=ROOT / "calib/ball_dynamics.json")
    parser.add_argument("--reload-brake", type=Path,
                        default=ROOT / "calib/reload_brake.json")
    parser.add_argument("--step-deg", type=float, default=0.25)
    parser.add_argument("--manual-max-deg", type=float, default=1.25)
    parser.add_argument("--max-reset-tilt-deg", type=float, default=1.25)
    parser.add_argument("--identify-tilt-deg", type=float, default=1.00)
    parser.add_argument("--identify-step-s", type=float, default=0.20)
    parser.add_argument("--identify-seconds", type=float, default=2.0,
                        help="active duration of each placement trial")
    parser.add_argument("--identify-trials", type=int, default=6,
                        help="reposition-and-Space trials in one log")
    parser.add_argument("--continuous-seconds", type=float, default=30.0)
    parser.add_argument("--continuous-alpha-hz", type=float, default=0.55)
    parser.add_argument("--continuous-beta-hz", type=float, default=0.73)
    parser.add_argument("--recenter-calibration", type=Path,
                        default=ROOT / "calib/ball_recenter.json")
    parser.add_argument("--recenter-kp", type=float, default=3.0,
                        help="desired acceleration per mm of position error")
    parser.add_argument("--recenter-kd", type=float, default=2.5,
                        help="desired acceleration per mm/s of velocity")
    parser.add_argument("--recenter-max-tilt-deg", type=float, default=0.8)
    parser.add_argument("--identify-position-mm", type=float, nargs=2,
                        default=(159.0, 109.0), metavar=("X", "Y"),
                        help="labelled placement target in the viewer")
    parser.add_argument("--occupied-map", type=Path, default=ROOT /
                        "artifacts/camera_maze/20260810_125246/occupied_inflated.png")
    parser.add_argument("--map-resolution-mm", type=float, default=1.0)
    parser.add_argument("--max-load", type=int, default=350)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()
    if not 0 < args.identify_tilt_deg <= 2.0:
        parser.error("--identify-tilt-deg must be in (0, 2.0]")
    if args.identify_step_s < 0.15 or args.identify_seconds <= 0:
        parser.error("identification timing is outside safe bounds")
    if args.identify_trials < 1:
        parser.error("--identify-trials must be positive")
    if args.mode == "reset" and not args.model.is_file():
        parser.error(f"reset mode requires fitted model {args.model}")
    if args.mode == "hard-reset" and not args.reload_brake.is_file():
        parser.error(f"hard-reset mode requires {args.reload_brake}")
    if args.execute and args.mode == "observe":
        parser.error("--execute has no meaning in observe mode")
    is_identification = args.mode in ("identify", "continuous-id")
    if is_identification and not args.occupied_map.is_file():
        parser.error(f"missing safety occupancy map: {args.occupied_map}")
    if (args.continuous_seconds <= 0 or args.continuous_alpha_hz <= 0
            or args.continuous_beta_hz <= 0):
        parser.error("continuous identification timing must be positive")
    if args.mode == "continuous-id" and not args.recenter_calibration.is_file():
        parser.error(f"missing recenter calibration: {args.recenter_calibration}")
    for path in (args.camera_zero, args.imu_zero):
        if not path.is_file():
            parser.error(f"missing joint zero: {path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / "artifacts/ball_control" / stamp
    output.mkdir(parents=True, exist_ok=False)
    geometry = BoardGeometry.load(args.board)
    estimator = BoardPoseEstimator(args.calib, geometry, min_tags=4)
    estimator.load_zero(args.camera_zero)
    detector = BlueBallDetector(estimator)
    imu = BNO086Stream(port=args.imu_port)
    imu.load_zero(args.imu_zero)
    reader = ImuReader(imu)
    reader.start()
    fusion = CameraImuFusion()

    capture = cv2.VideoCapture(camera_source(args.camera))
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    if not capture.isOpened():
        reader.stop(); imu.close()
        print("Could not open camera")
        return 1

    bus = None
    origin = None
    if args.execute:
        try:
            calibration = DirectionalMotorCalibration.from_json(
                args.motor_calibration)
            bus = STS3215Bus(port=args.servo_port)
            for servo_id in (1, 2):
                if not bus.ping(servo_id):
                    raise RuntimeError(f"servo {servo_id} did not answer")
                if bus.read_byte(servo_id, Register.MODE) != Mode.POSITION:
                    raise RuntimeError(f"servo {servo_id} not in POSITION mode")
            hold_current(bus)
            origin = DirectionalMotorOrigin(calibration)
        except Exception as exc:  # noqa: BLE001
            capture.release(); reader.stop(); imu.close()
            if bus is not None:
                release(bus); bus.close()
            print(f"Could not safely enable servos: {exc}")
            return 1

    brake = (AdaptiveResetBrake(
        BallDynamicsModel.load(args.model),
        max_tilt_deg=args.max_reset_tilt_deg) if args.mode == "reset" else None)
    fixed_brake = None
    if args.mode == "hard-reset":
        brake_data = json.loads(args.reload_brake.read_text(encoding="utf-8"))
        if brake_data.get("version") != "tag_reload_brake_v1":
            parser.error("unsupported reload brake calibration")
        fixed_brake = FixedResetBrake(
            brake_data["brake_tilt_deg"],
            trigger_speed_mm_s=brake_data["trigger_speed_mm_s"],
            settle_speed_mm_s=brake_data["settle_speed_mm_s"],
            settle_hold_s=brake_data["settle_hold_s"],
            max_brake_duration_s=brake_data["max_brake_duration_s"],
            minimum_brake_duration_s=brake_data.get(
                "minimum_brake_duration_s", 0.35),
            trigger_on_reappearance=brake_data.get(
                "trigger_on_reappearance", True))
    state_filter = BallStateFilter()
    previous_xy = None
    missed_ball_frames = 0
    target = np.zeros(2, dtype=np.float64)
    occupied = None
    recenter_matrix = None
    if is_identification:
        occupied = cv2.imread(str(args.occupied_map), cv2.IMREAD_GRAYSCALE)
        if occupied is None:
            parser.error(f"could not read {args.occupied_map}")
        occupied = occupied > 127
    if args.mode == "continuous-id":
        recenter_data = json.loads(args.recenter_calibration.read_text(
            encoding="utf-8"))
        if recenter_data.get("version") != "tag_ball_recenter_v1":
            parser.error("unsupported recenter calibration")
        recenter_matrix = np.asarray(
            recenter_data["acceleration_per_tilt"], dtype=np.float64)
        if recenter_matrix.shape != (2, 2) or abs(np.linalg.det(
                recenter_matrix)) < 1e-6:
            parser.error("invalid recenter acceleration matrix")
    # Each four-pulse block has zero integrated acceleration and zero ideal
    # displacement: +, -, -, +. This excites one axis without walking the
    # marble steadily out of a small calibration area. Reverse-order repeats
    # cancel direction-dependent friction and backlash bias.
    excitation = np.asarray([
        [1, 0], [-1, 0], [-1, 0], [1, 0], [0, 0],
        [0, 1], [0, -1], [0, -1], [0, 1], [0, 0],
        [-1, 0], [1, 0], [1, 0], [-1, 0], [0, 0],
        [0, -1], [0, 1], [0, 1], [0, -1], [0, 0],
    ], dtype=np.float64) * args.identify_tilt_deg
    start = time.monotonic()
    manifest = {"mode": args.mode, "execute": args.execute,
                "model": str(args.model) if brake else None,
                "motor_calibration": str(args.motor_calibration),
                "max_reset_tilt_deg": args.max_reset_tilt_deg}
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{args.mode.upper()} {'ACTIVE' if args.execute else 'PREVIEW'} -> {output}")
    print("A/D alpha, S/W beta, 0 level, Space start-identify, q quit")
    last_command = 0.0
    last_state_read = 0.0
    load_faults = 0
    identify_start = None
    identify_centre = None
    completed_trials = 0
    required_trials = (1 if args.no_window or args.mode == "continuous-id"
                       else args.identify_trials)
    # Headless operation cannot receive Space. Give the level command time to
    # settle; interactive operation waits indefinitely while the user places
    # the marble with the board powered and held.
    identify_auto_start = start + 3.0 if args.no_window else None
    try:
        with (output / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            while True:
                ok, native = capture.read()
                if not ok:
                    break
                now = time.monotonic()
                calibrated = cv2.resize(native, estimator.image_size,
                                        interpolation=cv2.INTER_AREA)
                pose = estimator.estimate(cv2.cvtColor(native, cv2.COLOR_BGR2GRAY))
                pose_ok = pose is not None and pose.reprojection_px <= 4.0
                ball = detector.detect(calibrated, pose, previous_xy_m=previous_xy) \
                    if pose_ok else None
                if ball is not None:
                    previous_xy = ball.board_xy_m.copy()
                    missed_ball_frames = 0
                else:
                    missed_ball_frames += 1
                    if missed_ball_frames > 3:
                        previous_xy = None
                ball_state = state_filter.update(
                    now, ball.board_xy_m * 1000.0 if ball is not None else None)

                latest = reader.latest()
                imu_angle = None
                if latest is not None:
                    imu_angle = relative_angles_deg(
                        [latest], imu.zero_rotation, imu.mount_rotation)[0]
                camera_angle = (np.array([pose.alpha_deg, pose.beta_deg])
                                if pose_ok else None)
                fused = fusion.update(
                    camera_angle, imu_angle, timestamp=now,
                    imu_timestamp=latest[0] if latest else None)
                fused_angle = fused.array if fused else np.full(2, np.nan)
                phase = args.mode
                if fixed_brake is not None:
                    reset_command = fixed_brake.update(ball_state, now)
                    target = reset_command.tilt_deg
                    phase = reset_command.phase.value
                elif brake is not None:
                    reset_command = brake.update(ball_state)
                    target = reset_command.tilt_deg
                    phase = reset_command.phase.value
                elif is_identification:
                    safe = False
                    if ball_state is not None and ball_state.measurement_age_s <= 0.12:
                        col = int(round(ball_state.position_mm[0]
                                        / args.map_resolution_mm))
                        row = int(round((geometry.board_height_m * 1000.0
                                         - ball_state.position_mm[1])
                                        / args.map_resolution_mm))
                        safe = (0 <= row < occupied.shape[0]
                                and 0 <= col < occupied.shape[1]
                                and not occupied[row, col])
                    if identify_start is None and identify_auto_start is not None \
                            and now >= identify_auto_start \
                            and ball_state is not None \
                            and ball_state.measurement_age_s <= 0.12:
                        identify_start = now
                        identify_centre = ball_state.position_mm.copy()
                    if identify_start is None:
                        target[:] = 0.0
                        phase = "identify_waiting_press_space"
                    elif safe and args.mode == "continuous-id":
                        elapsed = now - identify_start
                        excitation_target = args.identify_tilt_deg * np.array([
                            math.sin(2.0 * math.pi
                                     * args.continuous_alpha_hz * elapsed),
                            math.sin(2.0 * math.pi
                                     * args.continuous_beta_hz * elapsed
                                     + 0.5 * math.pi),
                        ])
                        position_error = (ball_state.position_mm
                                          - identify_centre)
                        desired_acceleration = (
                            -args.recenter_kp * position_error
                            -args.recenter_kd * ball_state.velocity_mm_s)
                        recenter_tilt = np.linalg.solve(
                            recenter_matrix, desired_acceleration)
                        recenter_tilt = np.clip(
                            recenter_tilt, -args.recenter_max_tilt_deg,
                            args.recenter_max_tilt_deg)
                        target = np.clip(excitation_target + recenter_tilt,
                                         -2.0, 2.0)
                        phase = "identify_active_continuous"
                    elif safe:
                        step = int((now - identify_start) / args.identify_step_s)
                        offset = (len(excitation) // 2
                                  if completed_trials % 2 else 0)
                        target = excitation[(step + offset)
                                            % len(excitation)].copy()
                        phase = f"identify_active_trial_{completed_trials + 1}"
                    else:
                        target[:] = 0.0
                        phase = "identify_paused_unsafe_or_lost"
                if bus is not None and now - last_command >= 0.10:
                    bus.sync_write_positions(origin.targets(*target))
                    last_command = now
                if bus is not None and now - last_state_read >= 0.20:
                    last_state_read = now
                    states = {servo_id: bus.read_state(servo_id)
                              for servo_id in (1, 2)}
                    high_and_still = any(
                        abs(state.load) > args.max_load and not state.moving
                        for state in states.values())
                    load_faults = load_faults + 1 if high_and_still else 0
                    if load_faults >= 3:
                        print("\nLOAD FAULT: releasing both servos")
                        break

                visible = ball_state is not None and ball is not None
                writer.writerow({
                    "time_s": now - start, "ball_visible": int(visible),
                    "x_mm": ball_state.position_mm[0] if visible else "",
                    "y_mm": ball_state.position_mm[1] if visible else "",
                    "raw_x_mm": ball.board_xy_m[0] * 1000.0 if ball is not None else "",
                    "raw_y_mm": ball.board_xy_m[1] * 1000.0 if ball is not None else "",
                    "ball_confidence": ball.confidence if ball is not None else "",
                    "vx_mm_s": ball_state.velocity_mm_s[0] if visible else "",
                    "vy_mm_s": ball_state.velocity_mm_s[1] if visible else "",
                    "speed_mm_s": ball_state.speed_mm_s if visible else "",
                    "camera_alpha_deg": camera_angle[0] if pose_ok else "",
                    "camera_beta_deg": camera_angle[1] if pose_ok else "",
                    "imu_alpha_deg": imu_angle[0] if imu_angle is not None else "",
                    "imu_beta_deg": imu_angle[1] if imu_angle is not None else "",
                    "fused_alpha_deg": fused_angle[0],
                    "fused_beta_deg": fused_angle[1],
                    "target_alpha_deg": target[0], "target_beta_deg": target[1],
                    "phase": phase})
                display = calibrated.copy()
                fresh_ball = bool(
                    ball is not None and ball_state is not None
                    and ball_state.measurement_age_s <= 0.12)
                speed = ball_state.speed_mm_s if fresh_ball else math.nan
                speed_text = f"{speed:.1f} mm/s" if fresh_ball else "LOST"
                text = (f"{phase}  ball speed {speed_text}  "
                        f"target {target[0]:+.2f}, {target[1]:+.2f} deg  "
                        f"{'ACTIVE' if args.execute else 'PREVIEW'}")
                cv2.putText(display, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 255, 255), 2)
                if fresh_ball and pose_ok:
                    filtered_board = np.array([[
                        ball_state.position_mm[0] / 1000.0,
                        ball_state.position_mm[1] / 1000.0,
                        detector.radius_m,
                    ]], dtype=np.float64)
                    filtered_pixel = detector._project(filtered_board, pose)[0]
                    cv2.circle(display,
                               tuple(np.rint(filtered_pixel).astype(int)),
                               8, (0, 255, 255), 2)
                if is_identification and pose_ok:
                    x_m, y_m = (value / 1000.0
                                for value in args.identify_position_mm)
                    target_points = np.array([
                        [x_m, y_m, detector.radius_m],
                        [x_m + 0.010, y_m, detector.radius_m],
                    ], dtype=np.float64)
                    target_pixels = detector._project(target_points, pose)
                    centre = tuple(np.rint(target_pixels[0]).astype(int))
                    radius_px = max(8, int(round(np.linalg.norm(
                        target_pixels[1] - target_pixels[0]))))
                    cv2.circle(display, centre, radius_px, (255, 0, 255), 3)
                    label = (f"PLACE BALL HERE  "
                             f"({args.identify_position_mm[0]:.0f}, "
                             f"{args.identify_position_mm[1]:.0f}) mm")
                    label_at = (max(10, centre[0] - 150),
                                max(65, centre[1] - radius_px - 12))
                    cv2.putText(display, label, label_at,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (255, 0, 255), 2, cv2.LINE_AA)
                    if identify_start is None:
                        cv2.putText(display,
                                    (f"TRIAL {completed_trials + 1}/"
                                     f"{required_trials}: PLACE, THEN SPACE"),
                                    (12, 68), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.70, (0, 255, 0), 2, cv2.LINE_AA)
                key = -1
                if not args.no_window:
                    cv2.imshow("real ball control", display)
                    key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if (is_identification and key == ord(" ")
                        and identify_start is None):
                    if ball_state is None or ball_state.measurement_age_s > 0.12:
                        print("\nCannot start: marble detection is not current")
                    else:
                        identify_start = now
                        identify_centre = ball_state.position_mm.copy()
                        print(f"\nIdentification started around "
                              f"({identify_centre[0]:.1f}, "
                              f"{identify_centre[1]:.1f}) mm")
                continuous_done = (
                    args.mode == "continuous-id" and identify_start is not None
                    and now - identify_start >= args.continuous_seconds)
                trials_done = (
                    args.mode == "identify" and identify_start is not None
                    and now - identify_start >= args.identify_seconds)
                if continuous_done:
                    print("\nContinuous identification complete")
                    break
                if trials_done:
                    completed_trials += 1
                    target[:] = 0.0
                    if bus is not None:
                        bus.sync_write_positions(origin.targets(0.0, 0.0))
                    if completed_trials >= required_trials:
                        print("\nIdentification trials complete")
                        break
                    identify_start = None
                    state_filter.reset()
                    print(f"\nTrial {completed_trials} complete. Reposition "
                          "the marble and press Space again.")
                if args.mode == "manual":
                    delta = {ord("a"): (-1, 0), ord("d"): (1, 0),
                             ord("s"): (0, -1), ord("w"): (0, 1)}.get(key)
                    if key == ord("0"):
                        target[:] = 0.0
                    elif delta is not None:
                        target += args.step_deg * np.asarray(delta)
                        target[:] = np.clip(target, -args.manual_max_deg,
                                            args.manual_max_deg)
    except KeyboardInterrupt:
        pass
    finally:
        if bus is not None:
            try:
                bus.sync_write_positions(origin.targets(0.0, 0.0))
                time.sleep(0.25)
            finally:
                release(bus); bus.close()
        capture.release(); reader.stop(); imu.close()
        cv2.destroyAllWindows()
    print(f"Saved {output / 'samples.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
