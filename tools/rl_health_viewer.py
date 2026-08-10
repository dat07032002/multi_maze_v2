#!/usr/bin/env python3
"""Live, observation-only health dashboard for the real RL state pipeline."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tag_vision.control.ball_state import BallStateFilter  # noqa: E402
from tag_vision.core.angle_fusion import CameraImuFusion  # noqa: E402
from tag_vision.core.ball_detection import BlueBallDetector  # noqa: E402
from tag_vision.core.board_geometry import BoardGeometry  # noqa: E402
from tag_vision.core.board_pose import BoardPoseEstimator  # noqa: E402
from tag_vision.hardware.imu import BNO086Stream  # noqa: E402
from tag_vision.rl.health import HealthLevel, HealthMonitor  # noqa: E402
from tag_vision.rl.task import MazeTask  # noqa: E402
from tools.camera_imu_check import (  # noqa: E402
    ImuReader, camera_source, relative_angles_deg)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="auto")
    parser.add_argument("--imu-port", default=None)
    parser.add_argument("--calib", type=Path,
                        default=ROOT / "calib/camera_calib.json")
    parser.add_argument("--board", type=Path,
                        default=ROOT / "calib/board_tags.json")
    parser.add_argument("--camera-zero", type=Path,
                        default=ROOT / "calib/board_zero.json")
    parser.add_argument("--imu-zero", type=Path,
                        default=ROOT / "calib/imu_zero.json")
    parser.add_argument("--map-dir", type=Path, default=ROOT /
                        "artifacts/camera_maze/20260810_125246")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="zero runs until q")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / "artifacts/rl/health" / stamp
    output.mkdir(parents=True, exist_ok=False)

    geometry = BoardGeometry.load(args.board)
    estimator = BoardPoseEstimator(args.calib, geometry, min_tags=4)
    estimator.load_zero(args.camera_zero)
    detector = BlueBallDetector(estimator)
    task = MazeTask.load(args.map_dir / "map.json",
                         args.map_dir / "occupied_inflated.png")
    health_monitor = HealthMonitor()
    state_filter = BallStateFilter()
    fusion = CameraImuFusion()
    imu = BNO086Stream(port=args.imu_port)
    imu.load_zero(args.imu_zero)
    reader = ImuReader(imu)
    reader.start()
    capture = cv2.VideoCapture(camera_source(args.camera))
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    if not capture.isOpened():
        reader.stop(); imu.close()
        print("Could not open camera")
        return 1

    history: deque[tuple[float, bool, bool]] = deque()
    previous_xy = None
    previous_fused = None
    previous_fused_time = None
    start = time.monotonic()
    last_print = 0.0
    log_path = output / "health.jsonl"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            while True:
                frame_start = time.monotonic()
                ok, native = capture.read()
                if not ok:
                    break
                now = time.monotonic()
                calibrated = cv2.resize(native, estimator.image_size,
                                        interpolation=cv2.INTER_AREA)
                pose = estimator.estimate(cv2.cvtColor(native, cv2.COLOR_BGR2GRAY))
                pose_ok = pose is not None and pose.reprojection_px <= 4.0
                ball = detector.detect(
                    calibrated, pose, previous_xy_m=previous_xy) if pose_ok else None
                if ball is not None:
                    previous_xy = ball.board_xy_m.copy()
                ball_state = state_filter.update(
                    now, ball.board_xy_m * 1000.0 if ball is not None else None)
                latest = reader.latest()
                imu_angle = (relative_angles_deg(
                    [latest], imu.zero_rotation, imu.mount_rotation)[0]
                    if latest is not None else None)
                camera_angle = (np.array([pose.alpha_deg, pose.beta_deg])
                                if pose_ok else None)
                fused = fusion.update(
                    camera_angle, imu_angle, timestamp=now,
                    imu_timestamp=latest[0] if latest else None)
                fused_angle = fused.array if fused else None
                angle_rate = np.zeros(2)
                if (fused_angle is not None and previous_fused is not None
                        and now > previous_fused_time):
                    angle_rate = (fused_angle - previous_fused) / (
                        now - previous_fused_time)
                if fused_angle is not None:
                    previous_fused = fused_angle.copy()
                    previous_fused_time = now

                fresh_ball = bool(ball is not None and ball_state is not None
                                  and ball_state.measurement_age_s <= 0.12)
                history.append((now, pose_ok, fresh_ball))
                while history and history[0][0] < now - 2.0:
                    history.popleft()
                span = max(1e-6, history[-1][0] - history[0][0])
                camera_fps = (len(history) - 1) / span
                pose_rate = sum(row[1] for row in history) / len(history)
                ball_rate = sum(row[2] for row in history) / len(history)
                imu_rate = len(reader.recent(1.0))
                residual = (max(abs(value) for value in fused.camera_residual_deg)
                            if fused is not None and camera_angle is not None else 0.0)
                ball_age = (ball_state.measurement_age_s
                            if ball_state is not None else math.inf)
                latency = time.monotonic() - frame_start
                health = health_monitor.classify(
                    camera_fps=camera_fps, pose_rate=pose_rate,
                    ball_rate=ball_rate, ball_age_s=ball_age,
                    imu_rate_hz=imu_rate, fusion_residual_deg=residual,
                    control_latency_s=latency)

                route = None
                observation = None
                if fresh_ball and fused_angle is not None:
                    observation, route = task.observation(
                        position_mm=ball_state.position_mm,
                        velocity_mm_s=ball_state.velocity_mm_s,
                        angles_deg=fused_angle, angle_rates_deg_s=angle_rate,
                        previous_action_deg=[0.0, 0.0],
                        stuck=ball_state.speed_mm_s < 3.0)
                record = {
                    "elapsed_s": now - start,
                    "level": health.level.name,
                    "reasons": health.reasons,
                    "camera_fps": camera_fps, "pose_rate": pose_rate,
                    "ball_rate": ball_rate, "ball_age_s": ball_age,
                    "imu_rate_hz": imu_rate,
                    "fusion_residual_deg": residual,
                    "processing_latency_s": latency,
                    "ball_xy_mm": (ball_state.position_mm.tolist()
                                   if fresh_ball else None),
                    "ball_speed_mm_s": (ball_state.speed_mm_s
                                        if fresh_ball else None),
                    "route_progress_mm": (route.progress_mm if route else None),
                    "cross_track_mm": (route.cross_track_mm if route else None),
                    "clearance_mm": (float(observation[
                        task.spec.index("clearance")]) if observation is not None
                                     else None),
                }
                log.write(json.dumps(record) + "\n")
                if now - last_print >= 1.0:
                    last_print = now
                    print(f"\r{health.level.name:6s} cam {camera_fps:4.1f}Hz "
                          f"pose {pose_rate:4.0%} ball {ball_rate:4.0%} "
                          f"imu {imu_rate:3d}Hz latency {latency*1000:4.0f}ms "
                          f"reasons {','.join(health.reasons) or '-'}",
                          end="", flush=True)

                if not args.no_window:
                    display = calibrated.copy()
                    colour = {HealthLevel.GREEN: (0, 255, 0),
                              HealthLevel.YELLOW: (0, 255, 255),
                              HealthLevel.RED: (0, 0, 255)}[health.level]
                    cv2.putText(display,
                                f"RL HEALTH {health.level.name}  "
                                f"cam {camera_fps:.1f} pose {pose_rate:.0%} "
                                f"ball {ball_rate:.0%} imu {imu_rate}Hz",
                                (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                colour, 2)
                    if route is not None:
                        cv2.putText(display,
                                    f"progress {route.progress_mm:.0f}/"
                                    f"{task.route_length_mm:.0f}mm  cross "
                                    f"{route.cross_track_mm:+.1f}mm  speed "
                                    f"{ball_state.speed_mm_s:.1f}mm/s",
                                    (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                    colour, 2)
                    cv2.imshow("RL health | observation only", display)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                if args.seconds > 0 and now - start >= args.seconds:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release(); reader.stop(); imu.close(); cv2.destroyAllWindows()
    print(f"\nhealth log: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
