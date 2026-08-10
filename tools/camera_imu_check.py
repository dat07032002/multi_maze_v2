#!/usr/bin/env python3
"""Cross-check AprilTag board angles against the BNO086 and track the ball.

Both sensors must be zeroed at the same physically level pose. Press ``z`` to
capture and save both references together, then hold the board still at several
positive and negative tilts on each axis. Comparisons are recorded only while
the IMU is settled, so camera latency during motion is not reported as error.

Examples:
    python3 tools/camera_imu_check.py
    python3 tools/camera_imu_check.py --zero-on-start --seconds 60
    python3 tools/camera_imu_check.py --camera 2 --no-window --seconds 30
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tag_vision.core.ball_detection import BlueBallDetector  # noqa: E402
from tag_vision.core.angle_fusion import CameraImuFusion  # noqa: E402
from tag_vision.core.board_geometry import BoardGeometry  # noqa: E402
from tag_vision.core.board_pose import (  # noqa: E402
    BoardPoseEstimator,
    angles_from_rotation,
)
from tag_vision.hardware.imu import BNO086Stream  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    matrix = np.mean(np.stack(rotations), axis=0)
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def camera_source(value: str) -> int | str:
    if value != "auto":
        return int(value) if value.isdigit() else value
    root = Path("/sys/class/video4linux")
    for device in sorted(root.glob("video*")) if root.is_dir() else ():
        try:
            name = (device / "name").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if "See3CAM_24CUG" in name:
            return int(device.name.removeprefix("video"))
    return 0


class ImuReader:
    """Continuously drain the 200 Hz serial stream without blocking video."""

    def __init__(self, imu: BNO086Stream) -> None:
        self.imu = imu
        self.samples: deque[tuple[float, np.ndarray, int]] = deque(maxlen=600)
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            sample = self.imu.read_sample(timeout=0.1)
            if sample is not None:
                with self.lock:
                    self.samples.append(
                        (sample.host_time, sample.rotation, sample.accuracy))

    def recent(self, seconds: float) -> list[tuple[float, np.ndarray, int]]:
        cutoff = time.time() - seconds
        with self.lock:
            return [row for row in self.samples if row[0] >= cutoff]

    def latest(self) -> tuple[float, np.ndarray, int] | None:
        """Newest unique IMU report, for low-latency fusion."""
        with self.lock:
            return self.samples[-1] if self.samples else None


def relative_angles_deg(
    rows: list[tuple[float, np.ndarray, int]], zero: np.ndarray,
    mount_rotation: np.ndarray | None = None,
) -> np.ndarray:
    mount = (np.eye(3, dtype=np.float64) if mount_rotation is None
             else np.asarray(mount_rotation, dtype=np.float64))
    return np.asarray([
        [math.degrees(v) for v in angles_from_rotation(
            mount @ (zero.T @ rotation) @ mount.T)]
        for _, rotation, _ in rows
    ])


def capture_joint_zero(
    estimator: BoardPoseEstimator,
    imu: BNO086Stream,
    reader: ImuReader,
    camera_rotations: deque[np.ndarray],
    camera_zero_path: Path,
    imu_zero_path: Path,
    imu_extra: dict | None = None,
) -> bool:
    imu_rows = reader.recent(0.75)
    if len(camera_rotations) < 10 or len(imu_rows) < 50:
        print("\nCannot zero yet: need 10 camera poses and 50 recent IMU samples.")
        return False
    estimator.set_zero(mean_rotation(list(camera_rotations)))
    imu.set_zero(mean_rotation([rotation for _, rotation, _ in imu_rows]))
    estimator.save_zero(camera_zero_path)
    extra = {
        "captured_with": "tools/camera_imu_check.py",
        "camera_samples_averaged": len(camera_rotations),
        "samples_averaged": len(imu_rows),
    }
    if imu_extra:
        extra.update(imu_extra)
    imu.save_zero(imu_zero_path, extra=extra)
    print(f"\nJoint level zero saved: {camera_zero_path} and {imu_zero_path}")
    return True


def report(pairs: list[tuple[float, float, float, float]]) -> None:
    print()
    if not pairs:
        print("No settled camera/IMU comparisons recorded.")
        return
    values = np.asarray(pairs)
    camera = values[:, :2]
    imu = values[:, 2:]
    delta = camera - imu
    span = np.ptp(imu, axis=0)
    print(f"Settled comparison samples: {len(pairs)}")
    for index, name in enumerate(("alpha", "beta")):
        bias = float(np.mean(delta[:, index]))
        rmse = float(np.sqrt(np.mean(delta[:, index] ** 2)))
        maximum = float(np.max(np.abs(delta[:, index])))
        print(
            f"  {name:5s}: IMU span {span[index]:5.2f} deg  "
            f"bias {bias:+6.3f}  RMSE {rmse:6.3f}  max {maximum:6.3f} deg"
        )
    if min(span) < 2.0:
        print("  More data needed: hold each axis at both signs for >=2 deg span.")
    elif float(np.max(np.sqrt(np.mean(delta ** 2, axis=0)))) <= 0.30:
        print("  CROSS-CHECK PASSED: worst-axis RMSE <= 0.30 deg.")
    else:
        print("  CROSS-CHECK FAILED: inspect IMU axis alignment and tag geometry.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="auto")
    parser.add_argument("--capture-width", type=int, default=1920)
    parser.add_argument("--capture-height", type=int, default=1200)
    parser.add_argument("--calib", type=Path,
                        default=ROOT / "calib/camera_calib.json")
    parser.add_argument("--board", type=Path,
                        default=ROOT / "calib/board_tags.json")
    parser.add_argument("--camera-zero", type=Path,
                        default=ROOT / "calib/board_zero.json")
    parser.add_argument("--imu-zero", type=Path,
                        default=ROOT / "calib/imu_zero.json")
    parser.add_argument("--imu-port", default=None)
    parser.add_argument("--ball-radius-mm", type=float, default=5.5)
    parser.add_argument("--max-reprojection", type=float, default=4.0)
    parser.add_argument("--settle-window", type=float, default=0.25)
    parser.add_argument("--settle-sd", type=float, default=0.05,
                        help="maximum IMU angle SD for a settled comparison")
    parser.add_argument("--fusion-time-constant", type=float, default=0.5,
                        help="seconds for camera to correct IMU drift")
    parser.add_argument("--fusion-camera-gate", type=float, default=2.0,
                        help="reject camera residuals larger than this (deg)")
    parser.add_argument("--zero-on-start", action="store_true")
    parser.add_argument("--ignore-saved-zero", action="store_true")
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    geometry = BoardGeometry.load(args.board)
    estimator = BoardPoseEstimator(args.calib, geometry, min_tags=4)
    ball_detector = BlueBallDetector(
        estimator, radius_m=args.ball_radius_mm / 1000.0)

    if not args.ignore_saved_zero and not args.zero_on_start:
        if args.camera_zero.is_file():
            estimator.load_zero(args.camera_zero)

    try:
        imu = BNO086Stream(port=args.imu_port)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not open BNO086 stream: {exc}")
        return 1
    if not args.ignore_saved_zero and not args.zero_on_start:
        if args.imu_zero.is_file():
            imu.load_zero(args.imu_zero)

    source = camera_source(args.camera)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        imu.close()
        print(f"Could not open camera {source!r}")
        return 1
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_height)
    actual = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    expected = estimator.image_size
    if actual[0] * expected[1] != actual[1] * expected[0]:
        capture.release()
        imu.close()
        print(f"Camera aspect ratio {actual} does not match calibration {expected}")
        return 1

    reader = ImuReader(imu)
    reader.start()
    fusion = CameraImuFusion(
        args.fusion_time_constant, args.fusion_camera_gate)
    camera_rotations: deque[np.ndarray] = deque(maxlen=30)
    pairs: list[tuple[float, float, float, float]] = []
    last_pair_time = 0.0
    jointly_zeroed = False
    frames = poses = balls = 0
    start = time.monotonic()
    print(f"Camera {source} {actual[0]}x{actual[1]}; IMU {imu.port}")
    print("Press z with the board physically level; q/Esc quits.")

    try:
        while not args.seconds or time.monotonic() - start < args.seconds:
            ok, frame = capture.read()
            if not ok:
                break
            detection_gray = None
            if actual != expected:
                detection_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, expected, interpolation=cv2.INTER_AREA)
            frames += 1
            gray = (detection_gray if detection_gray is not None
                    else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            pose = estimator.estimate(gray)
            ball = None
            imu_text = ("IMU unzeroed" if imu.zero_rotation is None
                        else "IMU waiting")
            delta_text = "not compared"
            pose_ok = bool(
                pose is not None
                and pose.reprojection_px <= args.max_reprojection)
            if pose_ok:
                poses += 1
                camera_rotations.append(pose.rotation.copy())
                ball = ball_detector.detect(frame, pose)
                balls += ball is not None

                if (args.zero_on_start and not jointly_zeroed
                        and len(camera_rotations) == camera_rotations.maxlen):
                    jointly_zeroed = capture_joint_zero(
                        estimator, imu, reader, camera_rotations,
                        args.camera_zero, args.imu_zero)
                    if jointly_zeroed:
                        pose = estimator.estimate(gray)
                        zero_latest = reader.latest()
                        fusion.reset(
                            [0.0, 0.0], [0.0, 0.0], timestamp=time.time(),
                            imu_timestamp=(zero_latest[0]
                                           if zero_latest else None))

                rows = reader.recent(args.settle_window)
                if imu.zero_rotation is not None and rows:
                    imu_angles = relative_angles_deg(
                        rows, imu.zero_rotation, imu.mount_rotation)
                    imu_mean = np.mean(imu_angles, axis=0)
                    imu_sd = np.std(imu_angles, axis=0)
                    imu_text = (f"IMU {imu_mean[0]:+.3f} {imu_mean[1]:+.3f} "
                                f"sd {max(imu_sd):.3f}")
                    if (estimator.zero_rotation is not None
                            and len(rows) >= 20
                            and max(imu_sd) <= args.settle_sd):
                        da = pose.alpha_deg - imu_mean[0]
                        db = pose.beta_deg - imu_mean[1]
                        delta_text = f"delta {da:+.3f} {db:+.3f}"
                        now = time.monotonic()
                        if now - last_pair_time >= 0.2:
                            pairs.append((pose.alpha_deg, pose.beta_deg,
                                          float(imu_mean[0]), float(imu_mean[1])))
                            last_pair_time = now

                ball_text = "ball --"
                if ball is not None:
                    x_mm, y_mm = ball.board_xy_m * 1000.0
                    ball_text = f"ball {x_mm:.1f},{y_mm:.1f} mm"
                camera_for_fusion = np.array(
                    [pose.alpha_deg, pose.beta_deg], dtype=np.float64)
            else:
                camera_for_fusion = None
                ball_text = "ball --"

            latest = reader.latest()
            latest_imu = None
            latest_imu_time = None
            if imu.zero_rotation is not None and latest is not None:
                latest_imu = relative_angles_deg(
                    [latest], imu.zero_rotation, imu.mount_rotation)[0]
                latest_imu_time = latest[0]
            fused = fusion.update(
                camera_for_fusion, latest_imu, timestamp=time.time(),
                imu_timestamp=latest_imu_time)
            fused_text = ("FUSED waiting" if fused is None else
                          f"FUSED {fused.alpha_deg:+.3f} {fused.beta_deg:+.3f}")
            if pose_ok:
                line = (f"CAM {pose.alpha_deg:+.3f} {pose.beta_deg:+.3f}  "
                        f"{imu_text}  {fused_text}  {delta_text}  {ball_text}")
            else:
                line = f"camera pose unavailable  {fused_text}"
            sys.stdout.write("\r" + line + "   ")
            sys.stdout.flush()

            if not args.no_window:
                display = estimator.undistort_frame(frame)
                cv2.putText(display, line[:110], (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.putText(display, "z: joint zero  q: quit", (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.imshow("camera + BNO086 cross-check", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("z"):
                    jointly_zeroed = capture_joint_zero(
                        estimator, imu, reader, camera_rotations,
                        args.camera_zero, args.imu_zero)
                    if jointly_zeroed:
                        fusion.reset([0.0, 0.0], [0.0, 0.0],
                                     timestamp=time.time(),
                                     imu_timestamp=(latest[0]
                                                    if latest else None))
                if key in (27, ord("q")):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        print()
        reader.stop()
        capture.release()
        imu.close()
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass

    print(f"Pose {poses}/{frames}; ball {balls}/{frames}; "
          f"IMU drop {imu.dropped}, CRC {imu.crc_errors}")
    report(pairs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
