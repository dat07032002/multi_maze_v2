#!/usr/bin/env python3
"""Measure board tilt and a blue marble's board coordinates from one camera.

No ROS installation is required. The board must be level once to define zero.
With a display, press ``z`` while the board is level. In a headless session,
use ``--zero-on-start`` and keep the board level during the initial samples.

Examples:
    python3 tools/manual_tilt_angle.py
    python3 tools/manual_tilt_angle.py --camera 1
    python3 tools/manual_tilt_angle.py --zero-on-start --no-window

Keys:
    z       save the current level reference
    q/Esc   quit
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tag_vision.core.board_geometry import BoardGeometry  # noqa: E402
from tag_vision.core.board_pose import BoardPoseEstimator  # noqa: E402
from tag_vision.core.ball_detection import BlueBallDetector  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    """Average rotation matrices and project the result back onto SO(3)."""
    matrix = np.mean(np.stack(rotations), axis=0)
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def camera_source(value: str):
    """Resolve ``auto``, a numeric index, or a video path for OpenCV."""
    if value != "auto":
        return int(value) if value.isdigit() else value

    video_root = Path("/sys/class/video4linux")
    if video_root.is_dir():
        for device in sorted(video_root.glob("video*")):
            try:
                name = (device / "name").read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if "See3CAM_24CUG" in name:
                return int(device.name.removeprefix("video"))
    return 0


class ManualTiltAngle:
    def __init__(
        self, args, estimator: BoardPoseEstimator, ball_detector: BlueBallDetector
    ):
        self.args = args
        self.estimator = estimator
        self.ball_detector = ball_detector
        self.window_ok = not args.no_window
        self.recent_rotations: deque[np.ndarray] = deque(maxlen=args.zero_samples)
        self.valid_frames = 0
        self.total_frames = 0
        self.zeroed_on_start = False
        self.ball_frames = 0
        self.filtered_ball_xy: np.ndarray | None = None
        self.missed_ball_frames = 0

    @staticmethod
    def _outlined_text(
        image: np.ndarray,
        text: str,
        position: tuple[int, int],
        scale: float,
        colour: tuple[int, int, int],
        thickness: int = 2,
    ) -> None:
        """Draw readable overlay text on both light and dark board regions."""
        cv2.putText(
            image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
            (0, 0, 0), thickness + 2, cv2.LINE_AA,
        )
        cv2.putText(
            image, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
            colour, thickness, cv2.LINE_AA,
        )

    def _draw_board_coordinates(self, image: np.ndarray, result) -> None:
        """Draw the board boundary and its metric XY coordinate frame."""
        geometry = self.estimator.geometry
        width = geometry.board_width_m
        height = geometry.board_height_m
        outline_board = np.array([
            [0.0, 0.0, 0.0],
            [width, 0.0, 0.0],
            [width, height, 0.0],
            [0.0, height, 0.0],
        ])
        outline = np.rint(
            self.estimator.project_undistorted(outline_board, result)
        ).astype(np.int32)
        cv2.polylines(image, [outline], True, (255, 0, 255), 2, cv2.LINE_AA)

        # A 50 mm axis is large enough to read while staying clear of the
        # opposite tags. Its label states the actual coordinate convention.
        axis_m = min(0.050, 0.25 * width, 0.25 * height)
        axes_board = np.array([
            [0.0, 0.0, 0.0],
            [axis_m, 0.0, 0.0],
            [0.0, axis_m, 0.0],
        ])
        origin, x_tip, y_tip = np.rint(
            self.estimator.project_undistorted(axes_board, result)
        ).astype(np.int32)
        origin_tuple = tuple(origin)
        x_tuple = tuple(x_tip)
        y_tuple = tuple(y_tip)
        cv2.circle(image, origin_tuple, 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, origin_tuple, 6, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.arrowedLine(
            image, origin_tuple, x_tuple, (0, 0, 255), 3, cv2.LINE_AA,
            tipLength=0.18,
        )
        cv2.arrowedLine(
            image, origin_tuple, y_tuple, (0, 255, 0), 3, cv2.LINE_AA,
            tipLength=0.18,
        )
        self._outlined_text(
            image, "O (0,0) mm", (int(origin[0]) + 8, int(origin[1]) + 22),
            0.55, (255, 255, 255), 1,
        )
        self._outlined_text(
            image, "+X", (int(x_tip[0]) + 5, int(x_tip[1]) + 5),
            0.55, (0, 0, 255), 2,
        )
        self._outlined_text(
            image, "+Y", (int(y_tip[0]) + 5, int(y_tip[1]) + 5),
            0.55, (0, 255, 0), 2,
        )

    def capture_zero(self) -> bool:
        """Save an averaged level reference from the most recent poses."""
        if not self.recent_rotations:
            print("\nCannot set zero: no complete board pose has been detected yet.")
            return False
        rotation = mean_rotation(list(self.recent_rotations))
        self.estimator.set_zero(rotation)
        self.estimator.save_zero(self.args.zero_file)
        print(
            f"\nZero saved from {len(self.recent_rotations)} pose(s) -> "
            f"{self.args.zero_file}"
        )
        return True

    def process(
        self, frame: np.ndarray, detection_gray: np.ndarray | None = None
    ) -> bool:
        """Process one BGR frame. Return False when the user asks to quit."""
        gray = (detection_gray if detection_gray is not None
                else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        self.total_frames += 1
        result = self.estimator.estimate(gray)
        ball = None

        if result is None or result.reprojection_px > self.args.max_reprojection:
            reason = "no pose" if result is None else (
                f"poor pose ({result.reprojection_px:.1f} px)")
            self.missed_ball_frames += 1
            if self.missed_ball_frames > self.args.ball_reset_frames:
                self.filtered_ball_xy = None
            line = (
                f"{reason}: need {self.estimator.min_tags} known tags  "
                f"frames {self.valid_frames}/{self.total_frames}"
            )
            result = None
        else:
            self.valid_frames += 1
            self.recent_rotations.append(result.rotation.copy())

            if (self.args.zero_on_start and not self.zeroed_on_start
                    and len(self.recent_rotations) == self.args.zero_samples):
                self.zeroed_on_start = self.capture_zero()
                # Re-estimate so this frame immediately reports zeroed angles.
                result = self.estimator.estimate(gray)

            ball = self.ball_detector.detect(
                frame, result, previous_xy_m=self.filtered_ball_xy)
            if ball is not None:
                self.ball_frames += 1
                self.missed_ball_frames = 0
                if self.filtered_ball_xy is None:
                    self.filtered_ball_xy = ball.board_xy_m.copy()
                else:
                    smoothing = self.args.ball_smoothing
                    self.filtered_ball_xy = (
                        smoothing * ball.board_xy_m
                        + (1.0 - smoothing) * self.filtered_ball_xy
                    )
            else:
                self.missed_ball_frames += 1
                if self.missed_ball_frames > self.args.ball_reset_frames:
                    self.filtered_ball_xy = None

            ids = ",".join(str(tag_id) for tag_id in result.tag_ids)
            zero_state = "zeroed" if self.estimator.zero_rotation is not None else "NO ZERO"
            if ball is None:
                ball_text = "ball NOT VISIBLE"
            else:
                x_mm, y_mm = self.filtered_ball_xy * 1000.0
                ball_text = (
                    f"ball x {x_mm:6.1f} y {y_mm:6.1f} mm "
                    f"conf {ball.confidence:4.0%}"
                )
            line = (
                f"alpha {result.alpha_deg:+7.3f} deg  "
                f"beta {result.beta_deg:+7.3f} deg  "
                f"tags [{ids}]  reproj {result.reprojection_px:4.2f} px  "
                f"{zero_state}  {ball_text}"
            )

        sys.stdout.write("\r" + line + "   ")
        sys.stdout.flush()

        if not self.window_ok:
            return True

        try:
            # Detection stays on the original calibrated camera image. Only
            # the viewer and all of its overlays are mapped to rectified pixels.
            display = self.estimator.undistort_frame(frame)
            # estimate() already detected the tags. Reusing those corners
            # avoids a second native-resolution detector pass per video frame.
            found = self.estimator.last_found
            for tag_id, points in found.items():
                display_points = np.rint(
                    self.estimator.undistort(points)
                ).astype(np.int32)
                cv2.polylines(display, [display_points], True,
                              (0, 255, 0), 2)
                centre = tuple(display_points.mean(axis=0).astype(int))
                cv2.putText(display, str(tag_id), centre,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            if result is not None:
                self._draw_board_coordinates(display, result)
                cv2.putText(
                    display,
                    f"alpha {result.alpha_deg:+.2f}  beta {result.beta_deg:+.2f} deg",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
            if ball is not None:
                display_contour = np.rint(self.estimator.undistort(
                    ball.contour.reshape(-1, 2)
                )).astype(np.int32).reshape(-1, 1, 2)
                cv2.drawContours(
                    display, [display_contour], -1, (255, 255, 0), 2)
                centre = tuple(np.rint(
                    self.estimator.undistort(ball.pixel_xy.reshape(1, 2))[0]
                ).astype(int))
                cv2.circle(display, centre, 3, (0, 0, 255), -1)
                x_mm, y_mm = self.filtered_ball_xy * 1000.0
                cv2.putText(
                    display,
                    f"ball x {x_mm:.1f}  y {y_mm:.1f} mm  {ball.confidence:.0%}",
                    (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
                )
            elif result is not None:
                cv2.putText(display, "ball not visible", (10, 64),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)
            cv2.putText(
                display, "undistorted | z: set level   q: quit",
                (10, display.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("manual board tilt (undistorted)", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("z"):
                self.capture_zero()
            return key not in (27, ord("q"))
        except Exception as exc:  # noqa: BLE001
            print(f"\nDisplay unavailable ({exc}); continuing in terminal.")
            self.window_ok = False
            return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="auto",
                        help="camera index, video path, or auto (default: auto)")
    parser.add_argument("--capture-width", type=int, default=1920)
    parser.add_argument("--capture-height", type=int, default=1200)
    parser.add_argument("--calib", default=str(ROOT / "calib" / "camera_calib.json"))
    parser.add_argument("--board", default=str(ROOT / "calib" / "board_tags.json"))
    parser.add_argument("--zero-file", default=str(ROOT / "calib" / "board_zero.json"))
    parser.add_argument("--ball-radius-mm", type=float, default=5.5)
    parser.add_argument("--ball-confidence", type=float, default=0.54)
    parser.add_argument("--ball-smoothing", type=float, default=0.45,
                        help="new-measurement weight from 0 (smooth) to 1 (raw)")
    parser.add_argument("--ball-reset-frames", type=int, default=3,
                        help="clear stale smoothing state after this many misses")
    parser.add_argument("--max-reprojection", type=float, default=4.0,
                        help="reject board poses above this RMS pixel error")
    parser.add_argument("--min-tags", type=int, choices=(3, 4), default=4,
                        help="known tags required for a pose (default: 4)")
    parser.add_argument("--zero-samples", type=int, default=20,
                        help="recent valid poses averaged when setting zero")
    parser.add_argument("--zero-on-start", action="store_true",
                        help="set level after the first --zero-samples valid poses")
    parser.add_argument("--ignore-saved-zero", action="store_true")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()
    if args.zero_samples < 1:
        parser.error("--zero-samples must be at least 1")
    if args.ball_radius_mm <= 0:
        parser.error("--ball-radius-mm must be positive")
    if not 0.0 < args.ball_confidence <= 1.0:
        parser.error("--ball-confidence must be in (0, 1]")
    if not 0.0 < args.ball_smoothing <= 1.0:
        parser.error("--ball-smoothing must be in (0, 1]")
    if args.ball_reset_frames < 0:
        parser.error("--ball-reset-frames cannot be negative")

    geometry = BoardGeometry.load(args.board)
    print(geometry.describe())
    for warning in geometry.validate():
        print("WARNING:", warning)

    estimator = BoardPoseEstimator(args.calib, geometry, min_tags=args.min_tags)
    ball_detector = BlueBallDetector(
        estimator,
        radius_m=args.ball_radius_mm / 1000.0,
        min_confidence=args.ball_confidence,
    )
    zero_path = Path(args.zero_file)
    if zero_path.is_file() and not args.ignore_saved_zero and not args.zero_on_start:
        estimator.load_zero(zero_path)
        print(f"Loaded level reference from {zero_path}")
    elif args.zero_on_start:
        print(f"Keep the board level for the first {args.zero_samples} valid poses.")
    else:
        print("No level reference loaded. Press z with the board level.")

    with open(args.calib, encoding="utf-8") as handle:
        calibration = json.load(handle)
    expected_width = int(calibration["image_width"])
    expected_height = int(calibration["image_height"])

    source = camera_source(args.camera)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(
            f"Could not open camera {source!r}. Try --camera 2, or check "
            "available devices with: ls /dev/video*"
        )
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_height)

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {source}: capture {actual_width}x{actual_height}")
    if actual_width * expected_height != actual_height * expected_width:
        capture.release()
        raise SystemExit(
            f"Camera aspect ratio {actual_width}x{actual_height} does not match "
            f"calibration {expected_width}x{expected_height}. Use the calibrated "
            "1920x1200 capture mode."
        )
    resize_frames = (actual_width, actual_height) != (expected_width, expected_height)
    if resize_frames:
        print(f"Resizing frames to calibrated {expected_width}x{expected_height}")
    print("Press z to set level; press q or Esc to quit.\n")

    app = ManualTiltAngle(args, estimator, ball_detector)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("\nCamera stopped returning frames.")
                break
            if resize_frames:
                # Preserve native pixels for AprilTag corner refinement. The
                # estimator scales detected corners into calibrated 1280x800
                # coordinates before PnP; ball processing stays at 1280x800.
                detection_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(
                    frame, (expected_width, expected_height),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                detection_gray = None
            if not app.process(frame, detection_gray=detection_gray):
                break
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass
        print(
            f"\nStopped: pose detected in {app.valid_frames}/{app.total_frames} "
            f"frames; ball detected in {app.ball_frames}/{app.total_frames}."
        )


if __name__ == "__main__":
    main()
