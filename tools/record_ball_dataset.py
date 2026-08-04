#!/usr/bin/env python3
"""Record frames and diagnostics for improving blue-marble detection.

Move the marble through the centre, edges, and corners; tilt the board; briefly
cover the marble; and let it touch the rails. Each retained sample contains the
camera frame, blue mask, pose quality, and current ball result.

Example:
    python3 tools/record_ball_dataset.py --frames 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tag_vision.core.ball_detection import BlueBallDetector  # noqa: E402
from tag_vision.core.board_geometry import BoardGeometry  # noqa: E402
from tag_vision.core.board_pose import BoardPoseEstimator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def camera_source(value: str):
    if value != "auto":
        return int(value) if value.isdigit() else value
    root = Path("/sys/class/video4linux")
    if root.is_dir():
        for device in sorted(root.glob("video*")):
            try:
                name = (device / "name").read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if "See3CAM_24CUG" in name:
                return int(device.name.removeprefix("video"))
    return 0


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot encode {type(value)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--rate", type=float, default=10.0,
                        help="retained samples per second (default: 10)")
    parser.add_argument("--camera", default="auto")
    parser.add_argument("--capture-width", type=int, default=1920)
    parser.add_argument("--capture-height", type=int, default=1200)
    parser.add_argument("--calib", default=str(ROOT / "calib" / "camera_calib.json"))
    parser.add_argument("--board", default=str(ROOT / "calib" / "board_tags.json"))
    parser.add_argument("--zero-file", default=str(ROOT / "calib" / "board_zero.json"))
    parser.add_argument("--ball-radius-mm", type=float, default=5.5)
    parser.add_argument("--ball-confidence", type=float, default=0.54)
    parser.add_argument("--min-tags", type=int, choices=(3, 4), default=4)
    parser.add_argument("--max-reprojection", type=float, default=4.0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    if args.rate <= 0:
        parser.error("--rate must be positive")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.out or ROOT / "artifacts" / "ball_capture" / stamp)
    frames_dir = output / "frames"
    masks_dir = output / "masks"
    frames_dir.mkdir(parents=True, exist_ok=False)
    masks_dir.mkdir(parents=True, exist_ok=False)

    geometry = BoardGeometry.load(args.board)
    estimator = BoardPoseEstimator(args.calib, geometry, min_tags=args.min_tags)
    zero_path = Path(args.zero_file)
    if zero_path.is_file():
        estimator.load_zero(zero_path)
    detector = BlueBallDetector(
        estimator,
        radius_m=args.ball_radius_mm / 1000.0,
        min_confidence=args.ball_confidence,
    )

    with open(args.calib, encoding="utf-8") as handle:
        calibration = json.load(handle)
    width = int(calibration["image_width"])
    height = int(calibration["image_height"])

    source = camera_source(args.camera)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"could not open camera {source!r}")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_height)
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_width * height != actual_height * width:
        capture.release()
        raise SystemExit(
            f"capture {actual_width}x{actual_height} does not match calibrated "
            f"aspect ratio {width}x{height}")

    manifest = {
        "created": datetime.now().isoformat(),
        "camera": source,
        "capture_size": [actual_width, actual_height],
        "saved_size": [width, height],
        "calibration": str(Path(args.calib).resolve()),
        "board": str(Path(args.board).resolve()),
        "zero_file": str(zero_path.resolve()) if zero_path.is_file() else None,
        "ball_radius_mm": args.ball_radius_mm,
        "ball_confidence": args.ball_confidence,
        "min_tags": args.min_tags,
        "max_reprojection": args.max_reprojection,
        "target_frames": args.frames,
        "sample_rate_hz": args.rate,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Recording {args.frames} samples at {args.rate:g} Hz -> {output}")
    print("Move the ball everywhere, tilt the board, touch the rails, and briefly")
    print("cover the ball. Press q or Esc to finish early.\n")

    metadata_path = output / "samples.jsonl"
    saved = 0
    pose_count = 0
    ball_count = 0
    previous_xy = None
    missed_ball_frames = 0
    next_sample = time.monotonic()
    try:
        with metadata_path.open("w", encoding="utf-8") as metadata:
            while saved < args.frames:
                ok, frame = capture.read()
                if not ok:
                    print("\ncamera stopped returning frames")
                    break
                now = time.monotonic()
                if now < next_sample:
                    if not args.no_window and cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                    continue
                next_sample = now + 1.0 / args.rate
                if (frame.shape[1], frame.shape[0]) != (width, height):
                    frame = cv2.resize(
                        frame, (width, height), interpolation=cv2.INTER_AREA)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                pose = estimator.estimate(gray)
                pose_ok = (
                    pose is not None
                    and pose.reprojection_px <= args.max_reprojection
                )
                ball = None
                if pose_ok:
                    pose_count += 1
                    ball = detector.detect(
                        frame, pose, previous_xy_m=previous_xy)
                    if ball is not None:
                        ball_count += 1
                        previous_xy = ball.board_xy_m.copy()
                        missed_ball_frames = 0
                    else:
                        missed_ball_frames += 1
                else:
                    missed_ball_frames += 1
                if missed_ball_frames > 3:
                    previous_xy = None

                name = f"{saved:06d}"
                cv2.imwrite(
                    str(frames_dir / f"{name}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
                mask = (
                    detector.last_mask if pose_ok and detector.last_mask is not None
                    else np.zeros((height, width), dtype=np.uint8)
                )
                cv2.imwrite(str(masks_dir / f"{name}.png"), mask)

                record = {
                    "index": saved,
                    "monotonic_s": now,
                    "pose_visible": pose_ok,
                    "tag_ids": list(pose.tag_ids) if pose is not None else [],
                    "reprojection_px": pose.reprojection_px if pose is not None else None,
                    "alpha_deg": pose.alpha_deg if pose_ok else None,
                    "beta_deg": pose.beta_deg if pose_ok else None,
                    "ball_visible": ball is not None,
                    "ball_pixel_xy": ball.pixel_xy if ball is not None else None,
                    "ball_board_xy_m": ball.board_xy_m if ball is not None else None,
                    "ball_radius_px": ball.radius_px if ball is not None else None,
                    "ball_confidence": ball.confidence if ball is not None else None,
                }
                metadata.write(json.dumps(record, default=json_value) + "\n")
                metadata.flush()
                saved += 1

                display = frame.copy()
                if pose is not None:
                    cv2.putText(
                        display,
                        f"a {pose.alpha_deg:+.2f} b {pose.beta_deg:+.2f} "
                        f"reproj {pose.reprojection_px:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if pose_ok else (0, 0, 255), 2)
                if ball is not None:
                    cv2.drawContours(display, [ball.contour], -1, (255, 255, 0), 2)
                    x_mm, y_mm = ball.board_xy_m * 1000.0
                    cv2.putText(
                        display, f"ball {x_mm:.1f}, {y_mm:.1f} mm",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2)
                cv2.putText(
                    display, f"saved {saved}/{args.frames}",
                    (10, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2)
                if not args.no_window:
                    cv2.imshow("ball dataset recorder", display)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
                print(
                    f"\rsaved {saved}/{args.frames}  poses {pose_count}  "
                    f"balls {ball_count}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass

    summary = {
        "saved_frames": saved,
        "valid_poses": pose_count,
        "detected_balls": ball_count,
        "pose_rate": pose_count / max(saved, 1),
        "ball_rate_overall": ball_count / max(saved, 1),
        "ball_rate_given_pose": ball_count / max(pose_count, 1),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n\nFinished -> {output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
