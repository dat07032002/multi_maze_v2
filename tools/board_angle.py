#!/usr/bin/env python3
"""Live board angle from the plate tags: zero it, then measure its precision.

This is the Phase 3 and Phase 4 tool. It does three jobs:

  --zero      capture the current plate rotation as level and save it
  --noise     log alpha/beta with the board still and report the standard
              deviation, which is the tilt precision you actually have
  (default)   live readout for eyeballing and for the shim check

The shim check is the one that matters: tilt the board to a known angle using
the table in the sysid protocol (1 mm of edge lift is 0.221 deg at the 259 mm
span) and compare. Reprojection residuals cannot tell you the angles are right,
only that the solve is self-consistent -- a wrongly measured tag layout fits
beautifully and reads wrong.

Usage:
    python3 tools/board_angle.py                 # live readout
    python3 tools/board_angle.py --zero          # board level, then save zero
    python3 tools/board_angle.py --noise 60      # 60 s static noise report
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tag_vision.core.board_geometry import BoardGeometry      # noqa: E402
from tag_vision.core.board_pose import BoardPoseEstimator     # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class BoardAngle(Node):
    def __init__(self, args, estimator):
        super().__init__("board_angle")
        self.args = args
        self.estimator = estimator
        self.bridge = CvBridge()
        self.samples: list[tuple[float, float]] = []
        self.rotations: list[np.ndarray] = []
        self.frames = 0
        self.detected = 0
        self.window_ok = not args.no_window
        self.create_subscription(Image, args.topic, self.on_image, 5)

    def on_image(self, message):
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frames += 1

        result = self.estimator.estimate(gray)
        if result is not None:
            self.detected += 1
            self.samples.append((result.alpha_deg, result.beta_deg))
            self.rotations.append(result.rotation)
            height_mm = float(np.linalg.norm(result.translation)) * 1000.0
            line = (f"tags {len(result.tag_ids)}  "
                    f"alpha {result.alpha_deg:+7.3f}  beta {result.beta_deg:+7.3f} deg  "
                    f"range {height_mm:7.1f} mm  reproj {result.reprojection_px:5.2f} px")
        else:
            line = f"no pose (need {self.estimator.min_tags}+ known tags)"
        sys.stdout.write("\r" + line + "   ")
        sys.stdout.flush()

        if self.window_ok:
            try:
                found = self.estimator.detect(gray)
                for tag_id, pts in found.items():
                    cv2.polylines(frame, [pts.astype(np.int32)], True,
                                  (0, 255, 0), 2)
                    centre = pts.mean(axis=0).astype(int)
                    cv2.putText(frame, str(tag_id), tuple(centre),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                if result is not None:
                    cv2.putText(
                        frame,
                        f"a {result.alpha_deg:+.2f}  b {result.beta_deg:+.2f} deg",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow("board angle", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    raise KeyboardInterrupt
            except KeyboardInterrupt:
                raise
            except Exception:  # noqa: BLE001
                self.window_ok = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/tag_camera/image")
    parser.add_argument("--calib", default=str(ROOT / "calib" / "camera_calib.json"))
    parser.add_argument("--board", default=str(ROOT / "calib" / "board_tags.json"))
    parser.add_argument("--zero-file", default=str(ROOT / "calib" / "board_zero.json"))
    parser.add_argument("--zero", action="store_true",
                        help="capture the current rotation as level and save it")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="seconds of static logging, then report sd")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    geometry = BoardGeometry.load(args.board)
    print(geometry.describe())
    warnings = geometry.validate()
    if warnings:
        print("\ngeometry warnings:")
        for warning in warnings:
            print("  -", warning)

    estimator = BoardPoseEstimator(args.calib, geometry)
    zero_path = Path(args.zero_file)
    if zero_path.is_file() and not args.zero:
        estimator.load_zero(zero_path)
        print(f"\nzero loaded from {zero_path}")
    elif not args.zero:
        print("\nNO ZERO SET: angles are plate-relative-to-camera, not to level."
              "\n  Level the board and run with --zero.")

    duration = args.noise if args.noise > 0 else (8.0 if args.zero else 0.0)
    print()

    rclpy.init()
    node = BoardAngle(args, estimator)
    try:
        if duration > 0:
            end = node.get_clock().now().nanoseconds + int(duration * 1e9)
            while rclpy.ok() and node.get_clock().now().nanoseconds < end:
                rclpy.spin_once(node, timeout_sec=0.1)
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print()
        _report(node, estimator, args)
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        rclpy.shutdown()


def _report(node, estimator, args):
    if not node.samples:
        print("no pose was recovered; check the tags are visible and the ids "
              "match calib/board_tags.json")
        return

    alphas = [a for a, _ in node.samples]
    betas = [b for _, b in node.samples]
    print(f"\nframes {node.frames}, pose in {node.detected} "
          f"({node.detected/max(node.frames,1):.0%})")
    print(f"  alpha  mean {statistics.fmean(alphas):+7.3f}  "
          f"sd {statistics.pstdev(alphas) if len(alphas) > 1 else 0.0:6.4f} deg")
    print(f"  beta   mean {statistics.fmean(betas):+7.3f}  "
          f"sd {statistics.pstdev(betas) if len(betas) > 1 else 0.0:6.4f} deg")

    if args.zero:
        # Average the rotations by averaging then re-orthonormalising: a plain
        # elementwise mean of rotation matrices is not itself a rotation.
        stacked = np.mean(np.stack(node.rotations), axis=0)
        u, _, vt = np.linalg.svd(stacked)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        estimator.set_zero(rotation)
        estimator.save_zero(args.zero_file)
        print(f"\nzero captured from {len(node.rotations)} frames -> "
              f"{args.zero_file}")
        print("  Valid only while the camera does not move. Re-zero after any"
              " disturbance.")
    elif args.noise > 0:
        sd = max(statistics.pstdev(alphas), statistics.pstdev(betas))
        print(f"\ntilt noise (worst axis sd): {sd:.4f} deg")
        print("  sysid needs to resolve actuator angles near 0.3-0.8 deg at "
              "command 80.")
        if sd < 0.05:
            print("  -> comfortable; coplanar tags are fine")
        elif sd < 0.15:
            print("  -> usable, but little margin at low commands")
        else:
            print("  -> too coarse: raise two diagonal tags on ~8-10 mm risers "
                  "and re-measure")


if __name__ == "__main__":
    main()
