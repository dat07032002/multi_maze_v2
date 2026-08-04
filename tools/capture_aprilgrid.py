#!/usr/bin/env python3
"""Capture AprilGrid views for camera calibration, with live feedback.

Unlike a checkerboard, the board does NOT have to be fully visible: each tag
carries its own id, so any subset that is seen still contributes correctly
placed 3D points. That is what makes it practical to push the target into the
image corners, where a wide lens distorts hardest.

Geometry follows Kalibr's GridCalibrationTargetAprilgrid: pitch is
(1 + tagSpacing) * tagSize, tag id 0 sits at the origin (bottom-left), and ids
increase left to right then upward.

Corner order from cv2.aruco is [top-left, top-right, bottom-right, bottom-left]
in image coordinates; with board +y up that is (0,s), (s,s), (s,0), (0,0).
Verified against a rendered reference sheet.

Usage:
    python3 tools/capture_aprilgrid.py --tag-size 30.0   # MEASURED, mm
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

GRID = 3
DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)


class Capture(Node):
    def __init__(self, args):
        super().__init__("aprilgrid_capture")
        self.args = args
        self.bridge = CvBridge()
        self.params = cv2.aruco.DetectorParameters_create()
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.coverage = np.zeros((GRID, GRID), dtype=int)
        self.views: list[tuple[np.ndarray, np.ndarray]] = []
        self.poses: list[np.ndarray] = []
        self.seen = 0
        self.hits = 0
        self.flash = 0
        self.shape = None
        self.window_ok = not args.no_window
        os.makedirs(args.out, exist_ok=True)

        size = args.tag_size / 1000.0
        pitch = (1.0 + args.tag_spacing) * size
        self.tag_world = {}
        for row in range(args.rows):
            for col in range(args.cols):
                x0, y0 = col * pitch, row * pitch
                self.tag_world[row * args.cols + col] = np.array([
                    [x0, y0 + size, 0.0],          # aruco corner 0: top-left
                    [x0 + size, y0 + size, 0.0],   # 1: top-right
                    [x0 + size, y0, 0.0],          # 2: bottom-right
                    [x0, y0, 0.0],                 # 3: bottom-left
                ], dtype=np.float32)
        self.create_subscription(Image, args.topic, self.on_image, 5)

    def _cell(self, points):
        centre = points.mean(axis=0)
        height, width = self.shape
        return (min(GRID - 1, int(centre[1] / height * GRID)),
                min(GRID - 1, int(centre[0] / width * GRID)))

    def _descriptor(self, points):
        centre = points.mean(axis=0)
        hull = cv2.convexHull(points.astype(np.float32))
        size = float(np.sqrt(max(cv2.contourArea(hull), 1.0)))
        return np.array([centre[0], centre[1], size])

    def _novel(self, descriptor):
        if not self.poses:
            return True
        return min(np.linalg.norm(descriptor - p)
                   for p in self.poses) > self.args.min_distance

    def _terminal(self):
        rows = []
        for r in range(GRID):
            rows.append("   ".join(
                "." if not self.coverage[r, c] else
                ("+" if self.coverage[r, c] >= 10 else str(self.coverage[r, c]))
                for c in range(GRID)))
        sys.stdout.write("\x1b[2J\x1b[H")
        print("aprilgrid capture\n")
        for line in rows:
            print("      " + line)
        print(f"\n  kept {len(self.views)}   tags seen in {self.hits}/{self.seen}")
        print(f"  empty cells {int((self.coverage == 0).sum())}/9")
        print("\n  Ctrl-C (or q in the window) to finish.")

    def on_image(self, message):
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.shape is None:
            self.shape = gray.shape
        self.seen += 1

        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, DICT, parameters=self.params)
        accepted = False
        n_tags = 0
        if ids is not None and len(ids):
            ids = ids.ravel()
            keep = [(i, c) for i, c in zip(ids, corners) if i in self.tag_world]
            n_tags = len(keep)
            if n_tags >= self.args.min_tags:
                self.hits += 1
                image_pts = np.concatenate(
                    [c[0] for _, c in keep], axis=0).astype(np.float32)
                world_pts = np.concatenate(
                    [self.tag_world[int(i)] for i, _ in keep], axis=0)
                descriptor = self._descriptor(image_pts)
                if self._novel(descriptor):
                    row, col = self._cell(image_pts)
                    self.coverage[row, col] += 1
                    self.poses.append(descriptor)
                    self.views.append((world_pts, image_pts))
                    accepted = True
                    self.flash = 5

        if self.window_ok:
            try:
                display = frame.copy()
                h, w = display.shape[:2]
                for i in range(1, GRID):
                    cv2.line(display, (w * i // GRID, 0), (w * i // GRID, h),
                             (90, 90, 90), 1)
                    cv2.line(display, (0, h * i // GRID), (w, h * i // GRID),
                             (90, 90, 90), 1)
                for r in range(GRID):
                    for c in range(GRID):
                        count = int(self.coverage[r, c])
                        cv2.putText(display, str(count),
                                    (c * w // GRID + 12, r * h // GRID + 32),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (60, 200, 60) if count else (60, 60, 220), 2)
                if ids is not None and len(ids):
                    cv2.aruco.drawDetectedMarkers(display, corners, ids.reshape(-1, 1))
                cv2.putText(display,
                            f"kept {len(self.views)}/{self.args.target}   "
                            f"tags {n_tags}",
                            (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (60, 220, 60) if n_tags >= self.args.min_tags
                            else (60, 60, 230), 2)
                if accepted or self.flash > 0:
                    cv2.rectangle(display, (2, 2), (w - 3, h - 3), (0, 255, 0), 6)
                    self.flash = max(0, self.flash - 1)
                cv2.imshow("aprilgrid capture", display)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    raise KeyboardInterrupt
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"\n(window unavailable: {exc})")
                self.window_ok = False

        if accepted or self.seen % 20 == 0:
            self._terminal()

    def save(self):
        if not self.views:
            print("\nno views kept")
            return
        path = os.path.join(self.args.out, "aprilgrid_views.npz")
        np.savez(
            path,
            world=np.array([v[0] for v in self.views], dtype=object),
            image=np.array([v[1] for v in self.views], dtype=object),
            counts=np.array([len(v[0]) for v in self.views]),
            image_shape=np.asarray(self.shape),
            tag_size_mm=self.args.tag_size,
            tag_spacing=self.args.tag_spacing,
            rows=self.args.rows, cols=self.args.cols,
            coverage=self.coverage,
            allow_pickle=True)
        total = sum(len(v[0]) for v in self.views)
        print(f"\nwrote {len(self.views)} views ({total} points) -> {path}")
        print("coverage:\n", self.coverage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/tag_camera/image")
    parser.add_argument("--out", default="artifacts/calibration/capture_april")
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--tag-size", type=float, default=30.0,
                        help="MEASURED outer black edge, mm")
    parser.add_argument("--tag-spacing", type=float, default=0.3)
    parser.add_argument("--min-tags", type=int, default=6,
                        help="tags needed before a view counts")
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--min-distance", type=float, default=40.0)
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = Capture(args)
    print(f"AprilGrid {args.rows}x{args.cols}, tagSize {args.tag_size} mm, "
          f"spacing {args.tag_spacing}")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save()
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
