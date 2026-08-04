#!/usr/bin/env python3
"""Validate a calibration by pointing the camera at the AprilGrid.

Solves the board pose live and reports three things the calibration maths
cannot fake:

  distance   compare against a tape measure. Tests the focal length directly,
             because distance scales inversely with fx.
  reprojection error, split centre vs edge. A wide-angle model that is wrong
             at the periphery shows a much larger edge residual, which is the
             failure the previous calibration hid.
  board tilt relative to the camera, useful if the board is on a known plane.

Distance also scales linearly with the assumed tag size, so measure a printed
tag with calipers and pass --tag-size before trusting the number.

Usage:
    python3 tools/check_calibration.py [--tag-size 30.0]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_APRILTAG_36h11)


class Check(Node):
    def __init__(self, args, calib):
        super().__init__("calibration_check")
        self.args = args
        self.bridge = CvBridge()
        self.params = cv2.aruco.DetectorParameters_create()
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.window_ok = not args.no_window

        kb = calib.get("kannala_brandt")
        self.fisheye = kb is not None and not args.pinhole
        source = kb if self.fisheye else calib
        self.K = np.array([[source["fx"], 0, source["cx"]],
                           [0, source["fy"], source["cy"]],
                           [0, 0, 1.0]])
        self.D = np.array(source["dist"], dtype=np.float64)
        if self.fisheye:
            self.D = self.D.reshape(4, 1)
        print(f"model: {'Kannala-Brandt' if self.fisheye else 'pinhole+radtan'}")
        print(f"  fx {self.K[0,0]:.3f}  fy {self.K[1,1]:.3f}  "
              f"cx {self.K[0,2]:.3f}  cy {self.K[1,2]:.3f}")
        print(f"  D  {self.D.ravel().round(6).tolist()}\n")

        size = args.tag_size / 1000.0
        pitch = (1.0 + args.tag_spacing) * size
        # Outer extent of the tag grid, used to locate the board centre.
        self.board_span = (max(args.rows, args.cols) - 1) * pitch + size
        self.world = {}
        for row in range(args.rows):
            for col in range(args.cols):
                x0, y0 = col * pitch, row * pitch
                self.world[row * args.cols + col] = np.array([
                    [x0, y0 + size, 0.0], [x0 + size, y0 + size, 0.0],
                    [x0 + size, y0, 0.0], [x0, y0, 0.0]], dtype=np.float32)
        self.create_subscription(Image, args.topic, self.on_image, 5)

    def _undistort(self, points):
        pts = points.reshape(-1, 1, 2).astype(np.float64)
        if self.fisheye:
            return cv2.fisheye.undistortPoints(pts, self.K, self.D, P=self.K)
        return cv2.undistortPoints(pts, self.K, self.D, P=self.K)

    def on_image(self, message):
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, DICT, parameters=self.params)

        line = "no board"
        if ids is not None and len(ids):
            keep = [(int(i), c) for i, c in zip(ids.ravel(), corners)
                    if int(i) in self.world]
            if len(keep) >= 4:
                image_pts = np.concatenate([c[0] for _, c in keep], 0)
                world_pts = np.concatenate([self.world[i] for i, _ in keep], 0)
                undist = self._undistort(image_pts).reshape(-1, 2)
                ok, rvec, tvec = cv2.solvePnP(
                    world_pts, undist.astype(np.float32), self.K,
                    np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    proj, _ = cv2.projectPoints(
                        world_pts, rvec, tvec, self.K, np.zeros(5))
                    err = np.linalg.norm(
                        proj.reshape(-1, 2) - undist, axis=1)
                    # Split by distance from the image centre: the periphery is
                    # where a wrong wide-angle model shows itself.
                    radius = np.linalg.norm(
                        image_pts - np.array([width / 2, height / 2]), axis=1)
                    limit = 0.6 * math.hypot(width / 2, height / 2)
                    inner = err[radius <= limit]
                    outer = err[radius > limit]
                    rot, _ = cv2.Rodrigues(rvec)
                    # tvec points at the board ORIGIN, which is tag 0's
                    # bottom-left corner, not the board centre. On a 186 mm
                    # board that offset is large enough to look like a
                    # calibration error, so report the centre instead.
                    span = self.board_span
                    centre_world = np.array([[span / 2.0], [span / 2.0], [0.0]])
                    centre_cam = rot @ centre_world + tvec
                    distance = float(np.linalg.norm(centre_cam)) * 1000.0
                    depth = float(centre_cam[2]) * 1000.0
                    tilt = math.degrees(math.acos(
                        min(1.0, abs(float(rot[2, 2])))))
                    line = (f"tags {len(keep):3d}  centre-range {distance:7.1f} mm"
                            f"  depth {depth:7.1f} mm  tilt {tilt:5.1f} deg"
                            f"  reproj c "
                            f"{inner.mean() if inner.size else float('nan'):5.2f}"
                            f" e {outer.mean() if outer.size else float('nan'):5.2f} px")
                    if self.window_ok:
                        cv2.aruco.drawDetectedMarkers(
                            frame, corners, ids.reshape(-1, 1))
                        cv2.putText(frame, f"{distance:.0f} mm", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (0, 255, 0), 2)
                        cv2.putText(frame,
                                    f"reproj c {inner.mean():.2f} / e "
                                    f"{outer.mean() if outer.size else 0:.2f} px",
                                    (10, height - 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0), 2)
        sys.stdout.write("\r" + line + "    ")
        sys.stdout.flush()

        if self.window_ok:
            try:
                cv2.imshow("calibration check", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    raise KeyboardInterrupt
            except KeyboardInterrupt:
                raise
            except Exception:  # noqa: BLE001
                self.window_ok = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/tag_camera/image")
    parser.add_argument("--calib",
                        default="artifacts/calibration/camera_calib.json")
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--tag-size", type=float, default=30.0,
                        help="MEASURED outer black edge, mm")
    parser.add_argument("--tag-spacing", type=float, default=0.3)
    parser.add_argument("--pinhole", action="store_true",
                        help="use the pinhole model instead of Kannala-Brandt")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    with open(args.calib) as handle:
        calib = json.load(handle)

    rclpy.init()
    node = Check(args, calib)
    print("Point the camera at the AprilGrid. Compare 'dist' with a tape.")
    print("Edge reprojection should stay close to centre reprojection.\n")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print()
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
