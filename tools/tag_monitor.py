#!/usr/bin/env python3
"""Live readout of every AprilTag the camera can see.

Reports each tag's id, apparent size in pixels, where it sits in the frame, and
how reliably it is being detected. Use it to check marker placement and to find
the smallest tag size that still detects at the board corners.

Detection rate matters more than a single hit: a tag that appears in 60% of
frames will drop out during motion, which is worse than it sounds when the
pose solve needs all four.

Usage:
    python3 tools/tag_monitor.py [--family tag36h11]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

FAMILIES = {
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
}


class Monitor(Node):
    def __init__(self, args):
        super().__init__("tag_monitor")
        self.args = args
        self.bridge = CvBridge()
        self.dict = cv2.aruco.getPredefinedDictionary(FAMILIES[args.family])
        self.params = cv2.aruco.DetectorParameters()
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
        self.detector = cv2.aruco.ArucoDetector(self.dict, self.params)
        self.frames = 0
        self.hits = defaultdict(int)
        self.size = defaultdict(list)
        self.pos = {}
        self.window_ok = not args.no_window
        self.create_subscription(Image, args.topic, self.on_image, 5)

    def on_image(self, message):
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        self.frames += 1

        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is not None and len(ids):
            for tag_id, corner in zip(ids.ravel(), corners):
                pts = corner[0]
                # Mean edge length: the tag's apparent size in pixels.
                edges = [np.linalg.norm(pts[(i + 1) % 4] - pts[i])
                         for i in range(4)]
                self.hits[int(tag_id)] += 1
                self.size[int(tag_id)].append(float(np.mean(edges)))
                self.pos[int(tag_id)] = pts.mean(axis=0)

        if self.frames % 10 == 0:
            sys.stdout.write("\x1b[2J\x1b[H")
            print(f"tag monitor  ({self.args.family})   frames {self.frames}\n")
            if not self.hits:
                print("  no tags detected")
            else:
                print("   id   seen    rate    size px   position (x, y)")
                for tag_id in sorted(self.hits):
                    rate = self.hits[tag_id] / self.frames
                    sizes = self.size[tag_id][-60:]
                    x, y = self.pos[tag_id]
                    flag = ""
                    if rate < 0.8:
                        flag = "  <- intermittent"
                    if np.mean(sizes) < 20:
                        flag += "  <- too small"
                    print(f"  {tag_id:3d}  {self.hits[tag_id]:5d}  "
                          f"{rate:5.0%}   {np.mean(sizes):7.1f}   "
                          f"({x:5.0f}, {y:5.0f}){flag}")
                print(f"\n  {len(self.hits)} distinct ids seen")
            print("\n  Ctrl-C (or q in the window) to stop.")

        if self.window_ok:
            try:
                if ids is not None and len(ids):
                    cv2.aruco.drawDetectedMarkers(
                        frame, corners, ids.reshape(-1, 1))
                cv2.putText(frame,
                            f"{0 if ids is None else len(ids)} tags",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
                cv2.imshow("tag monitor", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    raise KeyboardInterrupt
            except KeyboardInterrupt:
                raise
            except Exception:  # noqa: BLE001
                self.window_ok = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/tag_camera/image")
    parser.add_argument("--family", default="tag36h11", choices=list(FAMILIES))
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = Monitor(args)
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
