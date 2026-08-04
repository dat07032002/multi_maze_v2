#!/usr/bin/env python3
"""Live view plus a focus score, for setting an M12 lens by hand.

Turn the lens barrel slowly and watch the score. It peaks at best focus.
The score is the variance of the Laplacian over the centre region, which
responds to edge contrast: blurred images score near zero, sharp ones high.

A window is opened when a display is available, but the terminal readout is
the authoritative one -- GUI windows do not render reliably on this machine,
so the number is printed either way.

Usage:
    python3 tools/focus_assist.py [--topic /tag_camera/image] [--no-window]
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class FocusAssist(Node):
    def __init__(self, args):
        super().__init__("focus_assist")
        self.args = args
        self.bridge = CvBridge()
        self.best = 0.0
        self.count = 0
        self.window_ok = not args.no_window
        self.create_subscription(Image, args.topic, self.on_image, 5)

    @staticmethod
    def _score(gray):
        height, width = gray.shape
        # Centre half of the frame: corners are distorted and often darker.
        patch = gray[height // 4:3 * height // 4, width // 4:3 * width // 4]
        return float(cv2.Laplacian(patch, cv2.CV_64F).var())

    def on_image(self, message):
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = self._score(gray)
        self.best = max(self.best, score)
        self.count += 1

        if self.count % 3 == 0:
            share = score / self.best if self.best > 1e-6 else 0.0
            bar = "#" * int(share * 45)
            verdict = ("SHARP" if score > 500 else
                       "usable" if score > 150 else
                       "BLURRED")
            sys.stdout.write(
                f"\rfocus {score:9.1f}  best {self.best:9.1f}  "
                f"|{bar:<45}| {verdict}   ")
            sys.stdout.flush()

        if self.window_ok:
            try:
                display = frame.copy()
                cv2.putText(display, f"focus {score:.0f}  best {self.best:.0f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0), 2)
                width = int(min(1.0, score / max(self.best, 1e-6)) * (display.shape[1] - 20))
                cv2.rectangle(display, (10, 45), (10 + width, 65), (0, 255, 0), -1)
                cv2.rectangle(display, (10, 45),
                              (display.shape[1] - 10, 65), (0, 255, 0), 1)
                cv2.imshow("focus assist - turn the lens barrel", display)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    raise KeyboardInterrupt
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"\n(window unavailable: {exc}; continuing with numbers)")
                self.window_ok = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/tag_camera/image")
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = FocusAssist(args)
    print("Turn the lens barrel slowly. The score peaks at best focus.")
    print("Aim at the checkerboard at its real working distance. Ctrl-C to stop.\n")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print(f"\n\nbest focus score seen: {node.best:.1f}")
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:  # noqa: BLE001
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
