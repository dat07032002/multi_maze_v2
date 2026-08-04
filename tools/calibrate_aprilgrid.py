#!/usr/bin/env python3
"""Calibrate from captured AprilGrid views and compare models.

AprilGrid views have variable point counts, because partial views are valid and
contribute correctly placed points. cv2.calibrateCamera accepts per-view lists,
so that needs no special handling beyond not assuming a fixed grid.

The fisheye solver is seeded from the pinhole result. Started from zeros it
silently returns the initial guess with all distortion coefficients still zero,
which reads as a catastrophic fit rather than a failure to converge.

Usage:
    python3 tools/calibrate_aprilgrid.py [capture_dir]
"""
from __future__ import annotations

import argparse
import json
import math
import os

import cv2
import numpy as np


def fov(fx, fy, width, height):
    return (2 * math.degrees(math.atan(width / 2 / fx)),
            2 * math.degrees(math.atan(height / 2 / fy)),
            2 * math.degrees(math.atan(
                math.hypot(width / 2, height / 2) / ((fx + fy) / 2))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", nargs="?",
                        default="artifacts/calibration/capture_april")
    parser.add_argument("--out",
                        default="artifacts/calibration/aprilgrid_calib.json")
    parser.add_argument("--skip-fisheye", action="store_true")
    args = parser.parse_args()

    data = np.load(os.path.join(args.dir, "aprilgrid_views.npz"),
                   allow_pickle=True)
    world = [np.asarray(w, dtype=np.float32) for w in data["world"]]
    image = [np.asarray(i, dtype=np.float32) for i in data["image"]]
    height, width = (int(v) for v in data["image_shape"])
    n = len(world)
    print(f"{n} views, {sum(len(w) for w in world)} points, "
          f"image {width}x{height}")
    print(f"tag {float(data['tag_size_mm'])} mm, "
          f"spacing {float(data['tag_spacing'])}\n")

    # ---- pinhole + radial-tangential ------------------------------------
    p_rms, K, D, _, _ = cv2.calibrateCamera(
        world, image, (width, height), None, None)
    p_fov = fov(K[0, 0], K[1, 1], width, height)
    print("=== pinhole + radtan ===")
    print(f"  RMS {p_rms:.4f} px")
    print(f"  fx {K[0,0]:8.3f}  fy {K[1,1]:8.3f}")
    print(f"  cx {K[0,2]:8.3f}  cy {K[1,2]:8.3f}   "
          f"(frame centre {width/2:.1f}, {height/2:.1f})")
    print(f"  D  {np.round(D.ravel(), 6).tolist()}")
    print(f"  FOV  H {p_fov[0]:.1f}  V {p_fov[1]:.1f}  D {p_fov[2]:.1f} deg")

    payload = {
        "source": args.dir,
        "image_width": width, "image_height": height,
        "num_views": int(n),
        "num_points": int(sum(len(w) for w in world)),
        "tag_size_mm": float(data["tag_size_mm"]),
        "tag_spacing": float(data["tag_spacing"]),
        "pinhole_radtan": {
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "dist": [float(v) for v in D.ravel()],
            "rms_reprojection_px": float(p_rms),
            "fov_h_deg": p_fov[0], "fov_v_deg": p_fov[1],
            "fov_diag_deg": p_fov[2],
        },
    }

    # ---- Kannala-Brandt, seeded -----------------------------------------
    if not args.skip_fisheye:
        print("\n=== Kannala-Brandt (fisheye), seeded ===")
        fK = np.array([[K[0, 0], 0.0, K[0, 2]],
                       [0.0, K[1, 1], K[1, 2]],
                       [0.0, 0.0, 1.0]])
        fD = np.zeros((4, 1))
        f_world = [w.reshape(-1, 1, 3).astype(np.float64) for w in world]
        f_image = [i.reshape(-1, 1, 2).astype(np.float64) for i in image]
        flags = (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
                 | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                 | cv2.fisheye.CALIB_FIX_SKEW)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    100, 1e-8)
        try:
            f_rms, fK, fD, _, _ = cv2.fisheye.calibrate(
                f_world, f_image, (width, height), fK, fD,
                flags=flags, criteria=criteria)
            f_fov = fov(fK[0, 0], fK[1, 1], width, height)
            print(f"  RMS {f_rms:.4f} px")
            print(f"  fx {fK[0,0]:8.3f}  fy {fK[1,1]:8.3f}")
            print(f"  D  {np.round(fD.ravel(), 6).tolist()}")
            print(f"  FOV diagonal {f_fov[2]:.1f} deg")
            payload["kannala_brandt"] = {
                "fx": float(fK[0, 0]), "fy": float(fK[1, 1]),
                "cx": float(fK[0, 2]), "cy": float(fK[1, 2]),
                "dist": [float(v) for v in fD.ravel()],
                "rms_reprojection_px": float(f_rms),
                "fov_diag_deg": f_fov[2],
            }
            print(f"\n  residuals: pinhole {p_rms:.4f} vs fisheye {f_rms:.4f} px"
                  f" -> {'pinhole' if p_rms <= f_rms else 'fisheye'}")
        except cv2.error as exc:
            print(f"  failed: {str(exc)[:160]}")

    payload["recommended_model"] = (
        "pinhole_radtan" if p_fov[2] < 100 else "kannala_brandt")
    print(f"\nmeasured diagonal FOV {p_fov[2]:.1f} deg -> "
          f"recommend {payload['recommended_model']}")
    print("  (literature: pinhole+radtan adequate below ~100 deg DAOV)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
