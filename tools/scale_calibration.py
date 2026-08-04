#!/usr/bin/env python3
"""Rescale a camera calibration to a different published resolution.

Valid only for a pure uniform resize with no crop, which is what
fast_camera_publisher does: it captures 1920x1200 and resizes to the output
size in one step. Under that operation the pinhole terms scale linearly

    fx, fy, cx, cy  ->  multiplied by (new_width / old_width)

while the distortion coefficients are unchanged, because both the OpenCV
radial-tangential and the Kannala-Brandt models act on normalised coordinates
that a uniform resize leaves alone.

This does NOT hold if the capture is cropped or letterboxed. Check border_y
and the aspect ratio before trusting the result.

Usage:
    python3 tools/scale_calibration.py --width 1280 --height 800
"""
from __future__ import annotations

import argparse
import json
import math
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib",
                        default="artifacts/calibration/camera_calib.json")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.calib) as handle:
        calib = json.load(handle)

    old_w = int(calib["image_width"])
    old_h = int(calib["image_height"])
    sx = args.width / old_w
    sy = args.height / old_h
    if not math.isclose(sx, sy, rel_tol=1e-6):
        raise SystemExit(
            f"Non-uniform scale {sx:.4f} x {sy:.4f}: the aspect ratio changed, "
            "so this is a crop or letterbox, not a resize. Recalibrate instead.")

    print(f"{old_w}x{old_h} -> {args.width}x{args.height}   scale {sx:.4f}\n")
    scaled = dict(calib)
    scaled["image_width"] = args.width
    scaled["image_height"] = args.height
    scaled["scaled_from"] = {"width": old_w, "height": old_h, "factor": sx,
                             "source": os.path.basename(args.calib)}

    for key in ("kannala_brandt", "pinhole_radtan"):
        if key not in calib:
            continue
        model = dict(calib[key])
        for term in ("fx", "fy", "cx", "cy"):
            model[term] = float(calib[key][term]) * sx
        scaled[key] = model
        print(f"{key}:")
        print(f"  fx {calib[key]['fx']:8.3f} -> {model['fx']:8.3f}")
        print(f"  fy {calib[key]['fy']:8.3f} -> {model['fy']:8.3f}")
        print(f"  cx {calib[key]['cx']:8.3f} -> {model['cx']:8.3f}")
        print(f"  cy {calib[key]['cy']:8.3f} -> {model['cy']:8.3f}")
        print(f"  dist unchanged: "
              f"{[round(v, 6) for v in calib[key]['dist']]}")

    # Top-level intrinsics, if the file carries them flat as well.
    if "fx" in calib:
        for term in ("fx", "fy", "cx", "cy"):
            scaled[term] = float(calib[term]) * sx

    out = args.out or args.calib.replace(
        ".json", f"_{args.width}x{args.height}.json")
    with open(out, "w") as handle:
        json.dump(scaled, handle, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
