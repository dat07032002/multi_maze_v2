#!/usr/bin/env python3
"""Generate four minimal single-tag plates for the board corners.

Each plate carries one tag with a distinct id, so the pose solve knows which
corner it is looking at without any correspondence guessing.

Plates are kept as small as detection allows. An AprilTag needs a light quiet
zone around its black border or the detector cannot find the outer edge; one
cell width is the practical minimum, which for a 20 mm tag36h11 is 2.5 mm per
side and gives a 25 mm plate.

Two bodies are written so the top surface is flush rather than raised, which
avoids shadows at grazing light:

    *_white.stl   plates plus the WHITE cells
    *_black.stl   the BLACK cells

Usage:
    python3 tools/make_corner_tags_stl.py --tag-size 20 --quiet-cells 1
"""
from __future__ import annotations

import argparse
import os
import struct

import cv2
import numpy as np

FAMILIES = {
    "tag36h11": (cv2.aruco.DICT_APRILTAG_36h11, 8),
    "tag25h9": (cv2.aruco.DICT_APRILTAG_25h9, 7),
    "tag16h5": (cv2.aruco.DICT_APRILTAG_16h5, 6),
}


def box(x0, y0, z0, x1, y1, z1):
    c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
         (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7),
             (0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6),
             (1, 2, 6), (1, 6, 5), (3, 0, 4), (3, 4, 7)]
    return [(c[a], c[b], c[d]) for a, b, d in faces]


def write_stl(path, triangles):
    with open(path, "wb") as handle:
        handle.write(b"\0" * 80)
        handle.write(struct.pack("<I", len(triangles)))
        for a, b, c in triangles:
            n = np.cross(np.subtract(b, a), np.subtract(c, a))
            length = np.linalg.norm(n)
            n = n / length if length > 1e-12 else np.array([0.0, 0.0, 1.0])
            handle.write(struct.pack("<3f", *n))
            for point in (a, b, c):
                handle.write(struct.pack("<3f", *point))
            handle.write(struct.pack("<H", 0))
    print(f"  {path}  ({len(triangles)} tri, {os.path.getsize(path)/1024:.0f} KB)")


def runs(mask):
    out = []
    rows, cols = mask.shape
    for r in range(rows):
        c = 0
        while c < cols:
            if not mask[r, c]:
                c += 1
                continue
            start = c
            while c < cols and mask[r, c]:
                c += 1
            out.append((r, start, c))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-size", type=float, default=20.0, help="mm")
    parser.add_argument("--family", default="tag36h11", choices=list(FAMILIES))
    parser.add_argument("--ids", type=int, nargs=4, default=[0, 1, 2, 3])
    parser.add_argument("--quiet-cells", type=float, default=1.0,
                        help="quiet zone in tag cells; 1 is the minimum")
    parser.add_argument("--plate-thickness", type=float, default=1.5)
    parser.add_argument("--top-thickness", type=float, default=0.6)
    parser.add_argument("--gap", type=float, default=6.0,
                        help="spacing between plates on the print bed")
    parser.add_argument("--out", default="artifacts/calibration/corner_tags")
    args = parser.parse_args()

    dict_id, cells = FAMILIES[args.family]
    DICT = cv2.aruco.getPredefinedDictionary(dict_id)
    cell = args.tag_size / cells
    quiet = args.quiet_cells * cell
    plate = args.tag_size + 2 * quiet
    z0 = args.plate_thickness
    z1 = z0 + args.top_thickness

    print(f"{args.family}, tag {args.tag_size:.1f} mm, cell {cell:.2f} mm")
    print(f"quiet zone {quiet:.2f} mm per side -> plate "
          f"{plate:.1f} x {plate:.1f} x {z1:.1f} mm")
    print(f"ids {args.ids}\n")

    white, black = [], []
    for index, tag_id in enumerate(args.ids):
        ox = index * (plate + args.gap)
        # Plate body.
        white.extend(box(ox, 0, 0, ox + plate, plate, z0))
        # Quiet-zone ring on the top layer, so the surface finishes flush.
        for x_lo, y_lo, x_hi, y_hi in (
            (0, 0, plate, quiet),
            (0, plate - quiet, plate, plate),
            (0, quiet, quiet, plate - quiet),
            (plate - quiet, quiet, plate, plate - quiet),
        ):
            white.extend(box(ox + x_lo, y_lo, z0, ox + x_hi, y_hi, z1))
        # The tag itself.
        bitmap = cv2.aruco.generateImageMarker(DICT, tag_id, cells)
        is_black = bitmap < 128
        for target, mask in ((black, is_black), (white, ~is_black)):
            for r, c0, c1 in runs(mask):
                x_lo = ox + quiet + c0 * cell
                x_hi = ox + quiet + c1 * cell
                # Bitmap row 0 is the tag's top; flip so +y is up.
                y_lo = quiet + (cells - 1 - r) * cell
                target.extend(box(x_lo, y_lo, z0, x_hi, y_lo + cell, z1))
        print(f"  plate {index}: id {tag_id}   x {ox:.1f} -> {ox+plate:.1f} mm")

    total = 4 * plate + 3 * args.gap
    print(f"\nprint bed footprint: {total:.1f} x {plate:.1f} mm")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_stl(f"{args.out}_white.stl", white)
    write_stl(f"{args.out}_black.stl", black)
    print(f"\nfilament change at Z = {z0:.1f} mm")


if __name__ == "__main__":
    main()
