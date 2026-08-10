#!/usr/bin/env python3
"""Generate a 3D-printable Kalibr AprilGrid as two STL bodies.

Output is two solids that occupy the same coordinate space:

    *_white.stl   base plate plus the WHITE cells of every tag
    *_black.stl   the BLACK cells of every tag

Together they form a flush plate with a two-tone top surface, so there is no
relief to cast shadows and confuse the detector. Print the white body, insert a
filament change at the top-surface height, then print the black body -- or
assign the two bodies to two extruders.

Layout follows Kalibr's GridCalibrationTargetAprilgrid::createGridPoints():
pitch is (1 + tagSpacing) * tagSize, tag id 0 sits at the origin (bottom-left),
and ids increase left to right then upward.

Usage:
    python3 tools/make_aprilgrid_stl.py --rows 5 --cols 5 --tag-size 30
"""
from __future__ import annotations

import argparse
import os
import struct

import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
CELLS = 8  # tag36h11: 6x6 payload plus a one-cell black border


def box_triangles(x0, y0, z0, x1, y1, z1):
    """Twelve triangles for an axis-aligned box."""
    corners = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
               (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [(0, 3, 2), (0, 2, 1),      # bottom
             (4, 5, 6), (4, 6, 7),      # top
             (0, 1, 5), (0, 5, 4),      # front
             (2, 3, 7), (2, 7, 6),      # back
             (1, 2, 6), (1, 6, 5),      # right
             (3, 0, 4), (3, 4, 7)]      # left
    return [(corners[a], corners[b], corners[c]) for a, b, c in faces]


def write_stl(path, triangles):
    with open(path, "wb") as handle:
        handle.write(b"\0" * 80)
        handle.write(struct.pack("<I", len(triangles)))
        for a, b, c in triangles:
            u = np.subtract(b, a)
            v = np.subtract(c, a)
            n = np.cross(u, v)
            length = np.linalg.norm(n)
            n = n / length if length > 1e-12 else np.array([0.0, 0.0, 1.0])
            handle.write(struct.pack("<3f", *n))
            for point in (a, b, c):
                handle.write(struct.pack("<3f", *point))
            handle.write(struct.pack("<H", 0))
    print(f"  {path}  ({len(triangles)} triangles, "
          f"{os.path.getsize(path)/1024:.0f} KB)")


def merge_runs(mask):
    """Merge horizontally adjacent equal cells into runs, to cut triangle count."""
    runs = []
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
            runs.append((r, start, c))  # row, col_start, col_end (exclusive)
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5, help="tags per column")
    parser.add_argument("--cols", type=int, default=5, help="tags per row")
    parser.add_argument("--tag-size", type=float, default=30.0, help="mm")
    parser.add_argument("--tag-spacing", type=float, default=0.3)
    parser.add_argument("--plate-thickness", type=float, default=3.0, help="mm")
    parser.add_argument("--top-thickness", type=float, default=0.6, help="mm")
    parser.add_argument("--margin", type=float, default=6.0, help="mm border")
    parser.add_argument("--out", default="artifacts/calibration/aprilgrid_3dprint")
    args = parser.parse_args()

    pitch = (1.0 + args.tag_spacing) * args.tag_size
    grid_w = (args.cols - 1) * pitch + args.tag_size
    grid_h = (args.rows - 1) * pitch + args.tag_size
    plate_w = grid_w + 2 * args.margin
    plate_h = grid_h + 2 * args.margin
    cell = args.tag_size / CELLS
    z0 = args.plate_thickness
    z1 = args.plate_thickness + args.top_thickness

    print(f"AprilGrid {args.rows}x{args.cols} tags, tagSize {args.tag_size} mm, "
          f"spacing {args.tag_spacing}")
    print(f"  pattern {grid_w:.1f} x {grid_h:.1f} mm")
    print(f"  plate   {plate_w:.1f} x {plate_h:.1f} x "
          f"{z1:.1f} mm (cell {cell:.3f} mm)\n")

    white = box_triangles(0, 0, 0, plate_w, plate_h, z0)   # base plate
    black = []

    for row in range(args.rows):
        for col in range(args.cols):
            tag_id = row * args.cols + col          # id 0 bottom-left
            bitmap = cv2.aruco.generateImageMarker(DICT, tag_id, CELLS)
            is_black = bitmap < 128
            ox = args.margin + col * pitch
            oy = args.margin + row * pitch
            for target, mask in ((black, is_black), (white, ~is_black)):
                for r, c_start, c_end in merge_runs(mask):
                    # Bitmap row 0 is the tag's TOP; flip so +y is up.
                    x_lo = ox + c_start * cell
                    x_hi = ox + c_end * cell
                    y_lo = oy + (CELLS - 1 - r) * cell
                    y_hi = y_lo + cell
                    target.extend(box_triangles(x_lo, y_lo, z0, x_hi, y_hi, z1))

    # Margin ring, so the top surface is flush everywhere.
    ring = [(0, 0, plate_w, args.margin),
            (0, plate_h - args.margin, plate_w, plate_h),
            (0, args.margin, args.margin, plate_h - args.margin),
            (plate_w - args.margin, args.margin, plate_w, plate_h - args.margin)]
    for x_lo, y_lo, x_hi, y_hi in ring:
        white.extend(box_triangles(x_lo, y_lo, z0, x_hi, y_hi, z1))
    # Gaps between tags also belong to the white top layer.
    for row in range(args.rows):
        for col in range(args.cols):
            if col < args.cols - 1:
                x_lo = args.margin + col * pitch + args.tag_size
                white.extend(box_triangles(
                    x_lo, args.margin + row * pitch, z0,
                    args.margin + (col + 1) * pitch,
                    args.margin + row * pitch + args.tag_size, z1))
            if row < args.rows - 1:
                y_lo = args.margin + row * pitch + args.tag_size
                white.extend(box_triangles(
                    args.margin + col * pitch, y_lo, z0,
                    args.margin + col * pitch + args.tag_size,
                    args.margin + (row + 1) * pitch, z1))
    for row in range(args.rows - 1):
        for col in range(args.cols - 1):
            white.extend(box_triangles(
                args.margin + col * pitch + args.tag_size,
                args.margin + row * pitch + args.tag_size, z0,
                args.margin + (col + 1) * pitch,
                args.margin + (row + 1) * pitch, z1))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_stl(f"{args.out}_white.stl", white)
    write_stl(f"{args.out}_black.stl", black)

    with open(f"{args.out}.yaml", "w") as handle:
        handle.write(
            "target_type: 'aprilgrid'\n"
            f"tagCols: {args.cols}\n"
            f"tagRows: {args.rows}\n"
            f"tagSize: {args.tag_size/1000.0:.6f}\n"
            f"tagSpacing: {args.tag_spacing}\n"
            "# tagSize is the OUTER black edge of one tag, in metres.\n"
            "# Measure a printed tag with calipers and correct it here:\n"
            "# 3D printers shrink, typically a few tenths of a percent.\n")
    print(f"  {args.out}.yaml")
    print(f"\nfilament change at Z = {z0:.1f} mm")


if __name__ == "__main__":
    main()
