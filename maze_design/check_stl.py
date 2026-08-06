#!/usr/bin/env python3
"""Validate an exported maze STL against the layout it came from.

Mesh statistics alone do not prove a part is right: a maze whose holes were
never cut is perfectly watertight. So beyond the structural checks this casts
rays to ask whether specific points are inside the material, and compares the
answers against what the layout says should be there -- holes empty, floor solid
under the route, corridors clear above it, walls solid.

Solidity uses a signed winding count rather than crossing parity. The export is
a union of overlapping solids by design, and parity miscounts those: two nested
surfaces flip parity back to "outside". Summing the sign of each crossing
(from the triangle normal against the ray direction) counts containment depth
instead, which is correct for unions.

Usage:
    python3 maze_design/check_stl.py maze_design/maze_256x226.json
    python3 maze_design/check_stl.py maze_design/maze_256x226.json --split
"""
from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np

RAY = np.array([0.0, 0.0, 1.0])
EPSILON = 1e-9


def load_stl(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"{path}: too short to be an STL")
    count = struct.unpack("<I", data[80:84])[0]
    expected = 84 + 50 * count
    if len(data) != expected:
        raise ValueError(
            f"{path}: header claims {count} triangles ({expected} bytes) "
            f"but the file is {len(data)} bytes"
        )
    records = np.frombuffer(data[84:], dtype=np.uint8).reshape(count, 50)
    return records[:, 12:48].copy().view("<f4").reshape(count, 3, 3).astype(np.float64)


def signed_volume(triangles: np.ndarray) -> float:
    """Divergence-theorem volume. Negative means normals point inward."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def winding(point: np.ndarray, triangles: np.ndarray) -> int:
    """Signed count of +z crossings above `point`. Non-zero means inside."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    edge1, edge2 = b - a, c - a
    pvec = np.cross(RAY, edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)

    parallel = np.abs(det) < EPSILON
    safe = np.where(parallel, 1.0, det)
    tvec = point - a
    u = np.einsum("ij,ij->i", tvec, pvec) / safe
    qvec = np.cross(tvec, edge1)
    v = np.einsum("ij,ij->i", np.broadcast_to(RAY, tvec.shape), qvec) / safe
    t = np.einsum("ij,ij->i", edge2, qvec) / safe

    hit = (~parallel) & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > EPSILON)
    if not hit.any():
        return 0
    # Sign of the crossing: normal facing the ray means leaving the solid.
    normals = np.cross(edge1, edge2)
    facing = np.sign(normals[hit] @ RAY)
    return int(facing.sum())


def inside(point, triangles) -> bool:
    return winding(np.asarray(point, dtype=np.float64), triangles) > 0


def structural_report(name: str, triangles: np.ndarray) -> list[str]:
    problems = []
    vertices = triangles.reshape(-1, 3)
    lo, hi = vertices.min(0), vertices.max(0)

    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0],
                 triangles[:, 2] - triangles[:, 0]), axis=1)
    degenerate = int((areas < 1e-9).sum())

    edges = Counter()
    for tri in triangles:
        keys = [tuple(np.round(p, 4)) for p in tri]
        for i in range(3):
            edges[frozenset((keys[i], keys[(i + 1) % 3]))] += 1
    odd = sum(1 for n in edges.values() if n % 2)

    canonical = Counter(
        tuple(sorted(tuple(np.round(p, 4)) for p in tri)) for tri in triangles)
    duplicates = sum(n - 1 for n in canonical.values() if n > 1)

    volume = signed_volume(triangles)

    print(f"\n{name}")
    print(f"  triangles        {len(triangles)}")
    print(f"  bounding box     {hi[0]-lo[0]:.3f} x {hi[1]-lo[1]:.3f} x "
          f"{hi[2]-lo[2]:.3f} mm")
    print(f"  origin           {lo.round(4)}")
    print(f"  z range          {lo[2]:.3f} .. {hi[2]:.3f}")
    print(f"  degenerate       {degenerate}")
    print(f"  duplicate tris   {duplicates}")
    print(f"  odd-count edges  {odd}")
    print(f"  signed volume    {volume/1000.0:.2f} cm3 "
          f"({'outward normals' if volume > 0 else 'INWARD NORMALS'})")

    if degenerate:
        problems.append(f"{name}: {degenerate} degenerate triangles")
    if odd:
        problems.append(f"{name}: {odd} edges used an odd number of times "
                        "(open surface)")
    if volume <= 0:
        problems.append(f"{name}: normals point inward (negative volume)")
    if lo[2] < -1e-6:
        problems.append(f"{name}: geometry below z=0")
    return problems


def geometry_report(layout: dict, triangles: np.ndarray,
                    floor_mm: float) -> list[str]:
    """Ray-cast the places the layout says should be solid or empty."""
    problems = []
    width = layout["board_width"] * 1000.0
    height = layout["board_height"] * 1000.0
    wall_top = floor_mm + layout["wall_height"] * 1000.0
    mid_floor = floor_mm / 2.0
    mid_wall = floor_mm + (wall_top - floor_mm) / 2.0

    print("\nsolidity probes (ray-cast winding)")

    def check(label, point, want_solid, bucket):
        """Winding sign, not just non-zero.

        A solid whose faces are wound backwards still encloses a region, so a
        `!= 0` test calls it solid and the defect passes. Inside an inverted
        body the winding is negative, so requiring `> 0` catches it -- this is
        what the hole surrounds failed, while every other check passed.
        """
        count = winding(np.asarray(point, dtype=np.float64), triangles)
        got = count > 0
        ok = got == want_solid
        bucket.append(ok)
        if count < 0:
            problems.append(
                f"{label} at {np.round(point, 2).tolist()}: winding {count} "
                "-- this solid is inside-out")
        elif not ok:
            problems.append(
                f"{label} at {np.round(point, 2).tolist()}: expected "
                f"{'solid' if want_solid else 'empty'}, got "
                f"{'solid' if got else 'empty'}")
        return ok

    results = []
    for (hx, hy) in layout["holes"]:
        check("hole centre", [hx * 1000.0, hy * 1000.0, mid_floor], False, results)
    print(f"  hole centres empty in the floor      {sum(results)}/{len(results)}")

    results = []
    for (hx, hy), radius in zip(layout["holes"], layout["hole_radii"]):
        offset = radius * 1000.0 + 1.0
        check("hole surround", [hx * 1000.0 + offset, hy * 1000.0, mid_floor],
              True, results)
    print(f"  ring of material around each hole    {sum(results)}/{len(results)}")

    results = []
    step = max(1, len(layout["waypoints"]) // 12)
    for x, y in layout["waypoints"][::step]:
        check("floor under route", [x * 1000.0, y * 1000.0, mid_floor], True, results)
    print(f"  floor solid under the route          {sum(results)}/{len(results)}")

    results = []
    for x, y in layout["waypoints"][::step]:
        check("corridor", [x * 1000.0, y * 1000.0, mid_wall], False, results)
    print(f"  corridor clear above the floor       {sum(results)}/{len(results)}")

    results = []
    for x_lo, x_hi, y in layout["walls_h"][:14]:
        mx = (x_lo + x_hi) * 500.0
        check("wall_h", [mx, y * 1000.0, mid_wall], True, results)
    for y_lo, y_hi, x in layout["walls_v"][:14]:
        my = (y_lo + y_hi) * 500.0
        check("wall_v", [x * 1000.0, my, mid_wall], True, results)
    print(f"  walls solid at mid height            {sum(results)}/{len(results)}")

    results = []
    for pad in layout.get("tag_pads", []):
        cx, cy = pad["centre_mm"]
        solid = pad.get("mount") != "insert"
        check("tag pad centre", [cx, cy, mid_wall], solid, results)
    label = "solid" if layout.get("tag_pads", [{}])[0].get("mount") != "insert" \
        else "recessed"
    print(f"  tag pads {label:9s} at mid height       {sum(results)}/{len(results)}")

    results = []
    check("outside +x", [width + 5.0, height / 2, mid_floor], False, results)
    check("outside +y", [width / 2, height + 5.0, mid_floor], False, results)
    check("above walls", [width / 2, height / 2, wall_top + 5.0], False, results)
    print(f"  空 outside the board                  {sum(results)}/{len(results)}"
          .replace("空 ", ""))

    return problems


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("layout")
    parser.add_argument("--floor-mm", type=float, default=3.0)
    parser.add_argument("--split", action="store_true",
                        help="check the _floor/_walls pair instead of the "
                             "combined STL")
    args = parser.parse_args()

    layout_path = Path(args.layout)
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    stem = layout_path.with_suffix("")

    problems: list[str] = []
    if args.split:
        floor = load_stl(Path(f"{stem}_floor.stl"))
        walls = load_stl(Path(f"{stem}_walls.stl"))
        problems += structural_report(f"{stem.name}_floor.stl", floor)
        problems += structural_report(f"{stem.name}_walls.stl", walls)

        floor_top = floor.reshape(-1, 3)[:, 2].max()
        wall_bottom = walls.reshape(-1, 3)[:, 2].min()
        print(f"\ninterface: floor top {floor_top:.3f} mm, "
              f"wall bottom {wall_bottom:.3f} mm")
        if abs(floor_top - wall_bottom) > 1e-6:
            problems.append(
                f"floor top ({floor_top:.3f}) and wall bottom "
                f"({wall_bottom:.3f}) do not meet -- the parts would not bond")

        combined = np.concatenate([floor, walls])
    else:
        combined = load_stl(Path(f"{stem}.stl"))
        problems += structural_report(f"{stem.name}.stl", combined)

    problems += geometry_report(layout, combined, args.floor_mm)

    print()
    if problems:
        print(f"FAILED -- {len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("PASSED -- structure and geometry both match the layout")


if __name__ == "__main__":
    main()
