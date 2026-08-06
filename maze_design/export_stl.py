#!/usr/bin/env python3
"""Export a maze layout to a printable binary STL, in millimetres.

The existing `maze_print.stl` was produced outside this repository and is scaled
in **metres** (bounding box 0.259 x 0.229 x 0.011), so a slicer that assumes the
usual millimetre convention imports it as a quarter-millimetre speck. This
exporter emits millimetres and is reproducible from the layout JSON.

Construction: the part is a union of individually closed solids rather than one
CSG mesh. Slicers union overlapping solids reliably, so walls may abut and
overlap the floor without needing exact boolean geometry -- but a hole cannot be
made by union, so the floor is genuinely cut.

The floor is partitioned by coordinate compression on the hole squares and the
corner tag pads. Each resulting rectangle is emitted as a plain box, as an
annulus prism (square minus inscribed circle) where it carries a hole, or
skipped where a tag pad covers it.

Each hole's square is grown past the hole radius on purpose: were the square
exactly the circle's bounding box, the material would pinch to zero width at the
four edge midpoints and produce degenerate triangles.

Z convention, matching the layout:
    0                       bottom of the floor
    floor_thickness         playing surface
    floor + wall_height     top of the walls, and the tag pad top
Tag pads are solid unless the layout carries pocket fields, in which case a
recess is cut from the pad top so a tag drops in flush.

Usage:
    python3 maze_design/export_stl.py maze_design/maze_256x226.json
    python3 maze_design/export_stl.py maze_design/maze_256x226.json --split
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

FLOOR_THICKNESS_MM = 3.0
CIRCLE_SEGMENTS = 64          # multiple of 4 so square corners are hit exactly
SNAP_MM = 1e-3                # below print resolution; kills float-noise slivers


class Mesh:
    """Accumulates triangles. Normals are recomputed on write."""

    def __init__(self):
        self.triangles: list[tuple] = []

    def tri(self, a, b, c) -> None:
        self.triangles.append((a, b, c))

    def quad(self, a, b, c, d) -> None:
        """Counter-clockwise seen from outside."""
        self.tri(a, b, c)
        self.tri(a, c, d)

    def box(self, x0, y0, z0, x1, y1, z1) -> None:
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            return
        v = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        self.quad(v[0], v[3], v[2], v[1])   # bottom, normal -z
        self.quad(v[4], v[5], v[6], v[7])   # top, normal +z
        self.quad(v[0], v[1], v[5], v[4])   # -y
        self.quad(v[1], v[2], v[6], v[5])   # +x
        self.quad(v[2], v[3], v[7], v[6])   # +y
        self.quad(v[3], v[0], v[4], v[7])   # -x

    def annulus_prism(self, cx, cy, half, radius, z0, z1) -> None:
        """Square of side 2*half centred on (cx, cy), with a circular hole."""
        outer, inner = [], []
        for i in range(CIRCLE_SEGMENTS):
            theta = 2.0 * math.pi * i / CIRCLE_SEGMENTS
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            inner.append((cx + radius * cos_t, cy + radius * sin_t))
            # Radial projection of the direction onto the square boundary.
            reach = half / max(abs(cos_t), abs(sin_t))
            outer.append((cx + reach * cos_t, cy + reach * sin_t))

        for i in range(CIRCLE_SEGMENTS):
            j = (i + 1) % CIRCLE_SEGMENTS
            ix, iy = inner[i]
            jx, jy = inner[j]
            ox, oy = outer[i]
            px, py = outer[j]

            # Winding runs outer -> inner around the ring. Taking it the other
            # way (inner -> outer, which reads more naturally) puts every one
            # of these four faces backwards: the ring's signed volume comes out
            # negative and slicers see the hole surround as inside-out.
            self.quad((ox, oy, z1), (px, py, z1), (jx, jy, z1), (ix, iy, z1))
            self.quad((ix, iy, z0), (jx, jy, z0), (px, py, z0), (ox, oy, z0))
            # Hole wall: normal points into the hole, away from material.
            self.quad((ix, iy, z1), (jx, jy, z1), (jx, jy, z0), (ix, iy, z0))
            # Outer square wall, normal pointing away from the centre.
            self.quad((px, py, z0), (px, py, z1), (ox, oy, z1), (ox, oy, z0))

    def write_stl(self, path: Path, name: str = "maze") -> int:
        with open(path, "wb") as handle:
            handle.write(name.encode("ascii", "replace")[:80].ljust(80, b"\0"))
            handle.write(struct.pack("<I", len(self.triangles)))
            for a, b, c in self.triangles:
                ux, uy, uz = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
                vx, vy, vz = (c[0] - a[0], c[1] - a[1], c[2] - a[2])
                nx, ny, nz = (uy * vz - uz * vy,
                              uz * vx - ux * vz,
                              ux * vy - uy * vx)
                length = math.sqrt(nx * nx + ny * ny + nz * nz)
                if length > 0:
                    nx, ny, nz = nx / length, ny / length, nz / length
                handle.write(struct.pack("<12fH", nx, ny, nz,
                                         *a, *b, *c, 0))
        return len(self.triangles)


def hole_squares(holes, radii, width, height, pads):
    """Half-size of each hole's square, shrunk to avoid collisions."""
    halves = []
    for (hx, hy), radius in zip(holes, radii):
        half = radius * 1.4
        half = min(half, hx, width - hx, hy, height - hy)
        for (ox, oy), other_r in zip(holes, radii):
            if (ox, oy) == (hx, hy):
                continue
            gap = max(abs(ox - hx), abs(oy - hy)) / 2.0
            half = min(half, gap)
        for x0, y0, x1, y1 in pads:
            if x0 - half < hx < x1 + half and y0 - half < hy < y1 + half:
                if hx < x0:
                    half = min(half, x0 - hx)
                elif hx > x1:
                    half = min(half, hx - x1)
                if hy < y0:
                    half = min(half, y0 - hy)
                elif hy > y1:
                    half = min(half, hy - y1)
        halves.append(half)
    return halves


def build(layout: dict, floor_mm: float) -> tuple[Mesh, Mesh, dict]:
    """Return (floor_mesh, wall_mesh, stats).

    The split is at the playing surface: everything below is the plate, and
    everything above is walls and tag pads. Printing them as two bodies lets a
    slicer assign a different filament to each, and each body is independently
    closed so either can be printed alone.
    """
    width = layout["board_width"] * 1000.0
    height = layout["board_height"] * 1000.0
    wall_t = layout["wall_thickness"] * 1000.0
    wall_h = layout["wall_height"] * 1000.0
    holes = layout["holes"]
    radii = [r * 1000.0 for r in layout["hole_radii"]]
    holes_mm = [[x * 1000.0, y * 1000.0] for x, y in holes]

    top_of_wall = floor_mm + wall_h
    floor_mesh = Mesh()
    wall_mesh = Mesh()

    pad_rects = []
    for pad in layout.get("tag_pads", []):
        cx, cy = pad["centre_mm"]
        pw, ph = pad["pad_mm"]
        pad_rects.append((cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2))

    halves = hole_squares(holes_mm, radii, width, height, pad_rects)

    # ---- floor: everything except the hole squares and the tag pads --------
    #
    # A global grid does not work here. Compressing on every hole's edges
    # subdivides each hole's own square with other holes' coordinates, so a
    # "this rectangle is a hole square" test almost never matches and the holes
    # get emitted as solid floor. Instead, sweep in y bands and treat each hole
    # square and pad as an x-interval to skip; the squares then stay whole.
    exclusions = [(x0, y0, x1, y1) for x0, y0, x1, y1 in pad_rects]
    exclusions += [(hx - half, hy - half, hx + half, hy + half)
                   for (hx, hy), half in zip(holes_mm, halves)]

    y_edges = {0.0, height}
    for _, y0, _, y1 in exclusions:
        y_edges.update((y0, y1))
    y_edges = sorted(v for v in y_edges if -1e-9 <= v <= height + 1e-9)

    # Collapse band edges that differ by less than SNAP_MM. Without this, an
    # exclusion whose edge lands a hair off the board boundary produces a band
    # 5e-5 mm tall and a run of sliver triangles: not degenerate, so an
    # area test misses them, but junk geometry a strict validator will flag.
    # The residual mismatch lets a floor tile overlap a hole square by up to
    # SNAP_MM, which is harmless -- the export is a union of overlapping
    # solids by construction.
    snapped = []
    for value in y_edges:
        if not snapped or value - snapped[-1] > SNAP_MM:
            snapped.append(value)
        else:
            snapped[-1] = min(snapped[-1], value)
    y_edges = snapped

    plain = 0
    for j in range(len(y_edges) - 1):
        y0, y1 = y_edges[j], y_edges[j + 1]
        if y1 - y0 < SNAP_MM:
            continue
        mid_y = (y0 + y1) / 2

        spans = sorted(
            (ex0, ex1) for ex0, ey0, ex1, ey1 in exclusions
            if ey0 - 1e-6 <= mid_y <= ey1 + 1e-6
        )
        cursor = 0.0
        for span_start, span_end in spans:
            if span_start - cursor > SNAP_MM:
                floor_mesh.box(cursor, y0, 0.0, span_start, y1, floor_mm)
                plain += 1
            cursor = max(cursor, span_end)
        if width - cursor > SNAP_MM:
            floor_mesh.box(cursor, y0, 0.0, width, y1, floor_mm)
            plain += 1

    for (hx, hy), radius, half in zip(holes_mm, radii, halves):
        floor_mesh.annulus_prism(hx, hy, half, radius, 0.0, floor_mm)
    holed = len(holes_mm)
    skipped = len(pad_rects)

    # ---- walls -------------------------------------------------------------
    #
    # Clamped to the board. A wall centred half its thickness from the edge sat
    # exactly flush before rescaling; because positions scale and thickness does
    # not, its centreline moved inward and the wall now overhangs by the
    # difference. Small, but it would print as a lip outside the frame.
    # Walls whose centre lies inside a tag pad are skipped. The source layout
    # makes each pad solid by packing the corner cell with a comb of ~13
    # parallel walls about 2 mm apart, rather than emitting a block. Kept, they
    # fill the pocket straight back in. Walls sitting exactly on a pad boundary
    # are shared with the neighbouring cell and are retained.
    skipped_walls = 0

    def wall(x0, y0, x1, y1):
        nonlocal skipped_walls
        mid_x, mid_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        for px0, py0, px1, py1 in pad_rects:
            if px0 < mid_x < px1 and py0 < mid_y < py1:
                skipped_walls += 1
                return
        wall_mesh.box(max(0.0, x0), max(0.0, y0), floor_mm,
                      min(width, x1), min(height, y1), top_of_wall)

    for x_lo, x_hi, y in layout["walls_h"]:
        wall(x_lo * 1000.0, y * 1000.0 - wall_t / 2,
             x_hi * 1000.0, y * 1000.0 + wall_t / 2)
    for y_lo, y_hi, x in layout["walls_v"]:
        wall(x * 1000.0 - wall_t / 2, y_lo * 1000.0,
             x * 1000.0 + wall_t / 2, y_hi * 1000.0)

    # ---- tag pads, with a pocket for a drop-in tag -------------------------
    for pad, (x0, y0, x1, y1) in zip(layout.get("tag_pads", []), pad_rects):
        pad_top = pad.get("top_height_mm", top_of_wall)
        depth = pad.get("pocket_depth_mm", 3.0)
        pocket_w, pocket_h = pad.get("pocket_mm", [0.0, 0.0])
        floor_of_pocket = pad_top - depth
        cx, cy = pad["centre_mm"]

        # The pad spans both bodies, so it is cut at the playing surface: its
        # base prints in the floor colour, its raised part in the wall colour.
        floor_mesh.box(x0, y0, 0.0, x1, y1, floor_mm)
        if pocket_w <= 0 or pocket_h <= 0:
            wall_mesh.box(x0, y0, floor_mm, x1, y1, pad_top)
            continue
        wall_mesh.box(x0, y0, floor_mm, x1, y1, floor_of_pocket)
        px0, px1 = cx - pocket_w / 2, cx + pocket_w / 2
        py0, py1 = cy - pocket_h / 2, cy + pocket_h / 2
        # Frame of four rails around the recess.
        wall_mesh.box(x0, y0, floor_of_pocket, x1, py0, pad_top)
        wall_mesh.box(x0, py1, floor_of_pocket, x1, y1, pad_top)
        wall_mesh.box(x0, py0, floor_of_pocket, px0, py1, pad_top)
        wall_mesh.box(px1, py0, floor_of_pocket, x1, py1, pad_top)

    stats = {
        "width_mm": width, "height_mm": height,
        "floor_mm": floor_mm, "wall_top_mm": top_of_wall,
        "plain_rects": plain, "holed_rects": holed, "pad_rects": skipped,
        "holes": len(holes), "skipped_walls": skipped_walls,
        "min_hole_half_mm": min(halves) if halves else 0.0,
        "min_hole_radius_mm": min(radii) if radii else 0.0,
    }
    return floor_mesh, wall_mesh, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("layout")
    parser.add_argument("--out", default=None)
    parser.add_argument("--floor-mm", type=float, default=FLOOR_THICKNESS_MM)
    parser.add_argument("--split", action="store_true",
                        help="write floor and walls as two STLs for two-colour "
                             "printing, instead of one combined file")
    args = parser.parse_args()

    layout_path = Path(args.layout)
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    floor_mesh, wall_mesh, stats = build(layout, args.floor_mm)

    if args.split:
        stem = Path(args.out).with_suffix("") if args.out else layout_path.with_suffix("")
        floor_out = Path(f"{stem}_floor.stl")
        wall_out = Path(f"{stem}_walls.stl")
        floor_count = floor_mesh.write_stl(floor_out, name=f"{layout_path.stem}_floor")
        wall_count = wall_mesh.write_stl(wall_out, name=f"{layout_path.stem}_walls")
        print(f"wrote {floor_out}  ({floor_count} triangles, "
              f"{floor_out.stat().st_size/1024:.0f} kB)")
        print(f"wrote {wall_out}   ({wall_count} triangles, "
              f"{wall_out.stat().st_size/1024:.0f} kB)")
        print("  both share the same origin -- load them together and the "
              "slicer will line them up with no repositioning")
    else:
        combined = Mesh()
        combined.triangles = floor_mesh.triangles + wall_mesh.triangles
        out = Path(args.out) if args.out else layout_path.with_suffix(".stl")
        count = combined.write_stl(out, name=layout_path.stem)
        print(f"wrote {out}  ({count} triangles, {out.stat().st_size/1024:.0f} kB)")

    thin = stats["min_hole_half_mm"] - stats["min_hole_radius_mm"]
    print("  units      MILLIMETRES")
    print(f"  bounding   {stats['width_mm']:.2f} x {stats['height_mm']:.2f} x "
          f"{stats['wall_top_mm']:.2f} mm")
    print(f"  floor      {stats['floor_mm']:.2f} mm, walls to "
          f"{stats['wall_top_mm']:.2f} mm")
    print(f"  holes      {stats['holes']} cut through the floor")
    print(f"  floor tiles {stats['plain_rects']} plain, {stats['holed_rects']} holed, "
          f"{stats['pad_rects']} covered by tag pads")
    print(f"  walls      {stats['skipped_walls']} pad-fill walls dropped "
          f"(they would refill the tag pockets)")
    if thin < 1.0:
        print(f"  WARNING: thinnest ring of material around a hole is "
              f"{thin:.2f} mm; holes may be crowding each other or an edge")


if __name__ == "__main__":
    main()
