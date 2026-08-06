#!/usr/bin/env python3
"""Rescale a maze layout to a new board size, preserving the map exactly.

Regenerating from the seed is not an option here: `maze_final.json` was produced
by a pipeline that is not in this repository (`generate_maze.py` emits
`maze_v1.json`, with 23 horizontal wall runs against the final's 76), so a
re-run would yield a different maze. Scaling the existing layout keeps the map
byte-for-byte identical in topology.

What scales and what does not is a physical distinction, not a mathematical one:

* **Positions scale.** Wall endpoints, hole centres, waypoints, and tag pads are
  all fractions of the board, so they move with it.
* **Thicknesses do not.** Wall thickness, wall height, hole radius, and ball
  radius are set by the printer, the marble, and the physics -- shrinking the
  board by 3 mm does not make the marble smaller. Scaling them non-uniformly
  would also turn the circular holes into ellipses.

The consequence is that corridors get slightly narrower: the pitch shrinks but
the 3 mm walls do not, so all of the loss comes out of the clear span.

Tag pads are **solid** by default and the tag is glued to the top face, which is
how the board is actually built. That fixes where the tag face is, and
`calib/board_tags.json` has to agree: pad top 8 mm above the playing surface
plus a 3 mm tag puts the face at z = 0.011 m.

`--pocket` insets a recess instead, so a tag drops in flush at z = 0.008 m. It
is not the default because the pad is one grid cell wide (23.27 mm) and a pocket
big enough for a 23 mm tag leaves 0.14 mm of frame -- which, since the corner
pads sit on the board edge, opens the pocket to the outside entirely.

Usage:
    python3 maze_design/rescale_maze.py --width 256 --height 226
    python3 maze_design/rescale_maze.py --width 256 --height 226 --pocket
    python3 maze_design/rescale_maze.py --width 256 --height 226 --pocket-mm 20
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rescale(layout: dict, new_width_m: float, new_height_m: float) -> dict:
    sx = new_width_m / layout["board_width"]
    sy = new_height_m / layout["board_height"]

    out = dict(layout)
    out["board_width"] = new_width_m
    out["board_height"] = new_height_m

    # walls_h entries are [x_lo, x_hi, y]; walls_v are [y_lo, y_hi, x].
    out["walls_h"] = [[a * sx, b * sx, c * sy] for a, b, c in layout["walls_h"]]
    out["walls_v"] = [[a * sy, b * sy, c * sx] for a, b, c in layout["walls_v"]]
    out["walls_angled"] = []

    out["holes"] = [[x * sx, y * sy] for x, y in layout["holes"]]
    out["waypoints"] = [[x * sx, y * sy] for x, y in layout["waypoints"]]
    out["start_planned"] = out["waypoints"][0]
    out["goal_planned"] = out["waypoints"][-1]

    # Deliberately unscaled: printer- and marble-defined, not board-defined.
    for key in ("ball_radius", "wall_thickness", "wall_height"):
        out[key] = layout[key]
    out["hole_radii"] = list(layout["hole_radii"])

    # Computed against the old geometry by a validator that is not in this
    # repository. Carrying it forward would present a stale number as current.
    out.pop("min_clearance_m", None)

    out["rescaled_from"] = {
        "board_width": layout["board_width"],
        "board_height": layout["board_height"],
        "scale_x": sx,
        "scale_y": sy,
        "note": (
            "Positions scaled; wall thickness, wall height, hole radius and "
            "ball radius left at their physical values."
        ),
    }
    return out


def tag_pads(layout: dict, cols: int, rows: int, tag_mm: float,
             clearance_mm: float, thickness_mm: float,
             wall_top_mm: float, pocket_mm: float | None = None) -> list[dict]:
    """Corner-cell pads carrying an inset pocket for a drop-in tag.

    The pad is one grid cell, so the frame left around the pocket is
    (pitch - pocket) / 2 per side. Corner pads sit on the board edge, so an
    x-frame that goes to zero does not merely thin the wall -- it opens the
    pocket to the outside of the board and the tag is no longer retained.
    """
    pitch_x = layout["board_width"] * 1000.0 / cols
    pitch_y = layout["board_height"] * 1000.0 / rows
    pocket = pocket_mm if pocket_mm is not None else tag_mm + 2.0 * clearance_mm

    pads = []
    for col, row in ((0, rows - 1), (cols - 1, rows - 1), (0, 0), (cols - 1, 0)):
        centre_x = (col + 0.5) * pitch_x
        centre_y = (row + 0.5) * pitch_y
        pad = {
            "cell": [col, row],
            "centre_mm": [round(centre_x, 4), round(centre_y, 4)],
            "pad_mm": [round(pitch_x, 4), round(pitch_y, 4)],
            "top_height_mm": wall_top_mm,
        }
        if pocket > 0:
            pad.update({
                "pocket_mm": [round(pocket, 4), round(pocket, 4)],
                "pocket_depth_mm": thickness_mm,
                "pocket_floor_height_mm": round(wall_top_mm - thickness_mm, 4),
                "mount": "insert",
            })
        else:
            # Solid pad. The tag is glued to the top face, so it stands proud of
            # the pad by its own thickness -- which is where the tag plane is
            # for calib/board_tags.json.
            pad.update({
                "mount": "glue",
                "tag_face_height_mm": round(wall_top_mm + thickness_mm, 4),
            })
        pads.append(pad)
    return pads


def hole_clearance_mm(layout: dict) -> float:
    """Smallest gap between the route and any HOLE edge, minus the ball.

    Holes only -- this does not measure wall clearance, which is what the
    layout's old `min_clearance_m` recorded. Reported for information; the
    caller asked not to gate on route validation.
    """
    ball = layout["ball_radius"]
    best = math.inf
    for (hx, hy), radius in zip(layout["holes"], layout["hole_radii"]):
        for x, y in layout["waypoints"]:
            best = min(best, math.hypot(x - hx, y - hy) - radius - ball)
    return best * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=str(HERE / "maze_final.json"))
    parser.add_argument("--roles", default=str(HERE / "maze_final_roles.json"))
    parser.add_argument("--width", type=float, required=True, help="new board width, mm")
    parser.add_argument("--height", type=float, required=True, help="new board height, mm")
    parser.add_argument("--out", default=None,
                        help="output basename (default: maze_<W>x<H>)")
    parser.add_argument("--cols", type=int, default=11)
    parser.add_argument("--rows", type=int, default=9)
    parser.add_argument("--tag-mm", type=float, default=18.0)
    parser.add_argument("--tag-clearance-mm", type=float, default=0.25,
                        help="per-side gap around the tag in its pocket")
    parser.add_argument("--tag-thickness-mm", type=float, default=3.0)
    parser.add_argument("--pocket-mm", type=float, default=None,
                        help="set the pocket opening directly, overriding "
                             "tag size plus clearance")
    parser.add_argument("--pocket", action="store_true",
                        help="recess a pocket sized tag + 2x clearance "
                             "(default: solid pads, tag glued on top)")
    args = parser.parse_args()

    layout = json.loads(Path(args.source).read_text(encoding="utf-8"))
    out = rescale(layout, args.width / 1000.0, args.height / 1000.0)

    wall_top_mm = (layout["wall_height"] + 0.003) * 1000.0  # floor + wall
    out["tag_size_mm"] = args.tag_mm
    out["tag_pads"] = tag_pads(out, args.cols, args.rows, args.tag_mm,
                               args.tag_clearance_mm, args.tag_thickness_mm,
                               wall_top_mm,
                               args.pocket_mm if args.pocket_mm is not None
                               else (None if args.pocket else 0.0))

    basename = args.out or f"maze_{args.width:g}x{args.height:g}"
    layout_path = HERE / f"{basename}.json"
    layout_path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    roles_source = Path(args.roles)
    if roles_source.is_file():
        roles_path = HERE / f"{basename}_roles.json"
        roles_path.write_text(roles_source.read_text(encoding="utf-8"),
                              encoding="utf-8")
        print(f"roles  -> {roles_path}")

    pitch_x = args.width / args.cols
    pitch_y = args.height / args.rows
    wall_mm = out["wall_thickness"] * 1000.0
    old_clear = hole_clearance_mm(layout)
    new_clear = hole_clearance_mm(out)

    print(f"layout -> {layout_path}")
    print(f"\nboard      {layout['board_width']*1000:.0f} x "
          f"{layout['board_height']*1000:.0f}  ->  {args.width:g} x {args.height:g} mm")
    print(f"scale      x {out['rescaled_from']['scale_x']:.6f}  "
          f"y {out['rescaled_from']['scale_y']:.6f}")
    print(f"pitch      {pitch_x:.3f} x {pitch_y:.3f} mm")
    print(f"corridor   {pitch_x - wall_mm:.3f} x {pitch_y - wall_mm:.3f} mm "
          f"(ball {out['ball_radius']*2000:.0f} mm)")
    print(f"hole clr   {old_clear:.3f}  ->  {new_clear:.3f} mm  "
          f"({new_clear - old_clear:+.3f})")
    print(f"holes      {len(out['holes'])}   waypoints {len(out['waypoints'])}")
    pad = out["tag_pads"][0]
    if pad["mount"] == "glue":
        print(f"tag pads   SOLID {pad['pad_mm'][0]:.2f} x {pad['pad_mm'][1]:.2f} mm, "
              f"top at {pad['top_height_mm']:.1f} mm")
        print(f"           tag glued on top -> face at "
              f"{pad['tag_face_height_mm']:.1f} mm "
              f"({pad['tag_face_height_mm'] - 3.0:.1f} mm above the playing surface)")
        return
    frame_x = (pad["pad_mm"][0] - pad["pocket_mm"][0]) / 2.0
    frame_y = (pad["pad_mm"][1] - pad["pocket_mm"][1]) / 2.0
    print(f"tag pocket {pad['pocket_mm'][0]:.2f} mm square x "
          f"{pad['pocket_depth_mm']:.1f} mm deep, floor at "
          f"{pad['pocket_floor_height_mm']:.1f} mm, pad top {pad['top_height_mm']:.1f} mm")
    print(f"frame      {frame_x:.2f} mm on x sides, {frame_y:.2f} mm on y sides")
    if frame_x < 0.8 or frame_y < 0.8:
        print(f"  WARNING: a frame under ~0.8 mm will not print as a wall. The "
              f"corner pads sit on the board edge, so the pocket opens to the "
              f"outside and the tag is not retained. Largest pocket leaving a "
              f"1.6 mm frame: {min(pad['pad_mm']) - 3.2:.2f} mm.")


if __name__ == "__main__":
    main()
