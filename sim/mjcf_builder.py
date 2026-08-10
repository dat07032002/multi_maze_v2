"""``maze_256x226.json`` -> MJCF.

The layout is the source of truth and this file is a translator, not a second
opinion. Three of its rules are copied verbatim from ``maze_design/export_stl.py``
so the simulated board and the printed one cannot drift apart:

* a wall run ``(lo, hi, fixed)`` becomes a box spanning exactly ``lo..hi`` with
  the thickness straddling ``fixed`` -- the run is the extent, not a centreline
  to be extended;
* every wall is clamped to the board, because rescaling moved centrelines
  inward while thickness stayed put and an unclamped wall overhangs the frame;
* a wall whose **midpoint lies strictly inside a tag pad** is dropped. The
  layout makes each corner pad solid by packing the cell with ~13 parallel wall
  runs about 2 mm apart rather than emitting a block, and those combs are
  re-emitted here as four solid pads. Walls sitting exactly *on* a pad boundary
  are shared with the neighbouring cell and are kept.

Two things the layout does not contain and this file must add:

* **the outer frame.** ``border_walls_printed`` is false -- the physical picture
  frame retains the ball, and the data confirms it: there is no full-width
  ``walls_h`` run and no ``walls_v`` at the left or right edge. Without a frame
  the ball simply leaves the world.
* **holes as actual gaps.** MuJoCo has no boolean subtraction, and a mesh with
  holes cut in it would be collided as its convex hull -- a solid slab. So the
  floor is decomposed into y-strips and each strip has the holes' x-intervals
  subtracted from it, exactly in x and quantised in y by
  ``sim.floor_rim_step``. Bands with identical intervals are merged, which
  collapses everything away from a hole back into a handful of large boxes.

Coordinates: the layout's origin is the lower-left board corner, matching the
vision stack. MuJoCo needs the tilt hinges through the board centre, so every
geom is emitted shifted by ``(-W/2, -H/2)``. Use :func:`layout_to_model` and
:func:`model_to_layout` rather than open-coding the shift.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_LAYOUT = REPO / "maze_design" / "maze_256x226.json"
DEFAULT_PARAMETERS = HERE / "parameters.json"

# Below this, an interval is a sliver rather than a feature. Mirrors SNAP_MM in
# export_stl.py, in metres.
SNAP_M = 1e-5


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------
def load_parameters(path: str | Path | None = None) -> dict:
    """Return ``{name: value}``, dropping the status/source metadata.

    The metadata is what makes ``parameters.json`` worth having -- it records
    which numbers were measured -- but the builder only needs the values.
    """
    raw = json.loads(Path(path or DEFAULT_PARAMETERS).read_text(encoding="utf-8"))
    return {k: v["value"] for k, v in raw.items()
            if isinstance(v, dict) and "value" in v}


def load_layout(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or DEFAULT_LAYOUT).read_text(encoding="utf-8"))


def flat_board_layout(layout: dict | None = None) -> dict:
    """The same board with no walls, no holes and no pads.

    Used by the M0 physics tests, where a bare plate is the only place
    ``a = (5/7) g sin(theta)`` can be checked against a rolling ball, and again
    by the M3 ball-on-plate task.
    """
    base = layout if layout is not None else load_layout()
    flat = dict(base)
    flat.update({
        "walls_h": [], "walls_v": [], "walls_angled": [],
        "holes": [], "hole_radii": [], "tag_pads": [],
        "start_planned": [base["board_width"] / 2.0, base["board_height"] / 2.0],
    })
    return flat


def layout_to_model(x: float, y: float, layout: dict) -> tuple[float, float]:
    """Lower-left board frame -> MuJoCo board frame (origin at board centre)."""
    return (x - layout["board_width"] / 2.0, y - layout["board_height"] / 2.0)


def model_to_layout(x: float, y: float, layout: dict) -> tuple[float, float]:
    return (x + layout["board_width"] / 2.0, y + layout["board_height"] / 2.0)


# --------------------------------------------------------------------------
# geometry decomposition
# --------------------------------------------------------------------------
def tag_pad_rects(layout: dict) -> list[tuple[float, float, float, float]]:
    """Corner pad footprints as ``(x0, y0, x1, y1)`` in metres, layout frame."""
    rects = []
    W, H = layout["board_width"], layout["board_height"]
    for pad in layout.get("tag_pads", []):
        cx, cy = (v / 1000.0 for v in pad["centre_mm"])
        pw, ph = (v / 1000.0 for v in pad["pad_mm"])
        rects.append((max(0.0, cx - pw / 2.0), max(0.0, cy - ph / 2.0),
                      min(W, cx + pw / 2.0), min(H, cy + ph / 2.0)))
    return rects


def _free_intervals(blocked: list[tuple[float, float]],
                    width: float) -> tuple[tuple[float, float], ...]:
    """``[0, width]`` minus the union of ``blocked``."""
    if not blocked:
        return ((0.0, width),)
    merged: list[list[float]] = []
    for lo, hi in sorted(blocked):
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    free, cursor = [], 0.0
    for lo, hi in merged:
        if lo - cursor > SNAP_M:
            free.append((cursor, lo))
        cursor = max(cursor, hi)
    if width - cursor > SNAP_M:
        free.append((cursor, width))
    return tuple(free)


def _band_edges(layout: dict, rim_step: float) -> list[float]:
    """Y positions at which the floor is split, chosen per hole.

    Uniform strips are the wrong tool here. A hole's half-width is
    ``sqrt(r^2 - dy^2)``, whose slope runs away near the top and bottom of the
    circle, so uniform strips give a coarse rim exactly where the circle turns
    and waste boxes across its middle -- measured on this layout, going from
    2 mm to 0.25 mm strips cost 6.8x the boxes and improved the worst rim step
    only from 4.50 mm to 1.92 mm.

    Placing the edges at equal steps in *half-width* instead bounds the error
    directly: consecutive edges differ by ``rim_step`` of radius, so no part of
    the cut deviates from the true circle by more than that.
    """
    H = layout["board_height"]
    edges = {0.0, H}
    for (_, hy), r in zip(layout["holes"], layout["hole_radii"]):
        n = max(2, int(math.ceil(r / rim_step)))
        for k in range(n + 1):
            half = r * (1.0 - k / n)
            dy = math.sqrt(max(0.0, r * r - half * half))
            edges.add(hy - dy)
            edges.add(hy + dy)

    clamped = sorted(min(max(e, 0.0), H) for e in edges)
    snapped: list[float] = []
    for value in clamped:
        if not snapped or value - snapped[-1] > SNAP_M:
            snapped.append(value)
    return snapped


def floor_rects(layout: dict, rim_step: float) -> list[tuple[float, float, float, float]]:
    """Floor as ``(x0, y0, x1, y1)`` boxes with the 15 holes cut out.

    A hole blocks the x-interval it covers at the point in the band *closest*
    to the hole centre, so a band is cut wherever any part of it falls inside
    the circle. That errs toward removing slightly too much floor, which is the
    safe direction: the alternative leaves slivers of floor poking into a hole
    for the ball to catch on.
    """
    W = layout["board_width"]
    holes = [(hx, hy, r) for (hx, hy), r
             in zip(layout["holes"], layout["hole_radii"])]

    rows: list[tuple[float, float, tuple[tuple[float, float], ...]]] = []
    edges = _band_edges(layout, rim_step)
    for y0, y1 in zip(edges, edges[1:]):
        if y1 - y0 <= SNAP_M:
            continue
        blocked = []
        for hx, hy, r in holes:
            dy = 0.0 if y0 <= hy <= y1 else min(abs(hy - y0), abs(hy - y1))
            if dy < r:
                half = math.sqrt(r * r - dy * dy)
                blocked.append((hx - half, hx + half))
        rows.append((y0, y1, _free_intervals(blocked, W)))

    # Merge vertically adjacent strips that expose the same intervals. Away from
    # any hole every strip is ((0, W),), so this collapses the bulk of the floor
    # into a few large boxes instead of one box per millimetre.
    rects: list[tuple[float, float, float, float]] = []
    if not rows:
        return rects
    run_y0, run_y1, run_iv = rows[0]
    for y0, y1, iv in rows[1:]:
        if iv == run_iv:
            run_y1 = y1
            continue
        rects.extend((x0, run_y0, x1, run_y1) for x0, x1 in run_iv)
        run_y0, run_y1, run_iv = y0, y1, iv
    rects.extend((x0, run_y0, x1, run_y1) for x0, x1 in run_iv)
    return rects


def wall_rects(layout: dict) -> list[tuple[float, float, float, float]]:
    """Wall runs as ``(x0, y0, x1, y1)``, pad combs dropped, clamped to board."""
    W, H = layout["board_width"], layout["board_height"]
    half_t = layout["wall_thickness"] / 2.0
    pads = tag_pad_rects(layout)

    def keep(x0: float, y0: float, x1: float, y1: float) -> bool:
        mid_x, mid_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        return not any(px0 < mid_x < px1 and py0 < mid_y < py1
                       for px0, py0, px1, py1 in pads)

    rects = []
    for x_lo, x_hi, y in layout["walls_h"]:
        box = (x_lo, y - half_t, x_hi, y + half_t)
        if keep(*box):
            rects.append((max(0.0, box[0]), max(0.0, box[1]),
                          min(W, box[2]), min(H, box[3])))
    for y_lo, y_hi, x in layout["walls_v"]:
        box = (x - half_t, y_lo, x + half_t, y_hi)
        if keep(*box):
            rects.append((max(0.0, box[0]), max(0.0, box[1]),
                          min(W, box[2]), min(H, box[3])))
    return rects


# --------------------------------------------------------------------------
# MJCF emission
# --------------------------------------------------------------------------
def _camera(name: str, eye, target, fovy: float) -> str:
    """A fixed camera at ``eye`` looking at ``target``.

    Emitted into the model rather than built at render time: a free camera
    aimed straight down has a degenerate up-vector, and getting that wrong
    silently produces a blank image rather than an error.
    """
    ex, ey, ez = (float(v) for v in eye)
    tx, ty, tz = (float(v) for v in target)
    fx, fy, fz = tx - ex, ty - ey, tz - ez
    norm = math.sqrt(fx * fx + fy * fy + fz * fz)
    fx, fy, fz = fx / norm, fy / norm, fz / norm

    # World up, swapped when the view is near-vertical and the two would be
    # parallel -- which is exactly the top-down case.
    up = (0.0, 1.0, 0.0) if abs(fz) > 0.999 else (0.0, 0.0, 1.0)
    rx = fy * up[2] - fz * up[1]
    ry = fz * up[0] - fx * up[2]
    rz = fx * up[1] - fy * up[0]
    rn = math.sqrt(rx * rx + ry * ry + rz * rz)
    rx, ry, rz = rx / rn, ry / rn, rz / rn
    ux, uy, uz = ry * fz - rz * fy, rz * fx - rx * fz, rx * fy - ry * fx
    return (f'    <camera name="{name}" mode="fixed" fovy="{fovy}" '
            f'pos="{ex:.6f} {ey:.6f} {ez:.6f}" '
            f'xyaxes="{rx:.6f} {ry:.6f} {rz:.6f} {ux:.6f} {uy:.6f} {uz:.6f}"/>\n')


def _box(name: str, rect: tuple[float, float, float, float],
         z0: float, z1: float, layout: dict, cls: str, rgba: str = "") -> str:
    x0, y0, x1, y1 = rect
    cx, cy = layout_to_model((x0 + x1) / 2.0, (y0 + y1) / 2.0, layout)
    sx, sy, sz = (x1 - x0) / 2.0, (y1 - y0) / 2.0, (z1 - z0) / 2.0
    colour = f' rgba="{rgba}"' if rgba else ""
    return (f'      <geom name="{name}" class="{cls}" type="box" '
            f'pos="{cx:.6f} {cy:.6f} {(z0 + z1) / 2.0:.6f}" '
            f'size="{sx:.6f} {sy:.6f} {sz:.6f}"{colour}/>\n')


def build_mjcf(layout: dict | None = None, params: dict | None = None) -> str:
    layout = layout if layout is not None else load_layout()
    params = params if params is not None else load_parameters()

    W, H = layout["board_width"], layout["board_height"]
    floor_t = params["maze.floor_thickness"]
    wall_h = params["maze.wall_height"]
    pad_top = params["maze.pad_top_height"]
    frame_h = params["frame.height"]
    ball_r = params["ball.radius"]
    ball_m = params["ball.mass"]
    rim_step = params["sim.floor_rim_step"]
    dt = params["sim.timestep"]
    sref = params["sim.solref"]
    simp = params["sim.solimp"]
    wall_damp = params["sim.wall_dampratio"]

    mu = params["ball.floor_sliding_friction"]
    tor = params["ball.torsional_friction_length"]
    roll = params["ball.rolling_friction_length"]

    roll_lim = params["actuator.safe_travel_roll"]
    pitch_lim = params["actuator.safe_travel_pitch"]

    sx, sy = layout["start_planned"]
    start_x, start_y = layout_to_model(sx, sy, layout)

    # Named views. The two close-ups target the tightest hole clearance on the
    # route (8.38 mm) and the start cell -- the two places the floor
    # decomposition and the corridor width have to be right.
    if layout["holes"]:
        tight_x, tight_y = layout_to_model(*layout["holes"][-1], layout)
    else:
        tight_x, tight_y = 0.0, 0.0  # flat board: nothing to close up on
    cameras = (
        _camera("top", (0.0, 0.0, 0.60), (0.0, 0.0, 0.0), 27)
        + _camera("iso", (0.26, -0.30, 0.26), (0.0, 0.0, 0.0), 34)
        + _camera("hole_closeup", (tight_x + 0.035, tight_y - 0.040, 0.042),
                  (tight_x, tight_y, 0.0), 42)
        + _camera("start_closeup", (start_x + 0.032, start_y - 0.038, 0.038),
                  (start_x, start_y, 0.0), 42)
    )

    out = [f'''<mujoco model="maze_256x226">
  <compiler angle="radian" inertiafromgeom="auto"/>
  <option timestep="{dt}" gravity="0 0 -9.81" integrator="implicitfast"
          solver="Newton" iterations="20" tolerance="1e-10"/>
  <visual>
    <global offwidth="1600" offheight="1200"/>
    <quality shadowsize="4096"/>
    <map znear="0.005" zfar="4"/>
  </visual>

  <default>
    <!-- condim 6 is not optional. At condim 3 MuJoCo applies no rolling
         resistance at all and the ball coasts forever: plausible on screen,
         completely wrong. The friction triple is (sliding, torsional, rolling)
         with the last two in length units. -->
    <default class="board">
      <geom condim="6" friction="{mu} {tor} {roll}"
            solref="{sref[0]} {sref[1]}" solimp="{simp[0]} {simp[1]} {simp[2]}"
            rgba="0.82 0.80 0.76 1"/>
    </default>
    <default class="wall">
      <geom condim="6" friction="{mu} {tor} {roll}"
            solref="{sref[0]} {wall_damp}" solimp="{simp[0]} {simp[1]} {simp[2]}"
            rgba="0.30 0.34 0.42 1"/>
    </default>
    <default class="frame">
      <geom condim="6" friction="{mu} {tor} {roll}"
            solref="{sref[0]} {wall_damp}" solimp="{simp[0]} {simp[1]} {simp[2]}"
            rgba="0.18 0.18 0.20 1"/>
    </default>
  </default>

  <worldbody>
    <light name="key" pos="0 -0.20 0.60" dir="0 0.25 -1" diffuse="0.85 0.85 0.85"/>
    <light name="fill" pos="0.25 0.25 0.45" dir="-0.4 -0.4 -1" diffuse="0.40 0.40 0.40"/>
{cameras}
    <!-- Catches a ball that has gone through a hole, so it stops falling and
         the episode can end on a settled state rather than a runaway body. -->
    <geom name="catch_tray" type="box" pos="0 0 -0.09" size="0.20 0.18 0.005"
          rgba="0.06 0.06 0.07 1" condim="3" friction="0.6 0.005 0.0001"/>

    <body name="tilt_x_frame" pos="0 0 0">
      <inertial pos="0 0 0" mass="0.05" diaginertia="1e-4 1e-4 1e-4"/>
      <joint name="tilt_x" type="hinge" axis="1 0 0"
             range="{roll_lim[0]:.6f} {roll_lim[1]:.6f}" damping="1.0" armature="0.01"/>
      <body name="board" pos="0 0 0">
        <inertial pos="0 0 0" mass="0.25" diaginertia="1e-3 1e-3 2e-3"/>
        <joint name="tilt_y" type="hinge" axis="0 1 0"
               range="{pitch_lim[0]:.6f} {pitch_lim[1]:.6f}" damping="1.0" armature="0.01"/>
''']

    # ---- floor, holes cut out ------------------------------------------
    floors = floor_rects(layout, rim_step)
    for i, rect in enumerate(floors):
        out.append(_box(f"floor_{i}", rect, -floor_t, 0.0, layout, "board"))

    # ---- walls ---------------------------------------------------------
    walls = wall_rects(layout)
    for i, rect in enumerate(walls):
        out.append(_box(f"wall_{i}", rect, 0.0, wall_h, layout, "wall"))

    # ---- tag pads, solid from the underside of the floor to the pad top --
    pads = tag_pad_rects(layout)
    for i, rect in enumerate(pads):
        out.append(_box(f"pad_{i}", rect, -floor_t, pad_top, layout, "wall",
                        rgba="0.22 0.24 0.30 1"))

    # ---- outer frame, absent from the layout ----------------------------
    ft = 0.005
    frame = {
        "frame_x0": (-ft, -ft, 0.0, H + ft),
        "frame_x1": (W, -ft, W + ft, H + ft),
        "frame_y0": (0.0, -ft, W, 0.0),
        "frame_y1": (0.0, H, W, H + ft),
    }
    for name, rect in frame.items():
        out.append(_box(name, rect, -floor_t, frame_h, layout, "frame"))

    out.append(f'''      </body>
    </body>

    <body name="ball" pos="{start_x:.6f} {start_y:.6f} {ball_r:.6f}">
      <freejoint name="ball_free"/>
      <geom name="ball" type="sphere" size="{ball_r:.6f}" mass="{ball_m:.6f}"
            condim="6" friction="{mu} {tor} {roll}"
            solref="{sref[0]} {sref[1]}" solimp="{simp[0]} {simp[1]} {simp[2]}"
            rgba="0.15 0.35 0.85 1"/>
    </body>
  </worldbody>

  <actuator>
    <position name="roll" joint="tilt_x" kp="80" kv="6" ctrlrange="{roll_lim[0]:.6f} {roll_lim[1]:.6f}"/>
    <position name="pitch" joint="tilt_y" kp="80" kv="6" ctrlrange="{pitch_lim[0]:.6f} {pitch_lim[1]:.6f}"/>
  </actuator>
</mujoco>
''')
    return "".join(out)


def build_model(layout: dict | None = None, params: dict | None = None):
    """Compile the MJCF. Imports mujoco lazily so geometry tests stay cheap."""
    import mujoco
    return mujoco.MjModel.from_xml_string(build_mjcf(layout, params))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layout", default=None)
    parser.add_argument("--parameters", default=None)
    parser.add_argument("--out", default=None, help="write the MJCF here")
    args = parser.parse_args()

    layout = load_layout(args.layout)
    params = load_parameters(args.parameters)
    xml = build_mjcf(layout, params)
    if args.out:
        Path(args.out).write_text(xml, encoding="utf-8")

    floors = floor_rects(layout, params["sim.floor_rim_step"])
    walls = wall_rects(layout)
    dropped = len(layout["walls_h"]) + len(layout["walls_v"]) - len(walls)
    print(f"floor boxes {len(floors)}   wall boxes {len(walls)} "
          f"({dropped} pad-comb runs dropped)   pads {len(tag_pad_rects(layout))}")
    model = build_model(layout, params)
    print(f"compiled: {model.ngeom} geoms, {model.nbody} bodies, {model.njnt} joints")


if __name__ == "__main__":
    main()
