"""Build the stratified held-out maze set for generalisation evaluation.

Every result this project has produced is scored on one maze -- the one the
policy also trained on. That was fine while the map was fixed. It is not, so
this writes a set of mazes the policy has never seen, in two difficulty tiers
reported separately.

**This reuses the pipeline that produced the shipped maze rather than a new
one.** ``generate_maze.build`` emits waypoints at raw cell centres, which take
90-degree corners and run straight through dodge holes: measured on generated
mazes, 13 route samples sit at -12.3 mm clearance against a 5.5 mm ball, and the
analytic baseline fell on 24 of 24. The shipped maze has none, because
``maze_design/generate_samples.py`` throws those waypoints away and replaces
them with ``plan_safe_route``. Skipping that step does not make a harder maze,
it makes an incomparable one.

So the flow here is the flow there: carve, plan, validate, screen, rescale.

**Why two tiers.** The shipped maze is 14 blocking holes and 1 dodge. Dodges are
the hard feature -- the ball must leave the centreline and commit to a side. A
held-out set carrying more of them measures "harder maze" and "different maze"
at once, and the two cannot be separated afterwards. The matched tier holds the
shipped profile and carries the headline number; the harder tier is a stress
test reported beside it, never averaged in.

**Why rescale rather than generate small.** ``maze_256x226.json`` is
``maze_final.json`` scaled by 256/259 and 226/229 -- identical structure, 76/24
walls, 15 holes, 118 waypoints. Absolute quantities do not scale: wall
thickness, wall height, ball radius and hole radii are the same in both. This
reproduces that exactly, so a held-out maze differs from the shipped one only in
its carve.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "maze_design"))

import generate_maze as gm  # noqa: E402  (needs the path above)

from control.baseline import PurePursuitBaseline  # noqa: E402
from sim.mjcf_builder import DEFAULT_LAYOUT, load_parameters  # noqa: E402
from sim.rollout import run_closed_loop  # noqa: E402
from sim.route import Route  # noqa: E402

#: The route planner lives in the sibling Multi-maze checkout, not in this
#: repo. ``generate_samples.py`` hardcodes ``~/Desktop/TAG/Multi-maze``, which
#: no longer exists; the tree is at ``~/Desktop/Multi-maze``. Override with
#: ``--planner-path`` rather than editing this.
DEFAULT_PLANNER_PATH = Path.home() / "Desktop" / "Multi-maze" / "tag_mujoco"

#: Coordinates scale with the board; these do not. Verified against the
#: maze_final -> maze_256x226 pair, where all four are byte-identical.
ABSOLUTE_KEYS = ("ball_radius", "wall_thickness", "wall_height")

#: Planned-route length band, as a fraction of the shipped route. Outside this
#: a time comparison between mazes stops meaning anything. Applied to the
#: planned length, which corner-cuts well below the raw cell-centre length.
LENGTH_BAND = 0.25

#: A blocking hole belongs in a decoy branch. Within this distance of the route
#: it chokes the route itself. 7 mm is ``generate_samples``' own bar, calibrated
#: against the shipped mazes' tightest hole at 6.2 mm.
MIN_BLOCK_GAP_M = 0.007

SAFETY_MARGIN_M = 0.0007
GRID_RESOLUTION_M = 0.001
GENERATOR_VERSION = "generate_maze+plan_safe_route@259x229+rescale"


def load_planner(path: Path):
    """Import the external planner, or explain precisely what is missing."""
    if not (Path(path) / "route_planner.py").exists():
        raise SystemExit(
            f"route_planner.py not found under {path}.\n"
            "The held-out set needs the same planner that produced the shipped "
            "maze; pass --planner-path to point at the Multi-maze checkout.")
    sys.path.insert(0, str(path))
    from route_planner import (PlannerConfig, plan_safe_route,  # noqa: E402
                               validate_route)
    return PlannerConfig, plan_safe_route, validate_route


def route_length(layout: dict) -> float:
    points = layout["waypoints"]
    return sum(math.dist(points[i], points[i + 1])
               for i in range(len(points) - 1))


def rescale_layout(layout: dict, width: float, height: float) -> dict:
    """Squeeze a 259x229 design onto the shipped board.

    ``walls_h`` are ``[x0, x1, y]`` and ``walls_v`` are ``[y0, y1, x]``, so the
    two axes scale by different factors and the ordering matters.
    """
    rx = width / layout["board_width"]
    ry = height / layout["board_height"]
    out = dict(layout)
    out["board_width"], out["board_height"] = width, height
    out["walls_h"] = [[x0 * rx, x1 * rx, y * ry]
                      for x0, x1, y in layout["walls_h"]]
    out["walls_v"] = [[y0 * ry, y1 * ry, x * rx]
                      for y0, y1, x in layout["walls_v"]]
    out["walls_angled"] = list(layout.get("walls_angled", []))
    out["holes"] = [[x * rx, y * ry] for x, y in layout["holes"]]
    out["hole_radii"] = list(layout["hole_radii"])
    out["waypoints"] = [[x * rx, y * ry] for x, y in layout["waypoints"]]
    for key in ("start_planned", "goal_planned"):
        if key in layout:
            out[key] = [layout[key][0] * rx, layout[key][1] * ry]
    return out


def role_counts(roles: dict) -> tuple[int, int]:
    values = list(roles.values())
    return (sum(1 for v in values if v == "block"),
            sum(1 for v in values if v == "dodge"))


def classify(roles: dict) -> str | None:
    """Matched holds the shipped 1-dodge profile; harder carries more."""
    blocks, dodges = role_counts(roles)
    holes = len(roles)
    if dodges == 1 and 13 <= holes <= 17:
        return "matched"
    if dodges >= 2:
        return "harder"
    return None


def min_block_gap(layout: dict, roles: dict, planned) -> float:
    """Closest approach of any *blocking* hole to the planned route.

    Dodges are exempt: intruding on the route is their whole job. Mirrors
    ``generate_samples.min_block_gap`` so the two screens agree.
    """
    points = np.asarray(planned, dtype=float)
    segments = list(zip(points[:-1], points[1:]))
    gap = np.inf
    for index, (hx, hy) in enumerate(layout["holes"]):
        if roles.get(str(index)) != "block":
            continue
        centre = np.array([hx, hy], dtype=float)
        for a, b in segments:
            span = b - a
            length_sq = float(span @ span)
            t = 0.0 if length_sq == 0.0 else float(
                np.clip((centre - a) @ span / length_sq, 0.0, 1.0))
            gap = min(gap, float(np.linalg.norm(centre - (a + t * span))))
    return gap


def baseline_reference(layout: dict, max_seconds: float, seed: int) -> dict:
    """Time the analytic controller, as a per-maze yardstick.

    Recorded rather than gated on: this is a centreline follower and a dodge
    exists to force the ball off the centreline, so a fall here is expected on
    the harder tier and is not evidence the maze is unfair.
    """
    params = load_parameters()
    route = Route(layout, params)
    controller = PurePursuitBaseline(route, params["actuator.max_tilt"])
    result = run_closed_loop(controller, layout=layout, params=params,
                             seed=seed, max_seconds=max_seconds)
    return {
        "reached_goal": bool(result.reached_goal),
        "fell": bool(result.fell),
        "completion": float(
            route.project(result.track[-1])[0] / route.length),
        "seconds": result.steps / 20.0,
    }


def build_candidate(seed: int, shipped: dict, planner) -> dict | None:
    """Carve, plan, validate and screen one seed. ``None`` if it does not pass."""
    PlannerConfig, plan_safe_route, validate_route = planner
    config = PlannerConfig(safety_margin_m=SAFETY_MARGIN_M,
                           grid_resolution_m=GRID_RESOLUTION_M)
    try:
        layout, roles, route = gm.build(seed)
    except Exception:
        return None                       # a carve that placed no holes
    tier = classify(roles)
    if tier is None:
        return None
    try:
        planned = plan_safe_route(layout, config)
        result = validate_route(layout, planned, config)
    except Exception:
        return None                       # unsolvable at this margin
    if not result.passed:
        return None
    if min_block_gap(layout, roles, planned) < MIN_BLOCK_GAP_M:
        return None

    # The planned route replaces the cell-centre waypoints. This is the step
    # whose absence made the first attempt at this set incomparable.
    layout["waypoints"] = [list(map(float, point)) for point in planned]
    layout["start_planned"] = layout["waypoints"][0]
    layout["goal_planned"] = layout["waypoints"][-1]
    layout = rescale_layout(
        layout, shipped["board_width"], shipped["board_height"])

    length = route_length(layout)
    shipped_length = route_length(shipped)
    if not (shipped_length * (1 - LENGTH_BAND) <= length
            <= shipped_length * (1 + LENGTH_BAND)):
        return None

    blocks, dodges = role_counts(roles)
    return {
        "seed": seed, "tier": tier, "layout": layout, "roles": roles,
        "route_length_m": length, "route_points": len(planned),
        "blocks": blocks, "dodges": dodges, "holes": len(roles),
        "planner_min_clearance_m": float(result.minimum_clearance_m),
    }


def _work(task):
    seed, shipped, planner_path, max_seconds, gate_seed = task
    planner = load_planner(planner_path)
    candidate = build_candidate(seed, shipped, planner)
    if candidate is None:
        return None
    candidate["baseline"] = baseline_reference(
        candidate["layout"], max_seconds, gate_seed)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-tier", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=600)
    parser.add_argument("--gate-seed", type=int, default=90_000)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument("--planner-path", type=Path,
                        default=DEFAULT_PLANNER_PATH)
    parser.add_argument("--workers", type=int,
                        default=min(16, os.cpu_count() or 1))
    parser.add_argument("--out", default="artifacts/holdout_mazes")
    args = parser.parse_args()

    load_planner(args.planner_path)       # fail early, in the parent
    shipped = json.loads(Path(DEFAULT_LAYOUT).read_text(encoding="utf-8"))
    shipped_roles = json.loads(
        (Path(DEFAULT_LAYOUT).parent / "maze_256x226_roles.json")
        .read_text(encoding="utf-8"))

    print(f"shipped route {route_length(shipped) * 1000:.0f} mm, "
          f"profile {role_counts(shipped_roles)[0]} block / "
          f"{role_counts(shipped_roles)[1]} dodge")
    print(f"planning seeds {args.seed_start}-{args.seed_end} "
          f"on {args.workers} workers")

    found: dict[str, list[dict]] = {"matched": [], "harder": []}
    tasks = [(seed, shipped, args.planner_path, args.max_seconds,
              args.gate_seed)
             for seed in range(args.seed_start, args.seed_end)]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_work, task): task[0] for task in tasks}
        for future in as_completed(futures):
            candidate = future.result()
            if candidate is None:
                continue
            tier = candidate["tier"]
            if len(found[tier]) >= args.per_tier:
                continue
            found[tier].append(candidate)
            baseline = candidate["baseline"]
            print(f"  seed {candidate['seed']:>4} {tier:>7}: "
                  f"{candidate['route_length_m'] * 1000:>6.0f} mm  "
                  f"{candidate['blocks']}b/{candidate['dodges']}d  "
                  f"clear {candidate['planner_min_clearance_m'] * 1000:>4.2f} mm  "
                  f"baseline "
                  f"{'goal' if baseline['reached_goal'] else 'no':>4} "
                  f"{baseline['completion']:>6.1%}", flush=True)

    out = Path(args.out)
    manifest = {
        "schema": "holdout_mazes_v2",
        "generator": GENERATOR_VERSION,
        "planner": {"path": str(args.planner_path),
                    "safety_margin_m": SAFETY_MARGIN_M,
                    "grid_resolution_m": GRID_RESOLUTION_M},
        "shipped_layout": str(Path(DEFAULT_LAYOUT)),
        "shipped_route_length_m": route_length(shipped),
        "shipped_profile": dict(zip(("blocks", "dodges"),
                                    role_counts(shipped_roles))),
        "length_band": LENGTH_BAND,
        "min_block_gap_m": MIN_BLOCK_GAP_M,
        "note": ("Matched tier holds 12-13 blocking holes against the shipped "
                 "14: the generator's block count tops out at 13, so the "
                 "profiles are close but not identical."),
        "mazes": [],
    }

    for tier, rows in found.items():
        directory = out / tier
        directory.mkdir(parents=True, exist_ok=True)
        for row in sorted(rows, key=lambda r: r["seed"]):
            path = directory / f"maze_{row['seed']}.json"
            layout = dict(row["layout"])
            layout["_holdout_seed"] = row["seed"]
            layout["_holdout_tier"] = tier
            path.write_text(json.dumps(layout, indent=1), encoding="utf-8")
            (directory / f"maze_{row['seed']}_roles.json").write_text(
                json.dumps(row["roles"], indent=1), encoding="utf-8")
            manifest["mazes"].append({
                key: row[key] for key in (
                    "seed", "tier", "route_length_m", "route_points", "blocks",
                    "dodges", "holes", "planner_min_clearance_m", "baseline")
            # POSIX separators always: the set is generated on Windows and
            # consumed on the Linux training box, and a backslash here is not a
            # path separator there.
            } | {"path": path.relative_to(out).as_posix()})

    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(len(rows) for rows in found.values())
    print(f"\nwrote {total} mazes to {out} "
          f"({len(found['matched'])} matched, {len(found['harder'])} harder)")


if __name__ == "__main__":
    main()
