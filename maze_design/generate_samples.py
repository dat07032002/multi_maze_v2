#!/usr/bin/env python3
"""Generate several maze candidates that pass a stricter clearance bar.

Every candidate is validated by actually planning a route through it with
Multi-maze's planner at the requested margin. A design that only looks right in
the render is not accepted: two earlier versions of this maze were unsolvable
for a 12 mm ball while rendering perfectly.

Seeds are scored for edge endpoints, central coverage, and enough off-route
cells to hold decoys, then tried in that order until enough pass.

Usage:
    python3 generate_samples.py [--margin 0.0007] [--want 6]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

MULTI_MAZE = os.path.expanduser("~/Desktop/TAG/Multi-maze")
sys.path.insert(0, os.path.join(MULTI_MAZE, "tag_mujoco"))

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from generate_maze import MIDDLE, build, carve, longest_path  # noqa: E402
from route_planner import (PlannerConfig, plan_safe_route,      # noqa: E402
                           validate_route)


def min_block_gap(layout, roles, route):
    """Smallest gap between a blocking hole's rim and the planned route."""
    import numpy as np
    way = np.asarray(route, dtype=float)
    gaps = []
    for index, (hole, radius) in enumerate(
            zip(layout["holes"], layout["hole_radii"])):
        if roles.get(str(index)) != "block":
            continue
        point = np.asarray(hole, dtype=float)
        best = min(_point_to_segment(point, way[k], way[k + 1])
                   for k in range(len(way) - 1))
        gaps.append(best - radius - layout["ball_radius"])
    return min(gaps) if gaps else float("inf")


def _point_to_segment(p, a, b):
    import numpy as np
    d = b - a
    t = float(np.clip(np.dot(p - a, d) / max(float(np.dot(d, d)), 1e-12), 0, 1))
    return float(np.linalg.norm(p - (a + t * d)))


def rank_seeds(trials):
    ranked = []
    for seed in range(trials):
        route = longest_path(carve(random.Random(seed)))
        if not route:
            continue          # endpoints not connected in this spanning tree
        if not 22 <= len(route) <= 55:
            continue
        # Favour central coverage over raw length. A long snaking route leaves
        # no spare cells for dodge chambers, and dodges -- holes actually on
        # the path -- are what makes the maze interesting to drive.
        ranked.append((len(set(route) & MIDDLE) * 3, seed))
    ranked.sort(reverse=True)
    return [seed for _, seed in ranked]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin", type=float, default=0.0007,
                        help="required minimum ball clearance, metres")
    parser.add_argument("--want", type=int, default=6)
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--min-dodges", type=int, default=2)
    args = parser.parse_args()

    config = PlannerConfig(safety_margin_m=args.margin,
                           grid_resolution_m=0.001)
    print(f"looking for {args.want} designs that plan at "
          f"{args.margin*1000:.2f} mm clearance\n")

    samples = []
    for seed in rank_seeds(args.trials):
        if len(samples) >= args.want:
            break
        layout, roles, route = build(seed)
        try:
            planned = plan_safe_route(layout, config)
            result = validate_route(layout, planned, config)
        except Exception:
            continue                     # unsolvable at this margin
        if not result.passed:
            continue

        # A blocking hole belongs in a decoy branch, punishing a wrong turn.
        # If it sits within a few mm of the correct route it instead chokes the
        # route itself -- sample 4's hole #12 was 1.00 mm off it, and the
        # play-test drove into it. Dodges are exempt: intruding is their job.
        #
        # 7 mm matches the shipped mazes, whose tightest hole over 360 samples
        # sits 6.2 mm off the route (5th percentile 7.4 mm). The earlier 4 mm
        # bar was below the expert's own 4.4 mm median cross-track error, so it
        # wandered into blocking holes as a matter of course.
        if min_block_gap(layout, roles, planned) < 0.007:
            continue

        # Dodges are the point of the design, and a three-corridor pocket needs
        # both perpendicular neighbours free, so most seeds yield none. Insist
        # on them rather than silently shipping a maze with only decoys.
        if sum(1 for v in roles.values() if v == "dodge") < args.min_dodges:
            continue

        layout["waypoints"] = [list(map(float, p)) for p in planned]
        layout["start_planned"] = layout["waypoints"][0]
        layout["goal_planned"] = layout["waypoints"][-1]
        layout["seed"] = seed
        layout["min_clearance_m"] = float(result.minimum_clearance_m)

        index = len(samples) + 1
        json.dump(layout, open(f"{HERE}/sample_{index}.json", "w"), indent=1)
        json.dump(roles, open(f"{HERE}/sample_{index}_roles.json", "w"), indent=1)
        blocking = sum(1 for v in roles.values() if v == "block")
        samples.append({
            "index": index, "seed": seed,
            "route_mm": result.route_length_m * 1000,
            "clearance_mm": result.minimum_clearance_m * 1000,
            "holes": len(layout["holes"]),
            "blocking": blocking, "dodge": len(roles) - blocking,
            "cells": len(route),
            "middle": len(set(route) & MIDDLE),
        })
        print(f"  sample {index}: seed {seed:3d}  route {samples[-1]['route_mm']:6.0f} mm  "
              f"clearance {samples[-1]['clearance_mm']:.2f} mm  "
              f"holes {samples[-1]['holes']:2d} "
              f"({blocking}b/{len(roles)-blocking}d)  "
              f"middle {samples[-1]['middle']}/25", flush=True)

    json.dump(samples, open(f"{HERE}/samples_summary.json", "w"), indent=1)
    print(f"\n{len(samples)} samples written to {HERE}/sample_*.json")


if __name__ == "__main__":
    main()
