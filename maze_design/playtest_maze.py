#!/usr/bin/env python3
"""Play-test a maze in physics: can the marble actually get through?

Geometric clearance says a route exists for a ball of a given radius. It does
not say the ball can be driven along it -- a corridor can be wide enough to
contain the marble while being too tight to steer through at speed, and a hole
can be geometrically dodgeable but practically unavoidable once the ball is
rolling.

This runs Multi-maze's own route-following expert over many episodes and
reports how often it finishes, where it falls, and how much room it actually
used. That is the difference between "a path exists" and "the marble moves
comfortably".

Usage:
    python3 playtest_maze.py sample_4.json [--episodes 30]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

import numpy as np

MULTI_MAZE = os.path.expanduser("~/Desktop/TAG/Multi-maze")
sys.path.insert(0, MULTI_MAZE)
sys.path.insert(0, os.path.join(MULTI_MAZE, "tag_mujoco"))

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("layout", nargs="?", default=f"{HERE}/sample_4.json")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--margin", type=float, default=0.0007)
    args = parser.parse_args()

    from tag_mujoco.tag_env import TagMazeEnv
    from tag_mujoco.expert_controller import RouteExpertController

    env = TagMazeEnv(
        layout_paths=args.layout,
        planner_safety_margin_m=args.margin,
        planner_grid_resolution_m=0.001,
    )
    controller = RouteExpertController()
    layout_json = json.load(open(args.layout))
    holes = layout_json["holes"]
    roles_path = args.layout.replace(".json", "_roles.json")
    roles = json.load(open(roles_path)) if os.path.exists(roles_path) else {}

    finished = 0
    outcomes = collections.Counter()
    completions, clearances, steps_taken = [], [], []
    fall_points = []

    for episode in range(args.episodes):
        env.reset(seed=episode)
        info = {}
        for step in range(args.max_steps):
            action = controller.action(env.task)
            _, _, terminated, truncated, info = env.step(action)
            clearance = info.get("min_clearance_m")
            if clearance is not None and np.isfinite(clearance):
                clearances.append(float(clearance))
            if terminated or truncated:
                break

        completion = float(info.get("route_completion", 0.0) or 0.0)
        completions.append(completion)
        steps_taken.append(step + 1)
        if info.get("success"):
            outcomes["success"] += 1
            finished += 1
        elif info.get("fall_cost"):
            # The env reports a fall as `fall_cost`, not a boolean. Getting this
            # key wrong once made every fall look like a stall, which pointed
            # the blame at the controller instead of at the hole it drove into.
            outcomes["fell in a hole"] += 1
            ball = np.asarray(env.task.model.sim.ball_board_position()[:2])
            near = np.linalg.norm(np.asarray(holes) - ball, axis=1) * 1000
            index = int(np.argmin(near))
            fall_points.append((index, roles.get(str(index), "?"),
                                tuple(np.round(ball * 1000))))
        elif completion > 0.95:
            outcomes["reached the end"] += 1
            finished += 1
        else:
            outcomes["stalled"] += 1

    print(f"layout      : {os.path.basename(args.layout)}")
    print(f"episodes    : {args.episodes}\n")
    for name, count in outcomes.most_common():
        print(f"  {name:16} {count:3d}  ({count/args.episodes:.0%})")
    print()
    print(f"route completion : mean {np.mean(completions):.1%}  "
          f"median {np.median(completions):.1%}  best {max(completions):.1%}")
    print(f"steps            : median {np.median(steps_taken):.0f}")
    if clearances:
        print(f"ball clearance   : min {min(clearances)*1000:+.2f} mm  "
              f"5th pct {np.percentile(clearances, 5)*1000:.2f} mm  "
              f"median {np.median(clearances)*1000:.2f} mm")
    if fall_points:
        tally = collections.Counter((i, r) for i, r, _ in fall_points)
        print("\nwhere it fell:")
        for (index, role), count in tally.most_common(5):
            print(f"  hole #{index:2d} ({role:5s})  x{count}")

    rate = finished / args.episodes
    print()
    if rate >= 0.8:
        print(f"VERDICT: the marble moves through comfortably ({rate:.0%} finish)")
    elif rate >= 0.4:
        print(f"VERDICT: passable but demanding ({rate:.0%} finish) -- the expert "
              "is not a trained policy, so some failures are expected")
    else:
        print(f"VERDICT: only {rate:.0%} finish. Either the maze is too tight or "
              "the expert cannot drive it; check the fall locations above.")
    env.close()


if __name__ == "__main__":
    main()
