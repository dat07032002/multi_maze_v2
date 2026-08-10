"""Score a policy on mazes it has never seen.

Every number this project has produced comes from ten seeds of one maze -- the
maze the policy also trained on. The map changes at deployment, so that figure
answers a question nobody is asking. This answers the one that matters: what
happens on a different carve.

Reads the set written by ``tools.generate_holdout_mazes`` and sweeps
maze x physics scale x episode seed. Two rules are baked in rather than left to
the caller, because both have already cost a day of work here:

**Thirty episodes minimum per cell.** At ten, success rate moves in ten-point
steps and cannot resolve anything smaller. The v1 fine-tune's evaluation trace
read 80/80/80/90/70/80/90/90/90/90 -- noise around a flat line, and unreadable
at that sample size. It also revised the headline BC figure from 90% to 80%
once measured properly.

**The tiers are never averaged together.** ``matched`` holds the shipped
maze's 1-dodge profile and carries the headline; ``harder`` carries two or more
dodges and is a stress test. Merging them would report "the maze got harder" as
if it were "the policy does not generalise".

The shipped maze runs as a control. If it does not score near its known ~80%,
the harness is wrong and no other column should be believed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path

import numpy as np

from sim.maze_env import MazeEnv
from sim.mjcf_builder import DEFAULT_LAYOUT
from sim.randomization import (FULL_STRESS_SCALE, Randomizer,
                               REALISTIC_DR_SCALE, TEACHER_MAX_DR_SCALE)

DEFAULT_SCALES = (REALISTIC_DR_SCALE, TEACHER_MAX_DR_SCALE, FULL_STRESS_SCALE)

#: Disjoint from the ``seed + 20_000`` block ``FullMazeEvalCallback`` selects
#: ``best`` on. Reusing those would read checkpoint-selection noise back as a
#: result.
DEFAULT_SEED = 700_000

_MODEL = None
_KIND = None


def _initialize_worker(policy_path: str) -> None:
    """Load once per worker; the policy is the expensive part of a task."""
    global _MODEL, _KIND
    suffix = Path(policy_path).suffix
    if suffix == ".pt":
        from control.imitation_policy import load_policy
        _MODEL, _KIND = load_policy(policy_path, device="cpu"), "bc"
    elif suffix == ".zip":
        from stable_baselines3 import SAC
        _MODEL, _KIND = SAC.load(policy_path, device="cpu"), "sac"
    else:
        raise ValueError(
            f"unknown policy type {suffix!r}; expected .pt (BC) or .zip (SAC)")


def _act(observation: np.ndarray) -> np.ndarray:
    """One interface over both checkpoint kinds.

    The headline comparison spans a BC ``.pt`` and SAC ``.zip`` checkpoints, so
    a single command has to score both or the numbers are not comparable.
    """
    if _KIND == "bc":
        return np.asarray(_MODEL.predict(observation), dtype=np.float64)
    action, _ = _MODEL.predict(observation, deterministic=True)
    return np.asarray(action, dtype=np.float64)


def _run(task: dict) -> dict:
    scale = task["dr_scale"]
    env = MazeEnv(
        layout=task["layout"], max_seconds=task["max_seconds"],
        randomizer=Randomizer(scale=scale, enabled=scale > 0.0),
        start_fraction=0.0, seed=task["seed"])
    observation, _ = env.reset(seed=task["seed"])
    total = 0.0
    try:
        while True:
            observation, reward, terminated, truncated, info = env.step(
                _act(observation))
            total += reward
            if terminated or truncated:
                break
    finally:
        env.close()
    return {
        "maze": task["maze"], "tier": task["tier"], "dr_scale": scale,
        "seed": task["seed"], "outcome": info["outcome"],
        "completion": float(info["route_completion"]),
        "mean_cross_track_m": float(info["mean_cross_track"]),
        "reward": total, "seconds": info["steps"] / env.control_hz,
    }


def aggregate(rows: list[dict]) -> dict:
    outcomes = Counter(row["outcome"] for row in rows)
    goals = [row for row in rows if row["outcome"] == "goal"]
    return {
        "episodes": len(rows),
        "success_rate": outcomes["goal"] / len(rows),
        "fall_rate": outcomes["fell"] / len(rows),
        "timeout_rate": outcomes["timeout"] / len(rows),
        "mean_completion": float(np.mean([r["completion"] for r in rows])),
        "mean_cross_track_m": float(np.mean([
            r["mean_cross_track_m"] for r in rows])),
        "mean_reward": float(np.mean([r["reward"] for r in rows])),
        "mean_seconds_to_goal": (float(np.mean([r["seconds"] for r in goals]))
                                 if goals else None),
    }


def summarize(rows: list[dict], headline_scale: float) -> dict:
    by_maze, by_tier_scale = {}, {}
    for maze in sorted({row["maze"] for row in rows}):
        group = [row for row in rows if row["maze"] == maze]
        by_maze[maze] = {"tier": group[0]["tier"]} | aggregate(group)
    for tier in sorted({row["tier"] for row in rows}):
        for scale in sorted({row["dr_scale"] for row in rows}):
            group = [row for row in rows
                     if row["tier"] == tier and row["dr_scale"] == scale]
            if group:
                by_tier_scale[f"{tier}@{scale:g}"] = aggregate(group)

    def tier_at(tier, scale):
        group = [row for row in rows
                 if row["tier"] == tier and row["dr_scale"] == scale]
        return aggregate(group)["success_rate"] if group else None

    shipped = tier_at("shipped", headline_scale)
    matched = tier_at("matched", headline_scale)
    return {
        "by_maze": by_maze,
        "by_tier_and_scale": by_tier_scale,
        "headline": {
            "dr_scale": headline_scale,
            "shipped_success": shipped,
            "matched_success": matched,
            # The number the whole harness exists to produce.
            "generalization_gap": (None if shipped is None or matched is None
                                   else shipped - matched),
            "harder_success": tier_at("harder", headline_scale),
        },
    }


def build_tasks(directory: Path, scales, episodes: int, seed: int,
                max_seconds: float, include_shipped: bool) -> list[dict]:
    manifest = json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8"))
    mazes = [(f"{entry['tier']}/{entry['seed']}", entry["tier"],
              json.loads((directory / entry["path"]).read_text()))
             for entry in manifest["mazes"]]
    if include_shipped:
        mazes.append(("shipped", "shipped",
                      json.loads(Path(DEFAULT_LAYOUT).read_text())))
    return [{"maze": name, "tier": tier, "layout": layout, "dr_scale": scale,
             "seed": seed + index, "max_seconds": max_seconds}
            for name, tier, layout in mazes
            for scale in scales
            for index in range(episodes)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True,
                        help=".pt behaviour-cloning or .zip SAC checkpoint")
    parser.add_argument("--mazes", type=Path,
                        default=Path("artifacts/holdout_mazes"))
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--scales", type=float, nargs="+",
                        default=list(DEFAULT_SCALES))
    parser.add_argument("--headline-scale", type=float,
                        default=REALISTIC_DR_SCALE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-seconds", type=float, default=200.0)
    parser.add_argument("--no-shipped", action="store_true",
                        help="drop the shipped-maze control (not advised)")
    parser.add_argument("--workers", type=int,
                        default=min(32, os.cpu_count() or 1))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.episodes < 30:
        print(f"warning: {args.episodes} episodes per cell cannot resolve "
              f"less than a {100 / args.episodes:.0f}-point difference")

    tasks = build_tasks(args.mazes, args.scales, args.episodes, args.seed,
                        args.max_seconds, not args.no_shipped)
    print(f"{Path(args.policy).name}: {len(tasks):,} episodes "
          f"({len({t['maze'] for t in tasks})} mazes x {len(args.scales)} "
          f"scales x {args.episodes} seeds) on {args.workers} workers")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_initialize_worker,
                             initargs=(str(args.policy),)) as executor:
        futures = [executor.submit(_run, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if done % 50 == 0 or done == len(futures):
                print(f"  {done:>5}/{len(futures)}", flush=True)

    summary = summarize(rows, args.headline_scale)
    print(f"\n{'maze':<16} {'tier':<8} {'succ':>6} {'fall':>6} {'compl':>7} "
          f"{'xtrack':>8} {'s/goal':>7}")
    for maze, stats in summary["by_maze"].items():
        seconds = stats["mean_seconds_to_goal"]
        print(f"{maze:<16} {stats['tier']:<8} {stats['success_rate']:>6.0%} "
              f"{stats['fall_rate']:>6.0%} {stats['mean_completion']:>7.1%} "
              f"{stats['mean_cross_track_m'] * 1000:>7.2f}mm "
              f"{(f'{seconds:.1f}' if seconds else '-'):>7}")

    print(f"\n{'tier@scale':<20} {'succ':>6} {'fall':>6} {'compl':>7}")
    for key, stats in summary["by_tier_and_scale"].items():
        print(f"{key:<20} {stats['success_rate']:>6.0%} "
              f"{stats['fall_rate']:>6.0%} {stats['mean_completion']:>7.1%}")

    headline = summary["headline"]
    print(f"\nat dr_scale {headline['dr_scale']:g}:")
    print(f"  shipped (control) {headline['shipped_success']}")
    print(f"  matched held-out  {headline['matched_success']}")
    print(f"  harder held-out   {headline['harder_success']}")
    gap = headline["generalization_gap"]
    print(f"  GENERALIZATION GAP {gap:+.1%}" if gap is not None
          else "  GENERALIZATION GAP unavailable (shipped control excluded)")

    out = Path(args.out) if args.out else (
        Path(args.policy).parent / "maze_generalization.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": "maze_generalization_v1",
        "policy": str(Path(args.policy).resolve()),
        "episodes_per_cell": args.episodes, "scales": args.scales,
        "seed": args.seed, "max_seconds": args.max_seconds,
        "summary": summary, "rows": rows,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
