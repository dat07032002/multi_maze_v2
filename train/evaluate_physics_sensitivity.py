"""Rank full-maze policy sensitivity to individual uncertain physics values."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path

import numpy as np

from control.imitation_policy import load_policy
from sim.maze_env import MazeEnv
from sim.mjcf_builder import load_parameters
from sim.randomization import Randomizer


_MODEL = None


def build_conditions() -> list[dict]:
    """Nominal plus documented low/high values, one parameter at a time."""
    base = load_parameters()
    randomizer = Randomizer()
    conditions = [{
        "id": "nominal", "parameter": "nominal", "level": "nominal",
        "value": None, "params": base,
    }]
    for name, (nominal, bounds) in sorted(randomizer.spec.items()):
        if name == "actuator.centre_bias":
            sigma = base["actuator.level_repeatability"] / 2.0
            for axis, index in (("roll", 0), ("pitch", 1)):
                for sign, label in ((-1.0, "minus_2sigma"),
                                    (1.0, "plus_2sigma")):
                    params = dict(base)
                    value = list(nominal)
                    value[index] += sign * 2.0 * sigma
                    params[name] = value
                    conditions.append({
                        "id": f"{name}.{axis}.{label}", "parameter": name,
                        "axis": axis, "level": label, "value": value,
                        "params": params,
                    })
            continue
        if bounds is None:
            continue
        for value, level in zip(bounds, ("low", "high")):
            params = dict(base)
            params[name] = value
            if name == "ball.wall_restitution":
                params["sim.wall_dampratio"] = float(np.clip(
                    base["sim.wall_dampratio"] * nominal / max(value, 1e-3),
                    0.05, 2.0))
            conditions.append({
                "id": f"{name}.{level}", "parameter": name,
                "level": level, "value": value, "params": params,
            })
    return conditions


def _initialize_worker(policy_path: str) -> None:
    global _MODEL
    _MODEL = load_policy(policy_path, device="cpu")


def _run(task: tuple[dict, int, float]) -> dict:
    condition, seed, max_seconds = task
    env = MazeEnv(
        params=condition["params"], max_seconds=max_seconds,
        sensor_noise=True, start_fraction=0.0, seed=seed)
    observation, _ = env.reset(seed=seed)
    total_reward = 0.0
    try:
        while True:
            action = np.asarray(_MODEL.predict(observation), dtype=np.float64)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
    finally:
        env.close()
    return {
        "condition": condition["id"],
        "parameter": condition["parameter"],
        "axis": condition.get("axis"),
        "level": condition["level"],
        "value": condition["value"],
        "seed": seed,
        "outcome": info["outcome"],
        "completion": info["route_completion"],
        "mean_cross_track_m": info["mean_cross_track"],
        "reward": total_reward,
        "seconds": info["steps"] / env.control_hz,
    }


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for condition in sorted({row["condition"] for row in rows}):
        group = [row for row in rows if row["condition"] == condition]
        outcomes = Counter(row["outcome"] for row in group)
        goals = [row for row in group if row["outcome"] == "goal"]
        summaries.append({
            "condition": condition,
            "parameter": group[0]["parameter"],
            "axis": group[0]["axis"],
            "level": group[0]["level"],
            "value": group[0]["value"],
            "episodes": len(group),
            "success_rate": outcomes["goal"] / len(group),
            "fall_rate": outcomes["fell"] / len(group),
            "timeout_rate": outcomes["timeout"] / len(group),
            "mean_completion": float(np.mean([
                row["completion"] for row in group])),
            "mean_cross_track_m": float(np.mean([
                row["mean_cross_track_m"] for row in group])),
            "mean_seconds_to_goal": float(np.mean([
                row["seconds"] for row in goals])) if goals else None,
        })
    return sorted(
        summaries,
        key=lambda row: (row["success_rate"], row["mean_completion"],
                         -row["fall_rate"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        default="artifacts/local_segments/bc_policy_v1/best_model.pt")
    parser.add_argument("--episodes-per-condition", type=int, default=3)
    parser.add_argument("--seed", type=int, default=40_000)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--workers", type=int,
                        default=min(16, os.cpu_count() or 1))
    parser.add_argument(
        "--out",
        default="artifacts/local_segments/bc_policy_v1/physics_sensitivity.json")
    args = parser.parse_args()

    conditions = build_conditions()
    tasks = []
    for condition_index, condition in enumerate(conditions):
        for episode in range(args.episodes_per_condition):
            tasks.append((condition,
                          args.seed + condition_index * 100 + episode,
                          args.max_seconds))

    rows = []
    with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_initialize_worker,
            initargs=(args.policy,)) as executor:
        futures = [executor.submit(_run, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(
                f"[{completed:>3}/{len(tasks)}] {row['condition']:<48} "
                f"{row['outcome']:<7} {row['completion']:6.1%}", flush=True)

    rankings = summarize(rows)
    report = {
        "schema": "route_conditioned_bc_physics_sensitivity_v1",
        "policy": str(Path(args.policy).resolve()),
        "episodes_per_condition": args.episodes_per_condition,
        "max_seconds": args.max_seconds,
        "conditions": len(conditions),
        "episodes": rows,
        "ranking_worst_first": rankings,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(rankings, indent=2))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
