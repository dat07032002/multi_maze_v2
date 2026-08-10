"""Closed-loop evaluation of a behavior-cloned policy on held-out segments."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from control.imitation_policy import load_policy
from sim.segment_env import SegmentEnv
from train.generate_segment_demos import summarise
from train.segment_dataset import KINDS


def select_evaluation_episodes(dataset: dict, integrity_report: dict,
                               split: str, geometries_per_kind: int,
                               conditions_per_geometry: int):
    assigned = {
        row["id"] for row in integrity_report["geometry_assignment"]
        if row["split"] == split
    }
    geometries = {row["id"]: row for row in dataset["geometries"]}
    source_order = ("authentic", "mirrored_authentic", "procedural")
    chosen = []
    for kind in KINDS:
        candidates = [
            row for row in dataset["geometries"]
            if row["id"] in assigned and row["kind"] == kind
        ]
        by_source = {
            source: [row for row in candidates if row["source"] == source]
            for source in source_order
        }
        selected = []
        for source in source_order:
            if by_source[source] and len(selected) < geometries_per_kind:
                selected.append(by_source[source].pop(0))
        selected_ids = {row["id"] for row in selected}
        selected.extend(
            row for row in candidates if row["id"] not in selected_ids)
        chosen.extend(row["id"] for row in selected[:geometries_per_kind])

    counts = Counter()
    result = []
    chosen_set = set(chosen)
    for specification in dataset["episodes"]:
        geometry_id = specification["geometry_id"]
        if geometry_id not in chosen_set:
            continue
        if counts[geometry_id] >= conditions_per_geometry:
            continue
        counts[geometry_id] += 1
        result.append((geometries[geometry_id], specification))
    return result


def run_episode(model, geometry: dict, specification: dict,
                max_seconds: float = 60.0, sensor_noise: bool = True):
    env = SegmentEnv(
        geometry,
        randomization_scale=specification["randomization_scale"],
        max_seconds=max_seconds, seed=specification["physics_seed"],
        sensor_noise=sensor_noise)
    observation, _ = env.reset(
        seed=specification["sensor_seed"],
        options={"episode_spec": specification})
    total_reward = 0.0
    actions = []
    try:
        while True:
            action = model.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            actions.append(action)
            if terminated or truncated:
                break
    finally:
        env.close()
    action_array = np.asarray(actions)
    return {
        "episode_id": specification["id"],
        "geometry_id": geometry["id"],
        "kind": geometry["kind"],
        "source": geometry["source"],
        "randomization_scale": specification["randomization_scale"],
        "outcome": info["outcome"],
        "steps": info["steps"],
        "seconds": info["steps"] / 20.0,
        "completion": info["route_completion"],
        "cross_track_m": info["mean_cross_track"],
        "reward": total_reward,
        "mean_action_change": float(np.mean(np.linalg.norm(
            np.diff(action_array, axis=0), axis=1)))
            if len(action_array) > 1 else 0.0,
    }


def expanded_summary(results: list[dict]) -> dict:
    result = summarise(results)
    by_source = {}
    for source in sorted({row["source"] for row in results}):
        rows = [row for row in results if row["source"] == source]
        by_source[source] = {
            "episodes": len(rows),
            "success_rate": sum(row["outcome"] == "goal" for row in rows)
                            / len(rows),
            "fall_rate": sum(row["outcome"] == "fell" for row in rows)
                         / len(rows),
            "mean_completion": float(np.mean([
                row["completion"] for row in rows])),
        }
    result["by_source"] = by_source
    result["outcomes"] = dict(sorted(Counter(
        row["outcome"] for row in results).items()))
    result["mean_action_change"] = float(np.mean([
        row["mean_action_change"] for row in results]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--policy", default="artifacts/local_segments/bc_policy_v1/best_model.pt")
    parser.add_argument(
        "--dataset", default="artifacts/local_segments/balanced_dataset.json")
    parser.add_argument(
        "--integrity-report",
        default="artifacts/local_segments/imitation_v1/integrity_report.json")
    parser.add_argument("--split", choices=("validation", "test"),
                        default="validation")
    parser.add_argument("--geometries-per-kind", type=int, default=3)
    parser.add_argument("--conditions-per-geometry", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--no-sensor-noise", action="store_true")
    parser.add_argument(
        "--out", default="artifacts/local_segments/bc_policy_v1/closed_loop.json")
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    integrity = json.loads(Path(args.integrity_report).read_text(encoding="utf-8"))
    selected = select_evaluation_episodes(
        dataset, integrity, args.split, args.geometries_per_kind,
        args.conditions_per_geometry)
    model = load_policy(args.policy, device="cpu")
    results = []
    started = time.perf_counter()
    for number, (geometry, specification) in enumerate(selected, start=1):
        row = run_episode(
            model, geometry, specification, args.max_seconds,
            sensor_noise=not args.no_sensor_noise)
        results.append(row)
        print(f"[{number:>3}/{len(selected)}] {row['episode_id']} "
              f"{row['outcome']:<7} {row['completion']:6.1%} "
              f"{row['seconds']:5.1f}s sim "
              f"{time.perf_counter() - started:6.1f}s wall")
    report = {
        "policy": str(Path(args.policy).resolve()),
        "split": args.split,
        "episodes": results,
        "summary": expanded_summary(results),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
