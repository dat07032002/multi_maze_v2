"""Run the MPC teacher and save local-maneuver demonstration transitions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from control.mpc_teacher import CEMConfig, CEMMPCTeacher
from sim.randomization import PHYSICS_VECTOR_NAMES, physics_vector
from sim.segment_env import SegmentEnv
from train.segment_dataset import KINDS


def select_episodes(dataset: dict, geometries_per_kind: int,
                    conditions_per_geometry: int):
    """Select a class-balanced, source-diverse subset of episode specs.

    Dataset geometries are stored authentic-first, so taking a simple prefix
    can make a canary look healthy without ever exercising generated paths.
    Take one geometry from each available source before filling the remainder.
    """
    selected_geometries = {}
    source_order = ("authentic", "mirrored_authentic", "procedural")
    for kind in KINDS:
        matches = [g for g in dataset["geometries"] if g["kind"] == kind]
        by_source = {
            source: [g for g in matches if g["source"] == source]
            for source in source_order
        }
        chosen = []
        for source in source_order:
            if by_source[source] and len(chosen) < geometries_per_kind:
                chosen.append(by_source[source].pop(0))
        already_chosen = {g["id"] for g in chosen}
        chosen.extend(
            geometry for geometry in matches
            if geometry["id"] not in already_chosen
        )
        for geometry in chosen[:geometries_per_kind]:
            selected_geometries[geometry["id"]] = geometry
    counts = defaultdict(int)
    episodes = []
    for episode in dataset["episodes"]:
        geometry_id = episode["geometry_id"]
        if geometry_id not in selected_geometries:
            continue
        if counts[geometry_id] >= conditions_per_geometry:
            continue
        counts[geometry_id] += 1
        episodes.append((selected_geometries[geometry_id], episode))
    return episodes


def run_episode(geometry: dict, specification: dict, config: CEMConfig,
                max_seconds: float = 60.0, sensor_noise: bool = True):
    env = SegmentEnv(
        geometry,
        randomization_scale=specification["randomization_scale"],
        max_seconds=max_seconds,
        seed=specification["physics_seed"], sensor_noise=sensor_noise)
    observation, _ = env.reset(
        seed=specification["sensor_seed"],
        options={"episode_spec": specification})
    teacher = CEMMPCTeacher(
        env, seed=specification["physics_seed"], config=config)
    physics = physics_vector(env._params)

    rows = []
    total_reward = 0.0
    try:
        while True:
            action = teacher(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            rows.append((observation.copy(), action.copy(),
                         next_observation.copy(), reward,
                         terminated or truncated, truncated))
            total_reward += reward
            observation = next_observation
            if terminated or truncated:
                break
    finally:
        env.close()
    result = {
        "episode_id": specification["id"],
        "geometry_id": geometry["id"],
        "kind": geometry["kind"],
        "source": geometry["source"],
        # The plant this episode was actually flown on. Known exactly here and
        # never on the rig, which is the asymmetry a privileged teacher trains
        # against and an adaptation module learns to close. Recording it costs
        # eleven floats per episode; not recording it costs a regeneration of
        # the whole demonstration set.
        "physics": dict(zip(PHYSICS_VECTOR_NAMES,
                            (float(v) for v in physics))),
        "outcome": info["outcome"],
        "steps": info["steps"],
        "seconds": info["steps"] / 20.0,
        "completion": info["route_completion"],
        "cross_track_m": info["mean_cross_track"],
        "reward": total_reward,
    }
    return rows, result


def generate(dataset: dict, geometries_per_kind: int,
             conditions_per_geometry: int, config: CEMConfig,
             max_seconds: float = 60.0, sensor_noise: bool = True):
    selected = select_episodes(
        dataset, geometries_per_kind, conditions_per_geometry)
    return generate_selected(
        selected, config, max_seconds=max_seconds, sensor_noise=sensor_noise)


def generate_selected(selected, config: CEMConfig,
                      max_seconds: float = 60.0,
                      sensor_noise: bool = True):
    """Generate demonstrations for an already selected episode sequence."""
    transitions, results = [], []
    started = time.perf_counter()
    for number, (geometry, specification) in enumerate(selected, start=1):
        rows, result = run_episode(
            geometry, specification, config,
            max_seconds=max_seconds, sensor_noise=sensor_noise)
        transitions.extend((row, result["episode_id"], result["kind"],
                            result["outcome"] == "goal",
                            result["physics"]) for row in rows)
        results.append(result)
        elapsed = time.perf_counter() - started
        print(f"[{number:>3}/{len(selected)}] {result['episode_id']}  "
              f"{result['outcome']:<7} {result['completion']:6.1%}  "
              f"{result['seconds']:5.1f}s sim  {elapsed:6.1f}s wall")
    return transitions, results


def shard_selection(selected, episodes_per_shard: int):
    if episodes_per_shard < 1:
        raise ValueError("episodes_per_shard must be positive")
    return [selected[start:start + episodes_per_shard]
            for start in range(0, len(selected), episodes_per_shard)]


def partition_selection(selected, worker_index: int, workers: int):
    """Return one disjoint deterministic slice for parallel generation."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if not 0 <= worker_index < workers:
        raise ValueError("worker_index must be in [0, workers)")
    return selected[worker_index::workers]


def shard_paths(output: Path, number: int) -> tuple[Path, Path]:
    stem = f"{output.stem}.part-{number:04d}"
    data = output.with_name(stem + output.suffix)
    return data, data.with_suffix(".json")


def generate_sharded(dataset: dict, geometries_per_kind: int,
                     conditions_per_geometry: int, config: CEMConfig,
                     output: Path, episodes_per_shard: int,
                     max_seconds: float = 60.0,
                     sensor_noise: bool = True, resume: bool = False,
                     worker_index: int = 0, workers: int = 1):
    """Generate crash-safe shards and return the updated manifest."""
    all_selected = select_episodes(
        dataset, geometries_per_kind, conditions_per_geometry)
    selected = partition_selection(all_selected, worker_index, workers)
    shards = shard_selection(selected, episodes_per_shard)
    all_results, entries = [], []
    for number, episode_shard in enumerate(shards, start=1):
        data_path, report_path = shard_paths(output, number)
        if resume and data_path.exists() and report_path.exists():
            report_data = json.loads(report_path.read_text(encoding="utf-8"))
            results = report_data["episodes"]
            print(f"[{number:>3}/{len(shards)} shards] resume {data_path.name}  "
                  f"{len(results)} episodes")
        else:
            print(f"[{number:>3}/{len(shards)} shards] generating "
                  f"{len(episode_shard)} episodes")
            transitions, results = generate_selected(
                episode_shard, config, max_seconds=max_seconds,
                sensor_noise=sensor_noise)
            save(transitions, results, data_path)
        all_results.extend(results)
        entries.append({
            "number": number,
            "data": str(data_path.resolve()),
            "report": str(report_path.resolve()),
            "episodes": len(results),
        })
        manifest = {
            "worker_index": worker_index,
            "workers": workers,
            "global_target_episodes": len(all_selected),
            "target_episodes": len(selected),
            "completed_episodes": len(all_results),
            "complete": len(all_results) == len(selected),
            "shards": entries,
            "summary": summarise(all_results),
        }
        manifest_path = output.with_suffix(".manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path, manifest


def save(transitions, results, output: Path) -> tuple[Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [item[0] for item in transitions]
    np.savez_compressed(
        output,
        observations=np.asarray([row[0] for row in rows], dtype=np.float32),
        actions=np.asarray([row[1] for row in rows], dtype=np.float32),
        next_observations=np.asarray([row[2] for row in rows], dtype=np.float32),
        rewards=np.asarray([row[3] for row in rows], dtype=np.float32),
        dones=np.asarray([row[4] for row in rows], dtype=bool),
        truncated=np.asarray([row[5] for row in rows], dtype=bool),
        episode_ids=np.asarray([item[1] for item in transitions]),
        kinds=np.asarray([item[2] for item in transitions]),
        successful_episode=np.asarray([item[3] for item in transitions], dtype=bool),
        physics_params=np.asarray(
            [[item[4][name] for name in PHYSICS_VECTOR_NAMES]
             for item in transitions], dtype=np.float32),
        physics_param_names=np.asarray(PHYSICS_VECTOR_NAMES),
    )
    report = output.with_suffix(".json")
    report.write_text(json.dumps({
        "episodes": results,
        "summary": summarise(results),
    }, indent=2), encoding="utf-8")
    return output, report


def summarise(results: list[dict]) -> dict:
    by_kind = {}
    for kind in KINDS:
        rows = [row for row in results if row["kind"] == kind]
        if not rows:
            continue
        by_kind[kind] = {
            "episodes": len(rows),
            "success_rate": sum(row["outcome"] == "goal" for row in rows)
                            / len(rows),
            "fall_rate": sum(row["outcome"] == "fell" for row in rows)
                         / len(rows),
            "mean_completion": float(np.mean(
                [row["completion"] for row in rows])),
            "mean_seconds_to_goal": float(np.mean(
                [row["seconds"] for row in rows if row["outcome"] == "goal"]))
                if any(row["outcome"] == "goal" for row in rows) else None,
        }
    return {
        "episodes": len(results),
        "success_rate": sum(row["outcome"] == "goal" for row in results)
                        / max(len(results), 1),
        "fall_rate": sum(row["outcome"] == "fell" for row in results)
                     / max(len(results), 1),
        "by_kind": by_kind,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", default="artifacts/local_segments/balanced_dataset.json")
    parser.add_argument(
        "--out", default="artifacts/local_segments/mpc_canary.npz")
    parser.add_argument("--geometries-per-kind", type=int, default=10)
    parser.add_argument("--conditions-per-geometry", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--candidates", type=int, default=192)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--elites", type=int, default=24)
    parser.add_argument("--no-sensor-noise", action="store_true")
    parser.add_argument(
        "--shard-episodes", type=int, default=0,
        help="save every N episodes; use this for long crash-safe runs")
    parser.add_argument(
        "--resume", action="store_true",
        help="skip complete shard pairs already present on disk")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="number of disjoint generation workers")
    parser.add_argument(
        "--worker-index", type=int, default=0,
        help="zero-based disjoint worker slice")
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    config = CEMConfig(
        horizon_steps=args.horizon, candidates=args.candidates,
        iterations=args.iterations, elites=args.elites)
    if args.shard_episodes:
        manifest_path, manifest = generate_sharded(
            dataset, args.geometries_per_kind, args.conditions_per_geometry,
            config, Path(args.out), args.shard_episodes,
            max_seconds=args.max_seconds,
            sensor_noise=not args.no_sensor_noise, resume=args.resume,
            worker_index=args.worker_index, workers=args.workers)
        summary = manifest["summary"]
        print(f"manifest {manifest_path.resolve()}")
        print(f"completed {manifest['completed_episodes']}/"
              f"{manifest['target_episodes']} episodes")
    else:
        transitions, results = generate(
            dataset, args.geometries_per_kind, args.conditions_per_geometry,
            config, max_seconds=args.max_seconds,
            sensor_noise=not args.no_sensor_noise)
        output, report = save(transitions, results, Path(args.out))
        summary = summarise(results)
        print(f"saved {len(transitions):,} transitions to {output.resolve()}")
        print(f"saved report to {report.resolve()}")
    print(f"teacher success {summary['success_rate']:.0%}, "
          f"falls {summary['fall_rate']:.0%}")


if __name__ == "__main__":
    main()
