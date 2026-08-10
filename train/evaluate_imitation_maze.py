"""Evaluate the route-conditioned imitation policy on the complete maze."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from control.imitation_policy import load_policy
from sim.maze_env import MazeEnv
from sim.randomization import Randomizer
from train.evaluate import summarise


def evaluate(model, episodes: int = 5, randomization_scale: float = 0.0,
             seed: int = 30_000, max_seconds: float = 120.0,
             sensor_noise: bool = True) -> list[dict]:
    """Run deterministic policy actions from the full-route entrance."""
    randomizer = Randomizer(
        scale=randomization_scale, enabled=randomization_scale > 0.0)
    env = MazeEnv(
        max_seconds=max_seconds, sensor_noise=sensor_noise,
        randomizer=randomizer, start_fraction=0.0, seed=seed)
    rows = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            sampled_parameters = {
                name: env._params[name] for name in randomizer.spec
                if name in env._params
            }
            actions = []
            total_reward = 0.0
            while True:
                action = np.asarray(model.predict(observation), dtype=np.float64)
                observation, reward, terminated, truncated, info = env.step(action)
                actions.append(action)
                total_reward += reward
                if terminated or truncated:
                    break
            deltas = np.linalg.norm(np.diff(actions, axis=0), axis=1) \
                if len(actions) > 1 else np.zeros(1)
            row = {
                "episode": episode + 1,
                "seed": seed + episode,
                "outcome": info["outcome"],
                "completion": info["route_completion"],
                "mean_cross_track": info["mean_cross_track"],
                "reward": total_reward,
                "seconds": info["steps"] / env.control_hz,
                "action_rate": float(deltas.mean()),
                "sampled_parameters": sampled_parameters,
            }
            rows.append(row)
            print(
                f"[{episode + 1:>2}/{episodes}] {row['outcome']:<7} "
                f"completion {row['completion']:6.1%}  "
                f"cross-track {row['mean_cross_track'] * 1000:6.2f} mm  "
                f"{row['seconds']:6.1f} s",
                flush=True)
    finally:
        env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        default="artifacts/local_segments/bc_policy_v1/best_model.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--randomization-scale", type=float, default=0.0)
    parser.add_argument("--no-sensor-noise", action="store_true")
    parser.add_argument(
        "--out",
        default="artifacts/local_segments/bc_policy_v1/full_maze_nominal.json")
    args = parser.parse_args()

    model = load_policy(args.policy, device="cpu")
    rows = evaluate(
        model, episodes=args.episodes,
        randomization_scale=args.randomization_scale, seed=args.seed,
        max_seconds=args.max_seconds, sensor_noise=not args.no_sensor_noise)
    summary = summarise(rows, f"imitation policy {Path(args.policy).name}")
    report = {
        "schema": "route_conditioned_bc_full_maze_evaluation_v1",
        "policy": str(Path(args.policy).resolve()),
        "conditions": {
            "episodes": args.episodes,
            "seed": args.seed,
            "max_seconds": args.max_seconds,
            "sensor_noise": not args.no_sensor_noise,
            "randomization_scale": args.randomization_scale,
            "start_fraction": 0.0,
        },
        "episodes": rows,
        "summary": summary,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
