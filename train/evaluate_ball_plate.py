"""Evaluate the M3 policy and analytic setpoint controller head-to-head."""
from __future__ import annotations

import argparse

import numpy as np

from contract import policy_contract as pc
from sim.ball_plate import BallPlateEnv, SetpointBaseline
from sim.mjcf_builder import load_parameters
from sim.randomization import Randomizer


def policy_actor(path):
    from stable_baselines3 import SAC
    model = SAC.load(path, device="cpu")
    return lambda observation, env: model.predict(
        observation, deterministic=True)[0]


def baseline_actor():
    params = load_parameters()
    controller = SetpointBaseline(params["actuator.max_tilt"],
                                  params["actuator.centre_bias"])

    def act(_observation, env):
        position, velocity = env.estimator.state
        position, velocity = env.predictor.predict(
            position, velocity, env._command)
        angles = controller(position, velocity, env.target)
        return np.asarray(pc.angles_to_action(
            *angles, params["actuator.max_tilt"]), dtype=float)
    return act


def evaluate(actor, episodes=100, randomize=False, seed=20_000):
    env = BallPlateEnv(
        params=load_parameters(),
        randomizer=Randomizer(enabled=randomize), seed=seed)
    rows = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        actions, total = [], 0.0
        while True:
            action = np.asarray(actor(observation, env), dtype=float)
            observation, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            total += reward
            if terminated or truncated:
                break
        deltas = np.linalg.norm(np.diff(actions, axis=0), axis=1) \
            if len(actions) > 1 else np.zeros(1)
        rows.append({
            "success": info["outcome"] == "goal",
            "distance": info["target_distance"],
            "reward": total,
            "action_rate": float(deltas.mean()),
            "seconds": info["steps"] / env.control_hz,
        })
    return rows


def summarise(rows, label):
    goals = [r for r in rows if r["success"]]
    print(f"{label} ({len(rows)} episodes)")
    print(f"  success        {len(goals) / len(rows):.0%}")
    print(f"  final error    {np.mean([r['distance'] for r in rows]) * 1000:.2f} mm")
    print(f"  reward         {np.mean([r['reward'] for r in rows]):.2f}")
    print(f"  |delta action| {np.mean([r['action_rate'] for r in rows]):.4f} "
          f"(two quanta = {pc.ACTION_RATE_SCALE:.4f})")
    seconds = np.mean([r["seconds"] for r in goals]) if goals else float("nan")
    print(f"  time to hold   {seconds:.2f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--randomize", action="store_true")
    args = parser.parse_args()
    if not args.policy and not args.baseline:
        parser.error("provide --policy and/or --baseline")
    if args.baseline:
        summarise(evaluate(baseline_actor(), args.episodes, args.randomize),
                  "analytic setpoint baseline")
    if args.policy:
        summarise(evaluate(policy_actor(args.policy), args.episodes,
                           args.randomize), f"policy {args.policy}")


if __name__ == "__main__":
    main()
