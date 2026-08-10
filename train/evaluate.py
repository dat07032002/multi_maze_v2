"""Score a policy, or the analytic baseline, on the same terms.

    python -m train.evaluate --policy artifacts/sac/policy.zip --episodes 100
    python -m train.evaluate --baseline --episodes 20

Both run through ``MazeEnv``, so "the policy beat the baseline" is a claim about
the policy and not about two different plants. Reports the four numbers the
milestone gate asks for -- success, cross-track, hole-fall rate, mean |delta
action| -- and never route completion on its own: completion is a *projection*
onto the route, and a ball rattling around off-corridor can ratchet it without
having travelled anything.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from contract import policy_contract as pc
from sim.maze_env import MazeEnv
from sim.mjcf_builder import flat_board_layout, load_layout, load_parameters
from sim.randomization import Randomizer


def _baseline_actor(env, params):
    from control.baseline import PurePursuitBaseline
    controller = PurePursuitBaseline(env.route, params["actuator.max_tilt"])

    def act(_observation):
        position, velocity = env.estimator.state
        position, velocity = env.predictor.predict(position, velocity,
                                                   env._command)
        return pc.angles_to_action(*controller(position, velocity),
                                   params["actuator.max_tilt"])
    return act


def _policy_actor(path):
    from stable_baselines3 import SAC
    model = SAC.load(path, device="cpu")

    def act(observation):
        action, _ = model.predict(observation, deterministic=True)
        return action
    return act


def evaluate(actor_factory, episodes: int = 100, randomize: bool = False,
             flat: bool = False, seed: int = 1000, max_seconds: float = 60.0):
    layout = flat_board_layout() if flat else load_layout()
    params = load_parameters()
    env = MazeEnv(layout=layout, params=params, max_seconds=max_seconds,
                  randomizer=Randomizer(enabled=randomize), seed=seed)
    act = actor_factory(env, params)

    rows = []
    for episode in range(episodes):
        observation, info = env.reset(seed=seed + episode)
        actions, total = [], 0.0
        while True:
            action = np.asarray(act(observation), dtype=np.float64)
            observation, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            total += reward
            if terminated or truncated:
                break
        deltas = np.linalg.norm(np.diff(actions, axis=0), axis=1) \
            if len(actions) > 1 else np.zeros(1)
        rows.append({
            "outcome": info["outcome"],
            "completion": info["route_completion"],
            "mean_cross_track": info["mean_cross_track"],
            "reward": total,
            "seconds": info["steps"] / 20.0,
            "action_rate": float(deltas.mean()),
        })
    return rows


def summarise(rows, label: str) -> dict:
    n = len(rows)
    goals = [r for r in rows if r["outcome"] == "goal"]
    summary = {
        "success": len(goals) / n,
        "fell": sum(r["outcome"] == "fell" for r in rows) / n,
        "timeout": sum(r["outcome"] == "timeout" for r in rows) / n,
        "completion": float(np.mean([r["completion"] for r in rows])),
        "cross_track": float(np.mean([r["mean_cross_track"] for r in rows])),
        "reward": float(np.mean([r["reward"] for r in rows])),
        "action_rate": float(np.mean([r["action_rate"] for r in rows])),
        "seconds": float(np.mean([r["seconds"] for r in goals])) if goals else float("nan"),
    }
    print(f"{label}  ({n} episodes)")
    print(f"  success        {summary['success']:.0%}"
          f"   fell {summary['fell']:.0%}   timeout {summary['timeout']:.0%}")
    print(f"  completion     {summary['completion']:.1%} mean")
    print(f"  cross-track    {summary['cross_track'] * 1000:.2f} mm mean")
    print(f"  reward         {summary['reward']:.2f} of 40")
    print(f"  |delta action| {summary['action_rate']:.4f} "
          f"(one 40-count command = {pc.ACTION_QUANTUM:.4f})")
    print(f"  time to goal   {summary['seconds']:.1f} s")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", default=None)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--flat", action="store_true")
    args = parser.parse_args()

    if args.baseline:
        rows = evaluate(_baseline_actor, args.episodes, args.randomize, args.flat)
        summarise(rows, "analytic baseline" + (" (randomised)" if args.randomize else ""))
    if args.policy:
        factory = lambda env, params: _policy_actor(args.policy)  # noqa: E731
        rows = evaluate(factory, args.episodes, args.randomize, args.flat)
        summarise(rows, f"policy {Path(args.policy).name}"
                  + (" (randomised)" if args.randomize else ""))


if __name__ == "__main__":
    main()
