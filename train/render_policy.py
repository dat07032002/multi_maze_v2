"""Record one successful policy rollout as an MP4.

Searches the same deterministic seed sequence used by ``train.evaluate`` and
only writes a video after finding an episode that reaches the goal.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np
from stable_baselines3 import SAC

from sim.maze_env import MazeEnv


def run_episode(model: SAC, seed: int, capture: bool = False):
    env = MazeEnv(seed=seed)
    observation, info = env.reset(seed=seed)
    frames = []
    total_reward = 0.0
    renderer = mujoco.Renderer(env.model, height=720, width=960) if capture else None

    try:
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if renderer is not None:
                renderer.update_scene(env.data, camera="top")
                frame = renderer.render().copy()
                elapsed = info["steps"] / env.control_hz
                label = (f"t={elapsed:05.1f}s   progress={info['route_completion'] * 100:05.1f}%"
                         f"   outcome={info['outcome']}")
                cv2.putText(frame, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (20, 20, 20), 4, cv2.LINE_AA)
                cv2.putText(frame, label, (24, 42), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (245, 245, 245), 2, cv2.LINE_AA)
                frames.append(frame)

            if terminated or truncated:
                break
    finally:
        if renderer is not None:
            renderer.close()
        env.close()

    return info, total_reward, frames


def write_video(path: Path, frames: list[np.ndarray], fps: float = 20.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy", required=True)
    parser.add_argument("--out", default="artifacts/successful_rollout.mp4")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--attempts", type=int, default=20)
    args = parser.parse_args()

    model = SAC.load(args.policy, device="cpu")
    successful_seed = None
    for seed in range(args.seed, args.seed + args.attempts):
        info, reward, _ = run_episode(model, seed)
        print(f"seed {seed}: {info['outcome']}, "
              f"completion {info['route_completion']:.1%}, reward {reward:.2f}")
        if info["outcome"] == "goal":
            successful_seed = seed
            break

    if successful_seed is None:
        raise SystemExit(f"No success in {args.attempts} episodes")

    info, reward, frames = run_episode(model, successful_seed, capture=True)
    output = Path(args.out).resolve()
    write_video(output, frames)
    print(f"Saved successful seed {successful_seed} to {output}")
    print(f"  time {info['steps'] / 20.0:.1f}s, reward {reward:.2f}, "
          f"cross-track {info['mean_cross_track'] * 1000:.2f} mm")


if __name__ == "__main__":
    main()
