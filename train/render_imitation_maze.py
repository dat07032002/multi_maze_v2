"""Render a full-maze imitation-policy rollout as an accelerated GIF."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

from control.imitation_policy import load_policy
from sim.maze_env import MazeEnv


def labelled_frame(renderer, env: MazeEnv, info: dict) -> np.ndarray:
    renderer.update_scene(env.data, camera="top")
    frame = renderer.render().copy()
    elapsed = info["steps"] / env.control_hz
    label = (
        f"Best BC policy | t={elapsed:05.1f}s | "
        f"completion={info['route_completion'] * 100:05.1f}%")
    cv2.putText(frame, label, (18, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (15, 15, 15), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (18, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (250, 250, 250), 1, cv2.LINE_AA)
    return frame


def render(policy_path: str | Path, output: str | Path, seed: int = 30_000,
           max_seconds: float = 120.0, capture_every: int = 10,
           gif_fps: float = 10.0) -> dict:
    model = load_policy(policy_path, device="cpu")
    env = MazeEnv(max_seconds=max_seconds, sensor_noise=True,
                  start_fraction=0.0, seed=seed)
    observation, info = env.reset(seed=seed)
    frames = []
    renderer = mujoco.Renderer(env.model, height=480, width=540)
    total_reward = 0.0
    try:
        frames.append(labelled_frame(renderer, env, info))
        while True:
            action = np.asarray(model.predict(observation), dtype=np.float64)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if info["steps"] % capture_every == 0 or terminated or truncated:
                frames.append(labelled_frame(renderer, env, info))
            if terminated or truncated:
                break
    finally:
        renderer.close()
        env.close()

    # Hold the final state briefly so the stopping point is easy to inspect.
    frames.extend([frames[-1]] * int(round(gif_fps * 1.5)))
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame).convert("P", palette=Image.Palette.ADAPTIVE)
              for frame in frames]
    images[0].save(
        destination, save_all=True, append_images=images[1:], loop=0,
        duration=int(round(1000.0 / gif_fps)), optimize=False, disposal=2)
    return {
        "path": str(destination.resolve()),
        "seed": seed,
        "outcome": info["outcome"],
        "completion": info["route_completion"],
        "seconds": info["steps"] / 20.0,
        "mean_cross_track_m": info["mean_cross_track"],
        "reward": total_reward,
        "frames": len(frames),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        default="artifacts/local_segments/bc_policy_v1/best_model.pt")
    parser.add_argument(
        "--out",
        default="artifacts/local_segments/bc_policy_v1/best_full_maze.gif")
    parser.add_argument("--seed", type=int, default=30_000)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--capture-every", type=int, default=10)
    parser.add_argument("--gif-fps", type=float, default=10.0)
    args = parser.parse_args()
    print(render(args.policy, args.out, args.seed, args.max_seconds,
                 args.capture_every, args.gif_fps))


if __name__ == "__main__":
    main()
