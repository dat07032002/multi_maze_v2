"""Render representative MPC-teacher segment demonstrations as GIFs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

from control.mpc_teacher import CEMConfig, CEMMPCTeacher
from sim.segment_env import SegmentEnv


DEFAULT_EPISODES = (
    "straight-001-episode-01",
    "gentle_left-001-episode-01",
    "sharp_right-001-episode-01",
)


def render_demo(geometry: dict, specification: dict, output: Path,
                width: int = 640, height: int = 480,
                frame_stride: int = 2) -> dict:
    env = SegmentEnv(
        geometry,
        randomization_scale=specification["randomization_scale"],
        max_seconds=60.0,
        seed=specification["physics_seed"], sensor_noise=True)
    observation, _ = env.reset(
        seed=specification["sensor_seed"],
        options={"episode_spec": specification})
    teacher = CEMMPCTeacher(
        env, seed=specification["physics_seed"], config=CEMConfig())
    frames: list[Image.Image] = []
    total_reward = 0.0

    with mujoco.Renderer(env.model, height=height, width=width) as renderer:
        while True:
            action = teacher(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if info["steps"] % frame_stride == 0 or terminated or truncated:
                renderer.update_scene(env.data, camera="top")
                frame = renderer.render().copy()
                elapsed = info["steps"] / env.control_hz
                label = (f"MPC teacher | {geometry['kind'].replace('_', ' ')} | "
                         f"t={elapsed:04.1f}s | progress={info['route_completion']:.0%}")
                command = f"action: roll={action[0]:+.2f}  pitch={action[1]:+.2f}"
                for text, y in ((label, 30), (command, 56)):
                    cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.58, (15, 15, 15), 4, cv2.LINE_AA)
                    cv2.putText(frame, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.58, (245, 245, 245), 1, cv2.LINE_AA)
                frames.append(Image.fromarray(frame).convert(
                    "P", palette=Image.Palette.ADAPTIVE, colors=128))
            if terminated or truncated:
                break
    env.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output, save_all=True, append_images=frames[1:], optimize=True,
        duration=int(round(1000.0 * frame_stride / env.control_hz)), loop=0,
        disposal=2)
    return {
        "episode_id": specification["id"],
        "kind": geometry["kind"],
        "outcome": info["outcome"],
        "seconds": info["steps"] / env.control_hz,
        "completion": info["route_completion"],
        "cross_track_m": info["mean_cross_track"],
        "reward": total_reward,
        "frames": len(frames),
        "output": str(output.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset", default="artifacts/local_segments/balanced_dataset.json")
    parser.add_argument("--out-dir", default="artifacts/local_segments/mpc_gifs")
    parser.add_argument("--episodes", nargs="*", default=DEFAULT_EPISODES)
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    geometries = {row["id"]: row for row in dataset["geometries"]}
    episodes = {row["id"]: row for row in dataset["episodes"]}
    results = []
    for episode_id in args.episodes:
        specification = episodes[episode_id]
        geometry = geometries[specification["geometry_id"]]
        output = Path(args.out_dir) / f"{geometry['kind']}-mpc-teacher.gif"
        result = render_demo(geometry, specification, output)
        results.append(result)
        print(json.dumps(result))
    report = Path(args.out_dir) / "report.json"
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
