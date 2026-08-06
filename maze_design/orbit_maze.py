#!/usr/bin/env python3
"""Render an orbiting view of a maze as a GIF.

The interactive MuJoCo viewer comes up black on this machine (GLFW gets a
Wayland session and libEGL cannot create a dri2 screen), while the offscreen
EGL path renders fine. This gives the same sense of the 3-D shape without
needing a working GUI.

Usage:
    python3 orbit_maze.py [layout.json] [--frames 36] [--elevation -35]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

MULTI_MAZE = os.path.expanduser("~/Desktop/TAG/Multi-maze")
sys.path.insert(0, os.path.join(MULTI_MAZE, "tag_mujoco"))

import mujoco  # noqa: E402

from maze_layout import load_json_layout  # noqa: E402
from model_builder import build_mjcf      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("layout", nargs="?",
                        default=os.path.join(HERE, "maze_v1.json"))
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--elevation", type=float, default=-38.0)
    parser.add_argument("--width", type=int, default=620)
    parser.add_argument("--height", type=int, default=560)
    parser.add_argument("--out", default=os.path.join(HERE, "maze_v1_orbit.gif"))
    args = parser.parse_args()

    layout = load_json_layout(Path(args.layout))
    model = mujoco.MjModel.from_xml_string(build_mjcf(layout))
    data = mujoco.MjData(model)

    start = layout.get("start_planned") or layout["waypoints"][0]
    ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    if ball != -1:
        adr = model.jnt_qposadr[model.body_jntadr[ball]]
        data.qpos[adr:adr + 3] = [
            start[0] - layout["board_width"] / 2.0,
            start[1] - layout["board_height"] / 2.0,
            layout["ball_radius"] + 0.001,
        ]
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    positions = data.geom_xpos
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    centre = (lo + hi) / 2.0
    distance = float(np.linalg.norm(hi - lo)) * 1.25

    frames = []
    for index in range(args.frames):
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = centre
        camera.distance = distance
        camera.azimuth = 360.0 * index / args.frames
        camera.elevation = args.elevation
        renderer.update_scene(data, camera=camera)
        frames.append(Image.fromarray(renderer.render()).convert(
            "P", palette=Image.ADAPTIVE, colors=96))
        if (index + 1) % 12 == 0:
            print(f"  {index+1}/{args.frames}", flush=True)

    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=90, loop=0, optimize=True)
    size = os.path.getsize(args.out) / 1024
    print(f"wrote {args.out} ({len(frames)} frames, {size:.0f} KB)")


if __name__ == "__main__":
    main()
