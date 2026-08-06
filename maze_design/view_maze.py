#!/usr/bin/env python3
"""Build a maze layout in MuJoCo and view it.

Tries the interactive viewer, and always writes an offscreen render as well.
On this machine the interactive viewer has come up black before -- GLFW gets a
Wayland session and libEGL fails to create a dri2 screen -- while the offscreen
EGL path works reliably. So the render is the dependable output and the window
is a bonus.

Usage:
    python3 view_maze.py [layout.json] [--no-window] [--tilt ALPHA BETA]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

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
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--tilt", type=float, nargs=2, metavar=("ALPHA", "BETA"),
                        default=(0.0, 0.0), help="board tilt in degrees")
    parser.add_argument("--out", default=os.path.join(HERE, "maze_v1_3d.png"))
    args = parser.parse_args()

    from pathlib import Path
    layout = load_json_layout(Path(args.layout))
    model = mujoco.MjModel.from_xml_string(build_mjcf(layout))
    data = mujoco.MjData(model)

    print(f"{os.path.basename(args.layout)}: "
          f"{len(layout['holes'])} holes, "
          f"{len(layout['walls_h'])}H/{len(layout['walls_v'])}V walls")
    print(f"model: {model.ngeom} geoms, {model.nbody} bodies")

    # Apply tilt, and drop the ball at the route start.
    for name, value in zip(("tilt_x", "tilt_y"), args.tilt):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint != -1:
            data.qpos[model.jnt_qposadr[joint]] = math.radians(value)

    start = layout.get("start_planned") or layout["waypoints"][0]
    ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    if ball != -1:
        adr = model.jnt_qposadr[model.body_jntadr[ball]]
        # The MJCF centres the board on the origin; the layout uses a
        # lower-left origin, so shift by half the board.
        data.qpos[adr:adr + 3] = [
            start[0] - layout["board_width"] / 2.0,
            start[1] - layout["board_height"] / 2.0,
            layout["ball_radius"] + 0.001,
        ]
    mujoco.mj_forward(model, data)

    # Offscreen render, framed on the actual geometry.
    try:
        renderer = mujoco.Renderer(model, height=780, width=880)
        positions = data.geom_xpos
        lo, hi = positions.min(axis=0), positions.max(axis=0)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = (lo + hi) / 2.0
        camera.distance = float(np.linalg.norm(hi - lo)) * 1.15
        camera.azimuth, camera.elevation = 90, -89
        renderer.update_scene(data, camera=camera)
        from PIL import Image
        Image.fromarray(renderer.render()).save(args.out)
        print(f"render -> {args.out}")
    except Exception as exc:  # noqa: BLE001
        print(f"offscreen render failed: {exc}")

    if not args.no_window:
        print("\nopening the interactive viewer; close the window to exit")
        try:
            # `from ... import` rather than `import mujoco.viewer`: the latter
            # rebinds `mujoco` as a function local and shadows the module.
            from mujoco import viewer as mj_viewer
            mj_viewer.launch(model, data)
        except Exception as exc:  # noqa: BLE001
            print(f"viewer unavailable: {exc}")


if __name__ == "__main__":
    main()
