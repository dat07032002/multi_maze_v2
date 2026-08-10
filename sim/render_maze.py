"""Offscreen renders of the compiled maze, for checking it by eye.

The point of these is comparison against ``maze_design/maze_256x226.png``, which
is drawn straight from the layout. If a wall is missing, a hole is in the wrong
cell, or the pads have eaten a corridor, it shows up here in seconds -- and does
not show up at all in a policy that merely trains badly three milestones later.

    python3 -m sim.render_maze --out artifacts/sim_preview

Views come from cameras defined in the model (see ``_camera`` in
``mjcf_builder``), not from a free camera built here, because a free camera
aimed straight down has a degenerate up-vector and renders black rather than
raising.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
from PIL import Image

from .mjcf_builder import build_mjcf, load_layout, load_parameters

VIEWS = {
    "top_down": ("top", 1300, 1150),
    "isometric": ("iso", 1400, 1050),
    "hole_closeup": ("hole_closeup", 1200, 900),
    "start_closeup": ("start_closeup", 1200, 900),
}


def render_all(out_dir: Path, layout=None, params=None) -> list[Path]:
    layout = layout if layout is not None else load_layout()
    params = params if params is not None else load_parameters()
    model = mujoco.MjModel.from_xml_string(build_mjcf(layout, params))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, (camera, width, height) in VIEWS.items():
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            renderer.update_scene(data, camera=camera)
            path = out_dir / f"{stem}.png"
            Image.fromarray(renderer.render()).save(path)
            written.append(path)

    W, H = layout["board_width"], layout["board_height"]
    print(f"board {W * 1000:.0f} x {H * 1000:.0f} mm, {model.ngeom} geoms")
    for p in written:
        print(f"  wrote {p}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="artifacts/sim_preview")
    args = parser.parse_args()
    render_all(Path(args.out))


if __name__ == "__main__":
    main()
