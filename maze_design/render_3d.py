#!/usr/bin/env python3
"""Render a binary STL to a shaded 3-D image. No 3-D libraries required.

The project has no mesh viewer available -- trimesh, open3d, pyvista and mujoco
are all absent, and matplotlib's Poly3DCollection sorts whole polygons by depth,
which puts walls through the floor on a model like this one. So this is a small
z-buffer rasteriser: correct per-pixel depth, flat shading, supersampled.

Triangles are coloured by height so the structure reads at a glance: the
playing surface, the wall sides, and the wall/pad tops each land in a different
band. A recessed tag pocket, when the layout has one, gets its own colour too.

Usage:
    python3 maze_design/render_3d.py maze_design/maze_256x226.stl
    python3 maze_design/render_3d.py model.stl --azimuth 45 --elevation 30
    python3 maze_design/render_3d.py model.stl --out view.png --width 1600
"""
from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def load_stl(path: Path) -> np.ndarray:
    """Return an (N, 3, 3) array of triangle vertices from a binary STL."""
    data = path.read_bytes()
    count = struct.unpack("<I", data[80:84])[0]
    expected = 84 + 50 * count
    if len(data) != expected:
        raise ValueError(
            f"{path} is not a binary STL of {count} triangles "
            f"(expected {expected} bytes, got {len(data)})"
        )
    records = np.frombuffer(data[84:], dtype=np.uint8).reshape(count, 50)
    # Bytes 12..48 of each record are the nine vertex floats; the leading three
    # are the stored normal, which we recompute rather than trust.
    vertices = records[:, 12:48].copy().view("<f4").reshape(count, 3, 3)
    return vertices.astype(np.float64)


def look_at(azimuth_deg: float, elevation_deg: float):
    """Camera basis looking at the origin from the given spherical angles."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    eye = np.array([
        math.cos(el) * math.cos(az),
        math.cos(el) * math.sin(az),
        math.sin(el),
    ])
    forward = -eye
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return eye, np.stack([right, up, -forward])


def height_colour(z: float, floor_mm: float, top_mm: float,
                  pocket_mm: float | None) -> np.ndarray:
    """Colour by absolute height, against the real surface levels.

    Comparing against the mesh's minimum z instead of the actual floor
    thickness put the playing surface (z = floor_mm) into the wall band and
    made the whole model one colour.
    """
    tolerance = 0.05
    if abs(z - floor_mm) < tolerance:
        return np.array([236.0, 220.0, 184.0])      # playing surface
    if pocket_mm is not None and abs(z - pocket_mm) < tolerance:
        return np.array([196.0, 208.0, 232.0])      # tag pocket floor
    if abs(z - top_mm) < tolerance:
        return np.array([158.0, 116.0, 62.0])       # wall and pad tops
    if z < floor_mm:
        return np.array([96.0, 68.0, 34.0])         # underside and board edge
    return np.array([124.0, 88.0, 44.0])            # wall sides


def render(triangles: np.ndarray, width: int, height: int,
           azimuth: float, elevation: float, supersample: int,
           floor_mm: float = 3.0, pocket_mm: float | None = 8.0) -> Image.Image:
    W, H = width * supersample, height * supersample

    centre = (triangles.reshape(-1, 3).min(0) + triangles.reshape(-1, 3).max(0)) / 2.0
    centred = triangles - centre
    eye, basis = look_at(azimuth, elevation)

    camera = centred @ basis.T          # (N, 3, 3): x right, y up, z toward viewer
    extent = np.abs(camera[..., :2]).max()
    scale = 0.46 * min(W, H) / extent

    sx = camera[..., 0] * scale + W / 2.0
    sy = -camera[..., 1] * scale + H / 2.0
    depth = camera[..., 2]

    # Flat shading from the true geometric normal, lit slightly off the camera
    # axis so vertical faces stay distinguishable from horizontal ones.
    edge1 = centred[:, 1] - centred[:, 0]
    edge2 = centred[:, 2] - centred[:, 0]
    normals = np.cross(edge1, edge2)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals),
                        where=lengths > 0)
    light = eye + np.array([0.25, 0.15, 0.55])
    light /= np.linalg.norm(light)
    lambert = np.clip(normals @ light, 0.0, 1.0)
    shade = 0.28 + 0.72 * lambert

    zs = triangles[..., 2]
    top_mm = zs.max()

    frame = np.full((H, W, 3), 252.0)
    zbuffer = np.full((H, W), -np.inf)

    order = np.argsort(-depth.mean(axis=1))     # rough front-to-back
    for index in order:
        x0, x1, x2 = sx[index]
        y0, y1, y2 = sy[index]

        min_x = max(int(math.floor(min(x0, x1, x2))), 0)
        max_x = min(int(math.ceil(max(x0, x1, x2))), W - 1)
        min_y = max(int(math.floor(min(y0, y1, y2))), 0)
        max_y = min(int(math.ceil(max(y0, y1, y2))), H - 1)
        if min_x > max_x or min_y > max_y:
            continue

        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-12:
            continue

        ys, xs_grid = np.mgrid[min_y:max_y + 1, min_x:max_x + 1]
        px = xs_grid + 0.5
        py = ys + 0.5

        w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) / area
        w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        z = w0 * depth[index, 0] + w1 * depth[index, 1] + w2 * depth[index, 2]
        window = zbuffer[min_y:max_y + 1, min_x:max_x + 1]
        visible = inside & (z > window)
        if not visible.any():
            continue

        colour = height_colour(zs[index].mean(), floor_mm, top_mm,
                               pocket_mm) * shade[index]
        window[visible] = z[visible]
        frame[min_y:max_y + 1, min_x:max_x + 1][visible] = colour

    image = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))
    if supersample > 1:
        image = image.resize((width, height), Image.LANCZOS)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stl")
    parser.add_argument("--out", default=None)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--azimuth", type=float, default=-60.0)
    parser.add_argument("--elevation", type=float, default=38.0)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--floor-mm", type=float, default=3.0,
                        help="height of the playing surface (default: 3.0)")
    parser.add_argument("--pocket-mm", type=float, default=8.0,
                        help="height of the tag pocket floor (default: 8.0)")
    args = parser.parse_args()

    path = Path(args.stl)
    triangles = load_stl(path)
    lo = triangles.reshape(-1, 3).min(0)
    hi = triangles.reshape(-1, 3).max(0)
    print(f"{path.name}: {len(triangles)} triangles, "
          f"bbox {lo.round(2)} .. {hi.round(2)}")

    image = render(triangles, args.width, args.height,
                   args.azimuth, args.elevation, args.supersample,
                   args.floor_mm, args.pocket_mm)
    out = Path(args.out) if args.out else path.with_name(path.stem + "_3d.png")
    image.save(out)
    print(f"wrote {out} ({image.size[0]}x{image.size[1]})")


if __name__ == "__main__":
    main()
