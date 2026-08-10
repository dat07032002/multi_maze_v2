"""Run the analytic baseline controller over the maze and report.

    python -m sim.run_baseline --seeds 10
    python -m sim.run_baseline --seeds 10 --render artifacts/baseline

Its route completion is the bar every later RL result is compared against, so
this needs to stay runnable and reproducible rather than living in a notebook.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from control.baseline import PurePursuitBaseline
from sim.mjcf_builder import load_layout, load_parameters
from sim.rollout import run_closed_loop
from sim.route import Route


def evaluate(seeds: int = 10, max_seconds: float = 60.0, **controller_kwargs):
    layout = load_layout()
    params = load_parameters()
    route = Route(layout, params)

    rows = []
    for seed in range(seeds):
        controller = PurePursuitBaseline(route, params["actuator.max_tilt"],
                                         **controller_kwargs)
        result = run_closed_loop(controller, layout=layout, params=params,
                                 seed=seed, max_seconds=max_seconds)
        track = result.track
        completion = route.project(track[-1])[0] / route.length
        cross = np.array([abs(route.project(p)[1]) for p in track])
        rows.append({
            "seed": seed,
            "reached_goal": result.reached_goal,
            "fell": result.fell,
            "completion": completion,
            "cross_mean": float(cross.mean()),
            "cross_max": float(cross.max()),
            "seconds": result.steps / 20.0,
            "result": result,
        })
    return rows, route


def summarise(rows) -> None:
    successes = [r for r in rows if r["reached_goal"]]
    print(f"  success        {len(successes)}/{len(rows)}")
    print(f"  fell in a hole {sum(r['fell'] for r in rows)}/{len(rows)}")
    print(f"  completion     {100 * np.mean([r['completion'] for r in rows]):.1f}% mean, "
          f"{100 * min(r['completion'] for r in rows):.1f}% worst")
    print(f"  cross-track    {1000 * np.mean([r['cross_mean'] for r in rows]):.2f} mm mean, "
          f"{1000 * max(r['cross_max'] for r in rows):.2f} mm worst")
    if successes:
        print(f"  time to goal   {np.mean([r['seconds'] for r in successes]):.1f} s mean "
              f"(budget 60 s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--render", default=None,
                        help="write a track overlay PNG here")
    args = parser.parse_args()

    rows, route = evaluate(args.seeds, args.seconds)
    print(f"analytic baseline over {args.seeds} seeds")
    summarise(rows)

    if args.render:
        _render_tracks(rows, route, Path(args.render))


def _render_tracks(rows, route, out_dir: Path) -> None:
    """Route in grey, every run's track over it, holes marked."""
    from PIL import Image, ImageDraw

    layout = route.layout
    W, H = layout["board_width"], layout["board_height"]
    scale = 4000.0
    width, height = int(W * scale), int(H * scale)
    image = Image.new("RGB", (width, height), (250, 249, 245))
    draw = ImageDraw.Draw(image)

    def px(point):
        return (point[0] * scale, height - point[1] * scale)

    from sim.mjcf_builder import wall_rects
    for x0, y0, x1, y1 in wall_rects(layout):
        draw.rectangle([px((x0, y1)), px((x1, y0))], fill=(90, 74, 58))
    for (hx, hy), r in zip(layout["holes"], layout["hole_radii"]):
        draw.ellipse([px((hx - r, hy + r)), px((hx + r, hy - r))], fill=(20, 20, 20))

    draw.line([px(p) for p in route.points], fill=(170, 170, 175), width=6)
    for row in rows:
        colour = (30, 130, 220) if row["reached_goal"] else (220, 70, 60)
        draw.line([px(p) for p in row["result"].track], fill=colour, width=3)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "baseline_tracks.png"
    image.save(path)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
