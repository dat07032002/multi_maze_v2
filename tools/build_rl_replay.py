#!/usr/bin/env python3
"""Build a typed RL replay checkpoint from real ball-control logs."""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tag_vision.rl.import_logs import import_ball_control_logs  # noqa: E402
from tag_vision.rl.replay import ReplayBuffer  # noqa: E402
from tag_vision.rl.task import MazeTask  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*", default=[
        str(ROOT / "artifacts/ball_control/*/samples.csv")])
    parser.add_argument("--map-dir", type=Path, default=ROOT /
                        "artifacts/camera_maze/20260810_125246")
    parser.add_argument("--capacity", type=int, default=100_000)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "artifacts/rl/replay.npz")
    args = parser.parse_args()
    paths = []
    for pattern in args.logs:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    paths = [Path(path) for path in paths if Path(path).is_file()]
    if not paths:
        print("No input CSV logs found")
        return 1
    task = MazeTask.load(args.map_dir / "map.json",
                         args.map_dir / "occupied_inflated.png")
    replay = ReplayBuffer(args.capacity, task.spec.size)
    summary = import_ball_control_logs(paths, task, replay)
    if replay.size < 100:
        print(f"Only {replay.size} transitions; refusing a weak replay checkpoint")
        return 2
    replay.save(args.out)
    print(f"replay: {args.out}")
    print(f"files {summary.files}, rows {summary.rows}, valid "
          f"{summary.valid_rows}, segments {summary.segments}, "
          f"transitions {summary.transitions}")
    print(f"observation size {task.spec.size}, action size 2, capacity "
          f"{replay.capacity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
