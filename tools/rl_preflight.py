#!/usr/bin/env python3
"""Verify the local machine is ready for GPU real-hardware RL."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tag_vision.core.board_geometry import BoardGeometry  # noqa: E402
from tag_vision.hardware.imu import find_imu_port  # noqa: E402
from tag_vision.hardware.sts3215 import find_servo_port  # noqa: E402
from tag_vision.rl.task import MazeTask  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def command_output(command):
    try:
        result = subprocess.run(command, check=False, text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=5)
        return result.returncode, result.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return 127, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", type=Path, default=ROOT /
                        "artifacts/camera_maze/20260810_125246")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    checks = {}
    returncode, nvidia = command_output([
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader"])
    checks["nvidia_smi"] = {"ok": returncode == 0, "detail": nvidia}
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        detail = {"version": torch.__version__, "cuda": cuda,
                  "cuda_version": torch.version.cuda,
                  "device": torch.cuda.get_device_name(0) if cuda else None}
        checks["torch"] = {"ok": cuda, "detail": detail}
    except Exception as exc:  # noqa: BLE001
        checks["torch"] = {"ok": False, "detail": repr(exc)}
    try:
        task = MazeTask.load(args.map_dir / "map.json",
                             args.map_dir / "occupied_inflated.png")
        checks["maze"] = {"ok": True, "detail": {
            "route_mm": task.route_length_mm,
            "observation_size": task.spec.size}}
    except Exception as exc:  # noqa: BLE001
        checks["maze"] = {"ok": False, "detail": repr(exc)}
    required = [
        ROOT / "calib/camera_calib.json", ROOT / "calib/board_tags.json",
        ROOT / "calib/board_zero.json", ROOT / "calib/imu_zero.json",
        ROOT / "calib/directional_motor.json", ROOT / "calib/reload_brake.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    checks["calibration"] = {"ok": not missing, "detail": missing or "present"}
    checks["ports"] = {"ok": Path(find_imu_port()).exists()
                       and Path(find_servo_port()).exists(),
                       "detail": {"imu": find_imu_port(),
                                  "servo": find_servo_port()}}
    try:
        BoardGeometry.load(ROOT / "calib/board_tags.json")
        checks["board_geometry"] = {"ok": True, "detail": "valid"}
    except Exception as exc:  # noqa: BLE001
        checks["board_geometry"] = {"ok": False, "detail": repr(exc)}
    healthy = all(item["ok"] for item in checks.values())
    if args.json:
        print(json.dumps({"healthy": healthy, "checks": checks}, indent=2))
    else:
        for name, item in checks.items():
            print(f"{'PASS' if item['ok'] else 'FAIL':4s} {name}: {item['detail']}")
        print("READY" if healthy else "NOT READY")
    return 0 if healthy else 2


if __name__ == "__main__":
    raise SystemExit(main())
