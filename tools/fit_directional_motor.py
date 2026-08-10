#!/usr/bin/env python3
"""Fit direction-dependent motor lookup tables from a fused keyboard run.

The run must visit integer-degree targets in both logical directions and dwell
at each target. The fitter uses the final settled second before the next
command, averages repeated count positions, and writes a candidate calibration
beside the run. It never overwrites ``calib/servo_calibration.json``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contract.servo_contract import ServoContract  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _number(value: str):
    if value in ("", "nan"):
        return math.nan
    return float(value)


def extract_dwells(run: Path, settle_lag_s: float, window_s: float) -> list[dict]:
    events = [json.loads(line) for line in
              (run / "commands.jsonl").read_text(encoding="utf-8").splitlines()]
    with (run / "samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = [{key: (value if key == "detection_mode" else _number(value))
                 for key, value in row.items()} for row in csv.DictReader(handle)]

    points = []
    for index, event in enumerate(events):
        action = event["action"]
        axis = ("alpha" if action.startswith("alpha") else
                "beta" if action.startswith("beta") else None)
        if axis is None:
            continue
        target = float(event[f"target_{axis}_deg"])
        if abs(target - round(target)) > 1e-6:
            continue
        end = (float(events[index + 1]["host_time"])
               if index + 1 < len(events) else float(rows[-1]["host_time"]))
        start = float(event["host_time"])
        if end - start < settle_lag_s + 0.2:
            continue  # this was the first half of a one-degree key pair
        measure_start = max(start + settle_lag_s, end - window_s)
        window = [row for row in rows
                  if measure_start <= row["host_time"] < end
                  and row["settled"] == 1.0]
        values = [row[f"fused_{axis}_deg"] for row in window
                  if math.isfinite(row[f"fused_{axis}_deg"])]
        if len(values) < 5:
            continue
        other = "beta" if axis == "alpha" else "alpha"
        counts_key = "servo2_counts" if axis == "alpha" else "servo1_counts"
        points.append({
            "axis": axis,
            "direction": "up" if action.endswith("up") else "down",
            "requested_deg": target,
            "counts": float(np.median([row[counts_key] for row in window])),
            "measured_deg": float(np.mean(values)),
            "sd_deg": float(np.std(values)),
            "cross_axis_deg": float(np.mean([
                row[f"fused_{other}_deg"] for row in window])),
            "samples": len(values),
        })
    return points


def branch(points: list[dict]) -> tuple[dict, dict]:
    grouped: dict[float, list[dict]] = defaultdict(list)
    for point in points:
        grouped[point["counts"]].append(point)
    combined = []
    for counts, items in grouped.items():
        combined.append({
            "counts": counts,
            "angle": float(np.mean([p["measured_deg"] for p in items])),
            "repeat_span_deg": float(np.ptp([p["measured_deg"] for p in items])),
        })
    combined.sort(key=lambda item: item["angle"])
    if len(combined) < 4:
        raise ValueError("each direction needs at least four settled positions")
    angles = np.array([item["angle"] for item in combined])
    counts = np.array([item["counts"] for item in combined])
    linear = np.polyfit(counts, angles, 1)
    quadratic = np.polyfit(counts, angles, 2)
    metrics = {
        "linear_counts_per_deg": float(abs(1.0 / linear[0])),
        "linear_zero_count": float(-linear[1] / linear[0]),
        "linear_rms_deg": float(np.sqrt(np.mean(
            (angles - np.polyval(linear, counts)) ** 2))),
        "quadratic_angle_from_counts": quadratic.tolist(),
        "quadratic_rms_deg": float(np.sqrt(np.mean(
            (angles - np.polyval(quadratic, counts)) ** 2))),
        "max_repeat_span_deg": max(item["repeat_span_deg"] for item in combined),
    }
    table = {"angles_deg": angles.tolist(), "counts": counts.tolist()}
    return table, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--servo-calibration", type=Path,
                        default=ROOT / "calib/servo_calibration.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--settle-lag", type=float, default=0.8)
    parser.add_argument("--window", type=float, default=1.0)
    args = parser.parse_args()
    contract = ServoContract.from_json(args.servo_calibration)
    manifest = json.loads((args.run / "manifest.json").read_text())
    level = {int(key): int(value) for key, value in manifest["base_counts"].items()}
    points = extract_dwells(args.run, args.settle_lag, args.window)

    payload = {
        "version": "tag_directional_motor_v1",
        "source_run": str(args.run.resolve()),
        "method": "settled fused camera+IMU, piecewise-linear inverse by direction",
    }
    for name, calibration in zip(("alpha", "beta"), contract.axes):
        selected = [point for point in points if point["axis"] == name]
        payload[name] = {
            "servo_id": calibration.servo_id,
            "level_counts": level[calibration.servo_id],
            "min_counts": calibration.min_counts,
            "max_counts": calibration.max_counts,
            "points": selected,
        }
        for direction in ("up", "down"):
            table, metrics = branch([
                point for point in selected if point["direction"] == direction])
            payload[name][direction] = table
            payload[name][f"{direction}_fit"] = metrics

        print(f"{name} (servo {calibration.servo_id})")
        for direction in ("up", "down"):
            fit = payload[name][f"{direction}_fit"]
            print(f"  {direction:4s}: {fit['linear_counts_per_deg']:.1f} counts/deg, "
                  f"quadratic RMS {fit['quadratic_rms_deg']:.3f} deg")

    def mean_slope(axis: str, value: str) -> float:
        slopes = []
        for direction in ("up", "down"):
            selected = [point for point in points
                        if point["axis"] == axis
                        and point["direction"] == direction]
            slopes.append(float(np.polyfit(
                [point["counts"] for point in selected],
                [point[value] for point in selected], 1)[0]))
        return float(np.mean(slopes))

    # Columns follow logical alpha-servo and beta-servo count changes.
    payload["jacobian_deg_per_count"] = [
        [mean_slope("alpha", "measured_deg"),
         mean_slope("beta", "cross_axis_deg")],
        [mean_slope("alpha", "cross_axis_deg"),
         mean_slope("beta", "measured_deg")],
    ]

    output = args.output or args.run / "directional_motor_candidate.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"candidate written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
