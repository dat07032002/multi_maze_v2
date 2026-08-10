#!/usr/bin/env python3
"""Fit the physical marble model from a real-hardware controller CSV log."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tag_vision.control.ball_dynamics import fit_ball_dynamics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def local_derivatives(times: np.ndarray, positions: np.ndarray,
                      half_window: int = 5):
    """Quadratic local fits provide less noisy velocity/acceleration than diff."""
    velocity = np.full_like(positions, np.nan)
    acceleration = np.full_like(positions, np.nan)
    for index in range(half_window, len(times) - half_window):
        selected = slice(index - half_window, index + half_window + 1)
        local_t = times[selected] - times[index]
        if np.ptp(local_t) < 0.08:
            continue
        for axis in range(2):
            coefficient = np.polyfit(local_t, positions[selected, axis], 2)
            velocity[index, axis] = coefficient[1]
            acceleration[index, axis] = 2.0 * coefficient[0]
    return velocity, acceleration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "calib/ball_dynamics.json")
    parser.add_argument("--half-window", type=int, default=5)
    parser.add_argument("--max-speed", type=float, default=700.0)
    parser.add_argument("--max-acceleration", type=float, default=5000.0)
    args = parser.parse_args()
    with args.log.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    use_raw = bool(rows and "raw_x_mm" in rows[0])
    x_key, y_key = (("raw_x_mm", "raw_y_mm") if use_raw
                    else ("x_mm", "y_mm"))
    valid = []
    for row in rows:
        try:
            values = [float(row[key]) for key in (
                "time_s", x_key, y_key, "fused_alpha_deg",
                "fused_beta_deg")]
        except (KeyError, TypeError, ValueError):
            continue
        if row.get("ball_visible") == "1" and np.all(np.isfinite(values)):
            valid.append(row)
    if len(valid) < 30:
        print("Need at least 30 ball-visible fused-angle samples")
        return 1
    times = np.array([float(row["time_s"]) for row in valid])
    position = np.array([[float(row[x_key]), float(row[y_key])]
                         for row in valid])
    tilt = np.array([[float(row["fused_alpha_deg"]),
                      float(row["fused_beta_deg"])] for row in valid])
    active = np.array([
        row.get("phase", "").startswith("identify_active") for row in valid])
    target = np.array([[float(row["target_alpha_deg"]),
                        float(row["target_beta_deg"])] for row in valid])
    target_span = np.ptp(target, axis=0)
    # A numerically invertible regression is not evidence of both actuator
    # directions if camera noise supplied most of the apparent angle motion.
    if np.any(target_span < 0.5):
        print("Rejected: commanded tilt did not excite both axes by >=0.5 deg; "
              f"spans were {target_span.tolist()}")
        return 2
    active_count = int(np.count_nonzero(active))
    if active_count < 200:
        print(f"Rejected: only {active_count} safe active samples; need >=200")
        return 2
    # Split tracks at camera gaps; derivatives must never bridge a dropout or a
    # reload jump. Each contiguous segment is differentiated independently.
    breaks = np.flatnonzero(np.diff(times) > 0.12) + 1
    segments = np.split(np.arange(len(times)), breaks)
    all_velocity, all_acceleration, all_tilt = [], [], []
    for indices in segments:
        if len(indices) < 2 * args.half_window + 3:
            continue
        velocity, acceleration = local_derivatives(
            times[indices], position[indices], args.half_window)
        keep = (np.all(np.isfinite(velocity), axis=1)
                & np.all(np.isfinite(acceleration), axis=1)
                & active[indices]
                & (np.linalg.norm(velocity, axis=1) <= args.max_speed)
                & (np.linalg.norm(acceleration, axis=1)
                   <= args.max_acceleration))
        all_velocity.append(velocity[keep])
        all_acceleration.append(acceleration[keep])
        all_tilt.append(tilt[indices][keep])
    if not all_velocity or sum(map(len, all_velocity)) < 30:
        print("Not enough contiguous motion after derivative filtering")
        return 1
    model = fit_ball_dynamics(np.vstack(all_velocity), np.vstack(all_tilt),
                              np.vstack(all_acceleration), ridge=1e-3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"model: {args.out}")
    print("acceleration per tilt [mm/s^2 per deg], rows XY, columns alpha/beta:")
    print(model.acceleration_per_tilt)
    print("velocity damping [1/s]:")
    print(model.velocity_damping)
    print(f"fit RMSE: {model.fit_rmse_mm_s2:.1f} mm/s^2")
    condition = np.linalg.cond(model.acceleration_per_tilt)
    minimum_authority = float(np.min(np.linalg.svd(
        model.acceleration_per_tilt, compute_uv=False)))
    damping_stability = float(np.min(np.linalg.eigvalsh(
        0.5 * (model.velocity_damping + model.velocity_damping.T))))
    print(f"tilt-map condition number: {condition:.2f}")
    print(f"minimum identified tilt authority: {minimum_authority:.1f} "
          "mm/s^2/deg")
    print(f"minimum symmetric damping eigenvalue: {damping_stability:.2f} 1/s")
    if (condition > 8 or model.fit_rmse_mm_s2 > 500
            or minimum_authority < 10 or damping_stability < -2):
        print("WARNING: weak model; collect motion on both axes before reset mode")
        args.out.unlink(missing_ok=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
