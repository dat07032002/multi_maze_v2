"""Convert audited real-hardware CSV logs into 5 Hz RL transitions."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .replay import ReplayBuffer
from .task import MazeTask


@dataclass(frozen=True)
class ImportSummary:
    files: int
    rows: int
    valid_rows: int
    transitions: int
    segments: int


def _number(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def import_ball_control_logs(paths, task: MazeTask, replay: ReplayBuffer, *,
                             sample_period_s: float = 0.20,
                             max_gap_s: float = 0.35,
                             max_position_jump_mm: float = 35.0) -> ImportSummary:
    files = rows_seen = valid_seen = transitions = segments = 0
    for path_value in paths:
        path = Path(path_value)
        files += 1
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows_seen += len(rows)
        valid = []
        for row in rows:
            keys = ("time_s", "x_mm", "y_mm", "vx_mm_s", "vy_mm_s",
                    "fused_alpha_deg", "fused_beta_deg",
                    "target_alpha_deg", "target_beta_deg")
            values = [_number(row, key) for key in keys]
            if row.get("ball_visible") != "1" or any(
                    value is None for value in values):
                continue
            valid.append((row, np.asarray(values, dtype=np.float64)))
        valid_seen += len(valid)
        retained = []
        last_time = -math.inf
        for item in valid:
            timestamp = item[1][0]
            if timestamp - last_time >= sample_period_s:
                retained.append(item)
                last_time = timestamp
        previous = None
        previous_observation = None
        previous_route = None
        previous_action = None
        for row, values in retained:
            timestamp, x, y, vx, vy, alpha, beta, action_a, action_b = values
            position = np.array([x, y])
            action = np.array([action_a, action_b], dtype=np.float32)
            new_segment = previous is None
            if previous is not None:
                dt = timestamp - previous[0]
                jump = float(np.linalg.norm(position - previous[1]))
                new_segment = dt > max_gap_s or dt <= 0 or jump > max_position_jump_mm
            if new_segment:
                segments += 1
                task.reset(timestamp, position)
                angle_rate = np.zeros(2)
                previous_observation = previous_route = previous_action = None
            else:
                dt = timestamp - previous[0]
                angle_rate = (np.array([alpha, beta]) - previous[2]) / dt
            observation, route = task.observation(
                position_mm=position, velocity_mm_s=[vx, vy],
                angles_deg=[alpha, beta], angle_rates_deg_s=angle_rate,
                previous_action_deg=action,
                stuck=math.hypot(vx, vy) < 3.0)
            if previous_observation is not None:
                result = task.step_result(
                    observation, route, timestamp_s=timestamp)
                replay.add(previous_observation, previous_action,
                           result.reward, observation,
                           result.terminated, result.truncated, False)
                transitions += 1
            previous = (timestamp, position, np.array([alpha, beta]))
            previous_observation = observation
            previous_route = route
            previous_action = action
    return ImportSummary(files, rows_seen, valid_seen, transitions, segments)
