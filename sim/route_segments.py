"""Classify the planned maze route into straight and turning segments.

Classification is geometric rather than waypoint-based.  The stored route has
highly non-uniform waypoint spacing, so turn direction is measured from the
arc-length-resampled tangents supplied by :class:`sim.route.Route`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from sim.route import Route


@dataclass(frozen=True)
class RouteSegment:
    """One contiguous route region with a common maneuver class."""

    number: int
    kind: str
    start_index: int
    end_index: int  # exclusive
    start_s: float
    end_s: float
    mean_turn_deg: float
    peak_turn_deg: float
    min_clearance: float

    @property
    def length(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict:
        values = asdict(self)
        values["length"] = self.length
        return values


def signed_turn_degrees(route: Route, half_window_m: float = 0.006,
                        smoothing_samples: int = 3) -> np.ndarray:
    """Return signed tangent change over a local arc-length window.

    Positive is a left turn and negative is a right turn.  A 6 mm half-window
    gives a 12 mm measurement horizon, matching one policy look-ahead interval;
    the short three-sample mean removes classification flicker at resampling
    boundaries without erasing the maze's short corners.
    """
    if half_window_m <= 0:
        raise ValueError("half_window_m must be positive")
    if smoothing_samples < 1 or smoothing_samples % 2 == 0:
        raise ValueError("smoothing_samples must be a positive odd integer")

    span = max(1, int(round(half_window_m / route.spacing)))
    indices = np.arange(len(route.points))
    before = route.tangents[np.maximum(0, indices - span)]
    after = route.tangents[np.minimum(len(route.points) - 1, indices + span)]
    cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    dot = np.sum(before * after, axis=1)
    values = np.degrees(np.arctan2(cross, dot))

    if smoothing_samples > 1:
        pad = smoothing_samples // 2
        padded = np.pad(values, pad, mode="edge")
        values = np.convolve(
            padded, np.ones(smoothing_samples) / smoothing_samples,
            mode="valid")
    return values


def classify_route(route: Route, straight_max_deg: float = 10.0,
                   sharp_min_deg: float = 30.0,
                   half_window_m: float = 0.006,
                   smoothing_samples: int = 3) -> tuple[np.ndarray,
                                                        list[RouteSegment]]:
    """Classify and group the route into contiguous maneuver regions.

    A turn region is contiguous while its signed local turn remains on the
    same side.  Gentle shoulders around a sharp core therefore belong to the
    same sharp maneuver instead of becoming several tiny segments.
    """
    if not 0 < straight_max_deg < sharp_min_deg:
        raise ValueError("require 0 < straight_max_deg < sharp_min_deg")

    turns = signed_turn_degrees(
        route, half_window_m=half_window_m,
        smoothing_samples=smoothing_samples)
    direction = np.zeros(len(turns), dtype=np.int8)
    direction[turns >= straight_max_deg] = 1
    direction[turns <= -straight_max_deg] = -1

    starts = np.r_[0, np.flatnonzero(direction[1:] != direction[:-1]) + 1]
    ends = np.r_[starts[1:], len(direction)]
    boundaries = np.empty(len(direction) + 1, dtype=float)
    boundaries[0] = 0.0
    boundaries[-1] = route.length
    boundaries[1:-1] = (route.s[:-1] + route.s[1:]) / 2.0

    segments = []
    for number, (start, end) in enumerate(zip(starts, ends), start=1):
        local = turns[start:end]
        peak = float(np.max(np.abs(local)))
        side = int(direction[start])
        if side == 0:
            kind = "straight"
        else:
            strength = "sharp" if peak >= sharp_min_deg else "gentle"
            handedness = "left" if side > 0 else "right"
            kind = f"{strength}_{handedness}"
        segments.append(RouteSegment(
            number=number,
            kind=kind,
            start_index=int(start),
            end_index=int(end),
            start_s=float(boundaries[start]),
            end_s=float(boundaries[end]),
            mean_turn_deg=float(np.mean(local)),
            peak_turn_deg=peak,
            min_clearance=float(np.min(route.clearance[start:end])),
        ))
    return turns, segments


def summary(segments: list[RouteSegment]) -> dict[str, int]:
    """Count segments by maneuver class."""
    return {
        kind: sum(segment.kind == kind for segment in segments)
        for kind in ("straight", "gentle_left", "gentle_right",
                     "sharp_left", "sharp_right")
    }


if __name__ == "__main__":
    route = Route()
    _, found = classify_route(route)
    print(f"{len(found)} route segments over {route.length * 1000:.1f} mm")
    for kind, count in summary(found).items():
        print(f"  {kind:<13} {count}")
    print()
    for segment in found:
        print(f"{segment.number:02d} {segment.kind:<13} "
              f"{segment.start_s * 1000:7.1f}-{segment.end_s * 1000:7.1f} mm "
              f"peak {segment.peak_turn_deg:5.1f} deg  "
              f"clearance {segment.min_clearance * 1000:5.2f} mm")
