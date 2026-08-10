"""The maze's own route, made usable.

``maze_256x226.json`` already contains a valid path -- 118 waypoints, 1023 mm,
zero wall intersections, clearing every hole by at least 8.38 mm and every wall
by 1.88 mm. There is no reason to plan a new one and good reason not to: a
planner would only reproduce it less well.

What the stored waypoints are *not* is uniformly sampled. Spacing runs from
0.57 mm to 71.41 mm, a 125x range, with 14 segments over 20 mm. So a raw index
into that list means nothing physical: "five points ahead" is 2.8 mm in one
place and 350 mm in another. Everything here works off an **arc-length
resampling** onto a uniform grid, and nothing downstream should ever index the
original list.

Clearance is carried per point because the maze does not afford the same room
everywhere. At its tightest the route passes 1.88 mm from a wall; in an open
cell it has 10 mm. A cross-track error of 2 mm means "about to touch" in the
first case and "fine" in the second, and a cost function that cannot tell them
apart is weakest exactly where precision matters most.
"""
from __future__ import annotations

import numpy as np

from .mjcf_builder import load_layout, load_parameters, wall_rects


def _resample(points: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Uniform arc-length resampling of a polyline."""
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
    total = float(cumulative[-1])
    n = int(round(total / spacing)) + 1
    s = np.linspace(0.0, total, n)
    return (np.column_stack([np.interp(s, cumulative, points[:, axis])
                             for axis in (0, 1)]), s)


def _point_to_rect_distance(points: np.ndarray, rect) -> np.ndarray:
    x0, y0, x1, y1 = rect
    dx = np.maximum(np.maximum(x0 - points[:, 0], points[:, 0] - x1), 0.0)
    dy = np.maximum(np.maximum(y0 - points[:, 1], points[:, 1] - y1), 0.0)
    return np.hypot(dx, dy)


class Route:
    """Arc-length parameterised route with per-point clearance."""

    def __init__(self, layout: dict | None = None, params: dict | None = None,
                 spacing: float = 0.002):
        self.layout = layout if layout is not None else load_layout()
        params = params if params is not None else load_parameters()
        self.ball_radius = params["ball.radius"]
        self.spacing = spacing

        raw = np.asarray(self.layout["waypoints"], dtype=float)
        self.points, self.s = _resample(raw, spacing)
        self.length = float(self.s[-1])
        # The realised spacing, not the requested one. The route is divided into
        # a whole number of equal steps, so 2 mm over 1022.9 mm actually gives
        # 2.00176 mm. ``index_at`` divides by this, and using the requested
        # value instead drifts by nearly a millimetre by the end of the route.
        self.spacing = float(self.s[1] - self.s[0])

        tangents = np.gradient(self.points, axis=0)
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        self.tangents = tangents / np.maximum(norms, 1e-12)
        # Left-hand normal, so cross-track error is signed rather than absolute.
        self.normals = np.column_stack([-self.tangents[:, 1], self.tangents[:, 0]])

        self.clearance = self._compute_clearance()

    # ---- geometry ------------------------------------------------------
    def _compute_clearance(self) -> np.ndarray:
        """Room the ball has at each route point, to walls and to hole edges."""
        gaps = np.full(len(self.points), np.inf)
        for rect in wall_rects(self.layout):
            gaps = np.minimum(gaps, _point_to_rect_distance(self.points, rect))
        for (hx, hy), radius in zip(self.layout["holes"],
                                    self.layout["hole_radii"]):
            centre_distance = np.linalg.norm(self.points - [hx, hy], axis=1)
            gaps = np.minimum(gaps, centre_distance - radius)
        return gaps - self.ball_radius

    def project(self, xy) -> tuple[float, float, int]:
        """Arc length, signed cross-track error, and index of the nearest point.

        Refined onto the adjacent segments rather than snapped to the nearest
        sample: at 2 mm spacing, snapping alone would quantise cross-track error
        by up to 1 mm, which is half the clearance at the tightest point on this
        route.
        """
        point = np.asarray(xy, dtype=float)[:2]
        index = int(np.argmin(np.linalg.norm(self.points - point, axis=1)))

        best = (self.s[index], point - self.points[index], index)
        best_distance = np.linalg.norm(best[1])
        for other in (index - 1, index + 1):
            if not 0 <= other < len(self.points):
                continue
            lo, hi = min(index, other), max(index, other)
            a, b = self.points[lo], self.points[hi]
            segment = b - a
            length_sq = float(segment @ segment)
            if length_sq <= 0.0:
                continue
            t = float(np.clip((point - a) @ segment / length_sq, 0.0, 1.0))
            foot = a + t * segment
            distance = float(np.linalg.norm(point - foot))
            if distance < best_distance:
                best_distance = distance
                best = (float(self.s[lo] + t * (self.s[hi] - self.s[lo])),
                        point - foot, lo)

        offset = best[1]
        cross = float(offset @ self.normals[best[2]])
        return float(best[0]), cross, best[2]

    def index_at(self, s: float) -> int:
        return int(np.clip(round(s / self.spacing), 0, len(self.points) - 1))

    def point_at(self, s: float) -> np.ndarray:
        return self.points[self.index_at(s)]

    def clearance_at(self, s: float) -> float:
        return float(self.clearance[self.index_at(s)])

    def lookahead(self, s: float, spacing: float = 0.012,
                  count: int = 5) -> np.ndarray:
        """``count`` points ahead along the route, at fixed arc-length spacing.

        Clamped to the goal, so the horizon shortens as the ball arrives rather
        than running off the end and pointing nowhere.
        """
        targets = s + spacing * np.arange(1, count + 1)
        return np.stack([self.point_at(min(t, self.length)) for t in targets])

    @property
    def goal(self) -> np.ndarray:
        return self.points[-1]

    @property
    def start(self) -> np.ndarray:
        return self.points[0]
