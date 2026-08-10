"""M2/M4: the route the maze already carries, and the resampling it needs.

The numbers pinned here were measured against the layout when the plan was
written. They are regression guards for future mazes as much as checks on this
one -- ``rescale_maze.hole_clearance_mm`` checks holes only and says so in its
own docstring, so wall clearance had never been validated at all before this.
"""
from __future__ import annotations

import numpy as np
import pytest

from sim.mjcf_builder import load_layout, wall_rects
from sim.route import Route


@pytest.fixture(scope="module")
def route():
    return Route()


@pytest.fixture(scope="module")
def layout():
    return load_layout()


# -- resampling --------------------------------------------------------------
def test_arc_length_parameter_is_exactly_uniform(route):
    steps = np.diff(route.s)
    assert np.allclose(steps, route.spacing, atol=1e-12)
    # Uniform, but not exactly the requested 2 mm: the route is cut into a whole
    # number of equal steps, so 1022.9 mm gives 2.00176 mm. `spacing` reports
    # what was realised, because `index_at` divides by it.
    assert route.spacing == pytest.approx(0.002, rel=0.01)
    assert route.spacing != 0.002


def test_chord_spacing_is_uniform_apart_from_corner_cutting(route):
    """Straight-line distance between samples is slightly under the arc-length
    spacing wherever a pair straddles a vertex of the original polyline. That is
    unavoidable for any resampling of a polyline and is bounded by the turn."""
    chords = np.linalg.norm(np.diff(route.points, axis=0), axis=1)
    assert chords.max() <= route.spacing * 1.001
    assert chords.min() > route.spacing * 0.92


def test_resampling_fixes_the_125x_spacing_range(layout):
    """The reason none of this can index the stored waypoints directly."""
    raw = np.asarray(layout["waypoints"], dtype=float)
    gaps = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    assert gaps.min() == pytest.approx(0.00057, abs=1e-5)
    assert gaps.max() == pytest.approx(0.07141, abs=1e-5)
    assert gaps.max() / gaps.min() > 100


def test_route_length_is_preserved(route):
    assert route.length == pytest.approx(1.0229, abs=1e-3)


# -- clearance, the checks that had never been run ---------------------------
def test_route_never_intersects_a_wall(layout):
    """Zero, and it must stay zero: the ball has no way through a wall."""
    raw = np.asarray(layout["waypoints"], dtype=float)
    ball = 0.0055
    worst = np.inf
    for x0, y0, x1, y1 in wall_rects(layout):
        for a, b in zip(raw, raw[1:]):
            for t in np.linspace(0.0, 1.0, 21):
                p = a + t * (b - a)
                dx = max(x0 - p[0], p[0] - x1, 0.0)
                dy = max(y0 - p[1], p[1] - y1, 0.0)
                worst = min(worst, float(np.hypot(dx, dy)) - ball)
    assert worst > 0.0, f"route passes {worst * 1000:.2f} mm into a wall"


def test_minimum_clearances_match_the_measured_layout(route):
    """1.88 mm to a wall and 8.38 mm to a hole edge, measured on the raw
    waypoints. Resampling rounds corners very slightly, so the resampled route
    reads a touch more generous; both must stay above the ball's radius."""
    assert route.clearance.min() == pytest.approx(0.00195, abs=2e-4)
    assert route.clearance.min() > 0.0

    holes = np.asarray(route.layout["holes"])
    radii = np.asarray(route.layout["hole_radii"])
    gaps = np.linalg.norm(route.points[:, None, :] - holes[None, :, :], axis=2) \
        - radii[None, :] - route.ball_radius
    assert gaps.min() == pytest.approx(0.00838, abs=3e-4)


def test_clearance_varies_enough_to_matter(route):
    """Which is why cross-track cost is normalised by it rather than by a flat
    number: 1.95 mm at the tightest against 30 mm in an open cell."""
    assert route.clearance.max() / route.clearance.min() > 10


# -- projection and lookahead ------------------------------------------------
def test_projection_recovers_arc_length_and_offset(route):
    for index in (0, 137, 300, len(route.points) - 1):
        s, cross, _ = route.project(route.points[index])
        assert s == pytest.approx(route.s[index], abs=route.spacing)
        assert cross == pytest.approx(0.0, abs=1e-9)


def test_projection_signs_cross_track_by_the_left_normal(route):
    index = 200
    for offset in (0.004, -0.004):
        point = route.points[index] + offset * route.normals[index]
        _, cross, _ = route.project(point)
        assert cross == pytest.approx(offset, abs=1e-4)


def test_lookahead_is_spaced_by_arc_length_not_by_index(route):
    points = route.lookahead(0.2, spacing=0.012, count=5)
    assert points.shape == (5, 2)
    distances = [np.linalg.norm(p - route.point_at(0.2)) for p in points]
    assert all(b > a for a, b in zip(distances, distances[1:]))


def test_lookahead_clamps_at_the_goal(route):
    points = route.lookahead(route.length - 0.005, spacing=0.012, count=5)
    for point in points[1:]:
        assert np.allclose(point, route.goal)
