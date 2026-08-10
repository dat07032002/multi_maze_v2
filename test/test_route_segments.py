"""Route maneuver classification guards."""
from __future__ import annotations

import numpy as np

from sim.route import Route
from sim.route_segments import classify_route, signed_turn_degrees, summary


def test_signed_turn_matches_route_samples():
    route = Route()
    turns = signed_turn_degrees(route)
    assert turns.shape == route.s.shape
    assert np.all(np.isfinite(turns))
    assert turns.max() > 30.0
    assert turns.min() < -30.0


def test_segments_cover_route_exactly_without_gaps():
    route = Route()
    _, segments = classify_route(route)
    assert segments[0].start_s == 0.0
    assert segments[-1].end_s == route.length
    assert all(a.end_s == b.start_s for a, b in zip(segments, segments[1:]))
    assert sum(segment.length for segment in segments) == route.length


def test_current_maze_contains_both_sharp_turn_directions():
    _, segments = classify_route(Route())
    counts = summary(segments)
    assert counts["sharp_left"] > 0
    assert counts["sharp_right"] > 0
    assert counts["straight"] > counts["sharp_left"]
    assert counts["straight"] > counts["sharp_right"]


def test_turn_strength_threshold_changes_only_turn_strength():
    route = Route()
    _, ordinary = classify_route(route, sharp_min_deg=30.0)
    _, stricter = classify_route(route, sharp_min_deg=60.0)
    assert len(ordinary) == len(stricter)
    assert sum(s.kind.startswith("sharp") for s in stricter) \
        < sum(s.kind.startswith("sharp") for s in ordinary)
