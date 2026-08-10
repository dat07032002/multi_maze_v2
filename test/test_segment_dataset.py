"""Balanced local-maneuver dataset generation."""
from __future__ import annotations

from collections import Counter

import numpy as np

from sim.mjcf_builder import load_layout, load_parameters
from train.segment_dataset import (KINDS, generate_dataset,
                                   generate_geometries)


def test_geometry_classes_are_exactly_balanced():
    geometries = generate_geometries(per_kind=5, seed=7)
    assert Counter(g["kind"] for g in geometries) == {
        kind: 5 for kind in KINDS
    }
    assert len({g["id"] for g in geometries}) == len(geometries)


def test_procedural_turns_respect_direction_and_strength():
    geometries = generate_geometries(per_kind=25, seed=8)
    procedural = [g for g in geometries if g["source"] == "procedural"]
    assert procedural
    for geometry in procedural:
        peak = geometry["peak_local_turn_deg"]
        heading = geometry.get("total_heading_change_deg", 0.0)
        if geometry["kind"] == "straight":
            assert peak == 0.0
        elif geometry["kind"].startswith("gentle"):
            assert 10.0 <= peak < 30.0
        else:
            assert peak >= 30.0
        if geometry["kind"].endswith("left"):
            assert heading > 0.0
        elif geometry["kind"].endswith("right"):
            assert heading < 0.0


def test_all_generated_points_fit_on_board():
    layout = load_layout()
    geometries = generate_geometries(per_kind=10, seed=9)
    for geometry in geometries:
        points = np.asarray(geometry["points_m"])
        assert np.all(points[:, 0] >= 0.0)
        assert np.all(points[:, 0] <= layout["board_width"])
        assert np.all(points[:, 1] >= 0.0)
        assert np.all(points[:, 1] <= layout["board_height"])


def test_dataset_has_twenty_safe_starts_per_geometry():
    dataset = generate_dataset(per_kind=3, per_geometry=20, seed=10)
    assert len(dataset["geometries"]) == 15
    assert len(dataset["episodes"]) == 300
    counts = Counter(e["geometry_id"] for e in dataset["episodes"])
    assert set(counts.values()) == {20}

    params = load_parameters()
    layout = load_layout()
    positions = np.asarray([
        episode["initial_position_m"] for episode in dataset["episodes"]
    ])
    assert np.all(positions >= params["ball.radius"])
    assert np.all(positions <= np.array([
        layout["board_width"], layout["board_height"]
    ]) - params["ball.radius"])


def test_generation_is_deterministic():
    first = generate_dataset(per_kind=2, per_geometry=2, seed=11)
    second = generate_dataset(per_kind=2, per_geometry=2, seed=11)
    assert first == second
