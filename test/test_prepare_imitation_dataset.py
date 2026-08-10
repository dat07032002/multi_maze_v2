"""Geometry-level imitation corpus splitting."""
from __future__ import annotations

from collections import Counter

from train.prepare_imitation_dataset import build_geometry_split
from train.segment_dataset import generate_dataset, KINDS


def test_geometry_split_is_disjoint_balanced_and_deterministic():
    dataset = generate_dataset(per_kind=50, per_geometry=1, seed=71)
    first = build_geometry_split(dataset, seed=19)
    second = build_geometry_split(dataset, seed=19)
    assert first == second
    assert len(first) == 250
    assert Counter(first.values()) == {
        "train": 200, "validation": 25, "test": 25,
    }
    by_id = {row["id"]: row for row in dataset["geometries"]}
    for kind in KINDS:
        counts = Counter(
            split for geometry_id, split in first.items()
            if by_id[geometry_id]["kind"] == kind)
        assert counts == {"train": 40, "validation": 5, "test": 5}


def test_holdouts_retain_sources_with_enough_geometries():
    dataset = generate_dataset(per_kind=50, per_geometry=1, seed=73)
    split = build_geometry_split(dataset, seed=23)
    rows = {row["id"]: row for row in dataset["geometries"]}
    for kind in KINDS:
        source_counts = Counter(
            row["source"] for row in rows.values() if row["kind"] == kind)
        for source, count in source_counts.items():
            if count < 3:
                continue
            assert any(
                rows[geometry_id]["kind"] == kind
                and rows[geometry_id]["source"] == source
                and assigned == "validation"
                for geometry_id, assigned in split.items())
            assert any(
                rows[geometry_id]["kind"] == kind
                and rows[geometry_id]["source"] == source
                and assigned == "test"
                for geometry_id, assigned in split.items())
