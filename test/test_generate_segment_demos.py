"""Demonstration selection, storage and reporting."""
from __future__ import annotations

from collections import Counter

from train.generate_segment_demos import (
    partition_selection, select_episodes, shard_paths, shard_selection,
    summarise,
)
from train.segment_dataset import generate_dataset


def test_canary_selection_is_balanced():
    dataset = generate_dataset(per_kind=3, per_geometry=4, seed=61)
    selected = select_episodes(
        dataset, geometries_per_kind=2, conditions_per_geometry=3)
    assert len(selected) == 5 * 2 * 3
    assert Counter(geometry["kind"] for geometry, _ in selected) == {
        kind: 6 for kind in (
            "straight", "gentle_left", "gentle_right",
            "sharp_left", "sharp_right")
    }


def test_canary_selection_spans_available_geometry_sources():
    dataset = generate_dataset(per_kind=50, per_geometry=1, seed=61)
    selected = select_episodes(
        dataset, geometries_per_kind=3, conditions_per_geometry=1)
    for kind in (
            "straight", "gentle_left", "gentle_right",
            "sharp_left", "sharp_right"):
        sources = {
            geometry["source"] for geometry, _ in selected
            if geometry["kind"] == kind
        }
        assert sources == {"authentic", "mirrored_authentic", "procedural"}


def test_summary_is_success_first_and_per_class():
    rows = [
        {"kind": "straight", "outcome": "goal", "completion": 1.0,
         "seconds": 2.0},
        {"kind": "straight", "outcome": "timeout", "completion": 0.8,
         "seconds": 3.0},
    ]
    report = summarise(rows)
    assert report["success_rate"] == 0.5
    assert report["fall_rate"] == 0.0
    assert report["by_kind"]["straight"]["mean_completion"] == 0.9
    assert report["by_kind"]["straight"]["mean_seconds_to_goal"] == 2.0


def test_long_generation_is_split_into_stable_shards(tmp_path):
    episodes = list(range(11))
    assert list(map(len, shard_selection(episodes, 4))) == [4, 4, 3]
    data, report = shard_paths(tmp_path / "teacher.npz", 7)
    assert data.name == "teacher.part-0007.npz"
    assert report.name == "teacher.part-0007.json"


def test_parallel_worker_partitions_are_disjoint_and_complete():
    episodes = list(range(103))
    partitions = [partition_selection(episodes, index, 8)
                  for index in range(8)]
    assert sorted(item for partition in partitions for item in partition) \
        == episodes
    assert sum(map(len, partitions)) == len(episodes)
    assert all(set(partitions[a]).isdisjoint(partitions[b])
               for a in range(8) for b in range(a + 1, 8))
