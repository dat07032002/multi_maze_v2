"""Closed-loop imitation evaluation selection and reporting."""
from __future__ import annotations

from collections import Counter

from train.evaluate_imitation import expanded_summary, select_evaluation_episodes
from train.prepare_imitation_dataset import build_geometry_split
from train.segment_dataset import generate_dataset, KINDS


def test_evaluation_selection_is_held_out_and_balanced():
    dataset = generate_dataset(per_kind=50, per_geometry=6, seed=81)
    assignment = build_geometry_split(dataset, seed=29)
    report = {"geometry_assignment": [
        {"id": geometry_id, "split": split}
        for geometry_id, split in assignment.items()
    ]}
    selected = select_evaluation_episodes(
        dataset, report, "validation", geometries_per_kind=3,
        conditions_per_geometry=5)
    assert len(selected) == 75
    assert Counter(geometry["kind"] for geometry, _ in selected) == {
        kind: 15 for kind in KINDS
    }
    assert all(assignment[geometry["id"]] == "validation"
               for geometry, _ in selected)


def test_expanded_summary_reports_sources_and_outcomes():
    rows = [
        {"kind": "straight", "source": "authentic", "outcome": "goal",
         "completion": 0.9, "seconds": 2.0, "mean_action_change": 0.1},
        {"kind": "straight", "source": "authentic", "outcome": "timeout",
         "completion": 0.5, "seconds": 5.0, "mean_action_change": 0.2},
    ]
    summary = expanded_summary(rows)
    assert summary["outcomes"] == {"goal": 1, "timeout": 1}
    assert summary["by_source"]["authentic"]["success_rate"] == 0.5
