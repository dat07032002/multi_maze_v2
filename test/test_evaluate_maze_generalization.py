"""Aggregation and task construction for the held-out maze sweep."""
from __future__ import annotations

import json

import pytest

from train.evaluate_maze_generalization import (
    aggregate, build_tasks, summarize)


def row(maze, tier, scale, outcome, completion=0.9, seconds=100.0):
    return {"maze": maze, "tier": tier, "dr_scale": scale, "seed": 0,
            "outcome": outcome, "completion": completion,
            "mean_cross_track_m": 0.0015, "reward": 30.0, "seconds": seconds}


def test_aggregate_counts_outcomes_and_times_only_successes():
    rows = [row("m", "matched", 0.1, "goal", seconds=100.0),
            row("m", "matched", 0.1, "goal", seconds=140.0),
            row("m", "matched", 0.1, "timeout", seconds=200.0),
            row("m", "matched", 0.1, "fell", seconds=30.0)]
    stats = aggregate(rows)
    assert stats["success_rate"] == pytest.approx(0.5)
    assert stats["fall_rate"] == pytest.approx(0.25)
    assert stats["timeout_rate"] == pytest.approx(0.25)
    # A timeout is not a fast lap and must not drag the mean down.
    assert stats["mean_seconds_to_goal"] == pytest.approx(120.0)


def test_aggregate_reports_no_time_when_nothing_succeeded():
    stats = aggregate([row("m", "matched", 0.1, "fell")])
    assert stats["mean_seconds_to_goal"] is None


def test_headline_gap_compares_shipped_against_matched_only():
    rows = ([row("shipped", "shipped", 0.1, "goal")] * 8
            + [row("shipped", "shipped", 0.1, "timeout")] * 2
            + [row("a", "matched", 0.1, "goal")] * 5
            + [row("a", "matched", 0.1, "timeout")] * 5
            # The harder tier must not move the headline.
            + [row("b", "harder", 0.1, "fell")] * 10)
    headline = summarize(rows, 0.1)["headline"]
    assert headline["shipped_success"] == pytest.approx(0.8)
    assert headline["matched_success"] == pytest.approx(0.5)
    assert headline["generalization_gap"] == pytest.approx(0.3)
    assert headline["harder_success"] == pytest.approx(0.0)


def test_headline_is_taken_at_one_scale_not_pooled():
    rows = ([row("shipped", "shipped", 0.1, "goal")] * 10
            + [row("shipped", "shipped", 1.0, "fell")] * 10
            + [row("a", "matched", 0.1, "goal")] * 10
            + [row("a", "matched", 1.0, "fell")] * 10)
    headline = summarize(rows, 0.1)["headline"]
    # Pooling scales would report 50% and hide a perfect score at 0.1.
    assert headline["shipped_success"] == pytest.approx(1.0)
    assert headline["generalization_gap"] == pytest.approx(0.0)


def test_gap_is_unavailable_without_the_shipped_control():
    rows = [row("a", "matched", 0.1, "goal")] * 4
    assert summarize(rows, 0.1)["headline"]["generalization_gap"] is None


def test_build_tasks_covers_the_full_grid_and_adds_the_control(tmp_path):
    layout = {"board_width": 0.256, "board_height": 0.226}
    (tmp_path / "matched").mkdir()
    (tmp_path / "matched" / "maze_7.json").write_text(json.dumps(layout))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "mazes": [{"seed": 7, "tier": "matched",
                   "path": "matched/maze_7.json"}]}))

    tasks = build_tasks(tmp_path, [0.1, 1.0], episodes=3, seed=500,
                        max_seconds=200.0, include_shipped=True)
    assert len(tasks) == 2 * 2 * 3          # (holdout + shipped) x scales x seeds
    assert {t["tier"] for t in tasks} == {"matched", "shipped"}
    # Seeds repeat across cells on purpose: every maze faces the same episodes.
    assert {t["seed"] for t in tasks} == {500, 501, 502}

    without = build_tasks(tmp_path, [0.1], episodes=3, seed=500,
                          max_seconds=200.0, include_shipped=False)
    assert {t["tier"] for t in without} == {"matched"}
