"""Held-out maze set: the rescale, the tier screen, and layout validity.

The rescale check is the load-bearing one. ``maze_256x226.json`` is
``maze_final.json`` squeezed onto the smaller board, so that pair is a worked
example with a known answer -- if ``rescale_layout`` reproduces it exactly, then
held-out mazes differ from the shipped one only in their carve, which is the
whole premise of measuring generalisation with them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sim.maze_env import MazeEnv
from sim.mjcf_builder import DEFAULT_LAYOUT
from sim.route import Route
from tools.generate_holdout_mazes import (
    ABSOLUTE_KEYS, LENGTH_BAND, classify, rescale_layout, role_counts,
    route_length)

DESIGN = Path(DEFAULT_LAYOUT).parent


def _load(name):
    return json.loads((DESIGN / name).read_text(encoding="utf-8"))


def test_rescale_reproduces_the_shipped_layout_exactly():
    source = _load("maze_final.json")
    expected = _load("maze_256x226.json")

    result = rescale_layout(
        source, expected["board_width"], expected["board_height"])

    for key in ("walls_h", "walls_v", "holes", "waypoints"):
        assert np.allclose(result[key], expected[key], atol=1e-12), key
    assert result["board_width"] == expected["board_width"]
    assert result["board_height"] == expected["board_height"]


def test_absolute_quantities_survive_the_rescale():
    source = _load("maze_final.json")
    expected = _load("maze_256x226.json")

    result = rescale_layout(
        source, expected["board_width"], expected["board_height"])

    # Wall thickness, wall height, ball radius and hole radii are physical
    # parts, not board coordinates. Scaling them would quietly change the
    # clearances every hole placement was designed around.
    for key in ABSOLUTE_KEYS:
        assert result[key] == source[key] == expected[key], key
    assert result["hole_radii"] == expected["hole_radii"]


def test_rescale_uses_a_different_factor_per_axis():
    source = _load("maze_final.json")
    result = rescale_layout(source, 0.256, 0.226)
    # 256/259 != 226/229, so a single shared factor would fail one axis.
    assert result["board_width"] / source["board_width"] != pytest.approx(
        result["board_height"] / source["board_height"])


def test_tier_screen_matches_the_shipped_profile():
    roles = _load("maze_256x226_roles.json")
    blocks, dodges = role_counts(roles)
    assert (blocks, dodges) == (14, 1)
    # The shipped maze is the definition of the matched tier.
    assert classify(roles) == "matched"


def test_tier_screen_separates_harder_from_matched():
    assert classify({"0": "block", "1": "dodge", "2": "dodge"}) == "harder"
    # One dodge but far too few holes is neither tier: it is not the shipped
    # profile and it is not a stress test.
    assert classify({"0": "block", "1": "dodge"}) is None
    assert classify({str(i): "block" for i in range(15)}) is None


def _written_mazes():
    directory = Path("artifacts/holdout_mazes")
    if not (directory / "manifest.json").exists():
        pytest.skip("held-out set not generated; run tools.generate_holdout_mazes")
    manifest = json.loads((directory / "manifest.json").read_text())
    return directory, manifest


def test_written_mazes_drive_a_maze_env():
    directory, manifest = _written_mazes()
    for entry in manifest["mazes"]:
        layout = json.loads((directory / entry["path"]).read_text())
        env = MazeEnv(layout=layout, max_seconds=5.0, seed=0)
        try:
            observation, _ = env.reset(seed=0)
            assert observation.shape == env.observation_space.shape
            assert np.all(np.isfinite(observation)), entry["seed"]
            _, reward, _, _, info = env.step(np.zeros(2, dtype=np.float32))
            assert np.isfinite(reward)
            assert "route_completion" in info
        finally:
            env.close()


def test_written_routes_are_planned_not_cell_centres():
    """The step whose absence made the first attempt at this set unusable.

    Raw ``build`` waypoints run straight through dodge holes -- 13 samples at
    -12.3 mm against a 5.5 mm ball -- and the analytic baseline fell on 24 of
    24. A planned route keeps the centreline clear, as the shipped maze does.
    """
    directory, manifest = _written_mazes()
    shipped_clearance = np.asarray(Route(_load("maze_256x226.json")).clearance)
    assert (shipped_clearance < 0).sum() == 0

    for entry in manifest["mazes"]:
        layout = json.loads((directory / entry["path"]).read_text())
        clearance = np.asarray(Route(layout).clearance)
        assert (clearance < 0).sum() == 0, (
            f"seed {entry['seed']} routes through an obstacle")
        # A planned route is finely sampled; cell centres are one per cell.
        assert len(layout["waypoints"]) > 100, entry["seed"]


def test_written_mazes_sit_in_the_length_band_and_correct_tier():
    directory, manifest = _written_mazes()
    shipped_length = manifest["shipped_route_length_m"]
    for entry in manifest["mazes"]:
        assert abs(entry["route_length_m"] - shipped_length) <= (
            shipped_length * LENGTH_BAND), entry["seed"]
        expected = "matched" if entry["dodges"] == 1 else "harder"
        assert entry["tier"] == expected, entry["seed"]
        assert entry["path"].startswith(entry["tier"])


def test_manifest_paths_use_posix_separators():
    """Generated on Windows, consumed on the Linux training box.

    A backslash is a filename character there, not a separator, so a manifest
    written with ``str(Path(...))`` loads locally and fails with
    FileNotFoundError on the machine that actually runs the sweep.
    """
    _, manifest = _written_mazes()
    for entry in manifest["mazes"]:
        assert chr(92) not in entry["path"], entry["path"]
        assert "/" in entry["path"], entry["path"]


def test_holdout_seeds_are_disjoint_between_tiers():
    _, manifest = _written_mazes()
    by_tier: dict[str, set[int]] = {}
    for entry in manifest["mazes"]:
        by_tier.setdefault(entry["tier"], set()).add(entry["seed"])
    assert not (by_tier.get("matched", set()) & by_tier.get("harder", set()))


def test_generated_board_matches_the_shipped_board():
    import generate_maze as gm

    shipped = _load("maze_256x226.json")
    for seed in (7, 15, 140):
        layout, _, _ = gm.build(seed)
        layout = rescale_layout(
            layout, shipped["board_width"], shipped["board_height"])
        # Guards the rescale permanently: a policy must never be scored on a
        # board that is a different size from the one it trained on.
        assert layout["board_width"] == shipped["board_width"]
        assert layout["board_height"] == shipped["board_height"]
        for key in ABSOLUTE_KEYS:
            assert layout[key] == shipped[key], (seed, key)
        assert set(layout["hole_radii"]) == set(shipped["hole_radii"])
