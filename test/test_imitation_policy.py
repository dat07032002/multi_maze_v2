"""Route-conditioned behavior-cloning policy tests."""
from __future__ import annotations

import numpy as np
import torch

from contract import policy_contract as pc
from control.imitation_policy import RouteConditionedPolicy
from train.behavior_clone import (
    BalancedBatcher, load_split, select_policy_observations)
from train.segment_dataset import KINDS


def test_policy_shape_bounds_and_embedded_normalization():
    mean = np.linspace(-1.0, 1.0, pc.OBSERVATION_SIZE, dtype=np.float32)
    std = np.linspace(0.2, 2.0, pc.OBSERVATION_SIZE, dtype=np.float32)
    model = RouteConditionedPolicy(mean, std, hidden_sizes=(32, 16))
    output = model(torch.zeros(7, pc.OBSERVATION_SIZE))
    assert output.shape == (7, 2)
    assert torch.all(torch.isfinite(output))
    assert torch.max(torch.abs(output)) <= 1.0
    assert torch.allclose(model.observation_mean, torch.from_numpy(mean))
    assert torch.allclose(model.observation_std, torch.from_numpy(std))


def test_numpy_prediction_handles_one_or_many_observations():
    model = RouteConditionedPolicy(hidden_sizes=(16,))
    assert model.predict(np.zeros(pc.OBSERVATION_SIZE, dtype=np.float32)).shape == (2,)
    assert model.predict(np.zeros((5, pc.OBSERVATION_SIZE), dtype=np.float32)).shape == (5, 2)


def test_legacy_policy_ignores_appended_clearance_features():
    legacy_size = pc.OBSERVATION_SIZE - pc.LOOKAHEAD_COUNT
    model = RouteConditionedPolicy(
        np.zeros(legacy_size, dtype=np.float32),
        np.ones(legacy_size, dtype=np.float32),
        hidden_sizes=(16,))
    current = np.arange(pc.OBSERVATION_SIZE, dtype=np.float32)

    expected = model.predict(current[:legacy_size])

    assert np.allclose(model.predict(current), expected)


def test_balanced_batcher_equalizes_maneuver_classes():
    kinds = np.concatenate([
        np.repeat(kind, 10 * (index + 1))
        for index, kind in enumerate(KINDS)
    ])
    batcher = BalancedBatcher(kinds, seed=3)
    indices = batcher.sample(1000)
    selected = kinds[indices]
    assert {kind: int(np.sum(selected == kind)) for kind in KINDS} == {
        kind: 200 for kind in KINDS
    }


def test_load_split_uses_policy_contract_observation_size(tmp_path):
    transitions = 3
    path = tmp_path / "split.npz"
    np.savez_compressed(
        path,
        observations=np.zeros(
            (transitions, pc.OBSERVATION_SIZE), dtype=np.float32),
        actions=np.zeros((transitions, 2), dtype=np.float32),
        kinds=np.repeat(KINDS[0], transitions),
        episode_ids=np.repeat("episode-0", transitions),
        geometry_ids=np.repeat("geometry-0", transitions),
        sources=np.repeat("procedural", transitions),
    )

    loaded = load_split(path)

    assert loaded["observations"].shape == (
        transitions, pc.OBSERVATION_SIZE)


def test_clearance_ablation_keeps_legacy_observation_prefix():
    observations = np.arange(
        2 * pc.OBSERVATION_SIZE, dtype=np.float32).reshape(
            2, pc.OBSERVATION_SIZE)
    data = {"observations": observations, "actions": np.zeros((2, 2))}

    selected = select_policy_observations(data, omit_clearance=True)

    assert selected["observations"].shape == (
        2, pc.OBSERVATION_SIZE - pc.LOOKAHEAD_COUNT)
    assert np.array_equal(
        selected["observations"],
        observations[:, :-pc.LOOKAHEAD_COUNT])
    assert selected["actions"] is data["actions"]
