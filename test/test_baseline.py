"""M2 acceptance: does the hand-written controller actually drive the maze?

Deliberately a small number of seeds -- each closed-loop run is a full maze of
simulated time -- with the full ten-seed sweep left to
``python -m sim.run_baseline``. What is pinned here is that the controller
still completes the route and still tracks it, so a change to the estimator,
the predictor or the actuator cannot quietly break the bar that RL is measured
against.
"""
from __future__ import annotations

import numpy as np
import pytest

from control.baseline import PurePursuitBaseline
from sim.mjcf_builder import load_layout, load_parameters
from sim.rollout import run_closed_loop
from sim.route import Route


@pytest.fixture(scope="module")
def setup():
    layout = load_layout()
    params = load_parameters()
    return layout, params, Route(layout, params)


def _run(setup, seed, **kwargs):
    layout, params, route = setup
    controller = PurePursuitBaseline(route, params["actuator.max_tilt"])
    result = run_closed_loop(controller, layout=layout, params=params,
                             seed=seed, max_seconds=90.0, **kwargs)
    completion = route.project(result.track[-1])[0] / route.length
    cross = np.array([abs(route.project(p)[1]) for p in result.track])
    return result, completion, cross


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_baseline_completes_the_maze(setup, seed):
    result, completion, _ = _run(setup, seed)
    assert result.reached_goal, (
        f"seed {seed}: stopped at {completion:.1%} "
        f"({'fell in a hole' if result.fell else 'ran out of time'})")


def test_baseline_stays_in_the_corridor(setup):
    """Mean cross-track has to be small in absolute terms, not just on average
    over a route that is mostly open cells: the tightest point affords 1.95 mm."""
    _, _, cross = _run(setup, 0)
    assert cross.mean() < 0.005, f"mean cross-track {cross.mean() * 1000:.2f} mm"


def test_baseline_fits_inside_the_episode_budget(setup):
    """~67 s against a 90 s cap after the 2026-08-08 friction re-centre (was
    52.9 s / 60 s at the old machined-surface friction). If a change pushes
    this over, episodes start timing out rather than failing loudly."""
    result, _, _ = _run(setup, 0)
    assert result.steps / 20.0 < 85.0, (
        f"took {result.steps / 20.0:.1f} s of a 90 s budget")


def test_baseline_does_not_depend_on_a_perfect_sensor(setup):
    """It runs with 1 mm of detector noise and 2 % dropout by default. This
    pins that the noise is actually being applied, so a regression that
    silently disables it cannot make the bar look easier than it is."""
    noisy, _, _ = _run(setup, 3, sensor_noise=True)
    clean, _, _ = _run(setup, 3, sensor_noise=False)
    assert clean.reached_goal
    assert noisy.positions != clean.positions
