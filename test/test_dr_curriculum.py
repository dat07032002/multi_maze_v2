import json

import numpy as np
import pytest

from sim.randomization import REALISTIC_DR_SCALE, TEACHER_MAX_DR_SCALE
from train.sac_train import RandomizationCurriculum


class FakeVecEnv:
    def __init__(self):
        self.scales = []

    def env_method(self, name, value):
        assert name == "set_randomization_scale"
        self.scales.append(value)
        return [value]


class FakeLogger:
    def record(self, *args, **kwargs):
        pass


class FakeModel:
    """``training_env`` and ``logger`` are read-only views onto the model."""

    def __init__(self):
        self.env = FakeVecEnv()
        self.logger = FakeLogger()

    def get_env(self):
        return self.env


def make_curriculum(**kwargs):
    curriculum = RandomizationCurriculum(**kwargs)
    curriculum.model = FakeModel()
    curriculum.num_timesteps = 0
    curriculum._on_training_start()
    return curriculum


def feed(curriculum, successes, total):
    """Deliver ``total`` finished episodes, ``successes`` of them goals."""
    for index in range(total):
        outcome = "goal" if index < successes else "timeout"
        curriculum.locals = {"infos": [{"episode": {}, "outcome": outcome}]}
        curriculum.num_timesteps += 1
        curriculum._on_step()


def test_starts_at_the_requested_scale_and_pushes_it_to_the_envs():
    curriculum = make_curriculum(start=0.10, window=5)
    assert curriculum.scale == pytest.approx(0.10)
    assert curriculum.model.env.scales == [0.10]


def test_widens_when_the_window_clears_the_threshold():
    curriculum = make_curriculum(start=0.10, step=0.03, threshold=0.7,
                                 window=5)
    feed(curriculum, successes=5, total=5)
    assert curriculum.scale == pytest.approx(0.13)
    assert curriculum.model.env.scales[-1] == pytest.approx(0.13)


def test_holds_when_the_policy_cannot_carry_the_current_scale():
    curriculum = make_curriculum(start=0.10, step=0.03, threshold=0.7,
                                 window=5)
    feed(curriculum, successes=2, total=5)
    assert curriculum.scale == pytest.approx(0.10)
    assert curriculum.model.env.scales == [0.10]


def test_never_retreats_after_a_bad_window():
    curriculum = make_curriculum(start=0.10, step=0.03, threshold=0.7,
                                 window=5)
    feed(curriculum, successes=5, total=5)
    feed(curriculum, successes=0, total=5)
    assert curriculum.scale == pytest.approx(0.13)


def test_stops_at_the_teacher_coverage_ceiling():
    curriculum = make_curriculum(start=0.20, ceiling=0.25, step=0.03,
                                 threshold=0.7, window=5)
    for _ in range(6):
        feed(curriculum, successes=5, total=5)
    # Past TEACHER_MAX_DR_SCALE the BC base is outside its demonstrations.
    assert curriculum.scale == pytest.approx(0.25)


def test_gate_blocks_advance_until_the_reverse_curriculum_finishes():
    fraction = [0.5]
    curriculum = make_curriculum(start=0.10, step=0.03, threshold=0.7,
                                 window=5, gate=lambda: fraction[0] <= 0.0)
    feed(curriculum, successes=5, total=20)
    assert curriculum.scale == pytest.approx(0.10)

    fraction[0] = 0.0
    feed(curriculum, successes=5, total=5)
    assert curriculum.scale == pytest.approx(0.13)


def test_rejects_a_start_above_its_own_ceiling():
    with pytest.raises(ValueError, match="ceiling"):
        RandomizationCurriculum(start=0.5, ceiling=TEACHER_MAX_DR_SCALE)


def test_persists_progress_for_an_interrupted_run(tmp_path):
    path = tmp_path / "dr_progress.json"
    curriculum = make_curriculum(start=REALISTIC_DR_SCALE, step=0.03,
                                 threshold=0.7, window=5, progress_path=path)
    feed(curriculum, successes=5, total=5)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["current_dr_scale"] == pytest.approx(0.13)
    assert saved["history"][0][1] == pytest.approx(0.13)
