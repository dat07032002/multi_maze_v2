import numpy as np
import pytest

from tag_vision.rl.health import HealthLevel, HealthMonitor
from tag_vision.rl.replay import ReplayBuffer
from tag_vision.rl.task import MazeTask
from tag_vision.rl.import_logs import import_ball_control_logs
from tag_vision.rl.exploration import (
    SmoothRandomExploration, StuckRecovery, episode_policy)
from tag_vision.rl.online_training import (
    OnlineDynamicsTrainer, moving_transition_count)


def simple_task():
    route = np.array([[10., 50.], [50., 50.], [90., 50.]])
    occupied = np.zeros((100, 100), dtype=bool)
    occupied[[0, -1], :] = True
    occupied[:, [0, -1]] = True
    return MazeTask(route, occupied, board_size_mm=(100, 100), ray_count=8)


def test_task_observation_and_progress_reward():
    task = simple_task()
    task.reset(0.0, [10, 50])
    obs, route = task.observation(
        position_mm=[20, 52], velocity_mm_s=[10, 0], angles_deg=[0, 0],
        angle_rates_deg_s=[0, 0], previous_action_deg=[0, 0])
    assert obs.shape == (task.spec.size,)
    result = task.step_result(obs, route, timestamp_s=0.2)
    assert result.reward > 0
    assert not result.terminated
    goal_obs, goal_route = task.observation(
        position_mm=[89, 50], velocity_mm_s=[0, 0], angles_deg=[0, 0],
        angle_rates_deg_s=[0, 0], previous_action_deg=[0, 0])
    assert task.step_result(goal_obs, goal_route, timestamp_s=1.0).reason == "goal"


def test_task_reset_projects_globally_to_reload_position():
    route = np.column_stack((np.arange(0.0, 205.0, 5.0),
                             np.full(41, 50.0)))
    occupied = np.zeros((100, 220), dtype=bool)
    task = MazeTask(route, occupied, board_size_mm=(220, 100), ray_count=4)
    task.reset(3.0, [180.0, 51.0])
    state = task.route_state([180.0, 51.0])
    assert state.progress_mm > 175.0
    assert state.route_index >= 35


def test_replay_round_trip(tmp_path):
    replay = ReplayBuffer(8, 4, seed=3)
    for index in range(5):
        replay.add(np.full(4, index), [index, -index], index,
                   np.full(4, index + 1), index == 4, False, index == 2)
    path = tmp_path / "replay.npz"
    replay.save(path)
    loaded = ReplayBuffer.load(path, seed=4)
    assert loaded.size == 5 and loaded.cursor == 5
    assert np.array_equal(loaded.actions[:5], replay.actions[:5])
    assert loaded.sample(3)["observations"].shape == (3, 4)


def test_health_red_levels_and_rate_limits():
    monitor = HealthMonitor(max_delta_deg=0.5)
    healthy = monitor.classify(
        camera_fps=20, pose_rate=.95, ball_rate=.95, ball_age_s=.02,
        imu_rate_hz=200, fusion_residual_deg=.1, control_latency_s=.05)
    assert healthy.level == HealthLevel.GREEN
    limited = monitor.safe_action([4, -4], healthy)
    assert np.allclose(limited.executed_deg, [.5, -.5])
    assert limited.overridden
    stale = monitor.classify(
        camera_fps=20, pose_rate=.95, ball_rate=.95, ball_age_s=.4,
        imu_rate_hz=200, fusion_residual_deg=.1, control_latency_s=.05)
    assert stale.level == HealthLevel.RED
    assert np.allclose(monitor.safe_action([1, 1], stale).executed_deg, 0)


def test_smooth_random_exploration_holds_bounded_targets():
    policy = SmoothRandomExploration(max_tilt_deg=4.0, hold_s=0.8, seed=7)
    first = policy.command(1.0)
    held = policy.command(1.79)
    changed = policy.command(1.81)
    assert np.array_equal(first.target_deg, held.target_deg)
    assert first.segment == held.segment == 1
    assert changed.segment == 2
    assert np.max(np.abs(first.target_deg)) <= 4.0
    assert np.max(np.abs(changed.target_deg)) <= 4.0


def test_episode_policy_explores_before_model_and_every_fifth_episode():
    assert episode_policy(episode=1, learned_model_ready=False) == "explore"
    assert episode_policy(episode=4, learned_model_ready=True) == "cem"
    assert episode_policy(episode=5, learned_model_ready=True) == "explore"
    assert episode_policy(episode=10, learned_model_ready=True) == "explore"
    assert episode_policy(
        episode=5, learned_model_ready=True, explore_every=0) == "cem"


def test_stuck_recovery_requires_sustained_stationary_measurements():
    recovery = StuckRecovery(duration_s=3.0)
    assert not recovery.update(0.0, 10.0)
    assert not recovery.update(0.0, 12.9)
    assert recovery.update(0.0, 13.0)
    assert not recovery.update(4.0, 13.1)
    assert not recovery.update(0.0, 15.0)


def test_moving_transition_count_checks_both_sides_of_transition():
    task = simple_task()
    replay = ReplayBuffer(4, task.spec.size)
    stationary = np.zeros(task.spec.size, dtype=np.float32)
    moving = stationary.copy()
    moving[task.spec.index("vx")] = 5.0
    replay.add(stationary, [0, 0], 0, stationary, False, False)
    replay.add(stationary, [1, 0], 0, moving, False, False)
    replay.add(moving, [1, 0], 0, stationary, False, False)
    assert moving_transition_count(replay, task.spec) == 2


def test_online_trainer_schedule_uses_fresh_transition_intervals(tmp_path):
    trainer = OnlineDynamicsTrainer(tmp_path, tmp_path / "models")
    assert not trainer.should_start(299, minimum=300, every=100)
    assert trainer.should_start(300, minimum=300, every=100)
    trainer._submitted_size = 300
    assert not trainer.should_start(399, minimum=300, every=100)
    assert trainer.should_start(400, minimum=300, every=100)


def test_import_real_log_rows(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text(
        "time_s,ball_visible,x_mm,y_mm,vx_mm_s,vy_mm_s,"
        "fused_alpha_deg,fused_beta_deg,target_alpha_deg,target_beta_deg\n"
        "0.0,1,10,50,0,0,0,0,0,0\n"
        "0.2,1,12,50,10,0,0.1,0,0.2,0\n"
        "0.4,1,15,50,15,0,0.2,0,0.3,0\n",
        encoding="utf-8")
    task = simple_task()
    replay = ReplayBuffer(20, task.spec.size)
    summary = import_ball_control_logs([path], task, replay)
    assert summary.transitions == 2
    assert replay.size == 2
    assert np.all(np.isfinite(replay.observations[:2]))


def test_dynamics_checkpoint_round_trip(tmp_path):
    torch = pytest.importorskip("torch")
    from tag_vision.rl.dynamics_torch import EnsembleDynamics

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EnsembleDynamics(30, 2, members=2, hidden_size=16, device=device)
    path = tmp_path / "dynamics.pt"
    model.save_checkpoint(path, metadata={"purpose": "test"})
    restored, metadata = EnsembleDynamics.load_checkpoint(path, device=device)
    assert restored.observation_size == 30
    assert restored.member_count == 2
    assert metadata == {"purpose": "test"}


def test_dynamics_bootstrap_balances_moving_and_stationary_samples():
    from tag_vision.rl.dynamics_torch import motion_balanced_bootstrap

    replay = ReplayBuffer(20, 6)
    stationary = np.zeros(6, dtype=np.float32)
    moving = stationary.copy()
    moving[2] = 5.0
    for index in range(10):
        replay.add(stationary, [0, 0], 0,
                   moving if index == 0 else stationary, False, False)
    sampled = motion_balanced_bootstrap(
        replay, np.arange(10), np.random.default_rng(4))
    moving_count = np.count_nonzero(
        np.linalg.norm(replay.next_observations[sampled, 2:4], axis=1) >= 3)
    assert moving_count == 5
