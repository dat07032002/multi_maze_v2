import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import SAC

from contract import policy_contract as pc
from control.imitation_policy import RouteConditionedPolicy
from train.sac_finetune_bc import (
    BehaviorRegularizedSAC, RouteConditionedSACPolicy, actor_mse, checkpoint_score,
    transfer_route_policy)


def test_checkpoint_selection_prioritizes_success_and_safety():
    base = {
        "success_rate": 0.5, "fall_rate": 0.0, "mean_completion": 0.8,
        "mean_cross_track_m": 0.002, "mean_reward": 20.0,
    }
    more_success = {**base, "success_rate": 0.6, "mean_reward": -100.0}
    assert checkpoint_score(more_success) > checkpoint_score(base)
    safer = {**base, "fall_rate": 0.0, "mean_reward": -100.0}
    riskier = {**base, "fall_rate": 0.1, "mean_reward": 100.0}
    assert checkpoint_score(safer) > checkpoint_score(riskier)


def test_checkpoint_selection_uses_completion_before_reward():
    base = {
        "success_rate": 0.0, "fall_rate": 0.0, "mean_completion": 0.7,
        "mean_cross_track_m": 0.002, "mean_reward": 30.0,
    }
    progressed = {**base, "mean_completion": 0.8, "mean_reward": 0.0}
    assert checkpoint_score(progressed) > checkpoint_score(base)


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(-10, 10, shape=(pc.OBSERVATION_SIZE,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(pc.OBSERVATION_SIZE, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(pc.OBSERVATION_SIZE, dtype=np.float32), 0.0, False, True, {}


def test_route_policy_transfers_exactly_into_sac_actor(tmp_path):
    rng = np.random.default_rng(3)
    mean = rng.normal(size=pc.OBSERVATION_SIZE).astype(np.float32)
    std = rng.uniform(0.2, 2.0, size=pc.OBSERVATION_SIZE).astype(np.float32)
    teacher = RouteConditionedPolicy(mean, std, hidden_sizes=(32, 24, 16))
    checkpoint = tmp_path / "teacher.pt"
    import torch
    torch.save({
        "model_config": {"hidden_sizes": [32, 24, 16]},
        "observation_mean": mean, "observation_std": std,
        "model_state": teacher.state_dict(),
    }, checkpoint)
    model = SAC(
        RouteConditionedSACPolicy, TinyEnv(),
        policy_kwargs={"net_arch": [32, 24, 16]}, device="cpu")
    transfer_route_policy(model, checkpoint)
    observations = rng.normal(size=(128, pc.OBSERVATION_SIZE)).astype(np.float32)
    targets = teacher.predict(observations)
    assert actor_mse(model, observations, targets) < 1e-12
    assert not any(parameter.requires_grad
                   for parameter in model.actor.latent_pi.parameters())
    assert not any(parameter.requires_grad
                   for parameter in model.actor.mu.parameters())
    assert all(parameter.requires_grad
               for parameter in model.actor.residual_pi.parameters())
    sac_path = tmp_path / "residual_sac"
    model.save(sac_path)
    restored = SAC.load(sac_path, env=TinyEnv(), device="cpu")
    restored_actions, _ = restored.predict(observations, deterministic=True)
    assert np.max(np.abs(restored_actions - targets)) < 1e-6


def test_behavior_regularized_sac_retains_coefficient():
    model = BehaviorRegularizedSAC(
        RouteConditionedSACPolicy, TinyEnv(), bc_coef=321.0,
        policy_kwargs={"net_arch": [16, 12, 8]}, device="cpu")
    assert model.bc_coef == 321.0


def test_critic_carries_layer_norm_on_every_hidden_layer():
    env = TinyEnv()
    model = SAC(RouteConditionedSACPolicy, env, learning_starts=0,
                policy_kwargs={"net_arch": [16, 12],
                               "layer_norm_critic": True})
    for q_net in model.critic.q_networks:
        norms = [m for m in q_net if isinstance(m, torch.nn.LayerNorm)]
        assert [n.normalized_shape[0] for n in norms] == [16, 12]
    # The target critic must match, or polyak_update copies mismatched shapes.
    assert len(model.critic.state_dict()) == len(
        model.critic_target.state_dict())


def test_layer_norm_critic_can_be_switched_off():
    env = TinyEnv()
    model = SAC(RouteConditionedSACPolicy, env, learning_starts=0,
                policy_kwargs={"net_arch": [16, 12],
                               "layer_norm_critic": False})
    for q_net in model.critic.q_networks:
        assert not any(isinstance(m, torch.nn.LayerNorm) for m in q_net)


def test_critic_still_maps_observations_and_actions_to_q_values():
    env = TinyEnv()
    model = SAC(RouteConditionedSACPolicy, env, learning_starts=0,
                device="cpu", policy_kwargs={"net_arch": [16, 12]})
    obs = torch.zeros(4, pc.OBSERVATION_SIZE)
    actions = torch.zeros(4, 2)
    values = model.critic(obs, actions)
    assert len(values) == model.critic.n_critics
    assert all(value.shape == (4, 1) for value in values)
