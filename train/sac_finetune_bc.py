"""Fine-tune the route-conditioned BC policy with SAC on the full maze."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import Actor, SACPolicy

from control.imitation_policy import load_policy
from sim.maze_env import MazeEnv
from sim.randomization import (
    Randomizer, REALISTIC_DR_SCALE, TEACHER_MAX_DR_SCALE)
from train.imitation import seed_replay_buffer, warmup_critics
from train.nstep_buffer import NStepReplayBuffer
from train.sac_train import RandomizationCurriculum, ReverseCurriculum


def make_env(rank: int, seed: int, max_seconds: float,
             randomization_scale: float, start_fraction: float = 0.0):
    def _init():
        env = MazeEnv(
            max_seconds=max_seconds,
            randomizer=Randomizer(
                scale=randomization_scale,
                enabled=randomization_scale > 0.0),
            start_fraction=start_fraction,
            # Fine-tuning starts from a policy that can already finish, so the
            # whole route stays in the training distribution instead of only
            # the slice the curriculum currently points at.
            sample_start_fraction=True, seed=seed + rank)
        return Monitor(env, info_keywords=(
            "route_completion", "mean_cross_track", "outcome"))
    return _init


class RouteConditionedActor(Actor):
    """SAC actor with the exact BC mean path plus a learned exploration head."""

    def __init__(self, *args, residual_limit: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_sde:
            raise ValueError("route-conditioned transfer does not use gSDE")
        layers = []
        previous = self.features_dim
        for size in self.net_arch:
            layers.extend([nn.Linear(previous, size), nn.SiLU(),
                           nn.LayerNorm(size)])
            previous = size
        self.latent_pi = nn.Sequential(*layers)
        action_dim = int(np.prod(self.action_space.shape))
        self.mu = nn.Linear(previous, action_dim)
        self.log_std = nn.Linear(previous, action_dim)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -3.0)
        self.residual_limit = float(residual_limit)
        self.residual_pi = nn.Sequential(
            nn.Linear(self.features_dim, 128), nn.SiLU(), nn.LayerNorm(128),
            nn.Linear(128, 128), nn.SiLU(), nn.LayerNorm(128))
        self.residual_mu = nn.Linear(128, action_dim)
        # Exact BC behaviour at transfer time. SAC can subsequently move only
        # within residual_limit in pre-squash mean space.
        nn.init.zeros_(self.residual_mu.weight)
        nn.init.zeros_(self.residual_mu.bias)
        self.register_buffer(
            "observation_mean", torch.zeros(self.features_dim))
        self.register_buffer(
            "observation_std", torch.ones(self.features_dim))

    def _normalized_features(self, obs):
        features = self.extract_features(obs, self.features_extractor)
        return torch.clamp(
            (features - self.observation_mean) / self.observation_std,
            -10.0, 10.0)

    def base_action(self, obs):
        normalized = self._normalized_features(obs)
        return torch.tanh(self.mu(self.latent_pi(normalized)))

    def get_action_dist_params(self, obs):
        normalized = self._normalized_features(obs)
        latent = self.latent_pi(normalized)
        base_mean = self.mu(latent)
        residual = self.residual_limit * torch.tanh(
            self.residual_mu(self.residual_pi(normalized)))
        mean = base_mean + residual
        log_std = torch.clamp(self.log_std(latent), -20.0, 2.0)
        return mean, log_std, {}


class LayerNormCritic(ContinuousCritic):
    """Critic with LayerNorm on every hidden layer.

    Both RLPD and the residual fine-tuning line report this as the single
    change that removes Q divergence when a policy trained off demonstrations
    starts querying actions the critic has not seen. It is insurance rather
    than a fix here -- the v1 run's critic was stable at 0.009-0.023 -- but
    dropping the BC anchor and raising the update-to-data ratio are exactly
    the two changes that create the risk.
    """

    def __init__(self, *args, net_arch: list[int], features_dim: int,
                 activation_fn=nn.ReLU, **kwargs):
        super().__init__(*args, net_arch=net_arch, features_dim=features_dim,
                         activation_fn=activation_fn, **kwargs)
        action_dim = get_action_dim(self.action_space)
        self.q_networks = []
        for index in range(self.n_critics):
            layers = []
            previous = features_dim + action_dim
            for size in net_arch:
                layers.extend([nn.Linear(previous, size), activation_fn(),
                               nn.LayerNorm(size)])
                previous = size
            layers.append(nn.Linear(previous, 1))
            network = nn.Sequential(*layers)
            self.add_module(f"qf{index}", network)
            self.q_networks.append(network)


class RouteConditionedSACPolicy(SACPolicy):
    """SAC policy whose actor can receive the BC network without approximation."""

    def __init__(self, *args, residual_limit: float = 0.15,
                 layer_norm_critic: bool = True, **kwargs):
        self.residual_limit = float(residual_limit)
        self.layer_norm_critic = bool(layer_norm_critic)
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None) -> RouteConditionedActor:
        actor_kwargs = self._update_features_extractor(
            self.actor_kwargs, features_extractor)
        actor_kwargs["residual_limit"] = self.residual_limit
        return RouteConditionedActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor=None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor)
        critic_class = LayerNormCritic if self.layer_norm_critic \
            else ContinuousCritic
        return critic_class(**critic_kwargs).to(self.device)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["residual_limit"] = self.residual_limit
        data["layer_norm_critic"] = self.layer_norm_critic
        return data


class BehaviorRegularizedSAC(SAC):
    """SAC whose actor remains anchored to the frozen BC controller."""

    def __init__(self, *args, bc_coef: float = 1000.0, **kwargs):
        self.bc_coef = float(bc_coef)
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)
        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses, bc_losses, rl_losses = [], [], [], []

        for gradient_step in range(gradient_steps):
            replay = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env)
            stored_discounts = getattr(replay, "discounts", None)
            discounts = stored_discounts \
                if stored_discounts is not None else self.gamma
            actions_pi, log_prob = self.actor.action_log_prob(
                replay.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(
                    self.log_ent_coef
                    * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(float(ent_coef_loss.detach().cpu()))
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(float(ent_coef.detach().cpu()))

            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay.next_observations)
                next_q = torch.cat(self.critic_target(
                    replay.next_observations, next_actions), dim=1)
                next_q, _ = torch.min(next_q, dim=1, keepdim=True)
                next_q = next_q - ent_coef * next_log_prob.reshape(-1, 1)
                target_q = replay.rewards + (
                    1.0 - replay.dones) * discounts * next_q

            current_q = self.critic(replay.observations, replay.actions)
            critic_loss = 0.5 * sum(
                F.mse_loss(value, target_q) for value in current_q)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            critic_losses.append(float(critic_loss.detach().cpu()))

            q_pi = torch.cat(self.critic(
                replay.observations, actions_pi), dim=1)
            min_q_pi, _ = torch.min(q_pi, dim=1, keepdim=True)
            rl_loss = (ent_coef * log_prob - min_q_pi).mean()
            mean, _, _ = self.actor.get_action_dist_params(
                replay.observations)
            deterministic_action = torch.tanh(mean)
            with torch.no_grad():
                teacher_action = self.actor.base_action(replay.observations)
            bc_loss = F.mse_loss(deterministic_action, teacher_action)
            actor_loss = rl_loss + self.bc_coef * bc_loss
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            actor_losses.append(float(actor_loss.detach().cpu()))
            rl_losses.append(float(rl_loss.detach().cpu()))
            bc_losses.append(float(bc_loss.detach().cpu()))

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(),
                              self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats,
                              self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates,
                           exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/actor_rl_loss", np.mean(rl_losses))
        self.logger.record("train/actor_bc_loss", np.mean(bc_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if ent_coef_losses:
            self.logger.record("train/ent_coef_loss",
                               np.mean(ent_coef_losses))


def transfer_route_policy(model: SAC, checkpoint: str | Path) -> None:
    teacher = load_policy(checkpoint, device=model.device)
    actor = model.policy.actor
    if not isinstance(actor, RouteConditionedActor):
        raise TypeError("SAC actor is not route-conditioned")
    if tuple(actor.net_arch) != tuple(
            layer.out_features for layer in teacher.network
            if isinstance(layer, nn.Linear))[:-1]:
        raise ValueError("SAC and BC hidden architectures differ")
    teacher_hidden = {
        key: value for key, value in teacher.network.state_dict().items()
        if int(key.split(".", 1)[0]) < len(teacher.network) - 2
    }
    actor.latent_pi.load_state_dict(teacher_hidden)
    actor.mu.load_state_dict(teacher.network[-2].state_dict())
    actor.observation_mean.copy_(teacher.observation_mean)
    actor.observation_std.copy_(teacher.observation_std)
    for parameter in actor.latent_pi.parameters():
        parameter.requires_grad_(False)
    for parameter in actor.mu.parameters():
        parameter.requires_grad_(False)


def actor_mse(model: SAC, observations: np.ndarray,
              targets: np.ndarray, batch_size: int = 16_384) -> float:
    errors = []
    actor = model.policy.actor
    actor.eval()
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            obs = torch.as_tensor(
                observations[start:start + batch_size], device=model.device)
            target = torch.as_tensor(
                targets[start:start + batch_size], device=model.device)
            mean, _, _ = actor.get_action_dist_params(obs)
            errors.append(torch.square(torch.tanh(mean) - target).cpu().numpy())
    return float(np.mean(np.concatenate(errors)))


def collect_full_maze_replay(model: SAC, episodes: int, seed: int,
                             max_seconds: float,
                             randomization_scale: float) -> dict[str, np.ndarray]:
    env = MazeEnv(
        max_seconds=max_seconds,
        randomizer=Randomizer(
            scale=randomization_scale,
            enabled=randomization_scale > 0.0),
        start_fraction=0.0, seed=seed)
    arrays = {name: [] for name in (
        "observations", "next_observations", "actions", "rewards",
        "dones", "truncated")}
    outcomes = Counter()
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            while True:
                action, _ = model.predict(observation, deterministic=True)
                next_observation, reward, terminated, truncated, info = env.step(action)
                arrays["observations"].append(observation.copy())
                arrays["next_observations"].append(next_observation.copy())
                arrays["actions"].append(np.asarray(action, dtype=np.float32))
                arrays["rewards"].append(reward)
                arrays["dones"].append(terminated or truncated)
                arrays["truncated"].append(truncated)
                observation = next_observation
                if terminated or truncated:
                    outcomes[info["outcome"]] += 1
                    print(f"replay {episode + 1:>2}/{episodes}: "
                          f"{info['outcome']}, {info['route_completion']:.1%}",
                          flush=True)
                    break
    finally:
        env.close()
    print(f"collected {len(arrays['actions']):,} full-maze transitions; "
          f"outcomes {dict(outcomes)}")
    dtypes = {
        "observations": np.float32, "next_observations": np.float32,
        "actions": np.float32, "rewards": np.float32,
        "dones": np.float32, "truncated": bool,
    }
    return {name: np.asarray(values, dtype=dtypes[name])
            for name, values in arrays.items()}


def evaluate_model(model: SAC, episodes: int, seed: int,
                   max_seconds: float, randomization_scale: float) -> dict:
    env = MazeEnv(
        max_seconds=max_seconds,
        randomizer=Randomizer(
            scale=randomization_scale,
            enabled=randomization_scale > 0.0),
        start_fraction=0.0, seed=seed)
    rows = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            total = 0.0
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            rows.append({
                "seed": seed + episode, "outcome": info["outcome"],
                "completion": info["route_completion"],
                "mean_cross_track_m": info["mean_cross_track"],
                "reward": total, "seconds": info["steps"] / env.control_hz,
            })
    finally:
        env.close()
    n = len(rows)
    goals = [row for row in rows if row["outcome"] == "goal"]
    return {
        "episodes": n,
        "success_rate": len(goals) / n,
        "fall_rate": sum(row["outcome"] == "fell" for row in rows) / n,
        "timeout_rate": sum(row["outcome"] == "timeout" for row in rows) / n,
        "mean_completion": float(np.mean([row["completion"] for row in rows])),
        "mean_cross_track_m": float(np.mean([
            row["mean_cross_track_m"] for row in rows])),
        "mean_reward": float(np.mean([row["reward"] for row in rows])),
        "mean_seconds_to_goal": float(np.mean([
            row["seconds"] for row in goals])) if goals else None,
        "rows": rows,
    }


def checkpoint_score(metrics: dict) -> tuple[float, ...]:
    """Safety/completion lexicographic order; reward is only the last tie-break."""
    return (metrics["success_rate"], -metrics["fall_rate"],
            metrics["mean_completion"], -metrics["mean_cross_track_m"],
            metrics["mean_reward"])


class FullMazeEvalCallback(BaseCallback):
    def __init__(self, out: Path, interval: int, episodes: int, seed: int,
                 max_seconds: float, randomization_scale: float,
                 verbose: int = 1):
        super().__init__(verbose)
        self.out = out
        self.interval = interval
        self.episodes = episodes
        self.seed = seed
        self.max_seconds = max_seconds
        self.randomization_scale = randomization_scale
        self.next_evaluation = 0
        self.best_score = None
        self.history = []

    def _evaluate(self) -> None:
        metrics = evaluate_model(
            self.model, self.episodes, self.seed, self.max_seconds,
            self.randomization_scale)
        metrics["timesteps"] = self.num_timesteps
        self.history.append(metrics)
        with (self.out / "evaluation.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics) + "\n")
        score = checkpoint_score(metrics)
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            (self.out / "best").mkdir(parents=True, exist_ok=True)
            self.model.save(self.out / "best" / "best_model")
            (self.out / "best" / "metrics.json").write_text(
                json.dumps(metrics, indent=2), encoding="utf-8")
            marker = " NEW BEST"
        else:
            marker = ""
        print(f"evaluation @{self.num_timesteps:,}: "
              f"success {metrics['success_rate']:.0%}, "
              f"falls {metrics['fall_rate']:.0%}, "
              f"completion {metrics['mean_completion']:.1%}{marker}",
              flush=True)

    def _on_training_start(self) -> None:
        self._evaluate()
        self.next_evaluation = self.interval

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_evaluation:
            self._evaluate()
            while self.next_evaluation <= self.num_timesteps:
                self.next_evaluation += self.interval
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bc-checkpoint",
        default="artifacts/local_segments/bc_policy_v1/best_model.pt")
    parser.add_argument(
        "--imitation-validation",
        default="artifacts/local_segments/imitation_v1/validation.npz")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--envs", type=int, default=8)
    # UTD = gradient_steps / envs, because train_freq is one step per env.
    # v1 ran gradient_steps=1 at 8 envs, a UTD of 0.125; the residual
    # fine-tuning literature ablates this and lands on 4.
    parser.add_argument("--gradient-steps", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--residual-limit", type=float, default=0.15)
    # The residual bound is the constraint. v1 stacked a 1000x MSE anchor on
    # top of it and pinned the policy 2.9e-4 from BC for 500k steps.
    parser.add_argument("--bc-coef", type=float, default=0.0)
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--no-layer-norm-critic", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=200.0)
    parser.add_argument("--dr-scale", type=float, default=REALISTIC_DR_SCALE)
    parser.add_argument("--dr-ceiling", type=float,
                        default=TEACHER_MAX_DR_SCALE)
    parser.add_argument("--dr-step", type=float, default=0.03)
    # Reverse curriculum must have reached this fraction before ADR may widen.
    # 0.0 waits for the full route; 1.0 opens the gate immediately. Sampled
    # starts already keep the whole route in the training distribution, so
    # waiting for the curriculum to bottom out is far more conservative than it
    # was when every episode began at a fixed fraction.
    parser.add_argument("--dr-gate-fraction", type=float, default=0.0)
    parser.add_argument("--no-dr-curriculum", action="store_true")
    parser.add_argument("--train-start-fraction", type=float, default=0.80)
    parser.add_argument("--replay-episodes", type=int, default=10)
    parser.add_argument("--critic-warmup-steps", type=int, default=2_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--out", default="artifacts/sac_bc_finetune_v1")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    factories = [make_env(
        rank, args.seed, args.max_seconds, args.dr_scale,
        args.train_start_fraction)
        for rank in range(args.envs)]
    vec = SubprocVecEnv(factories) if args.envs > 1 else DummyVecEnv(factories)
    gamma = 0.995
    model = BehaviorRegularizedSAC(
        RouteConditionedSACPolicy, vec, learning_rate=args.learning_rate,
        buffer_size=1_000_000, batch_size=512, tau=0.005, gamma=gamma,
        train_freq=(1, "step"), gradient_steps=args.gradient_steps,
        learning_starts=0,
        replay_buffer_class=NStepReplayBuffer,
        replay_buffer_kwargs={"n_steps": args.n_step, "gamma": gamma},
        policy_kwargs={
            "net_arch": [256, 256, 128],
            "residual_limit": args.residual_limit,
            "layer_norm_critic": not args.no_layer_norm_critic},
        ent_coef=args.ent_coef, bc_coef=args.bc_coef,
        verbose=1, seed=args.seed, device="auto",
        tensorboard_log=str(out / "tb"))

    transfer_route_policy(model, args.bc_checkpoint)
    with np.load(args.imitation_validation, allow_pickle=False) as data:
        validation_observations = np.asarray(
            data["observations"], dtype=np.float32)
    if len(validation_observations) > 100_000:
        validation_observations = validation_observations[:100_000]
    teacher = load_policy(args.bc_checkpoint, device=model.device)
    validation_targets = teacher.predict(validation_observations)
    transfer_mse = actor_mse(
        model, validation_observations, validation_targets)
    print(f"held-out BC-to-SAC exact-transfer MSE {transfer_mse:.10f}")
    if transfer_mse > 1e-10:
        raise RuntimeError(f"actor transfer changed BC actions: MSE {transfer_mse}")
    model.save(out / "protected_bc_init")

    replay = collect_full_maze_replay(
        model, args.replay_episodes, args.seed + 10_000,
        args.max_seconds, args.dr_scale)
    seeded = seed_replay_buffer(model, replay)
    if args.critic_warmup_steps:
        warmup_critics(model, args.critic_warmup_steps, batch_size=512)

    evaluation = FullMazeEvalCallback(
        out, args.eval_freq, args.eval_episodes, args.seed + 20_000,
        args.max_seconds, args.dr_scale)
    checkpoint = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq // args.envs),
        save_path=str(out), name_prefix="checkpoint")
    curriculum = ReverseCurriculum(
        start=args.train_start_fraction, step=0.10, threshold=0.60,
        window=25, progress_path=out / "curriculum_progress.json",
        verbose=1)
    callbacks = [curriculum, evaluation, checkpoint]
    randomization = None
    if not args.no_dr_curriculum:
        randomization = RandomizationCurriculum(
            start=args.dr_scale, ceiling=args.dr_ceiling, step=args.dr_step,
            threshold=0.70, window=25,
            # Widening physics while the route is still unrolling confounds two
            # curricula, so ADR waits for the reverse one to come down far
            # enough. See --dr-gate-fraction for why "far enough" is not
            # always zero.
            gate=lambda: curriculum.fraction <= args.dr_gate_fraction,
            progress_path=out / "dr_progress.json", verbose=1)
        callbacks.insert(1, randomization)
    model.learn(
        total_timesteps=args.steps,
        callback=CallbackList(callbacks),
        progress_bar=False)
    model.save(out / "last_model")
    (out / "report.json").write_text(json.dumps({
        "schema": "bc_initialized_sac_finetune_v1",
        "bc_checkpoint": str(Path(args.bc_checkpoint).resolve()),
        "actor_transfer_validation_mse": transfer_mse,
        "seeded_replay_transitions": seeded,
        "steps": args.steps, "envs": args.envs,
        "gradient_steps": args.gradient_steps,
        "update_to_data_ratio": args.gradient_steps / args.envs,
        "residual_limit": args.residual_limit,
        "bc_coef": args.bc_coef,
        "n_step": args.n_step,
        "layer_norm_critic": not args.no_layer_norm_critic,
        "max_seconds": args.max_seconds, "dr_scale": args.dr_scale,
        "dr_ceiling": args.dr_ceiling if randomization else None,
        "dr_gate_fraction": args.dr_gate_fraction if randomization else None,
        "final_dr_scale": randomization.scale if randomization else args.dr_scale,
        "dr_curriculum_history": randomization.history if randomization else [],
        "initial_train_start_fraction": args.train_start_fraction,
        "final_train_start_fraction": curriculum.fraction,
        "curriculum_history": curriculum.history,
        "evaluation_history": evaluation.history,
    }, indent=2), encoding="utf-8")
    vec.close()
    print(f"saved {out / 'last_model.zip'}; protected initial and best retained")


if __name__ == "__main__":
    main()
