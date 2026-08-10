"""Train the M3 random-setpoint ball-on-plate SAC policy.

    python -m train.ball_plate_train --steps 300000 --out artifacts/sac_m3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (CallbackList,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from contract import policy_contract as pc
from sim.ball_plate import BallPlateEnv, SetpointBaseline
from sim.mjcf_builder import load_parameters
from sim.randomization import Randomizer
from train.imitation import pretrain_actor


def make_env(rank: int, randomize: bool, seed: int):
    def _init():
        env = BallPlateEnv(
            params=load_parameters(),
            randomizer=Randomizer(enabled=randomize),
            seed=seed + rank)
        return Monitor(env, info_keywords=("outcome", "target_distance"))
    return _init


def collect_demonstrations(episodes: int, seed: int):
    """Roll out M2's analytic controller for M3 behaviour cloning."""
    params = load_parameters()
    env = BallPlateEnv(params=params, seed=seed)
    controller = SetpointBaseline(params["actuator.max_tilt"],
                                  params["actuator.centre_bias"])
    observations, actions = [], []
    successes = 0
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        while True:
            position, velocity = env.estimator.state
            position, velocity = env.predictor.predict(
                position, velocity, env._command)
            angles = controller(position, velocity, env.target)
            action = np.asarray(pc.angles_to_action(
                *angles, params["actuator.max_tilt"]), dtype=np.float32)
            observations.append(observation.copy())
            actions.append(action)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                successes += info["outcome"] == "goal"
                break
    print(f"collected {len(observations):,} demonstration transitions; "
          f"controller success {successes}/{episodes}")
    return (np.asarray(observations, dtype=np.float32),
            np.asarray(actions, dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=300_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--gradient-steps", type=int, default=4)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--demo-episodes", type=int, default=100)
    parser.add_argument("--bc-epochs", type=int, default=15)
    parser.add_argument("--out", default="artifacts/sac_m3")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    factories = [make_env(i, args.randomize, args.seed)
                 for i in range(args.envs)]
    vec = SubprocVecEnv(factories) if args.envs > 1 else DummyVecEnv(factories)
    eval_env = DummyVecEnv([make_env(0, args.randomize, args.seed + 10_000)])

    model = SAC(
        "MlpPolicy", vec,
        learning_rate=3e-4,
        buffer_size=500_000,
        batch_size=512,
        tau=0.005,
        gamma=0.995,
        train_freq=(1, "step"),
        gradient_steps=args.gradient_steps,
        learning_starts=0 if args.demo_episodes else 10_000,
        policy_kwargs={"net_arch": [256, 256]},
        ent_coef="auto_0.01",
        verbose=1,
        seed=args.seed,
        device="auto",
        tensorboard_log=str(out / "tb"),
    )

    if args.demo_episodes:
        demo_obs, demo_actions = collect_demonstrations(
            args.demo_episodes, args.seed + 30_000)
        pretrain_actor(model, demo_obs, demo_actions, args.bc_epochs)
        model.save(out / "pretrained_policy")

    checkpoint = CheckpointCallback(
        save_freq=max(1, 25_000 // args.envs),
        save_path=str(out), name_prefix="checkpoint")
    evaluation = EvalCallback(
        eval_env,
        best_model_save_path=str(out / "best"),
        log_path=str(out / "eval"),
        eval_freq=max(1, 25_000 // args.envs),
        n_eval_episodes=20,
        deterministic=True,
        render=False)
    model.learn(args.steps, callback=CallbackList([checkpoint, evaluation]),
                progress_bar=False)
    model.save(out / "policy")
    vec.close()
    eval_env.close()
    print(f"saved {out / 'policy.zip'} and best evaluation policy under "
          f"{out / 'best'}")


if __name__ == "__main__":
    main()
