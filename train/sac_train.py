"""SAC on the maze, with a reverse curriculum.

    python -m train.sac_train --flat --steps 200000     # smoke test, no maze
    python -m train.sac_train --steps 3000000

SAC rather than PPO because the hardware phase depends on it: bounded residual
learning, symmetric sampling between simulated and real data, and a replay
buffer that real transitions can be poured into are all off-policy techniques,
and PPO can do none of them. Sample efficiency in simulation is a bonus, not
the reason.

**The reverse curriculum is not optional.** A 1023 mm route past 15 holes is a
brutal cold start, and the reward arithmetic makes it worse: falling early
scores below standing still until enough progress is available to outweigh the
fall. Episodes therefore begin near the goal and the start walks backwards as
the success rate holds up, so there is always progress within reach.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (BaseCallback, CallbackList,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from sim.maze_env import MazeEnv
from sim.mjcf_builder import flat_board_layout, load_layout, load_parameters
from sim.randomization import (
    Randomizer, REALISTIC_DR_SCALE, TEACHER_MAX_DR_SCALE)
from train.ball_plate_train import collect_demonstrations as collect_m3_demonstrations
from train.imitation import (pretrain_actor, seed_replay_buffer,
                             warmup_critics)


def make_env(rank: int, flat: bool, randomize: bool, start_fraction: float,
             seed: int):
    def _init():
        layout = flat_board_layout() if flat else load_layout()
        env = MazeEnv(layout=layout, params=load_parameters(),
                      randomizer=Randomizer(enabled=randomize),
                      start_fraction=start_fraction, seed=seed + rank)
        return Monitor(env, info_keywords=("route_completion", "outcome"))
    return _init


def collect_demonstrations(episodes: int, seed: int):
    """Use M2's full-route controller to initialise M4's policy."""
    from contract import policy_contract as pc
    from control.baseline import PurePursuitBaseline

    params = load_parameters()
    env = MazeEnv(layout=load_layout(), params=params, seed=seed)
    controller = PurePursuitBaseline(env.route, params["actuator.max_tilt"])
    observations, next_observations, actions = [], [], []
    rewards, dones, truncateds = [], [], []
    successes = 0
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        while True:
            position, velocity = env.estimator.state
            position, velocity = env.predictor.predict(
                position, velocity, env._command)
            action = np.asarray(pc.angles_to_action(
                *controller(position, velocity), params["actuator.max_tilt"]),
                dtype=np.float32)
            previous = observation.copy()
            observation, reward, terminated, truncated, info = env.step(action)
            observations.append(previous)
            next_observations.append(observation.copy())
            actions.append(action)
            rewards.append(reward)
            dones.append(terminated or truncated)
            truncateds.append(truncated)
            if terminated or truncated:
                successes += info["outcome"] == "goal"
                break
    print(f"collected {len(observations):,} maze demonstrations; "
          f"controller success {successes}/{episodes}")
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "next_observations": np.asarray(next_observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=np.float32),
        "truncated": np.asarray(truncateds, dtype=bool),
    }


class BehaviorRehearsal(BaseCallback):
    """Periodically anchor M4's actor to retained M3 and maze skills."""

    def __init__(self, observations, actions, interval: int = 10_000,
                 epochs: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.observations = observations
        self.actions = actions
        self.interval = interval
        self.epochs = epochs
        self.next_rehearsal = interval

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_rehearsal:
            return True
        pretrain_actor(self.model, self.observations,
                       self.actions, self.epochs)
        self.logger.record("rehearsal/updates", 1)
        while self.next_rehearsal <= self.num_timesteps:
            self.next_rehearsal += self.interval
        return True


class ReverseCurriculum(BaseCallback):
    """Walk the episode start backwards from the goal as success holds up.

    Advances on a *windowed* success rate rather than a single episode, and
    never retreats: a stage that has been cleared stays cleared, so noise in one
    window cannot undo progress. Retreating on a bad window was tried in the
    previous project and produced a curriculum that oscillated instead of
    advancing.
    """

    def __init__(self, start: float = 0.92, step: float = 0.06,
                 threshold: float = 0.6, window: int = 25,
                 progress_path: Path | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.fraction = start
        self.step_size = step
        self.threshold = threshold
        self.window = window
        self.progress_path = progress_path
        self.outcomes: list[bool] = []
        self.history: list[tuple[int, float, float]] = []
        self.last_success_rate = 0.0

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_start_fraction", self.fraction)
        self._record()
        self._persist()

    def _record(self) -> None:
        """Expose the state that determines curriculum transitions."""
        self.logger.record("curriculum/start_fraction", self.fraction)
        self.logger.record("curriculum/window_success_rate",
                           self.last_success_rate)

    def _persist(self) -> None:
        """Keep stage history even when a long run is interrupted."""
        if self.progress_path is None:
            return
        self.progress_path.write_text(json.dumps({
            "history": self.history,
            "current_start_fraction": self.fraction,
            "window_success_rate": self.last_success_rate,
            "pending_outcomes": len(self.outcomes),
        }, indent=2), encoding="utf-8")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            self.outcomes.append(info.get("outcome") == "goal")
            self.last_success_rate = float(np.mean(
                self.outcomes[-self.window:]))
            self._record()
            self._persist()
            if len(self.outcomes) < self.window:
                continue
            rate = self.last_success_rate
            if rate >= self.threshold and self.fraction > 0.0:
                self.fraction = max(0.0, self.fraction - self.step_size)
                self.training_env.env_method("set_start_fraction", self.fraction)
                self.history.append((self.num_timesteps, self.fraction, rate))
                self.outcomes.clear()
                self.last_success_rate = 0.0
                self._record()
                self._persist()
                if self.verbose:
                    print(f"  [{self.num_timesteps:>9,}] success {rate:.0%} "
                          f"-> start fraction {self.fraction:.2f}")
        return True


class RandomizationCurriculum(BaseCallback):
    """Widen domain randomisation to the edge of what the policy can hold.

    Automatic domain randomisation, in the shape ``ReverseCurriculum`` already
    established: advance on a windowed success rate, never retreat. Fixed-scale
    randomisation trains one narrow band of physics and under-samples the cases
    that decide a transfer; ADR keeps the range as wide as competence allows.

    The ceiling is a real boundary, not caution. ``TEACHER_MAX_DR_SCALE`` is
    where the MPC demonstrations stop, so a base policy pushed past it is being
    asked about plants its teacher never saw.
    """

    def __init__(self, start: float = REALISTIC_DR_SCALE,
                 ceiling: float = TEACHER_MAX_DR_SCALE, step: float = 0.03,
                 threshold: float = 0.7, window: int = 25,
                 gate=None, progress_path: Path | None = None,
                 verbose: int = 0):
        super().__init__(verbose)
        if start > ceiling:
            raise ValueError(
                f"start scale {start} is already above ceiling {ceiling}")
        #: Optional predicate that must hold before the scale may advance.
        #: Two curricula moving at once would confound each other, so the
        #: caller gates this one on the reverse curriculum being finished.
        self.gate = gate
        self.scale = float(start)
        self.ceiling = float(ceiling)
        self.step_size = float(step)
        self.threshold = float(threshold)
        self.window = int(window)
        self.progress_path = progress_path
        self.outcomes: list[bool] = []
        self.history: list[tuple[int, float, float]] = []
        self.last_success_rate = 0.0

    def _on_training_start(self) -> None:
        self.training_env.env_method("set_randomization_scale", self.scale)
        self._record()
        self._persist()

    def _record(self) -> None:
        self.logger.record("curriculum/dr_scale", self.scale)
        self.logger.record("curriculum/dr_window_success_rate",
                           self.last_success_rate)

    def _persist(self) -> None:
        if self.progress_path is None:
            return
        self.progress_path.write_text(json.dumps({
            "history": self.history,
            "current_dr_scale": self.scale,
            "ceiling": self.ceiling,
            "window_success_rate": self.last_success_rate,
            "pending_outcomes": len(self.outcomes),
        }, indent=2), encoding="utf-8")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            if self.gate is not None and not self.gate():
                # Hold the window empty while gated, so the first eligible
                # decision is made on episodes from the gated-open regime.
                self.outcomes.clear()
                continue
            self.outcomes.append(info.get("outcome") == "goal")
            self.last_success_rate = float(np.mean(
                self.outcomes[-self.window:]))
            self._record()
            self._persist()
            if len(self.outcomes) < self.window:
                continue
            rate = self.last_success_rate
            if rate >= self.threshold and self.scale < self.ceiling:
                self.scale = min(self.ceiling, self.scale + self.step_size)
                self.training_env.env_method(
                    "set_randomization_scale", self.scale)
                self.history.append((self.num_timesteps, self.scale, rate))
                self.outcomes.clear()
                self.last_success_rate = 0.0
                self._record()
                self._persist()
                if self.verbose:
                    print(f"  [{self.num_timesteps:>9,}] success {rate:.0%} "
                          f"-> dr scale {self.scale:.3f}")
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--flat", action="store_true",
                        help="bare plate: same route, no walls or holes")
    parser.add_argument("--no-randomize", action="store_true")
    parser.add_argument("--start-fraction", type=float, default=0.92)
    parser.add_argument("--out", default="artifacts/sac")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo-episodes", type=int, default=20)
    parser.add_argument("--bc-epochs", type=int, default=15)
    parser.add_argument("--init-actor",
                        default="artifacts/sac_m3_v4/pretrained_policy.zip")
    parser.add_argument("--m3-rehearsal-episodes", type=int, default=30)
    parser.add_argument("--rehearsal-freq", type=int, default=10_000)
    parser.add_argument("--critic-warmup-steps", type=int, default=2_000)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--gradient-steps", type=int, default=None,
                        help="updates per vec step; defaults to --envs, i.e. "
                             "one gradient per environment step. Lower it to "
                             "trade sample efficiency for wall clock -- the "
                             "gradient loop, not the simulator, is the "
                             "bottleneck here, with the envs about 90 percent "
                             "idle at the default.")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    factory = [make_env(i, args.flat, not args.no_randomize,
                        args.start_fraction, args.seed)
               for i in range(args.envs)]
    vec = SubprocVecEnv(factory) if args.envs > 1 else DummyVecEnv(factory)

    model = SAC(
        "MlpPolicy", vec,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        batch_size=512,
        tau=0.005,
        gamma=0.995,               # ~10 s horizon at 20 Hz
        train_freq=(1, "step"),
        gradient_steps=args.gradient_steps or args.envs,
        learning_starts=0 if args.demo_episodes else 10_000,
        policy_kwargs={"net_arch": [256, 256]},
        ent_coef=args.ent_coef,
        verbose=1, seed=args.seed, device="auto",
        tensorboard_log=str(out / "tb"),
    )

    if args.init_actor:
        source = SAC.load(args.init_actor, device=model.device)
        model.actor.load_state_dict(source.actor.state_dict())
        print(f"loaded M3 actor from {args.init_actor}")

    anchor_obs = anchor_actions = None
    if args.demo_episodes:
        maze_demo = collect_demonstrations(
            args.demo_episodes, args.seed + 40_000)
        maze_obs = maze_demo["observations"]
        maze_actions = maze_demo["actions"]
        if args.m3_rehearsal_episodes:
            m3_obs, m3_actions = collect_m3_demonstrations(
                args.m3_rehearsal_episodes, args.seed + 60_000)
            # Balance the two tasks during cloning even though maze episodes
            # contain many more control steps.
            indices = np.resize(np.arange(len(m3_obs)), len(maze_obs))
            anchor_obs = np.concatenate([maze_obs, m3_obs[indices]])
            anchor_actions = np.concatenate(
                [maze_actions, m3_actions[indices]])
        else:
            anchor_obs, anchor_actions = maze_obs, maze_actions
        pretrain_actor(model, anchor_obs, anchor_actions, args.bc_epochs)
        seed_replay_buffer(model, maze_demo)
        if args.critic_warmup_steps:
            warmup_critics(model, args.critic_warmup_steps,
                           batch_size=512)
        model.save(out / "pretrained_policy")

    curriculum = ReverseCurriculum(
        start=args.start_fraction,
        progress_path=out / "curriculum_progress.json",
        verbose=1)
    # Checkpoints so a long run can be evaluated while it is still going, and
    # so an interrupted one is not a total loss.
    checkpoint = CheckpointCallback(
        save_freq=max(1, 50_000 // args.envs), save_path=str(out),
        name_prefix="checkpoint")
    callbacks = [curriculum, checkpoint]
    if anchor_obs is not None and args.rehearsal_freq > 0:
        callbacks.append(BehaviorRehearsal(
            anchor_obs, anchor_actions, interval=args.rehearsal_freq))
    eval_env = None
    if args.eval_freq > 0:
        eval_env = DummyVecEnv([make_env(
            0, args.flat, False, 0.0, args.seed + 50_000)])
        evaluation = EvalCallback(
            eval_env,
            best_model_save_path=str(out / "best"),
            log_path=str(out / "eval"),
            eval_freq=max(1, args.eval_freq // args.envs),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False)
        # Register the transferred/cloned actor as the first candidate.  A
        # worse first SAC checkpoint must not become "best" merely because it
        # is the first periodic evaluation.
        rewards, _ = evaluate_policy(
            model, eval_env, n_eval_episodes=args.eval_episodes,
            deterministic=True, return_episode_rewards=True, warn=False)
        evaluation.best_mean_reward = float(np.mean(rewards))
        (out / "best").mkdir(parents=True, exist_ok=True)
        model.save(out / "best" / "best_model")
        print(f"initial full-route evaluation mean reward "
              f"{evaluation.best_mean_reward:.2f}; registered as best")
        callbacks.append(evaluation)
    model.learn(total_timesteps=args.steps,
                callback=CallbackList(callbacks),
                progress_bar=False)

    model.save(out / "policy")
    (out / "curriculum.json").write_text(json.dumps({
        "history": curriculum.history,
        "final_start_fraction": curriculum.fraction,
        "flat": args.flat,
        "randomized": not args.no_randomize,
        "steps": args.steps,
    }, indent=2), encoding="utf-8")
    print(f"saved {out / 'policy.zip'}, final start fraction "
          f"{curriculum.fraction:.2f}")
    vec.close()
    if eval_env is not None:
        eval_env.close()


if __name__ == "__main__":
    main()
