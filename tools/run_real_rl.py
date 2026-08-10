#!/usr/bin/env python3
"""Hardware-safe full-maze model-based RL runner.

The default is PREVIEW: camera, IMU, model and planner run, but the servo bus is
never opened.  ``--execute`` enables motion; Space is still required to arm the
first episode.  Q/Escape always levels, releases torque and exits.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tag_vision.control.ball_state import BallStateFilter  # noqa: E402
from tag_vision.control.directional_calibration import (  # noqa: E402
    DirectionalMotorCalibration, DirectionalMotorOrigin)
from tag_vision.control.fixed_reset_brake import (  # noqa: E402
    FixedResetBrake, FixedResetPhase)
from tag_vision.core.angle_fusion import CameraImuFusion  # noqa: E402
from tag_vision.core.ball_detection import BlueBallDetector  # noqa: E402
from tag_vision.core.board_geometry import BoardGeometry  # noqa: E402
from tag_vision.core.board_pose import BoardPoseEstimator  # noqa: E402
from tag_vision.hardware.imu import BNO086Stream  # noqa: E402
from tag_vision.hardware.sts3215 import Mode, Register, STS3215Bus  # noqa: E402
from tag_vision.rl.cem_torch import TorchCEMPlanner  # noqa: E402
from tag_vision.rl.dynamics_torch import EnsembleDynamics  # noqa: E402
from tag_vision.rl.exploration import (  # noqa: E402
    SmoothRandomExploration, StuckRecovery, episode_policy)
from tag_vision.rl.health import HealthLevel, HealthMonitor  # noqa: E402
from tag_vision.rl.online_training import (  # noqa: E402
    OnlineDynamicsTrainer, moving_transition_count)
from tag_vision.rl.replay import ReplayBuffer  # noqa: E402
from tag_vision.rl.task import MazeTask  # noqa: E402
from tools.camera_imu_check import (  # noqa: E402
    ImuReader, camera_source, relative_angles_deg)
from tools.keyboard_motor_compare import hold_current, release  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIELDS = (
    "time_s", "episode", "phase", "health", "health_reasons",
    "ball_visible", "x_mm", "y_mm", "vx_mm_s", "vy_mm_s", "speed_mm_s",
    "fused_alpha_deg", "fused_beta_deg", "progress_mm", "cross_track_mm",
    "clearance_mm", "proposed_alpha_deg", "proposed_beta_deg",
    "executed_alpha_deg", "executed_beta_deg", "action_overridden",
    "planner_cost", "planner_uncertainty", "planner_latency_ms",
    "camera_fps", "pose_rate", "ball_rate", "imu_rate_hz",
    "fusion_residual_deg", "servo_load_max", "reward", "termination",
)


def load_brake(path: Path) -> FixedResetBrake:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != "tag_reload_brake_v1":
        raise ValueError("unsupported reload brake calibration")
    return FixedResetBrake(
        data["brake_tilt_deg"],
        trigger_speed_mm_s=data["trigger_speed_mm_s"],
        settle_speed_mm_s=data["settle_speed_mm_s"],
        settle_hold_s=data["settle_hold_s"],
        max_brake_duration_s=data["max_brake_duration_s"],
        minimum_brake_duration_s=data.get("minimum_brake_duration_s", 0.35),
        trigger_on_reappearance=data.get("trigger_on_reappearance", True),
    )


def open_camera(source):
    # OpenCV defaults to GStreamer on this host, but its v4l2src pipeline
    # rejects this camera's otherwise valid 1920x1200 MJPEG negotiation. USB
    # device indices work reliably through the direct V4L2 backend. Explicit
    # pipeline/file sources retain automatic backend selection.
    capture = (cv2.VideoCapture(source, cv2.CAP_V4L2)
               if isinstance(source, int) else cv2.VideoCapture(source))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    # Width/height changes can renegotiate the device back to raw UYVY. Set
    # compression and rate afterwards so USB transfer remains MJPEG.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FPS, 60)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--camera", default="auto")
    parser.add_argument("--tag-detection", choices=("native", "calibrated"),
                        default="native",
                        help="detect tags at 1920x1200 or resized calibration size")
    parser.add_argument("--imu-port", default=None)
    parser.add_argument("--servo-port", default=None)
    parser.add_argument("--calib", type=Path,
                        default=ROOT / "calib/camera_calib.json")
    parser.add_argument("--board", type=Path,
                        default=ROOT / "calib/board_tags.json")
    parser.add_argument("--camera-zero", type=Path,
                        default=ROOT / "calib/board_zero.json")
    parser.add_argument("--imu-zero", type=Path,
                        default=ROOT / "calib/imu_zero.json")
    parser.add_argument("--motor-calibration", type=Path,
                        default=ROOT / "calib/directional_motor.json")
    parser.add_argument("--reload-brake", type=Path,
                        default=ROOT / "calib/reload_brake.json")
    parser.add_argument("--map-dir", type=Path, default=ROOT /
                        "artifacts/camera_maze/20260810_125246")
    parser.add_argument("--checkpoint", type=Path, default=ROOT /
                        "artifacts/rl/models/20260810_150107/dynamics.pt")
    parser.add_argument("--replay", type=Path,
                        default=ROOT / "artifacts/rl/replay.npz")
    parser.add_argument("--control-hz", type=float, default=5.0)
    parser.add_argument("--policy", choices=("explore", "cem"),
                        default="explore")
    parser.add_argument("--continuous-training", action="store_true",
                        help="train in background and hot-swap between episodes")
    parser.add_argument("--fresh-replay", action="store_true",
                        help="start with an empty replay (requires continuous training)")
    parser.add_argument("--train-min-transitions", type=int, default=300)
    parser.add_argument("--train-min-moving-transitions", type=int, default=120,
                        help="moving samples required before first model activation")
    parser.add_argument("--train-every-transitions", type=int, default=100)
    parser.add_argument("--train-epochs", type=int, default=50)
    parser.add_argument("--exploration-hold-s", type=float, default=0.8)
    parser.add_argument("--explore-every-episodes", type=int, default=5,
                        help="one random episode per N after learning; 0 disables")
    parser.add_argument("--cem-stuck-recovery-s", type=float, default=3.0,
                        help="switch a stationary CEM episode to random recovery")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cem-candidates", type=int, default=256)
    parser.add_argument("--cem-iterations", type=int, default=3)
    parser.add_argument("--min-camera-fps", type=float, default=10.0,
                        help="yellow below this measured processing rate")
    parser.add_argument("--max-tilt-deg", type=float, default=4.0)
    parser.add_argument("--policy-max-tilt-deg", type=float, default=4.0,
                        help="RL envelope inside the absolute ±4 degree limit")
    parser.add_argument("--max-delta-deg", type=float, default=0.5)
    parser.add_argument("--max-load", type=float, default=350.0)
    parser.add_argument("--seconds", type=float, default=0.0)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.control_hz <= 0 or args.control_hz > 10:
        parser.error("--control-hz must be in (0, 10]")
    if args.cem_candidates < 64 or args.cem_iterations < 2:
        parser.error("CEM needs at least 64 candidates and 2 iterations")
    if args.exploration_hold_s < 1.0 / args.control_hz:
        parser.error("--exploration-hold-s must span at least one control step")
    if args.explore_every_episodes < 0:
        parser.error("--explore-every-episodes must be nonnegative")
    if args.cem_stuck_recovery_s <= 0:
        parser.error("--cem-stuck-recovery-s must be positive")
    if args.fresh_replay and not args.continuous_training:
        parser.error("--fresh-replay requires --continuous-training")
    if (args.train_min_transitions < 100
            or args.train_min_moving_transitions < 1
            or args.train_every_transitions < 1 or args.train_epochs < 1):
        parser.error("invalid continuous-training schedule")
    if args.min_camera_fps < 2.0 * args.control_hz:
        parser.error("--min-camera-fps must be at least twice --control-hz")
    if args.max_tilt_deg != 4.0:
        parser.error("this first hardware protocol pins --max-tilt-deg to 4.0")
    if not 0 < args.policy_max_tilt_deg <= args.max_tilt_deg:
        parser.error("--policy-max-tilt-deg must be in (0, --max-tilt-deg]")
    required = [args.camera_zero, args.imu_zero, args.motor_calibration,
                args.reload_brake, args.checkpoint,
                 args.map_dir / "map.json",
                args.map_dir / "occupied_inflated.png"]
    if not args.fresh_replay:
        required.append(args.replay)
    for path in required:
        if not path.is_file():
            parser.error(f"missing required file: {path}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or ROOT / "artifacts/rl/runs" / stamp
    output.mkdir(parents=True, exist_ok=False)
    task = MazeTask.load(args.map_dir / "map.json",
                         args.map_dir / "occupied_inflated.png")
    replay = (ReplayBuffer(100_000, task.spec.size)
              if args.fresh_replay else ReplayBuffer.load(args.replay))
    ensemble, model_metadata = EnsembleDynamics.load_checkpoint(
        args.checkpoint, device="cuda")
    if ensemble.observation_size != task.spec.size:
        parser.error("checkpoint and maze observation layouts differ")
    planner = TorchCEMPlanner(
        ensemble, task.spec, horizon=10, candidates=args.cem_candidates,
        iterations=args.cem_iterations,
        max_tilt_deg=args.policy_max_tilt_deg,
        max_delta_deg=args.max_delta_deg)
    # Pay CUDA graph/kernel initialization latency before camera acquisition
    # and before the user can arm an episode. The first live plan otherwise
    # stalls one frame for roughly 180 ms on this machine.
    warm_observation = (replay.observations[0] if replay.size
                        else np.zeros(task.spec.size, dtype=np.float32))
    warm_action = (replay.actions[0] if replay.size
                   else np.zeros(2, dtype=np.float32))
    planner.command(warm_observation, warm_action)
    import torch
    torch.cuda.synchronize()
    planner.reset()
    exploration = SmoothRandomExploration(
        max_tilt_deg=args.policy_max_tilt_deg,
        hold_s=args.exploration_hold_s, seed=args.seed)
    stuck_recovery = StuckRecovery(duration_s=args.cem_stuck_recovery_s)
    trainer = (OnlineDynamicsTrainer(
        ROOT, output / "online_models", epochs=args.train_epochs)
        if args.continuous_training else None)
    health_monitor = HealthMonitor(
        max_tilt_deg=args.max_tilt_deg, max_delta_deg=args.max_delta_deg,
        min_camera_fps=args.min_camera_fps, max_servo_load=args.max_load)
    brake = load_brake(args.reload_brake)

    geometry = BoardGeometry.load(args.board)
    estimator = BoardPoseEstimator(args.calib, geometry, min_tags=4)
    estimator.load_zero(args.camera_zero)
    detector = BlueBallDetector(estimator)
    state_filter = BallStateFilter()
    fusion = CameraImuFusion()
    imu = BNO086Stream(port=args.imu_port)
    imu.load_zero(args.imu_zero)
    reader = ImuReader(imu)
    reader.start()
    camera_input = camera_source(args.camera)
    capture = open_camera(camera_input)
    if not capture.isOpened():
        reader.stop(); imu.close()
        print("Could not open camera")
        return 1

    bus = None
    origin = None
    if args.execute:
        try:
            calibration = DirectionalMotorCalibration.from_json(
                args.motor_calibration)
            bus = STS3215Bus(port=args.servo_port)
            for servo_id in (1, 2):
                if not bus.ping(servo_id):
                    raise RuntimeError(f"servo {servo_id} did not answer")
                if bus.read_byte(servo_id, Register.MODE) != Mode.POSITION:
                    raise RuntimeError(f"servo {servo_id} not in POSITION mode")
            hold_current(bus)
            origin = DirectionalMotorOrigin(calibration)
        except Exception as exc:  # noqa: BLE001
            capture.release(); reader.stop(); imu.close()
            if bus is not None:
                release(bus); bus.close()
            print(f"Could not safely enable servos: {exc}")
            return 1

    manifest = {
        "version": "tag_real_mbrl_run_v1", "execute": args.execute,
        "checkpoint": str(args.checkpoint), "model_metadata": model_metadata,
        "map_dir": str(args.map_dir), "control_hz": args.control_hz,
        "tag_detection": args.tag_detection,
        "cem_candidates": args.cem_candidates,
        "cem_iterations": args.cem_iterations,
        "min_camera_fps": args.min_camera_fps,
        "policy": args.policy,
        "continuous_training": args.continuous_training,
        "fresh_replay": args.fresh_replay,
        "train_min_transitions": args.train_min_transitions,
        "train_min_moving_transitions": args.train_min_moving_transitions,
        "train_every_transitions": args.train_every_transitions,
        "train_epochs": args.train_epochs,
        "exploration_hold_s": args.exploration_hold_s,
        "explore_every_episodes": args.explore_every_episodes,
        "cem_stuck_recovery_s": args.cem_stuck_recovery_s,
        "seed": args.seed,
        "max_tilt_deg": args.max_tilt_deg,
        "policy_max_tilt_deg": args.policy_max_tilt_deg,
        "max_delta_deg": args.max_delta_deg,
        "initial_replay_size": replay.size,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"RL {'ACTIVE' if args.execute else 'PREVIEW'} -> {output}")
    print(f"Tag detection: {args.tag_detection} "
          f"({'1920x1200' if args.tag_detection == 'native' else '1280x800'})")
    print("Reload/place marble anywhere, wait for GREEN, press Space; q quits")

    history: deque[tuple[float, bool, bool]] = deque()
    previous_xy = None
    missed_ball_frames = 0
    previous_fused = None
    previous_fused_time = None
    target = np.zeros(2, dtype=np.float64)
    proposed = np.zeros(2, dtype=np.float64)
    armed = False
    resetting = False
    episode_started = False
    waiting_for_training = False
    # In a continuous run, explicit exploration means the supplied checkpoint
    # is only a bootstrap object for constructing CUDA planner resources.  It
    # receives no control authority until a motion-rich online update succeeds.
    learned_model_ready = (not args.continuous_training
                           or (not args.fresh_replay and args.policy == "cem"))
    learned_episode_count = 0
    active_policy = ("explore" if args.fresh_replay else args.policy)

    def policy_for_episode(_number: int) -> str:
        nonlocal learned_episode_count
        if not args.continuous_training:
            return args.policy
        if not learned_model_ready:
            return "explore"
        learned_episode_count += 1
        return episode_policy(
            episode=learned_episode_count, learned_model_ready=True,
            explore_every=args.explore_every_episodes)

    pending_checkpoint = None
    online_update = 0
    episode = 0
    previous_transition = None
    last_control = 0.0
    last_command = 0.0
    last_servo_read = 0.0
    servo_load_max = 0.0
    servo_stalled_overload = False
    servo_timeouts = 0
    planner_cost = planner_uncertainty = planner_latency_ms = math.nan
    red_streak = 0
    start = time.monotonic()
    last_print = 0.0
    new_transitions = 0
    try:
        with (output / "samples.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            while True:
                frame_start = time.monotonic()
                ok, native = capture.read()
                if not ok:
                    print("\nCamera frame failed; leveling and reconnecting")
                    target[:] = 0.0
                    proposed[:] = 0.0
                    health_monitor.reset_action()
                    if bus is not None:
                        try:
                            bus.sync_write_positions(origin.targets(0.0, 0.0))
                        except Exception as exc:  # noqa: BLE001
                            print(f"\nCould not level after camera failure: {exc}")
                            break
                    capture.release()
                    recovered = False
                    for attempt in range(1, 6):
                        time.sleep(0.25)
                        capture = open_camera(camera_input)
                        if capture.isOpened():
                            ok, native = capture.read()
                            if ok:
                                print(f"Camera recovered on attempt {attempt}")
                                recovered = True
                                frame_start = time.monotonic()
                                break
                        capture.release()
                    if not recovered:
                        print("Camera unavailable after 5 reconnect attempts; "
                              "stopping safely")
                        break
                now = time.monotonic()
                if trainer is not None:
                    result = trainer.poll()
                    if result is not None:
                        if result.checkpoint is not None:
                            pending_checkpoint = result.checkpoint
                            print(f"\nOnline update {result.update_index} trained "
                                  f"from {result.replay_size} transitions")
                        else:
                            print(f"\nOnline training update "
                                  f"{result.update_index} failed; keeping "
                                  f"current policy (see {result.log_path})")
                calibrated = cv2.resize(native, estimator.image_size,
                                        interpolation=cv2.INTER_AREA)
                tag_frame = (native if args.tag_detection == "native"
                             else calibrated)
                pose = estimator.estimate(cv2.cvtColor(
                    tag_frame, cv2.COLOR_BGR2GRAY))
                pose_ok = pose is not None and pose.reprojection_px <= 4.0
                ball = detector.detect(
                    calibrated, pose, previous_xy_m=previous_xy) if pose_ok else None
                if ball is not None:
                    previous_xy = ball.board_xy_m.copy()
                    missed_ball_frames = 0
                else:
                    missed_ball_frames += 1
                    # A physical reload is a legitimate large position jump.
                    # Drop the local tracking gate after a short absence so
                    # colour/shape detection can reacquire anywhere on board.
                    if missed_ball_frames > 3:
                        previous_xy = None
                ball_state = state_filter.update(
                    now, ball.board_xy_m * 1000.0 if ball is not None else None)
                fresh_ball = bool(ball is not None and ball_state is not None
                                  and ball_state.measurement_age_s <= 0.12)

                latest = reader.latest()
                imu_angle = (relative_angles_deg(
                    [latest], imu.zero_rotation, imu.mount_rotation)[0]
                    if latest is not None else None)
                camera_angle = (np.array([pose.alpha_deg, pose.beta_deg])
                                if pose_ok else None)
                fused = fusion.update(
                    camera_angle, imu_angle, timestamp=now,
                    imu_timestamp=latest[0] if latest else None)
                fused_angle = fused.array if fused is not None else None
                angle_rate = np.zeros(2)
                if (fused_angle is not None and previous_fused is not None
                        and now > previous_fused_time):
                    angle_rate = ((fused_angle - previous_fused)
                                  / (now - previous_fused_time))
                if fused_angle is not None:
                    previous_fused = fused_angle.copy()
                    previous_fused_time = now

                history.append((now, pose_ok, fresh_ball))
                while history and history[0][0] < now - 2.0:
                    history.popleft()
                span = max(1e-6, history[-1][0] - history[0][0])
                camera_fps = (len(history) - 1) / span
                pose_rate = sum(row[1] for row in history) / len(history)
                ball_rate = sum(row[2] for row in history) / len(history)
                imu_rate = len(reader.recent(1.0))
                residual = (max(abs(value) for value in fused.camera_residual_deg)
                            if fused is not None and camera_angle is not None
                            else 0.0)
                ball_age = (ball_state.measurement_age_s
                            if ball_state is not None else math.inf)
                health = health_monitor.classify(
                    camera_fps=camera_fps, pose_rate=pose_rate,
                    ball_rate=ball_rate, ball_age_s=ball_age,
                    imu_rate_hz=imu_rate, fusion_residual_deg=residual,
                    control_latency_s=time.monotonic() - frame_start,
                    servo_timeout_count=servo_timeouts,
                    # Moving servos can report a transient high load. Treat it
                    # as a fault only when the same sample says they are stalled.
                    servo_load_max=(servo_load_max
                                    if servo_stalled_overload else 0.0))
                red_streak = red_streak + 1 if health.level == HealthLevel.RED else 0

                observation = route = None
                if fresh_ball and fused_angle is not None:
                    observation, route = task.observation(
                        position_mm=ball_state.position_mm,
                        velocity_mm_s=ball_state.velocity_mm_s,
                        angles_deg=fused_angle, angle_rates_deg_s=angle_rate,
                        previous_action_deg=target,
                        stuck=ball_state.speed_mm_s < 3.0)

                reward = math.nan
                termination = ""
                phase = "DISARMED"
                overridden = False
                reset_command = (brake.update(ball_state, now)
                                 if episode_started else None)
                if (episode_started and not resetting
                        and not waiting_for_training and ball_age > 0.25):
                    # Level immediately on a detection gap, but do not end the
                    # episode until the brake's longer absence debounce has
                    # positively confirmed a physical drop. If detection
                    # returns first, FixedResetBrake clears its pending timer
                    # and normal control resumes on the next fresh observation.
                    previous_transition = None
                    target[:] = 0.0
                    proposed[:] = 0.0
                    health_monitor.reset_action()
                    phase = "CONFIRMING_DROP"
                    if (reset_command is not None
                            and reset_command.phase == FixedResetPhase.ARMED):
                        armed = False
                        resetting = True
                        phase = "BALL_LOST"
                        # Confirmed absence is an episode boundary and a safe
                        # time to checkpoint without delaying active control.
                        replay.save(output / "replay_after_run.npz")
                        handle.flush()
                        if (trainer is not None and trainer.should_start(
                                replay.size, minimum=args.train_min_transitions,
                                every=args.train_every_transitions)
                                and moving_transition_count(
                                    replay, task.spec) >=
                                args.train_min_moving_transitions):
                            trainer.start(replay, replay.size)
                            print(f"\nStarted online dynamics update from "
                                  f"{replay.size} fresh transitions ("
                                  f"{moving_transition_count(replay, task.spec)} "
                                  "moving)")

                if resetting:
                    if reset_command is None:
                        reset_command = brake.update(ball_state, now)
                    target = np.clip(reset_command.tilt_deg,
                                     -args.max_tilt_deg, args.max_tilt_deg)
                    proposed = target.copy()
                    phase = reset_command.phase.value.upper()
                    if reset_command.episode_ready:
                        resetting = False
                        target[:] = 0.0
                        if trainer is not None and trainer.running:
                            armed = False
                            waiting_for_training = True
                            phase = "TRAINING_WAIT"
                        else:
                            if pending_checkpoint is not None:
                                ensemble, model_metadata = (
                                    EnsembleDynamics.load_checkpoint(
                                        pending_checkpoint, device="cuda"))
                                planner = TorchCEMPlanner(
                                    ensemble, task.spec, horizon=10,
                                    candidates=args.cem_candidates,
                                    iterations=args.cem_iterations,
                                    max_tilt_deg=args.policy_max_tilt_deg,
                                    max_delta_deg=args.max_delta_deg)
                                warm = (observation if observation is not None
                                        else np.zeros(task.spec.size,
                                                      dtype=np.float32))
                                planner.command(warm, target)
                                torch.cuda.synchronize(); planner.reset()
                                learned_model_ready = True
                                online_update += 1
                                print(f"\nActivated online model "
                                      f"{pending_checkpoint}")
                                pending_checkpoint = None
                            armed = True
                            episode += 1
                            active_policy = policy_for_episode(episode)
                            task.reset(now, ball_state.position_mm)
                            planner.reset(); exploration.reset()
                            stuck_recovery.reset()
                            health_monitor.reset_action()
                            previous_transition = None
                            phase = "EPISODE_READY"
                elif waiting_for_training:
                    phase = "TRAINING_WAIT"
                    target[:] = 0.0
                    proposed[:] = 0.0
                    if trainer is None or not trainer.running:
                        if pending_checkpoint is not None:
                            ensemble, model_metadata = (
                                EnsembleDynamics.load_checkpoint(
                                    pending_checkpoint, device="cuda"))
                            planner = TorchCEMPlanner(
                                ensemble, task.spec, horizon=10,
                                candidates=args.cem_candidates,
                                iterations=args.cem_iterations,
                                max_tilt_deg=args.policy_max_tilt_deg,
                                max_delta_deg=args.max_delta_deg)
                            warm = (observation if observation is not None
                                    else np.zeros(task.spec.size,
                                                  dtype=np.float32))
                            planner.command(warm, target)
                            torch.cuda.synchronize(); planner.reset()
                            learned_model_ready = True
                            online_update += 1
                            print(f"\nActivated online model "
                                  f"{pending_checkpoint}")
                            pending_checkpoint = None
                        waiting_for_training = False
                        if fresh_ball:
                            armed = True
                            episode += 1
                            active_policy = policy_for_episode(episode)
                            task.reset(now, ball_state.position_mm)
                            exploration.reset(); stuck_recovery.reset()
                            health_monitor.reset_action()
                            previous_transition = None
                            phase = "EPISODE_READY"
                        else:
                            resetting = True
                elif armed and observation is not None and route is not None:
                    if (active_policy == "cem"
                            and stuck_recovery.update(
                                ball_state.speed_mm_s, now)):
                        active_policy = "recovery"
                        exploration.reset()
                        print("\nCEM remained stationary; switching this "
                              "episode to random recovery exploration")
                    phase = ({"explore": "EXPLORE",
                              "recovery": "RECOVERY_EXPLORE"}.get(
                                  active_policy, "RL_EPISODE"))
                    if now - last_control >= 1.0 / args.control_hz:
                        if previous_transition is not None:
                            old_obs, old_action, old_overridden = previous_transition
                            step = task.step_result(
                                observation, route, timestamp_s=now,
                                safety_abort=red_streak >= 3)
                            reward = step.reward
                            termination = step.reason or ""
                            replay.add(old_obs, old_action, reward, observation,
                                       step.terminated, step.truncated,
                                       old_overridden)
                            new_transitions += 1
                            if step.terminated or step.truncated:
                                armed = False
                                target[:] = 0.0
                                previous_transition = None
                                phase = (step.reason or "DONE").upper()
                                # A timeout commonly means the marble is
                                # wedged but still visible.  Continuous mode
                                # must not fall back to a manual Space press:
                                # checkpoint the completed episode, train if
                                # due, then resume from the current position.
                                if (args.continuous_training
                                        and step.reason == "timeout"):
                                    replay.save(output / "replay_after_run.npz")
                                    handle.flush()
                                    if (trainer is not None
                                            and trainer.should_start(
                                                replay.size,
                                                minimum=args.train_min_transitions,
                                                every=args.train_every_transitions)
                                            and moving_transition_count(
                                                replay, task.spec) >=
                                            args.train_min_moving_transitions):
                                        trainer.start(replay, replay.size)
                                        print(f"\nStarted online dynamics update "
                                              f"from {replay.size} fresh "
                                              "transitions after timeout ("
                                              f"{moving_transition_count(replay, task.spec)} "
                                              "moving)")
                                    if trainer is not None and trainer.running:
                                        waiting_for_training = True
                                        phase = "TRAINING_WAIT"
                                    else:
                                        armed = True
                                        episode += 1
                                        active_policy = policy_for_episode(
                                            episode)
                                        task.reset(now,
                                                   ball_state.position_mm)
                                        planner.reset(); exploration.reset()
                                        stuck_recovery.reset()
                                        health_monitor.reset_action()
                                        phase = "EPISODE_READY"
                                        print(f"\nEpisode {episode} restarted "
                                              "automatically after timeout")
                        if armed:
                            if active_policy in ("explore", "recovery"):
                                proposed = exploration.command(now).target_deg
                                planner_cost = planner_uncertainty = math.nan
                                planner_latency_ms = 0.0
                                phase = ("RECOVERY_EXPLORE"
                                         if active_policy == "recovery"
                                         else "EXPLORE")
                            else:
                                plan_start = time.perf_counter()
                                plan = planner.command(observation, target)
                                torch.cuda.synchronize()
                                planner_latency_ms = 1000.0 * (
                                    time.perf_counter() - plan_start)
                                planner_cost = plan.cost
                                planner_uncertainty = plan.uncertainty
                                proposed = plan.action_deg.astype(np.float64)
                            action = health_monitor.safe_action(proposed, health)
                            target = (action.executed_deg if bus is not None
                                      else np.zeros(2, dtype=np.float64))
                            overridden = action.overridden or bus is None
                            if bus is None:
                                health_monitor.reset_action()
                            previous_transition = (
                                observation.copy(), target.copy(), overridden)
                        last_control = now

                if bus is not None and now - last_command >= 0.10:
                    try:
                        bus.sync_write_positions(origin.targets(*target))
                        last_command = now
                    except Exception as exc:  # noqa: BLE001
                        servo_timeouts += 1
                        print(f"\nServo command failed; stopping safely: {exc}")
                        break
                if bus is not None and now - last_servo_read >= 0.20:
                    last_servo_read = now
                    try:
                        states = [bus.read_state(index) for index in (1, 2)]
                        servo_load_max = max(abs(state.load) for state in states)
                        servo_stalled_overload = any(
                            abs(state.load) > args.max_load and not state.moving
                            for state in states)
                        if servo_stalled_overload:
                            servo_timeouts += 1
                            print("\nSERVO LOAD FAULT")
                            break
                    except Exception as exc:  # noqa: BLE001
                        servo_timeouts += 1
                        print(f"\nServo state read failed; stopping safely: {exc}")
                        break

                writer.writerow({
                    "time_s": now - start, "episode": episode,
                    "phase": phase, "health": health.level.name,
                    "health_reasons": ",".join(health.reasons),
                    "ball_visible": int(fresh_ball),
                    "x_mm": ball_state.position_mm[0] if fresh_ball else "",
                    "y_mm": ball_state.position_mm[1] if fresh_ball else "",
                    "vx_mm_s": ball_state.velocity_mm_s[0] if fresh_ball else "",
                    "vy_mm_s": ball_state.velocity_mm_s[1] if fresh_ball else "",
                    "speed_mm_s": ball_state.speed_mm_s if fresh_ball else "",
                    "fused_alpha_deg": fused_angle[0] if fused_angle is not None else "",
                    "fused_beta_deg": fused_angle[1] if fused_angle is not None else "",
                    "progress_mm": route.progress_mm if route else "",
                    "cross_track_mm": route.cross_track_mm if route else "",
                    "clearance_mm": (observation[task.spec.index("clearance")]
                                     if observation is not None else ""),
                    "proposed_alpha_deg": proposed[0],
                    "proposed_beta_deg": proposed[1],
                    "executed_alpha_deg": target[0],
                    "executed_beta_deg": target[1],
                    "action_overridden": int(overridden),
                    "planner_cost": planner_cost,
                    "planner_uncertainty": planner_uncertainty,
                    "planner_latency_ms": planner_latency_ms,
                    "camera_fps": camera_fps, "pose_rate": pose_rate,
                    "ball_rate": ball_rate, "imu_rate_hz": imu_rate,
                    "fusion_residual_deg": residual,
                    "servo_load_max": servo_load_max, "reward": reward,
                    "termination": termination,
                })

                if now - last_print >= 1.0:
                    last_print = now
                    print(f"\r{phase:18s} {health.level.name:6s} "
                          f"progress {route.progress_mm if route else 0:6.1f}/"
                          f"{task.route_length_mm:.1f} mm action "
                          f"{target[0]:+.2f},{target[1]:+.2f} "
                          f"plan {planner_latency_ms:5.1f}ms "
                          f"{','.join(health.reasons) or '-'}",
                          end="", flush=True)

                key = -1
                if not args.no_window:
                    display = calibrated.copy()
                    colour = {HealthLevel.GREEN: (0, 255, 0),
                              HealthLevel.YELLOW: (0, 255, 255),
                              HealthLevel.RED: (0, 0, 255)}[health.level]
                    cv2.putText(display,
                                f"{phase} | {health.level.name} | episode {episode}",
                                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                                colour, 2)
                    cv2.putText(display,
                                f"proposal {proposed[0]:+.2f}, {proposed[1]:+.2f}  "
                                f"sent {target[0]:+.2f}, {target[1]:+.2f} deg  "
                                f"plan {planner_latency_ms:.0f} ms",
                                (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                colour, 2)
                    cv2.putText(display,
                                f"health: {','.join(health.reasons) or 'all checks pass'}",
                                (12, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                                colour, 2)
                    if route is not None:
                        cv2.putText(display,
                                    f"progress {route.progress_mm:.0f}/"
                                    f"{task.route_length_mm:.0f} mm  cross "
                                    f"{route.cross_track_mm:+.1f} mm  speed "
                                    f"{ball_state.speed_mm_s:.1f} mm/s",
                                    (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                    colour, 2)
                    if not armed and not resetting and not waiting_for_training:
                        cv2.putText(display,
                                    "CURRENT POSITION IS EPISODE START; PRESS SPACE",
                                    (12, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                    (255, 0, 255), 2)
                    cv2.imshow("real model-based RL", display)
                    key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord(" ") and not armed and not resetting:
                    if not fresh_ball or fused_angle is None:
                        print("\nCannot arm: marble/angle observation is stale")
                    elif health.level != HealthLevel.GREEN:
                        print(f"\nCannot arm: health {health.level.name} "
                              f"{health.reasons}")
                    else:
                        episode += 1
                        active_policy = policy_for_episode(episode)
                        armed = True
                        episode_started = True
                        task.reset(now, ball_state.position_mm)
                        planner.reset(); exploration.reset()
                        stuck_recovery.reset()
                        health_monitor.reset_action()
                        previous_transition = None
                        target[:] = 0.0
                        reset_route = task.route_state(ball_state.position_mm)
                        clearance = task.clearance_rays(
                            ball_state.position_mm)[0]
                        print(f"\nEpisode {episode} armed at "
                              f"({ball_state.position_mm[0]:.1f}, "
                              f"{ball_state.position_mm[1]:.1f}) mm; route "
                              f"progress {reset_route.progress_mm:.1f} mm, "
                              f"cross {reset_route.cross_track_mm:+.1f} mm, "
                              f"clearance {clearance:.1f} mm")
                if args.seconds > 0 and now - start >= args.seconds:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if bus is not None:
            try:
                bus.sync_write_positions(origin.targets(0.0, 0.0))
                time.sleep(0.25)
            finally:
                release(bus); bus.close()
        if trainer is not None:
            trainer.shutdown()
        capture.release(); reader.stop(); imu.close(); cv2.destroyAllWindows()
        replay.save(output / "replay_after_run.npz")
    print(f"\nrun: {output}")
    print(f"new transitions: {new_transitions}; replay: "
          f"{output / 'replay_after_run.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
