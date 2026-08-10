"""Background dynamics training with atomic episode snapshots."""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def moving_transition_count(replay, observation_spec, *,
                            minimum_speed_mm_s: float = 3.0) -> int:
    """Count transitions with measured motion on either side of the step."""
    vx = observation_spec.index("vx")
    vy = observation_spec.index("vy")
    indices = [vx, vy]
    before = np.linalg.norm(replay.observations[:replay.size, indices], axis=1)
    after = np.linalg.norm(
        replay.next_observations[:replay.size, indices], axis=1)
    return int(np.count_nonzero(
        np.maximum(before, after) >= float(minimum_speed_mm_s)))


@dataclass(frozen=True)
class TrainingResult:
    update_index: int
    replay_size: int
    checkpoint: Path | None
    log_path: Path
    returncode: int


class OnlineDynamicsTrainer:
    """Run the existing audited trainer outside the real-time control loop."""

    def __init__(self, root: Path, output: Path, *, device: str = "cuda",
                 members: int = 5, epochs: int = 50,
                 batch_size: int = 256) -> None:
        self.root = Path(root)
        self.output = Path(output)
        self.device = str(device)
        self.members = int(members)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.process: subprocess.Popen | None = None
        self._log_handle = None
        self._update_index = 0
        self._submitted_size = 0
        self._model_dir: Path | None = None
        self._log_path: Path | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    @property
    def submitted_size(self) -> int:
        return self._submitted_size

    def should_start(self, replay_size: int, *, minimum: int,
                     every: int) -> bool:
        return (not self.running and replay_size >= minimum
                and replay_size - self._submitted_size >= every)

    def start(self, replay, replay_size: int) -> bool:
        if self.running:
            return False
        self._update_index += 1
        update = self.output / f"update_{self._update_index:03d}"
        snapshot = update.with_suffix(".npz")
        log_path = update.with_suffix(".log")
        replay.save(snapshot)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable, str(self.root / "tools/train_rl_dynamics.py"),
            "--replay", str(snapshot), "--device", self.device,
            "--members", str(self.members), "--epochs", str(self.epochs),
            "--batch-size", str(self.batch_size), "--out", str(update),
        ]
        self.process = subprocess.Popen(
            command, cwd=self.root, stdout=self._log_handle,
            stderr=subprocess.STDOUT, text=True)
        self._submitted_size = int(replay_size)
        self._model_dir = update
        self._log_path = log_path
        return True

    def poll(self) -> TrainingResult | None:
        if self.process is None:
            return None
        returncode = self.process.poll()
        if returncode is None:
            return None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        checkpoint = self._model_dir / "dynamics.pt"
        result = TrainingResult(
            self._update_index, self._submitted_size,
            checkpoint if returncode == 0 and checkpoint.is_file() else None,
            self._log_path, int(returncode))
        self.process = None
        return result

    def shutdown(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        self.process = None
