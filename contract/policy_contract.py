"""Canonical policy interface shared by simulation and the TAG hardware stack.

The physical TAG estimator publishes board angles and marble coordinates centered
on the board.  Its Dreamer environment shifts those coordinates into the
lower-left board frame before normalization.  MuJoCo already uses that
lower-left frame, so both systems can use the same functions below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


CONTRACT_VERSION = "tag_hardware_policy_v2"


@dataclass(frozen=True)
class TagPolicyContract:
    """Numerical observation and action contract for deployment on TAG."""

    board_width_m: float = 0.259
    board_height_m: float = 0.229
    angle_scale_rad: float = math.radians(12.0)
    relative_goal_points: int = 5
    relative_goal_spacing_m: float = 0.012
    policy_command_scale: tuple[float, float] = (800.0 / 3.0, 800.0 / 3.0)
    policy_command_sign: tuple[float, float] = (-1.0, -1.0)
    bridge_command_limit: tuple[float, float] = (800.0 / 3.0, 800.0 / 3.0)
    servo_home: tuple[float, float] = (500.0, 500.0)
    servo_command_scale: tuple[float, float] = (1.5, 1.5)
    servo_limits: tuple[tuple[float, float], tuple[float, float]] = (
        (100.0, 900.0),
        (100.0, 900.0),
    )

    @property
    def observation_keys(self) -> tuple[str, str, str]:
        return ("image", "states", "goal")

    def hardware_centered_to_lower_left(
        self, centered_xy_m: Iterable[float]
    ) -> np.ndarray:
        centered = np.asarray(tuple(centered_xy_m), dtype=np.float32)
        if centered.shape != (2,):
            raise ValueError("TAG centered marble position must be XY")
        return centered + 0.5 * np.asarray(
            (self.board_width_m, self.board_height_m), dtype=np.float32
        )

    @staticmethod
    def grayscale_patch(image: np.ndarray) -> np.ndarray:
        """Apply TAG's arithmetic channel mean to an already rectified patch."""

        pixels = np.asarray(image)
        if pixels.shape == (64, 64):
            gray = pixels
        elif pixels.shape == (64, 64, 1):
            gray = pixels[..., 0]
        elif pixels.ndim == 3 and pixels.shape[:2] == (64, 64):
            gray = pixels.mean(axis=2)
        else:
            raise ValueError(f"TAG rectified patch must be 64 x 64, got {pixels.shape}")
        return np.clip(gray, 0, 255).astype(np.uint8).reshape(64, 64, 1)

    def normalize_states(
        self,
        board_angles_rad: Iterable[float],
        marble_xy_lower_left_m: Iterable[float],
    ) -> np.ndarray:
        """Return ``[alpha, beta, x, y]`` exactly as the TAG policy sees it.

        Angles are divided by 12 degrees.  Positions are divided by the board
        dimensions and therefore nominally occupy [0, 1], with the board center
        at (0.5, 0.5).  Values are intentionally not clipped, matching TAG's
        real-robot environment and preserving out-of-range diagnostics.
        """

        angles = np.asarray(tuple(board_angles_rad), dtype=np.float32)
        position = np.asarray(tuple(marble_xy_lower_left_m), dtype=np.float32)
        if angles.shape != (2,) or position.shape != (2,):
            raise ValueError("TAG states require two angles and one XY position")
        scales = np.asarray(
            (self.angle_scale_rad, self.angle_scale_rad,
             self.board_width_m, self.board_height_m),
            dtype=np.float32,
        )
        return np.concatenate((angles, position)).astype(np.float32) / scales

    def normalize_relative_goal(self, relative_points_m: np.ndarray) -> np.ndarray:
        """Normalize each lookahead point by its own 12 mm path horizon."""

        points = np.asarray(relative_points_m, dtype=np.float32)
        expected = (self.relative_goal_points, 2)
        if points.shape != expected:
            raise ValueError(f"Expected relative route points {expected}, got {points.shape}")
        horizons = (
            self.relative_goal_spacing_m
            * np.arange(1, self.relative_goal_points + 1, dtype=np.float32)
        )[:, None]
        return (points / horizons).reshape(-1).astype(np.float32)

    def make_observation(
        self,
        image: np.ndarray,
        board_angles_rad: Iterable[float],
        marble_xy_lower_left_m: Iterable[float],
        relative_points_m: np.ndarray,
    ) -> dict[str, np.ndarray]:
        image = self.grayscale_patch(image)
        return {
            "image": image,
            "states": self.normalize_states(board_angles_rad, marble_xy_lower_left_m),
            "goal": self.normalize_relative_goal(relative_points_m),
        }

    def action_to_hiwonder_command(self, action: Iterable[float]) -> np.ndarray:
        """Apply the learner scaling, signs, and the effective bridge clamp."""

        action_array = np.asarray(tuple(action), dtype=np.float32)
        if action_array.shape != (2,):
            raise ValueError(f"TAG action must have shape (2,), got {action_array.shape}")
        if not np.all(np.isfinite(action_array)):
            raise ValueError(f"TAG action must be finite, received {action_array!r}")
        normalized = np.clip(action_array, -1.0, 1.0)
        command = (
            normalized
            * np.asarray(self.policy_command_scale, dtype=np.float32)
            * np.asarray(self.policy_command_sign, dtype=np.float32)
        )
        limit = np.asarray(self.bridge_command_limit, dtype=np.float32)
        return np.clip(command, -limit, limit).astype(np.float32)

    def hiwonder_command_to_servo_target(
        self, command: Iterable[float]
    ) -> np.ndarray:
        command_array = np.asarray(tuple(command), dtype=np.float32)
        if command_array.shape != (2,):
            raise ValueError(f"Hiwonder command must have shape (2,), got {command_array.shape}")
        target = (
            np.asarray(self.servo_home, dtype=np.float32)
            + command_array * np.asarray(self.servo_command_scale, dtype=np.float32)
        )
        limits = np.asarray(self.servo_limits, dtype=np.float32)
        return np.clip(target, limits[:, 0], limits[:, 1]).astype(np.float32)

    def validate_observation(self, observation: Mapping[str, np.ndarray]) -> None:
        missing = set(self.observation_keys).difference(observation)
        extra_policy_fields = {"ball_visible"}.intersection(observation)
        if missing:
            raise ValueError(f"TAG policy observation is missing {sorted(missing)}")
        if extra_policy_fields:
            raise ValueError("ball_visible is not part of the deployed TAG policy input")
        if np.asarray(observation["image"]).shape != (64, 64, 1):
            raise ValueError("TAG image shape mismatch")
        if np.asarray(observation["states"]).shape != (4,):
            raise ValueError("TAG state shape mismatch")
        if np.asarray(observation["goal"]).shape != (10,):
            raise ValueError("TAG route shape mismatch")
        for key in self.observation_keys:
            if not np.all(np.isfinite(observation[key])):
                raise ValueError(f"TAG policy observation {key!r} contains non-finite values")
