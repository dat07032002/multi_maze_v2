"""Vendored copy of the simulator's policy contract, pinned to its version.

`policy_contract.py` here is a copy of `Multi-maze/tag_mujoco/policy_contract.py`.
It is vendored so this project has no import dependency on the simulator repo,
but a vendored copy can drift, and a drifted contract does not raise -- it
silently feeds the policy differently-scaled observations than it was trained
on, which looks like a policy that mysteriously performs worse on hardware.

`check_contract_matches_simulator()` compares against the simulator's file when
it is present, so drift fails loudly during development. It cannot help on a
machine without the simulator checkout, hence EXPECTED_VERSION as a second,
weaker guard.
"""
from __future__ import annotations

import hashlib
import os
import re

from .policy_contract import CONTRACT_VERSION, TagPolicyContract  # noqa: F401

# Bump deliberately, and only together with the simulator.
EXPECTED_VERSION = "tag_hardware_policy_v2"

SIMULATOR_CONTRACT = os.path.expanduser(
    "~/Desktop/TAG/Multi-maze/tag_mujoco/policy_contract.py"
)

# Constants that must agree with the simulator. A version match alone does not
# prove these are unchanged: someone can edit a value without bumping the
# version, and that is the drift most likely to go unnoticed.
PINNED = {
    "board_width_m": 0.259,
    "board_height_m": 0.229,
    "angle_scale_rad": 0.20943951023931953,   # 12 degrees
    "relative_goal_points": 5,
    "relative_goal_spacing_m": 0.012,
}


def check_version() -> None:
    """Fail if the vendored contract is not the version this code expects."""
    if CONTRACT_VERSION != EXPECTED_VERSION:
        raise RuntimeError(
            f"Vendored contract is {CONTRACT_VERSION!r} but this project "
            f"expects {EXPECTED_VERSION!r}. Re-vendor and re-check the "
            "observation scaling before running on hardware."
        )


def check_pinned_values() -> None:
    """Fail if a contract constant changed without the version changing."""
    contract = TagPolicyContract()
    for name, expected in PINNED.items():
        actual = getattr(contract, name)
        if abs(float(actual) - float(expected)) > 1e-9:
            raise RuntimeError(
                f"Contract {name} is {actual!r}, expected {expected!r}. "
                "A constant changed without a version bump; observations "
                "would be scaled differently from training."
            )


def check_contract_matches_simulator(path: str = SIMULATOR_CONTRACT) -> str:
    """Compare the vendored copy against the simulator's, when available.

    Returns a human-readable status. Absence of the simulator checkout is not
    an error -- this project is meant to run on the robot as well.
    """
    here = os.path.join(os.path.dirname(__file__), "policy_contract.py")
    if not os.path.isfile(path):
        return f"simulator contract not found at {path}; skipped byte comparison"

    def digest(target: str) -> str:
        with open(target, "rb") as handle:
            return hashlib.sha256(handle.read().replace(b"\r\n", b"\n")).hexdigest()

    if digest(here) == digest(path):
        return "vendored contract is byte-identical to the simulator's"

    with open(path) as handle:
        found = re.search(r'CONTRACT_VERSION\s*=\s*"([^"]+)"', handle.read())
    theirs = found.group(1) if found else "unknown"
    raise RuntimeError(
        f"Vendored contract differs from {path}. "
        f"Vendored version {CONTRACT_VERSION!r}, simulator {theirs!r}. "
        "Re-vendor before running on hardware: a silent difference in the "
        "observation scaling breaks checkpoint transfer with no error."
    )


def verify(strict: bool = True) -> str:
    """Run every available check. Call this at estimator startup."""
    check_version()
    check_pinned_values()
    status = check_contract_matches_simulator()
    if strict and "not found" in status:
        pass  # absence is tolerated; the pinned checks still ran
    return status
