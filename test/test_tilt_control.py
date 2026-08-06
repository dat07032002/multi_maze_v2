"""Tests for the feedforward tilt layer.

The backlash inverse is easy to get subtly wrong -- an offset applied with the
wrong sign makes the error twice as large instead of zero, and nothing raises.
So the cases here are mostly about sign and about state across reversals.
"""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from contract.servo_contract import AxisCalibration, ServoContract
from tag_vision.control.tilt import (
    AxisBacklash,
    BacklashCompensator,
    TiltController,
    TiltLimits,
    load_backlash,
)

BACKLASH_DEG = 1.35


def compensator(deg: float = BACKLASH_DEG) -> BacklashCompensator:
    return BacklashCompensator(
        AxisBacklash(lost_motion_rad=math.radians(deg), measured=True))


def measured_contract() -> ServoContract:
    """A contract with both axes measured, near this rig's real numbers."""
    roll = AxisCalibration(servo_id=1, counts_per_rad=11776.0,
                           center_counts=2215, sign=1,
                           min_counts=929, max_counts=3329, measured=True)
    pitch = AxisCalibration(servo_id=2, counts_per_rad=11412.0,
                            center_counts=2053, sign=1,
                            min_counts=902, max_counts=3302, measured=True)
    return ServoContract(roll=roll, pitch=pitch,
                         max_tilt_rad=math.radians(4.0))


class FakeBus:
    def __init__(self) -> None:
        self.sync_writes: list[dict[int, int]] = []
        self.torque: dict[int, bool] = {}

    def sync_write_positions(self, targets):
        self.sync_writes.append(dict(targets))

    def torque_enable(self, servo_id):
        self.torque[servo_id] = True

    def torque_disable(self, servo_id):
        self.torque[servo_id] = False


class TestBacklashCompensator(unittest.TestCase):
    def test_first_command_applies_no_correction(self):
        """Direction is unknown until the axis has moved.

        Guessing would be wrong half the time, and being wrong doubles the
        error rather than leaving it alone.
        """
        c = compensator()
        self.assertAlmostEqual(c.target_for(math.radians(1.0)),
                               math.radians(1.0))

    def test_upward_move_needs_no_correction_against_an_up_branch_fit(self):
        """sysid conditions from below, so the fit already IS the up branch.

        Treating it as the band centre and adding B/2 put every command half a
        backlash high -- measured at +0.85 deg on roll against a B/2 of 0.66.
        """
        c = compensator()
        c.target_for(0.0)
        aimed = c.target_for(math.radians(1.0))
        self.assertAlmostEqual(math.degrees(aimed), 1.0, places=6)

    def test_downward_move_gives_away_the_whole_band(self):
        c = compensator()
        c.target_for(0.0)
        aimed = c.target_for(math.radians(-1.0))
        self.assertAlmostEqual(math.degrees(aimed), -1.0 - BACKLASH_DEG,
                               places=6)

    def test_sign_is_not_inverted(self):
        """The correction must lead the motion, not oppose it.

        An inverted sign still produces plausible counts and would double the
        positioning error instead of removing it.
        """
        c = compensator()
        c.target_for(0.0)
        self.assertAlmostEqual(c.target_for(math.radians(1.0)),
                               math.radians(1.0))
        c2 = compensator()
        c2.target_for(0.0)
        self.assertLess(c2.target_for(math.radians(-1.0)), math.radians(-1.0))

    def test_reference_conventions(self):
        """Which face the calibration sits on decides the whole correction."""
        B = math.radians(BACKLASH_DEG)
        cases = {
            "up":     (0.0, -B),
            "centre": (B / 2, -B / 2),
            "down":   (B, 0.0),
        }
        for reference, (up_offset, down_offset) in cases.items():
            back = AxisBacklash(lost_motion_rad=B, measured=True,
                                reference=reference)
            self.assertAlmostEqual(back.offset_for(+1), up_offset, places=9)
            self.assertAlmostEqual(back.offset_for(-1), down_offset, places=9)
            self.assertEqual(back.offset_for(0), 0.0)

    def test_rejects_an_unknown_reference(self):
        with self.assertRaises(ValueError):
            AxisBacklash(lost_motion_rad=0.02, reference="middle")

    def test_direction_persists_without_a_reversal(self):
        c = compensator()
        c.target_for(0.0)
        c.target_for(math.radians(1.0))
        aimed = c.target_for(math.radians(2.0))
        self.assertAlmostEqual(math.degrees(aimed), 2.0, places=6)
        self.assertEqual(c.reversals, 0)

    def test_repeated_target_holds_direction(self):
        """A no-op request does not move the plate off the face it rests on."""
        c = compensator()
        c.target_for(0.0)
        first = c.target_for(math.radians(1.0))
        again = c.target_for(math.radians(1.0))
        self.assertAlmostEqual(first, again)
        self.assertEqual(c.reversals, 0)

    def test_reversal_swings_the_correction_by_the_full_lost_motion(self):
        """Up-then-down at the same angle differ by B, which is the point."""
        c = compensator()
        c.target_for(math.radians(-1.0))
        up = c.target_for(math.radians(0.0))
        c.target_for(math.radians(1.0))
        down = c.target_for(math.radians(0.0))
        self.assertAlmostEqual(math.degrees(up - down), BACKLASH_DEG, places=6)
        self.assertEqual(c.reversals, 1)

    def test_unmeasured_backlash_degrades_to_plain_feedforward(self):
        c = BacklashCompensator(AxisBacklash())
        c.target_for(0.0)
        self.assertAlmostEqual(c.target_for(math.radians(1.0)),
                               math.radians(1.0))

    def test_reset_forgets_direction(self):
        c = compensator()
        c.target_for(0.0)
        c.target_for(math.radians(1.0))
        c.reset()
        self.assertAlmostEqual(c.target_for(math.radians(2.0)),
                               math.radians(2.0))


class TestTiltLimits(unittest.TestCase):
    def limits(self) -> TiltLimits:
        return TiltLimits(min_counts={1: 900, 2: 900},
                          max_counts={1: 3300, 2: 3300})

    def test_clamps_and_reports(self):
        lim = self.limits()
        value, hit = lim.clamp(1, 5000)
        self.assertEqual(value, 3300)
        self.assertTrue(hit)

    def test_passes_through_in_range(self):
        value, hit = self.limits().clamp(1, 2000)
        self.assertEqual(value, 2000)
        self.assertFalse(hit)

    def test_round_trips_through_json(self):
        payload = {
            "roll": {"servo_id": 1, "safe": {"min_counts": 371, "max_counts": 3730}},
            "pitch": {"servo_id": 2, "safe": {"min_counts": 347, "max_counts": 3794}},
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "limits.json"
            path.write_text(json.dumps(payload))
            lim = TiltLimits.from_json(path)
        self.assertEqual(lim.min_counts[1], 371)
        self.assertEqual(lim.max_counts[2], 3794)


class TestTiltController(unittest.TestCase):
    def controller(self, **kw) -> tuple[TiltController, FakeBus]:
        bus = FakeBus()
        return TiltController(bus, measured_contract(), **kw), bus

    def test_refuses_an_unmeasured_contract(self):
        """Without a measured contract angle_to_counts would raise anyway."""
        with self.assertRaises(ValueError):
            TiltController(FakeBus(), ServoContract())

    def test_commands_both_axes_in_one_packet(self):
        """Sequential writes put a bus round trip between the two axes."""
        ctl, bus = self.controller()
        ctl.command_angles(math.radians(1.0), math.radians(-1.0))
        self.assertEqual(len(bus.sync_writes), 1)
        self.assertEqual(set(bus.sync_writes[0]), {1, 2})

    def test_zero_action_lands_on_the_measured_centres(self):
        ctl, bus = self.controller()
        ctl.command_action((0.0, 0.0))
        self.assertEqual(bus.sync_writes[0], {1: 2215, 2: 2053})

    def test_reversal_changes_the_counts_by_the_lost_motion(self):
        back = AxisBacklash(lost_motion_rad=math.radians(BACKLASH_DEG),
                            measured=True)
        ctl, bus = self.controller(roll_backlash=back, pitch_backlash=back)
        ctl.command_angles(math.radians(-1.0), 0.0)   # establish downward
        ctl.command_angles(math.radians(1.0), 0.0)    # upward
        up_counts = bus.sync_writes[-1][1]
        ctl.command_angles(math.radians(2.0), 0.0)
        ctl.command_angles(math.radians(1.0), 0.0)    # back down to the same angle
        down_counts = bus.sync_writes[-1][1]
        expected = math.radians(BACKLASH_DEG) * 11776.0
        self.assertAlmostEqual(up_counts - down_counts, expected, delta=2)

    def test_limits_clamp_and_are_reported(self):
        limits = TiltLimits(min_counts={1: 2000, 2: 2000},
                            max_counts={1: 2300, 2: 2300})
        ctl, bus = self.controller(limits=limits)
        command = ctl.command_angles(math.radians(4.0), math.radians(4.0))
        self.assertTrue(command.clamped)
        self.assertLessEqual(max(bus.sync_writes[-1].values()), 2300)

    def test_compensating_flag_reflects_measurement(self):
        ctl, _ = self.controller()
        self.assertFalse(ctl.compensating)
        back = AxisBacklash(lost_motion_rad=0.02, measured=True)
        ctl2, _ = self.controller(roll_backlash=back, pitch_backlash=back)
        self.assertTrue(ctl2.compensating)

    def test_reports_which_axes_reversed(self):
        back = AxisBacklash(lost_motion_rad=math.radians(BACKLASH_DEG),
                            measured=True)
        ctl, _ = self.controller(roll_backlash=back, pitch_backlash=back)
        ctl.command_angles(0.0, 0.0)
        ctl.command_angles(math.radians(1.0), math.radians(1.0))
        command = ctl.command_angles(math.radians(0.5), math.radians(2.0))
        self.assertEqual(command.reversed_axes, (True, False))


class TestLoadBacklash(unittest.TestCase):
    def test_prefers_the_unconditioned_figure(self):
        """The conditioned number describes calibration, not operation."""
        payload = {
            "roll": {"backlash_rad": 0.0005,
                     "backlash_unconditioned_rad": 0.0185},
            "pitch": {"backlash_rad": 0.0024,
                      "backlash_unconditioned_rad": 0.0164},
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cal.json"
            path.write_text(json.dumps(payload))
            roll, pitch = load_backlash(path)
        self.assertAlmostEqual(roll.lost_motion_rad, 0.0185)
        self.assertAlmostEqual(pitch.lost_motion_rad, 0.0164)
        self.assertTrue(roll.measured)

    def test_falls_back_and_marks_unmeasured_when_absent(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cal.json"
            path.write_text(json.dumps({"roll": {}, "pitch": {}}))
            roll, pitch = load_backlash(path)
        self.assertEqual(roll.lost_motion_rad, 0.0)
        self.assertFalse(roll.measured)


if __name__ == "__main__":
    unittest.main()
