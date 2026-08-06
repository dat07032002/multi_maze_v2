"""Tests for the STS3215 wire-level decoding.

Focused on the sign conventions, because they are the part that fails silently:
a wrong sign bit does not raise, it produces plausible numbers that are wrong by
a constant, and every conclusion drawn from them inherits the error.
"""
from __future__ import annotations

import unittest

from tag_vision.hardware.sts3215 import (
    CANONICAL_CONFIG,
    COUNTS_PER_REV,
    Register,
    _to_signed,
    decode_status,
    find_port,
)


class TestSignedDecoding(unittest.TestCase):
    """SMS/STS encodes sign as a magnitude flag, not two's complement."""

    def test_speed_uses_bit_15(self):
        self.assertEqual(_to_signed(100), 100)
        self.assertEqual(_to_signed(0x8000 | 100), -100)
        self.assertEqual(_to_signed(0), 0)

    def test_load_uses_bit_10(self):
        """PRESENT_LOAD's magnitude is 10 bits, so bit 10 is the direction.

        Decoding it as bit 15 leaves the direction bit in the magnitude, so
        every load in one direction reads as 1024 plus its true value. On this
        rig that turned a gentle -24 into a reported +1048 -- 105% of the
        servo's rated torque, from a servo sitting at 25 C doing nothing.
        """
        self.assertEqual(_to_signed(24, sign_bit=10), 24)
        self.assertEqual(_to_signed(1024 + 24, sign_bit=10), -24)
        self.assertEqual(_to_signed(1024, sign_bit=10), 0)
        self.assertEqual(_to_signed(1000, sign_bit=10), 1000)

    def test_every_load_seen_on_this_rig_decodes_small(self):
        """The observed raw values are all small loads, in two directions.

        These are the exact readings that drove a session's worth of wrong
        conclusions about a heavy plate. Under the correct sign bit none of them
        exceeds 10% of the servo's 1000-count rating.
        """
        observed_raw = [0, 24, 68, 72, 1044, 1048, 1052, 1056, 1072]
        decoded = [_to_signed(v, sign_bit=10) for v in observed_raw]
        self.assertTrue(all(abs(v) <= 100 for v in decoded), decoded)
        # Both directions are represented, at comparable magnitude.
        self.assertTrue(any(v > 0 for v in decoded))
        self.assertTrue(any(v < 0 for v in decoded))

    def test_load_magnitude_cannot_exceed_the_rated_range(self):
        """A 10-bit magnitude spans 0-1023, which covers the documented 0-1000."""
        for raw in range(0, 2048, 37):
            self.assertLessEqual(abs(_to_signed(raw, sign_bit=10)), 1023)


class TestStatusDecoding(unittest.TestCase):
    def test_clear_status_is_empty(self):
        self.assertEqual(decode_status(0), [])

    def test_known_flags_are_named(self):
        self.assertIn("overload", decode_status(0x20))
        self.assertIn("voltage out of range", decode_status(0x01))

    def test_unknown_bits_are_surfaced_not_dropped(self):
        self.assertTrue(any("unknown" in n for n in decode_status(0x80)))


class TestPortResolution(unittest.TestCase):
    def test_falls_back_when_no_match(self):
        """A missing by-id entry must fall back, not raise."""
        self.assertEqual(
            find_port("definitely-not-a-real-usb-device", "/dev/ttyUSB9"),
            "/dev/ttyUSB9")


class TestCanonicalConfig(unittest.TestCase):
    def test_holds_the_measured_motion_settings(self):
        self.assertEqual(CANONICAL_CONFIG[Register.ACCELERATION], 50)
        self.assertGreater(CANONICAL_CONFIG[Register.GOAL_SPEED], 0)

    def test_goal_speed_is_never_zero(self):
        """0 is not 'maximum' on this firmware -- it nearly stops the servo."""
        self.assertNotEqual(CANONICAL_CONFIG[Register.GOAL_SPEED], 0)

    def test_counts_per_rev(self):
        self.assertEqual(COUNTS_PER_REV, 4096)


if __name__ == "__main__":
    unittest.main()
