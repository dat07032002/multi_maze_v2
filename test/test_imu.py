"""Synthetic tests for the BNO086 stream reader.

Everything here runs without hardware. The cases that matter are the ones a
healthy bench session will never show you: a corrupted frame, a dropped byte
mid-stream, a sequence gap. This link already demonstrated it drops bytes above
115200, so the parser's behaviour under corruption is not hypothetical.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from tag_vision.core.board_pose import angles_from_rotation, rotation_from_angles
from tag_vision.hardware.imu import (
    SYNC0,
    SYNC1,
    TYPE_PONG,
    TYPE_SAMPLE,
    TYPE_STATUS,
    BNO086Stream,
    ImuError,
    crc8,
    quat_to_rotation,
)


def rotation_to_quat(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 -> (i, j, k, real). Test-side inverse of quat_to_rotation."""
    r = np.asarray(rotation, dtype=np.float64)
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    return (x, y, z, w)


def build_frame(ftype: int, payload: bytes) -> bytes:
    header = bytes([ftype, len(payload)])
    return bytes([SYNC0, SYNC1]) + header + payload + bytes(
        [crc8(header + payload)])


def sample_frame(esp_micros: int, quat, accuracy: int = 3, seq: int = 0) -> bytes:
    payload = esp_micros.to_bytes(4, "little")
    for value in quat:
        payload += int(round(value * 16384.0)).to_bytes(2, "little", signed=True)
    payload += bytes([accuracy, seq])
    return build_frame(TYPE_SAMPLE, payload)


class FakeSerial:
    """Serial-like object that replays a fixed byte string."""

    def __init__(self, data: bytes) -> None:
        self.data = bytearray(data)
        self.written = bytearray()
        self.pos = 0

    @property
    def in_waiting(self) -> int:
        return len(self.data) - self.pos

    def read(self, size: int = 1) -> bytes:
        chunk = bytes(self.data[self.pos:self.pos + size])
        self.pos += len(chunk)
        return chunk

    def write(self, data: bytes) -> int:
        self.written.extend(data)
        return len(data)

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        self.pos = len(self.data)

    def close(self) -> None:
        pass


def stream(data: bytes) -> BNO086Stream:
    return BNO086Stream(transport=FakeSerial(data))


class TestCrc(unittest.TestCase):
    def test_matches_known_vectors(self):
        # Dallas/Maxim CRC-8 poly 0x31. Pinned so a firmware change that alters
        # the polynomial fails here rather than silently rejecting every frame.
        self.assertEqual(crc8(b""), 0x00)
        self.assertEqual(crc8(b"\x00"), 0x00)
        self.assertEqual(crc8(b"\x01\x02\x03"), 0xCC)


class TestQuaternion(unittest.TestCase):
    def test_identity(self):
        self.assertTrue(np.allclose(quat_to_rotation((0, 0, 0, 1)), np.eye(3)))

    def test_rejects_zero_quaternion(self):
        with self.assertRaises(ImuError):
            quat_to_rotation((0, 0, 0, 0))

    def test_normalises_non_unit_input(self):
        # The Q14 round trip leaves the quaternion slightly off unit length.
        scaled = quat_to_rotation((0, 0, 0, 2))
        self.assertTrue(np.allclose(scaled, np.eye(3)))

    def test_round_trips_board_angles(self):
        """quat -> rotation -> angles must recover the simulator's convention.

        This is the test that catches a transposed rotation or a swapped
        quaternion component order: both still produce a valid rotation matrix,
        so nothing else would notice.
        """
        for alpha_deg, beta_deg in [(0, 0), (3, 0), (0, -4), (2.5, 1.5),
                                    (-5, 6)]:
            alpha = math.radians(alpha_deg)
            beta = math.radians(beta_deg)
            rotation = rotation_from_angles(alpha, beta)
            recovered = quat_to_rotation(rotation_to_quat(rotation))
            got_alpha, got_beta = angles_from_rotation(recovered)
            self.assertAlmostEqual(got_alpha, alpha, places=6)
            self.assertAlmostEqual(got_beta, beta, places=6)


class TestFraming(unittest.TestCase):
    def test_parses_a_clean_sample(self):
        quat = rotation_to_quat(rotation_from_angles(math.radians(2.0), 0.0))
        with stream(sample_frame(12345, quat, accuracy=3, seq=7)) as imu:
            sample = imu.read_sample(timeout=0.5)
        self.assertIsNotNone(sample)
        self.assertEqual(sample.esp_micros, 12345)
        self.assertEqual(sample.accuracy, 3)
        self.assertEqual(sample.seq, 7)

    def test_recovers_angles_through_the_wire_format(self):
        """End to end: known angle -> Q14 frame -> parsed angle.

        Q14 quantises each quaternion component to 1/16384 = 6.1e-5, which puts
        a floor of roughly 0.007 deg on any angle recovered from the wire. That
        floor is asserted here rather than papered over with a loose tolerance:
        it is the resolution limit of the telemetry format, and it needs to stay
        far below the BNO's own noise for the format to be the right choice.
        """
        quant_floor_deg = math.degrees(2.0 / 16384.0)
        self.assertLess(quant_floor_deg, 0.01)

        alpha, beta = math.radians(3.0), math.radians(-2.0)
        quat = rotation_to_quat(rotation_from_angles(alpha, beta))
        with stream(sample_frame(1, quat)) as imu:
            sample = imu.read_sample(timeout=0.5)
            got_alpha, got_beta = imu.angles(sample)
        self.assertAlmostEqual(math.degrees(got_alpha), 3.0,
                               delta=quant_floor_deg)
        self.assertAlmostEqual(math.degrees(got_beta), -2.0,
                               delta=quant_floor_deg)

    def test_rejects_corrupted_frame(self):
        frame = bytearray(sample_frame(1, (0, 0, 0, 1)))
        frame[-1] ^= 0xFF  # break the CRC
        with stream(bytes(frame)) as imu:
            self.assertIsNone(imu.read_sample(timeout=0.2))
            self.assertGreater(imu.crc_errors, 0)

    def test_resyncs_after_leading_garbage(self):
        """A dropped byte must not cost more than the frame it landed in."""
        good = sample_frame(99, (0, 0, 0, 1), seq=1)
        with stream(b"\x11\x22\x33" + good) as imu:
            sample = imu.read_sample(timeout=0.5)
        self.assertIsNotNone(sample)
        self.assertEqual(sample.esp_micros, 99)
        self.assertGreater(imu.resyncs, 0)

    def test_skips_status_frames(self):
        data = build_frame(TYPE_STATUS, b"BNO086 ready") + sample_frame(
            5, (0, 0, 0, 1))
        with stream(data) as imu:
            sample = imu.read_sample(timeout=0.5)
            self.assertEqual(sample.esp_micros, 5)
            self.assertEqual(imu.status_messages, ["BNO086 ready"])

    def test_counts_dropped_samples_from_sequence_gap(self):
        """Sequence gaps must be reported, never interpolated over.

        A missing report during a step response is a hole in the measurement.
        Filling it silently would corrupt rise_time_s with invented data.
        """
        data = (sample_frame(1, (0, 0, 0, 1), seq=10)
                + sample_frame(2, (0, 0, 0, 1), seq=14))
        with stream(data) as imu:
            imu.read_sample(timeout=0.5)
            imu.read_sample(timeout=0.5)
        self.assertEqual(imu.dropped, 3)

    def test_sequence_wraps_without_false_drops(self):
        data = (sample_frame(1, (0, 0, 0, 1), seq=255)
                + sample_frame(2, (0, 0, 0, 1), seq=0))
        with stream(data) as imu:
            imu.read_sample(timeout=0.5)
            imu.read_sample(timeout=0.5)
        self.assertEqual(imu.dropped, 0)

    def test_link_duplicate_is_discarded_not_counted_as_drops(self):
        """This link re-delivers frames; a repeat must not read as 255 drops.

        Measured on the bench: the port yielded 1298 frames' worth of bytes in
        5 s when the device had sent 1000, the excess being byte-for-byte
        repeats. The old accounting scored each repeat as
        ``(seq - last - 1) % 256 == 255`` dropped frames, turning a 2% duplicate
        rate into a reported 40546 drops and failing the health gate on a
        perfectly good stream.
        """
        frame = sample_frame(1000, (0, 0, 0, 1), seq=5)
        data = frame + frame + sample_frame(1005, (0, 0, 0, 1), seq=6)
        with stream(data) as imu:
            first = imu.read_sample(timeout=0.5)
            second = imu.read_sample(timeout=0.5)
        self.assertEqual(first.seq, 5)
        self.assertEqual(second.seq, 6)      # the repeat was skipped, not served
        self.assertEqual(imu.duplicates, 1)
        self.assertEqual(imu.dropped, 0)

    def test_duplicate_sample_is_never_handed_to_the_caller(self):
        """A repeated timestamp would flatten a step-response derivative."""
        frame = sample_frame(2000, (0, 0, 0, 1), seq=9)
        with stream(frame + frame + frame) as imu:
            got = []
            while True:
                sample = imu.read_sample(timeout=0.05)
                if sample is None:
                    break
                got.append(sample.esp_micros)
        self.assertEqual(got, [2000])
        self.assertEqual(imu.duplicates, 2)

    def test_same_seq_different_timestamp_is_corruption_not_a_duplicate(self):
        """The firmware increments seq per send, so this cannot be genuine."""
        data = (sample_frame(3000, (0, 0, 0, 1), seq=11)
                + sample_frame(3005, (0, 0, 0, 1), seq=11))
        with stream(data) as imu:
            first = imu.read_sample(timeout=0.5)
            second = imu.read_sample(timeout=0.05)
        self.assertEqual(first.esp_micros, 3000)
        self.assertIsNone(second)
        self.assertEqual(imu.duplicates, 0)
        self.assertGreater(imu.crc_errors, 0)

    def test_drops_are_still_counted_alongside_duplicates(self):
        frame = sample_frame(1, (0, 0, 0, 1), seq=20)
        data = frame + frame + sample_frame(2, (0, 0, 0, 1), seq=24)
        with stream(data) as imu:
            imu.read_sample(timeout=0.5)
            imu.read_sample(timeout=0.5)
        self.assertEqual(imu.duplicates, 1)
        self.assertEqual(imu.dropped, 3)

    def test_returns_none_when_stream_is_silent(self):
        with stream(b"") as imu:
            self.assertIsNone(imu.read_sample(timeout=0.05))


class TestPing(unittest.TestCase):
    def test_matches_token_and_ignores_stale_pong(self):
        """A stale pong from an earlier ping must not be accepted.

        Taking one would report a round trip far shorter than the real link
        latency, which would then be subtracted from step_latency_s and make the
        servo look faster than it is.
        """
        imu = stream(b"")
        stale = build_frame(TYPE_PONG, (0xDEADBEEF).to_bytes(4, "little")
                            + (42).to_bytes(4, "little"))
        imu.serial.data.extend(stale)
        imu.serial.pos = 0

        with self.assertRaises(ImuError):
            imu.ping(timeout=0.05)

    def test_accepts_matching_token(self):
        imu = stream(b"")

        real_write = imu.serial.write

        def write_and_reply(data: bytes) -> int:
            real_write(data)
            token = data[1:5]
            imu.serial.data.extend(
                build_frame(TYPE_PONG, token + (777).to_bytes(4, "little")))
            return len(data)

        imu.serial.write = write_and_reply
        rtt, esp_micros = imu.ping(timeout=0.5)
        self.assertEqual(esp_micros, 777)
        self.assertGreaterEqual(rtt, 0.0)


class TestZeroing(unittest.TestCase):
    def test_zero_makes_the_level_pose_read_zero(self):
        tilted = rotation_from_angles(math.radians(1.5), math.radians(-0.8))
        quat = rotation_to_quat(tilted)
        # Distinct seq values: the firmware increments per send, and the reader
        # now rejects a repeated seq as corruption.
        with stream(sample_frame(1, quat, seq=0)
                    + sample_frame(2, quat, seq=1)) as imu:
            first = imu.read_sample(timeout=0.5)
            imu.set_zero(first.rotation)
            second = imu.read_sample(timeout=0.5)
            alpha, beta = imu.angles(second)
        self.assertAlmostEqual(alpha, 0.0, places=5)
        self.assertAlmostEqual(beta, 0.0, places=5)

    def test_zero_matches_the_camera_convention(self):
        """IMU zeroing must use zero.T @ current, as BoardPoseEstimator does.

        If the two differed, IMU and camera angles could not be compared, and
        the camera cross-check that validates the mount matrix would fail for
        the wrong reason.
        """
        zero = rotation_from_angles(math.radians(0.5), math.radians(0.2))
        current = rotation_from_angles(math.radians(2.0), math.radians(0.2))
        expected = angles_from_rotation(zero.T @ current)

        with stream(sample_frame(1, rotation_to_quat(current))) as imu:
            imu.set_zero(zero)
            sample = imu.read_sample(timeout=0.5)
            got = imu.angles(sample)
        # Tolerance is the Q14 floor: the sample crossed the wire, the zero
        # did not.
        quant_floor_rad = 2.0 / 16384.0
        self.assertAlmostEqual(got[0], expected[0], delta=quant_floor_rad)
        self.assertAlmostEqual(got[1], expected[1], delta=quant_floor_rad)

    def test_capture_zero_averages_and_handles_sign_flips(self):
        """q and -q are the same rotation; averaging must not cancel them."""
        rotation = rotation_from_angles(math.radians(1.0), math.radians(-1.0))
        quat = rotation_to_quat(rotation)
        flipped = tuple(-v for v in quat)
        data = (sample_frame(1, quat, seq=0)
                + sample_frame(2, flipped, seq=1)
                + sample_frame(3, quat, seq=2))
        with stream(data) as imu:
            captured, count = imu.capture_zero(seconds=0.3)
        self.assertEqual(count, 3)
        self.assertTrue(np.allclose(captured, rotation, atol=1e-3))

    def test_mount_rotation_maps_sensor_axes_into_board_axes(self):
        """A 180-degree Y mount flips alpha but preserves beta."""
        board = rotation_from_angles(math.radians(2.0), math.radians(-1.0))
        mount = np.diag([-1.0, 1.0, -1.0])
        sensor = mount.T @ board @ mount
        with stream(sample_frame(1, rotation_to_quat(sensor))) as imu:
            imu.mount_rotation = mount
            sample = imu.read_sample(timeout=0.5)
            alpha, beta = imu.angles(sample)
        self.assertAlmostEqual(math.degrees(alpha), 2.0, delta=0.01)
        self.assertAlmostEqual(math.degrees(beta), -1.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
