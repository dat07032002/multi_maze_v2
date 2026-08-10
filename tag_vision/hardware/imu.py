"""Host reader for the BNO086 tilt stream from ``firmware/esp32_imu``.

Structured like ``sts3215.py``: pyserial, a context manager, a dataclass of
state, and no vendor SDK. The wire format is defined by the firmware in
``firmware/esp32_imu/esp32_imu.ino``; the two must be changed together.

    [0xA5][0x5A][type][len][payload...][crc8]

Angle extraction deliberately calls ``board_pose.angles_from_rotation`` rather
than deriving alpha and beta here. That function already implements the
simulator's hinge order ``R = Rx(alpha) @ Ry(beta)``, and a second
implementation of the same convention is exactly the drift the servo contract
rewrite was meant to end. If the convention ever changes it must change in one
place.

Zeroing follows ``BoardPoseEstimator``: the level reference is a rotation, and
the reported angles come from ``zero.T @ current``. Same convention, same
meaning, so IMU and camera angles are directly comparable.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import serial

from ..core.board_pose import angles_from_rotation
from .sts3215 import IMU_USB_MARKER, find_port

SYNC0 = 0xA5
SYNC1 = 0x5A

TYPE_SAMPLE = 0x01
TYPE_PONG = 0x02
TYPE_STATUS = 0x03

# Resolved by USB identity, not by number. The old CP2102 now runs the marble
# reload controller; matching the Nano ESP32 explicitly prevents an IMU tool
# from silently opening that unrelated firmware.
def find_imu_port() -> str:
    return find_port(IMU_USB_MARKER, "/dev/ttyACM0")


DEFAULT_PORT = None
# Fixed at 115200 because this link is demonstrably unreliable above it: flash
# reads at 921600 and 460800 both stalled at exactly 0x6000. Raising this
# without re-testing the link will produce jitter that looks like servo latency.
DEFAULT_BAUDRATE = 115200

Q14 = 16384.0


class ImuError(RuntimeError):
    """Raised when the stream cannot be read or the firmware reports failure."""


def crc8(data: bytes) -> int:
    """Dallas/Maxim CRC-8, poly 0x31. Must match the firmware bit for bit."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x31) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


@dataclass(frozen=True)
class ImuSample:
    """One game-rotation-vector report.

    ``esp_micros`` is the device clock and ``host_time`` the host clock at the
    moment the frame finished arriving. They are kept separate on purpose: the
    device clock has the good relative timing, the host clock is what servo
    commands are stamped with, and the offset between them is measured by
    ``ping`` rather than assumed to be zero.
    """

    esp_micros: int
    quat: tuple[float, float, float, float]  # (i, j, k, real)
    accuracy: int
    seq: int
    host_time: float

    @property
    def rotation(self) -> np.ndarray:
        return quat_to_rotation(self.quat)


def quat_to_rotation(quat) -> np.ndarray:
    """(i, j, k, real) -> 3x3 rotation matrix.

    Normalised first: the Q14 fixed-point round trip leaves the quaternion
    slightly off unit length, and feeding that straight into the matrix puts a
    small scale error into every angle.
    """
    x, y, z, w = (float(v) for v in quat)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ImuError("zero-length quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class BNO086Stream:
    """Binary frame reader for the ESP32 IMU firmware."""

    def __init__(
        self,
        port: str | None = None,
        baudrate: int = DEFAULT_BAUDRATE,
        timeout: float = 1.0,
        transport=None,
    ) -> None:
        """``transport`` substitutes an already-open serial-like object.

        This exists so the frame parser can be tested against synthetic byte
        streams -- including corruption and dropped bytes, which are the cases
        that matter and which real hardware will not produce on demand.
        """
        port = port if port is not None else find_imu_port()
        self.port = port
        self.serial = transport if transport is not None else serial.Serial(
            port, baudrate, timeout=timeout)
        self._buffer = bytearray()
        self.zero_rotation: np.ndarray | None = None
        # Proper rotation mapping sensor coordinates into board coordinates.
        # Zeroing removes a fixed attitude offset but cannot remove an axis
        # rotation: relative_sensor = M.T @ relative_board @ M.
        self.mount_rotation = np.eye(3, dtype=np.float64)

        # Stream health. Surfaced, never smoothed over: a dropped report during
        # a step response is a missing measurement, not a value to interpolate.
        self.dropped = 0
        self.crc_errors = 0
        self.resyncs = 0
        # This USB-serial link re-delivers data. Measured on the bench: over 5 s
        # the port yielded 1298 frames' worth of bytes when the device had sent
        # exactly 1000, and the excess were byte-for-byte repeats of the frame
        # before them. The same link stalls at 0x6000 on flash reads above
        # 115200, so this is of a piece with a path that is not entirely well.
        #
        # The firmware increments seq on every transmission, so two frames
        # sharing a seq *and* a device timestamp cannot both be real. That makes
        # the duplicate provable rather than heuristic, and safe to discard.
        self.duplicates = 0
        self._last_seq: int | None = None
        self._last_micros: int | None = None
        self.status_messages: list[str] = []

    # ---- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self.serial.close()

    def __enter__(self) -> "BNO086Stream":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ---- framing -----------------------------------------------------------
    def _next_frame(self, deadline: float) -> tuple[int, bytes] | None:
        """Pull one CRC-valid frame, resyncing past corruption."""
        while time.monotonic() < deadline:
            # Need sync + type + len before the length is even known.
            if len(self._buffer) < 4:
                self._fill()
                continue

            start = self._buffer.find(bytes([SYNC0, SYNC1]))
            if start < 0:
                # Keep one byte: the sync word may straddle this read and the
                # next, and discarding everything would drop the leading 0xA5.
                del self._buffer[:-1]
                self._fill()
                continue
            if start > 0:
                del self._buffer[:start]
                self.resyncs += 1
                continue

            if len(self._buffer) < 4:
                self._fill()
                continue
            ftype = self._buffer[2]
            length = self._buffer[3]
            total = 4 + length + 1
            if len(self._buffer) < total:
                self._fill()
                continue

            payload = bytes(self._buffer[4:4 + length])
            received = self._buffer[4 + length]
            if crc8(bytes([ftype, length]) + payload) != received:
                # Drop only the sync byte, not the whole frame: the real frame
                # may start inside what we mis-parsed as this one.
                del self._buffer[:1]
                self.crc_errors += 1
                continue

            del self._buffer[:total]
            return ftype, payload
        return None

    def _fill(self) -> None:
        waiting = self.serial.in_waiting or 1
        chunk = self.serial.read(waiting)
        if chunk:
            self._buffer.extend(chunk)

    # ---- reading -----------------------------------------------------------
    def read_sample(self, timeout: float = 1.0) -> ImuSample | None:
        """Return the next IMU sample, or None if none arrived in time."""
        deadline = time.monotonic() + timeout
        while True:
            frame = self._next_frame(deadline)
            if frame is None:
                return None
            ftype, payload = frame

            if ftype == TYPE_STATUS:
                self.status_messages.append(
                    payload.decode("utf-8", "replace"))
                continue
            if ftype == TYPE_PONG:
                continue  # handled by ping(), which reads frames itself
            if ftype != TYPE_SAMPLE or len(payload) != 14:
                continue

            host_time = time.time()
            esp_micros = int.from_bytes(payload[0:4], "little")
            quat = tuple(
                int.from_bytes(payload[i:i + 2], "little", signed=True) / Q14
                for i in (4, 6, 8, 10)
            )
            accuracy = payload[12]
            seq = payload[13]

            if self._last_seq is not None:
                if seq == self._last_seq:
                    # Same frame delivered twice by the link. Discard it and
                    # fetch the next rather than handing the caller a sample
                    # that duplicates the previous timestamp -- a repeated
                    # reading would flatten the derivative in a step response
                    # and inflate every rate estimate that counts samples.
                    if esp_micros == self._last_micros:
                        self.duplicates += 1
                        continue
                    # Same seq, different timestamp: the firmware cannot do
                    # that, so treat it as corruption rather than a duplicate.
                    self.crc_errors += 1
                    continue
                gap = (seq - self._last_seq - 1) % 256
                self.dropped += gap

            self._last_seq = seq
            self._last_micros = esp_micros

            return ImuSample(
                esp_micros=esp_micros,
                quat=quat,  # type: ignore[arg-type]
                accuracy=accuracy,
                seq=seq,
                host_time=host_time,
            )

    def drain(self) -> None:
        """Discard buffered samples so the next read reflects the present.

        Needed before a step: a backlog of stale samples would otherwise be
        timestamped as if they described the board after the command.
        """
        self.serial.reset_input_buffer()
        self._buffer.clear()
        self._last_seq = None
        self._last_micros = None

    # ---- timing ------------------------------------------------------------
    def ping(self, timeout: float = 1.0) -> tuple[float, int]:
        """Measure link round-trip. Returns (rtt_seconds, esp_micros_at_pong).

        Every latency figure sysid produces includes the transport delay of this
        link. Measuring it is what makes ``step_latency_s`` a servo property
        rather than a property of the USB cable.
        """
        token = int(time.monotonic_ns()) & 0xFFFFFFFF
        self.serial.write(b"P" + token.to_bytes(4, "little"))
        self.serial.flush()
        sent = time.monotonic()

        deadline = sent + timeout
        while True:
            frame = self._next_frame(deadline)
            if frame is None:
                raise ImuError("no PONG within timeout; is the firmware running?")
            ftype, payload = frame
            if ftype != TYPE_PONG or len(payload) != 8:
                continue
            if int.from_bytes(payload[0:4], "little") != token:
                continue  # a stale pong from an earlier ping
            rtt = time.monotonic() - sent
            return rtt, int.from_bytes(payload[4:8], "little")

    def estimate_clock_offset(self, samples: int = 21) -> tuple[float, float]:
        """Offset from ESP micros to host seconds, and its spread.

        Uses the minimum-RTT ping, not the mean: the shortest round trip is the
        least contaminated by scheduling and buffering, so it gives the tightest
        bound on the true offset. Returns (offset_s, best_rtt_s) where
        ``host_time ~= esp_micros * 1e-6 + offset_s``.
        """
        best_rtt = float("inf")
        best_offset = 0.0
        for _ in range(samples):
            before = time.time()
            rtt, esp_micros = self.ping()
            if rtt < best_rtt:
                # Assume a symmetric link: the pong was generated about rtt/2
                # after the send.
                best_rtt = rtt
                best_offset = (before + rtt / 2.0) - esp_micros * 1e-6
        return best_offset, best_rtt

    # ---- angles ------------------------------------------------------------
    def angles(self, sample: ImuSample) -> tuple[float, float]:
        """(alpha, beta) in radians, relative to the captured level zero."""
        rotation = sample.rotation
        relative_sensor = rotation if self.zero_rotation is None else (
            self.zero_rotation.T @ rotation)
        relative_board = (
            self.mount_rotation @ relative_sensor @ self.mount_rotation.T)
        return angles_from_rotation(relative_board)

    # ---- zeroing -----------------------------------------------------------
    def set_zero(self, rotation: np.ndarray) -> None:
        self.zero_rotation = np.asarray(rotation, dtype=np.float64).copy()

    def capture_zero(self, seconds: float = 2.0) -> tuple[np.ndarray, int]:
        """Average samples over a still interval and adopt the result as level.

        Averaging matters: a single sample carries the sensor's noise straight
        into every angle measured afterwards. Quaternions are averaged by
        summing with sign alignment and renormalising, which is valid here only
        because the samples are all near-identical -- true for a still board,
        and not a general-purpose quaternion mean.
        """
        deadline = time.monotonic() + seconds
        accum = np.zeros(4)
        reference: np.ndarray | None = None
        count = 0
        while time.monotonic() < deadline:
            sample = self.read_sample(timeout=0.5)
            if sample is None:
                continue
            q = np.array(sample.quat, dtype=np.float64)
            if reference is None:
                reference = q
            # q and -q are the same rotation; summing without aligning signs
            # would cancel them.
            if float(np.dot(q, reference)) < 0:
                q = -q
            accum += q
            count += 1
        if count == 0:
            raise ImuError("no samples received while capturing zero")
        mean = accum / np.linalg.norm(accum)
        rotation = quat_to_rotation(tuple(mean))
        self.set_zero(rotation)
        return rotation, count

    def save_zero(self, path: str | Path, extra: dict | None = None) -> None:
        if self.zero_rotation is None:
            raise ImuError("no zero captured")
        payload = {
            "zero_rotation": self.zero_rotation.tolist(),
            "source": "bno086_game_rotation_vector",
            "port": self.port,
            "mount_rotation_sensor_to_board": self.mount_rotation.tolist(),
            "note": (
                "Board rotation treated as level, captured in place while the "
                "board was physically level with the IMU mounted on it. Unlike "
                "the camera zero this does not depend on the camera, but it "
                "does depend on the IMU not having been re-mounted. Re-capture "
                "after any disturbance to the plate or the sensor."
            ),
        }
        if extra:
            payload.update(extra)
        Path(path).write_text(json.dumps(payload, indent=2) + "\n",
                              encoding="utf-8")

    def load_zero(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.zero_rotation = np.asarray(data["zero_rotation"], dtype=np.float64)
        self.mount_rotation = np.asarray(
            data.get("mount_rotation_sensor_to_board", np.eye(3)),
            dtype=np.float64,
        )
