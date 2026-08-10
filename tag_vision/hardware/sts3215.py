"""FEETECH STS3215 serial-bus servo driver.

Written against the FEETECH SMS/STS protocol rather than a vendor SDK, because no
SDK is installed and the protocol is small. Two details are easy to get wrong and
both are verified against the official ``FTServo_Python`` sources rather than
recalled:

* **Byte order is little-endian.** ``sms_sts.__init__`` passes ``protocol_end=0``
  to ``protocol_packet_handler``, and with ``scs_end == 0`` its ``scs_lobyte``
  returns ``w & 0xFF`` -- low byte first. The older SCS series uses the opposite
  convention, so a driver that guesses produces plausible-looking garbage instead
  of an obvious failure.
* **Length counts parameters plus two.** The SDK writes ``length + 3`` for a WRITE
  whose parameter list is one address byte plus ``length`` data bytes, which is
  the same rule.

Signed quantities (present speed, present load) carry their sign in bit 15 as a
sign-magnitude flag, not as two's complement. ``_to_signed`` handles that.

The register map is the STS3215 control table; ``Register`` documents the subset
this project uses. Position is 0-4095 over 360 degrees, roughly 0.088 deg/count.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import serial

BROADCAST_ID = 0xFE
DEFAULT_BAUDRATE = 1_000_000

# The FEETECH bus is behind a CH340; the IMU is behind a CP2102. Which one gets
# ttyUSB0 depends on enumeration order, and they have already swapped once when
# the rig was unplugged and reconnected. Resolving by USB identity rather than
# by number is what stops a servo tool opening the ESP32, or a flashing tool
# opening the servo bus.
BY_ID_DIR = "/dev/serial/by-id"
SERVO_USB_MARKER = "1a86_USB_Serial"       # CH340
IMU_USB_MARKER = "Arduino_Nano_ESP32"      # Nano ESP32 native USB CDC


def find_port(marker: str, fallback: str) -> str:
    """Resolve a serial port by USB identity, falling back to a fixed path."""
    directory = Path(BY_ID_DIR)
    if directory.is_dir():
        for entry in sorted(directory.iterdir()):
            if marker in entry.name:
                return str(entry.resolve())
    return fallback


def find_servo_port() -> str:
    return find_port(SERVO_USB_MARKER, "/dev/ttyUSB0")

# Baud-rate register values, index is the value written to Register.BAUD_RATE.
BAUDRATE_TABLE = {
    0: 1_000_000,
    1: 500_000,
    2: 250_000,
    3: 128_000,
    4: 115_200,
    5: 76_800,
    6: 57_600,
    7: 38_400,
}


class Instruction(IntEnum):
    PING = 0x01
    READ = 0x02
    WRITE = 0x03
    REG_WRITE = 0x04
    ACTION = 0x05
    RESET = 0x06
    SYNC_READ = 0x82
    SYNC_WRITE = 0x83


class Register(IntEnum):
    """STS3215 control table addresses used by this project."""

    # --- EEPROM, persistent -------------------------------------------------
    FIRMWARE_MAJOR = 0
    FIRMWARE_MINOR = 1
    MODEL = 3               # 2 bytes
    ID = 5
    BAUD_RATE = 6
    RETURN_DELAY = 7
    RESPONSE_LEVEL = 8
    MIN_ANGLE_LIMIT = 9     # 2 bytes
    MAX_ANGLE_LIMIT = 11    # 2 bytes
    MAX_TEMP_LIMIT = 13
    MAX_VOLTAGE_LIMIT = 14
    MIN_VOLTAGE_LIMIT = 15
    MAX_TORQUE_LIMIT = 16   # 2 bytes
    PHASE = 18
    UNLOAD_CONDITION = 19
    LED_ALARM = 20
    P_COEFFICIENT = 21
    D_COEFFICIENT = 22
    I_COEFFICIENT = 23
    MIN_STARTUP_FORCE = 24  # 2 bytes
    CW_DEAD_ZONE = 26
    CCW_DEAD_ZONE = 27
    PROTECTION_CURRENT = 28  # 2 bytes
    ANGULAR_RESOLUTION = 30
    POSITION_OFFSET = 31    # 2 bytes
    MODE = 33
    PROTECTION_TORQUE = 34
    PROTECTION_TIME = 35
    OVERLOAD_TORQUE = 36

    # --- SRAM, volatile -----------------------------------------------------
    TORQUE_ENABLE = 40
    ACCELERATION = 41
    GOAL_POSITION = 42      # 2 bytes
    GOAL_TIME = 44          # 2 bytes
    GOAL_SPEED = 46         # 2 bytes
    TORQUE_LIMIT = 48       # 2 bytes
    LOCK = 55
    PRESENT_POSITION = 56   # 2 bytes
    PRESENT_SPEED = 58      # 2 bytes, signed
    PRESENT_LOAD = 60       # 2 bytes, signed
    PRESENT_VOLTAGE = 62    # x0.1 V
    PRESENT_TEMPERATURE = 63
    ASYNC_WRITE_FLAG = 64
    SERVO_STATUS = 65
    MOVING = 66
    PRESENT_CURRENT = 69    # 2 bytes, x6.5 mA


class Mode(IntEnum):
    """Register.MODE values."""

    POSITION = 0
    CONSTANT_SPEED = 1
    PWM = 2
    STEP = 3


COUNTS_PER_REV = 4096
DEGREES_PER_COUNT = 360.0 / COUNTS_PER_REV

# The motion settings every measurement on this rig was taken under. Pinned
# rather than assumed, because the measured latency and rise time are properties
# of these registers as much as of the servo.
#
# ACCELERATION = 50 comes from measurement, not taste. Stepping servo 2 by 160
# counts gave a board rise time of 270 ms at the old value of 20 and 146 ms at
# 50, with 150 no better than 50 -- so 20 was costing roughly 120 ms for nothing.
#
# GOAL_SPEED was measured as *not* limiting: rise time was flat at 300 ms across
# 100, 200, 500 and 1000. It is pinned only so the conditions stay reproducible.
#
# Do NOT set GOAL_SPEED to 0. On this firmware that is not "maximum speed" -- a
# 160-count step under it moved the board 0.031 deg in 1.26 s, i.e. barely at
# all.
CANONICAL_CONFIG = {
    Register.ACCELERATION: 50,
    Register.GOAL_SPEED: 500,
    Register.P_COEFFICIENT: 32,
    Register.D_COEFFICIENT: 32,
    Register.I_COEFFICIENT: 0,
}


class STSError(Exception):
    """Any protocol-level failure: bad framing, bad checksum, or no reply."""


class STSTimeout(STSError):
    """No status packet arrived within the timeout."""


# Status byte flags. A set flag is servo *health*, not a failed transaction: the
# servo received the instruction, acted on it, and replied. Treating these as
# errors makes a servo with a low-voltage warning look like it is not on the bus.
STATUS_FLAGS = {
    0x01: "voltage out of range",
    0x02: "angle/sensor",
    0x04: "overheat",
    0x08: "overcurrent",
    0x10: "angle limit",
    0x20: "overload",
    0x40: "instruction",
}


def decode_status(error: int) -> list[str]:
    """Human-readable names for the set status bits."""
    if not error:
        return []
    names = [name for bit, name in STATUS_FLAGS.items() if error & bit]
    unknown = error & ~sum(STATUS_FLAGS)
    if unknown:
        names.append(f"unknown bits 0x{unknown:02x}")
    return names


@dataclass(frozen=True)
class ServoState:
    """One snapshot of a servo's SRAM, read in a single bus transaction."""

    position: int
    speed: int
    load: int
    voltage_v: float
    temperature_c: int
    moving: bool
    current_ma: float | None = None


def _to_signed(value: int, sign_bit: int = 15) -> int:
    """SMS/STS encodes sign as a magnitude flag in the top bit, not two's complement."""
    mask = 1 << sign_bit
    if value & mask:
        return -(value & (mask - 1))
    return value


def _checksum(payload: bytes) -> int:
    """Bitwise-not of the sum from ID through the final parameter."""
    return (~sum(payload)) & 0xFF


class STS3215Bus:
    """Half-duplex serial bus carrying one or more STS3215 servos.

    Read operations are safe to run against unknown hardware. Write operations
    can move a servo, so they are kept explicit and are not used by discovery.
    """

    def __init__(
        self,
        port: str | None = None,
        baudrate: int = DEFAULT_BAUDRATE,
        timeout_s: float = 0.05,
    ):
        port = port if port is not None else find_servo_port()
        self.port_name = port
        self.baudrate = int(baudrate)
        self.timeout_s = float(timeout_s)
        # Most recent status byte per servo id, updated on every reply.
        self.last_status: dict[int, int] = {}
        self.serial = serial.Serial(
            port=port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout_s,
        )

    # ---- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        if self.serial.is_open:
            self.serial.close()

    def __enter__(self) -> "STS3215Bus":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ---- framing -----------------------------------------------------------
    def _transact(
        self, servo_id: int, instruction: Instruction, params: bytes = b""
    ) -> bytes:
        """Send one instruction packet and return the status packet's parameters."""
        body = bytes([servo_id, len(params) + 2, int(instruction)]) + params
        packet = b"\xff\xff" + body + bytes([_checksum(body)])

        self.serial.reset_input_buffer()
        self.serial.write(packet)
        self.serial.flush()

        if servo_id == BROADCAST_ID and instruction != Instruction.PING:
            return b""  # broadcast writes are not acknowledged
        return self._read_status(servo_id)

    def _read_status(self, servo_id: int) -> bytes:
        """Resynchronise on the 0xFF 0xFF header, then validate one status packet.

        Some USB adapters echo the transmitted bytes back. Scanning for a header
        whose ID matches the addressed servo skips an echoed instruction packet
        without needing to know whether this adapter echoes.
        """
        deadline = time.monotonic() + self.timeout_s
        window = bytearray()

        while time.monotonic() < deadline:
            chunk = self.serial.read(1)
            if not chunk:
                continue
            window += chunk
            if len(window) > 4:
                del window[:-4]
            if len(window) < 4 or window[0] != 0xFF or window[1] != 0xFF:
                continue

            packet_id, length = window[2], window[3]
            if length < 2:
                continue

            remaining = self.serial.read(length)
            if len(remaining) < length:
                raise STSTimeout(
                    f"servo {servo_id}: truncated status packet "
                    f"({len(remaining)}/{length} bytes)"
                )

            body = bytes([packet_id, length]) + remaining
            if _checksum(body[:-1]) != body[-1]:
                window.clear()
                continue  # corrupt or an echoed frame; keep scanning
            if packet_id != servo_id:
                window.clear()
                continue  # echo of our own packet, or another servo

            self.last_status[servo_id] = remaining[0]
            return bytes(remaining[1:-1])

        raise STSTimeout(
            f"servo {servo_id}: no status packet within {self.timeout_s:.3f}s"
        )

    # ---- primitives --------------------------------------------------------
    def ping(self, servo_id: int) -> bool:
        """Return True if the servo answers, whatever its health flags say.

        A servo reporting a low-voltage warning is present, not absent. Check
        ``last_status[servo_id]`` for the flags.
        """
        try:
            self._transact(servo_id, Instruction.PING)
            return True
        except STSError:
            return False

    def read(self, servo_id: int, address: int, length: int) -> bytes:
        return self._transact(
            servo_id, Instruction.READ, bytes([int(address), int(length)])
        )

    def read_byte(self, servo_id: int, address: int) -> int:
        return self.read(servo_id, address, 1)[0]

    def read_word(self, servo_id: int, address: int) -> int:
        data = self.read(servo_id, address, 2)
        return data[0] | (data[1] << 8)  # little-endian, verified against the SDK

    def write(self, servo_id: int, address: int, data: bytes) -> None:
        self._transact(
            servo_id, Instruction.WRITE, bytes([int(address)]) + bytes(data)
        )

    def write_byte(self, servo_id: int, address: int, value: int) -> None:
        self.write(servo_id, address, bytes([int(value) & 0xFF]))

    def write_word(self, servo_id: int, address: int, value: int) -> None:
        value = int(value) & 0xFFFF
        self.write(servo_id, address, bytes([value & 0xFF, (value >> 8) & 0xFF]))

    # ---- discovery ---------------------------------------------------------
    def scan(self, id_range=range(0, 21)) -> list[int]:
        """PING a range of IDs. Read-only; safe against unknown hardware."""
        return [servo_id for servo_id in id_range if self.ping(servo_id)]

    def model_number(self, servo_id: int) -> int:
        return self.read_word(servo_id, Register.MODEL)

    def firmware(self, servo_id: int) -> tuple[int, int]:
        data = self.read(servo_id, Register.FIRMWARE_MAJOR, 2)
        return data[0], data[1]

    # ---- telemetry ---------------------------------------------------------
    def read_state(self, servo_id: int) -> ServoState:
        """Read position through moving flag in one transaction.

        Registers 56..66 are contiguous, so this is a single bus round trip
        rather than six. At 1 Mbps that difference matters for sysid sampling.
        """
        block = self.read(servo_id, Register.PRESENT_POSITION, 11)
        position = block[0] | (block[1] << 8)
        speed = _to_signed(block[2] | (block[3] << 8))
        # PRESENT_LOAD carries its direction in **bit 10**, not bit 15: the
        # magnitude field is 10 bits, which is what the documented 0-1000 =
        # 0-100% range needs and no more.
        #
        # Decoding it as bit 15 made every load in one direction read as
        # 1024 + its true value. That is where "lifting draws ~1050 against ~72
        # lowering" came from -- the real figures are about -24 and +72, small
        # and comparable. Three things gave it away: readings clustered only in
        # 0-72 and 1044-1072 with nothing between, a servo sitting at a supposed
        # 105% of rated torque held 25 C flat for eight seconds, and 1048 is
        # exactly 1024 + 24.
        load = _to_signed(block[4] | (block[5] << 8), sign_bit=10)
        voltage = block[6] * 0.1
        temperature = block[7]
        moving = bool(block[10])
        return ServoState(
            position=position,
            speed=speed,
            load=load,
            voltage_v=voltage,
            temperature_c=temperature,
            moving=moving,
        )

    def read_current_ma(self, servo_id: int) -> float:
        return _to_signed(self.read_word(servo_id, Register.PRESENT_CURRENT)) * 6.5

    def read_config(self, servo_id: int) -> dict:
        """Read the settings that change what a step response looks like.

        Recorded before sysid so a later session can reproduce the conditions.
        """
        return {
            "id": servo_id,
            "model": self.model_number(servo_id),
            "firmware": self.firmware(servo_id),
            "mode": self.read_byte(servo_id, Register.MODE),
            "p_coefficient": self.read_byte(servo_id, Register.P_COEFFICIENT),
            "d_coefficient": self.read_byte(servo_id, Register.D_COEFFICIENT),
            "i_coefficient": self.read_byte(servo_id, Register.I_COEFFICIENT),
            "acceleration": self.read_byte(servo_id, Register.ACCELERATION),
            "goal_speed": self.read_word(servo_id, Register.GOAL_SPEED),
            "min_angle_limit": self.read_word(servo_id, Register.MIN_ANGLE_LIMIT),
            "max_angle_limit": self.read_word(servo_id, Register.MAX_ANGLE_LIMIT),
            "max_torque_limit": self.read_word(servo_id, Register.MAX_TORQUE_LIMIT),
            "torque_limit": self.read_word(servo_id, Register.TORQUE_LIMIT),
            "torque_enable": self.read_byte(servo_id, Register.TORQUE_ENABLE),
            "cw_dead_zone": self.read_byte(servo_id, Register.CW_DEAD_ZONE),
            "ccw_dead_zone": self.read_byte(servo_id, Register.CCW_DEAD_ZONE),
            "min_startup_force": self.read_word(servo_id, Register.MIN_STARTUP_FORCE),
            "position_offset": self.read_word(servo_id, Register.POSITION_OFFSET),
            "phase": self.read_byte(servo_id, Register.PHASE),
            "response_level": self.read_byte(servo_id, Register.RESPONSE_LEVEL),
            "baud_rate_index": self.read_byte(servo_id, Register.BAUD_RATE),
            "present_voltage_v": self.read_byte(servo_id, Register.PRESENT_VOLTAGE) * 0.1,
            "min_voltage_v": self.read_byte(servo_id, Register.MIN_VOLTAGE_LIMIT) * 0.1,
            "max_voltage_v": self.read_byte(servo_id, Register.MAX_VOLTAGE_LIMIT) * 0.1,
            "present_temperature_c": self.read_byte(
                servo_id, Register.PRESENT_TEMPERATURE
            ),
            "servo_status": self.read_byte(servo_id, Register.SERVO_STATUS),
            "packet_status": self.last_status.get(servo_id, 0),
        }

    # ---- EEPROM ------------------------------------------------------------
    def write_eeprom(self, servo_id: int, address: int, value: int, width: int = 1) -> int:
        """Write a persistent register, handling the lock, and verify the result.

        EEPROM writes survive power cycles, so they are kept explicit and always
        read back: a silently rejected write leaves the servo in its old mode
        while the caller believes otherwise.

        Returns the previous value so it can be restored.
        """
        reader = self.read_byte if width == 1 else self.read_word
        writer = self.write_byte if width == 1 else self.write_word

        previous = reader(servo_id, address)
        if previous == value:
            return previous

        self.write_byte(servo_id, Register.LOCK, 0)
        try:
            writer(servo_id, address, value)
            time.sleep(0.02)  # EEPROM commit
        finally:
            self.write_byte(servo_id, Register.LOCK, 1)

        actual = reader(servo_id, address)
        if actual != value:
            raise STSError(
                f"servo {servo_id}: EEPROM write to register {int(address)} "
                f"did not take (wrote {value}, read back {actual}). "
                "Check that torque is disabled and the register is writable."
            )
        return previous

    def apply_config(self, servo_id: int, config: dict | None = None) -> dict:
        """Write the canonical motion settings and verify they took.

        The measured dynamics are a property of these registers, not of the
        servo alone, so they have to be pinned rather than assumed. Before this
        existed ``ramp_to`` rewrote acceleration and goal speed on every call,
        which meant any caller passing different values silently invalidated the
        calibration with nothing to notice.

        Returns the previous values so a caller can restore them.
        """
        settings = dict(CANONICAL_CONFIG if config is None else config)
        previous = {}
        for register, value in settings.items():
            width = 2 if register in (Register.GOAL_SPEED,
                                      Register.TORQUE_LIMIT) else 1
            reader = self.read_byte if width == 1 else self.read_word
            writer = self.write_byte if width == 1 else self.write_word
            previous[register] = reader(servo_id, register)
            writer(servo_id, register, value)

        mismatched = {
            r: (v, (self.read_byte if r not in (Register.GOAL_SPEED,
                                                Register.TORQUE_LIMIT)
                    else self.read_word)(servo_id, r))
            for r, v in settings.items()
        }
        bad = {r: pair for r, pair in mismatched.items() if pair[0] != pair[1]}
        if bad:
            raise STSError(
                f"servo {servo_id}: configuration did not take: "
                + ", ".join(f"reg {int(r)} wrote {w} read {g}"
                            for r, (w, g) in bad.items()))
        return previous

    def check_config(self, servo_id: int, config: dict | None = None) -> dict:
        """Return {register: (expected, actual)} for anything that has drifted.

        Empty means the servo still matches the conditions the calibration was
        measured under.
        """
        settings = CANONICAL_CONFIG if config is None else config
        drift = {}
        for register, expected in settings.items():
            reader = (self.read_word
                      if register in (Register.GOAL_SPEED, Register.TORQUE_LIMIT)
                      else self.read_byte)
            actual = reader(servo_id, register)
            if actual != expected:
                drift[register] = (expected, actual)
        return drift

    def calibrate_middle(self, servo_id: int) -> int:
        """Renumber the servo so its current physical position reads 2048.

        This is FEETECH's own middle-position calibration: writing 128 to
        TORQUE_ENABLE makes the servo compute the POSITION_OFFSET that puts it
        at mid-scale and commit it to EEPROM. Matches ``CalibrationOfs`` in
        FEETECH's FTServo_Python rather than being derived here.

        Nothing moves. The shaft, the horn, and the board stay exactly where
        they are -- only the number the servo reports for that position changes.
        Use it when the working position has landed near 0 or 4095, where the
        servo cannot be commanded past the end of its own scale: on this rig
        board-level put servo 1 at 4093 of 4095, leaving two counts of travel in
        one direction and no way to ask for more.

        The offset persists across power cycles, and it invalidates any stored
        calibration that records servo counts. Returns the new offset.
        """
        self.write_byte(servo_id, Register.TORQUE_ENABLE, 128)
        time.sleep(0.05)  # EEPROM commit
        offset = _to_signed(self.read_word(servo_id, Register.POSITION_OFFSET))
        position = self.read_word(servo_id, Register.PRESENT_POSITION)
        if abs(position - 2048) > 16:
            raise STSError(
                f"servo {servo_id}: middle calibration did not take -- position "
                f"reads {position}, expected about 2048. The servo may not "
                "support this command."
            )
        return offset

    def set_mode(self, servo_id: int, mode: Mode) -> int:
        """Switch operating mode. Returns the previous mode.

        Torque must be off: changing mode under load is a way to get a sudden
        uncommanded movement.
        """
        if self.read_byte(servo_id, Register.TORQUE_ENABLE):
            raise STSError(
                f"servo {servo_id}: disable torque before changing mode"
            )
        return self.write_eeprom(servo_id, Register.MODE, int(mode))

    # ---- motion ------------------------------------------------------------
    def torque_enable(self, servo_id: int) -> None:
        self.write_byte(servo_id, Register.TORQUE_ENABLE, 1)

    def torque_disable(self, servo_id: int) -> None:
        """Release the servo. This is the emergency stop."""
        self.write_byte(servo_id, Register.TORQUE_ENABLE, 0)

    def sync_write_positions(self, targets: dict[int, int]) -> None:
        """Set several goal positions in one broadcast packet.

        One packet means every servo latches its target in the same instant.
        Writing them one at a time puts a bus round trip between the two axes,
        so the board takes a slightly skewed path and the gap lands inside the
        control period -- small, but it is exactly the sort of error that a
        policy learns around in simulation and then meets differently on
        hardware.

        Broadcast writes are not acknowledged, so this cannot report a per-servo
        failure. Read the positions back if that matters.
        """
        if not targets:
            return
        params = bytearray([int(Register.GOAL_POSITION), 2])
        for servo_id, counts in sorted(targets.items()):
            value = int(min(max(int(counts), 0), COUNTS_PER_REV - 1))
            params += bytes([servo_id, value & 0xFF, (value >> 8) & 0xFF])
        self._transact(BROADCAST_ID, Instruction.SYNC_WRITE, bytes(params))

    def set_goal_position(self, servo_id: int, counts: int) -> None:
        counts = int(counts)
        if not 0 <= counts < COUNTS_PER_REV:
            raise ValueError(
                f"goal position {counts} outside 0..{COUNTS_PER_REV - 1}"
            )
        self.write_word(servo_id, Register.GOAL_POSITION, counts)


def counts_to_degrees(counts: float) -> float:
    return counts * DEGREES_PER_COUNT


def degrees_to_counts(degrees: float) -> float:
    return degrees / DEGREES_PER_COUNT
