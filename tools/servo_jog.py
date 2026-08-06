#!/usr/bin/env python3
"""Command the tilt servos by hand. Interactive keyboard jog.

This is the bring-up tool: it commands the board to tilt before any calibration
exists, so it deliberately works in **raw servo counts** rather than board
angles. The count->angle mapping is what sysid measures, and you cannot use a
mapping to produce the data that defines it.

Safety, in order of importance:

* Torque starts **off**. It energises only when you press a direction key or
  ``t`` -- starting the tool never moves the board.
* ``space`` is the emergency stop: torque off on both servos, immediately.
* Travel is clamped to +-``--max-travel`` counts around the home position
  captured at startup, so a held key cannot walk the board into a hard stop.
* Torque is released on exit, on exception, and on Ctrl-C.
* Goal speed and acceleration are set conservatively so every move is slow
  enough to interrupt.

Keys:
    a / d       servo 1 (roll)  -/+   one small step
    s / w       servo 2 (pitch) -/+   one small step
    [ / ]       step size down/up
    h           return both servos to home
    z           adopt the current position as the new home
    t           torque on/off (a/d/s/w turn it on for you)
    space       EMERGENCY STOP (torque off)
    q / Esc     quit

Examples:
    python3 tools/servo_jog.py
    python3 tools/servo_jog.py --step 5 --max-travel 150
    python3 tools/servo_jog.py --pose          # also show camera alpha/beta
"""
from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tag_vision.hardware.sts3215 import (  # noqa: E402
    COUNTS_PER_REV,
    Mode,
    Register,
    STS3215Bus,
    STSError,
    counts_to_degrees,
    decode_status,
)

ROOT = Path(__file__).resolve().parents[1]


class RawTerminal:
    """Single-key input without Enter, restored on the way out."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc_info):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    @staticmethod
    def key(timeout: float = 0.0) -> str | None:
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if ready else None


class ServoJog:
    def __init__(self, bus: STS3215Bus, args):
        self.bus = bus
        self.args = args
        self.ids = {"roll": args.roll_id, "pitch": args.pitch_id}
        self.torque = False
        self.step = args.step
        self.message = ""

        self.home: dict[str, int] = {}
        self.target: dict[str, int] = {}
        for axis, servo_id in self.ids.items():
            position = bus.read_word(servo_id, Register.PRESENT_POSITION)
            self.home[axis] = position
            self.target[axis] = position

    # ---- limits ------------------------------------------------------------
    def clamp(self, axis: str, counts: int) -> int:
        home = self.home[axis]
        low = max(0, home - self.args.max_travel)
        high = min(COUNTS_PER_REV - 1, home + self.args.max_travel)
        return int(min(max(counts, low), high))

    def headroom(self, axis: str) -> tuple[int, int]:
        """Counts available below and above home before hitting a limit."""
        home = self.home[axis]
        return (
            home - max(0, home - self.args.max_travel),
            min(COUNTS_PER_REV - 1, home + self.args.max_travel) - home,
        )

    # ---- motion ------------------------------------------------------------
    def set_torque(self, enabled: bool) -> None:
        for axis, servo_id in self.ids.items():
            if enabled:
                # Re-issue the current position as the goal before energising,
                # so enabling torque never causes a jump to a stale target.
                position = self.bus.read_word(servo_id, Register.PRESENT_POSITION)
                self.target[axis] = self.clamp(axis, position)
                self.bus.write_byte(servo_id, Register.ACCELERATION, self.args.accel)
                self.bus.write_word(servo_id, Register.GOAL_SPEED, self.args.speed)
                self.bus.set_goal_position(servo_id, self.target[axis])
                self.bus.torque_enable(servo_id)
            else:
                self.bus.torque_disable(servo_id)
        self.torque = enabled
        self.message = "torque ON" if enabled else "torque off"

    def nudge(self, axis: str, delta: int) -> None:
        # Pressing a direction key is an unambiguous request to move, so it
        # energises the servos rather than complaining that torque is off.
        if not self.torque:
            self.set_torque(True)
        wanted = self.target[axis] + delta
        clamped = self.clamp(axis, wanted)
        if clamped != wanted:
            self.message = f"{axis} at travel limit"
        self.target[axis] = clamped
        self.bus.set_goal_position(self.ids[axis], clamped)

    def go_home(self) -> None:
        if not self.torque:
            self.message = "torque is off - press t first"
            return
        for axis, servo_id in self.ids.items():
            self.target[axis] = self.home[axis]
            self.bus.set_goal_position(servo_id, self.home[axis])
        self.message = "returning to home"

    def set_home(self) -> None:
        for axis, servo_id in self.ids.items():
            position = self.bus.read_word(servo_id, Register.PRESENT_POSITION)
            self.home[axis] = position
            self.target[axis] = position
        self.message = "home set to current position"

    def stop(self) -> None:
        self.set_torque(False)
        self.message = "*** EMERGENCY STOP ***"

    def check_load(self) -> None:
        """Back off and release torque if a servo is straining.

        Servo 1 was measured reaching load 1044 -- above its own 1000 limit --
        about 40 counts below its resting position, so holding a key down can
        drive it hard into a mechanical stop. Backing off before cutting torque
        means it relaxes off the stop rather than staying wedged against it.
        """
        if not self.torque:
            return
        for axis, servo_id in self.ids.items():
            state = self.bus.read_state(servo_id)
            if abs(state.load) <= self.args.max_load:
                continue
            retreat = self.clamp(axis, self.home[axis])
            self.bus.set_goal_position(servo_id, retreat)
            self.target[axis] = retreat
            time.sleep(0.3)
            self.set_torque(False)
            self.message = (
                f"*** {axis} load {state.load} > {self.args.max_load} - "
                "backed off, torque OFF ***"
            )
            return

    # ---- display -----------------------------------------------------------
    def status_line(self, pose_text: str) -> str:
        """One compact line. Offsets are shown relative to home, which is what
        you actually care about while jogging -- the absolute count only matters
        when you are near a travel limit."""
        parts = []
        for axis, servo_id in self.ids.items():
            state = self.bus.read_state(servo_id)
            offset = state.position - self.home[axis]
            status = self.bus.last_status.get(servo_id, 0)
            flag = "" if not status else f" !{','.join(decode_status(status))}"
            parts.append(
                f"{axis:5s} {offset:+5d} ({counts_to_degrees(offset):+6.2f} deg)"
                f" load {state.load:+5d}{flag}"
            )
        torque = "ON " if self.torque else "off"
        return (
            f"[{torque}] step {self.step:3d}  "
            + "   ".join(parts)
            + (f"   {pose_text}" if pose_text else "")
            + f"   {self.message}"
        )


def make_pose_reader(args):
    """Optional camera readout, so you can see the board actually tilt.

    Returns a callable giving a short text summary, or a no-op if the camera or
    calibration is unavailable. A camera failure must not take down the jog
    tool -- commanding the motors is the point, the pose is a bonus.
    """
    if not args.pose:
        return lambda: ""

    try:
        import cv2
        from tag_vision.core.board_geometry import BoardGeometry
        from tag_vision.core.board_pose import BoardPoseEstimator

        geometry = BoardGeometry.load(str(ROOT / "calib" / "board_tags.json"))
        estimator = BoardPoseEstimator(
            str(ROOT / "calib" / "camera_calib.json"), geometry, min_tags=4
        )
        zero_path = ROOT / "calib" / "board_zero.json"
        if zero_path.is_file():
            estimator.load_zero(zero_path)

        capture = cv2.VideoCapture(args.camera)
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            print("camera did not open; continuing without pose")
            return lambda: ""

        width, height = estimator.image_size

        def read() -> str:
            ok, frame = capture.read()
            if not ok:
                return "pose: no frame"
            if (frame.shape[1], frame.shape[0]) != (width, height):
                frame = cv2.resize(frame, (width, height),
                                   interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = estimator.estimate(gray)
            if result is None:
                return "pose: --"
            return (
                f"alpha {result.alpha_deg:+6.2f} beta {result.beta_deg:+6.2f} "
                f"({result.reprojection_px:.1f}px)"
            )

        return read
    except Exception as exc:  # noqa: BLE001
        print(f"pose readout unavailable ({exc}); continuing without it")
        return lambda: ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", default=None, help="auto-resolved by USB id; override only if needed")
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument("--roll-id", type=int, default=1)
    parser.add_argument("--pitch-id", type=int, default=2)
    parser.add_argument("--step", type=int, default=3,
                        help="counts per key press (default: 3, about 0.26 deg)")
    parser.add_argument("--max-travel", type=int, default=200,
                        help="max counts from home in either direction "
                             "(default: 200, about 17.6 deg)")
    parser.add_argument("--speed", type=int, default=300,
                        help="goal speed register, 0 = unlimited (default: 300)")
    parser.add_argument("--accel", type=int, default=20,
                        help="acceleration register, 0 = instant (default: 20)")
    parser.add_argument("--max-load", type=int, default=350,
                        help="back off and cut torque above this load "
                             "(default: 350)")
    parser.add_argument("--pose", action="store_true",
                        help="also show camera-measured board angles")
    parser.add_argument("--camera", default=0, help="camera index for --pose")
    args = parser.parse_args()

    if args.max_travel < 1:
        parser.error("--max-travel must be at least 1")
    if args.step < 1:
        parser.error("--step must be at least 1")

    pose_reader = make_pose_reader(args)

    with STS3215Bus(args.port, args.baud) as bus:
        for name, servo_id in (("roll", args.roll_id), ("pitch", args.pitch_id)):
            if not bus.ping(servo_id):
                raise SystemExit(
                    f"{name} servo id {servo_id} did not answer on {args.port}. "
                    "Run tools/servo_scan.py to find the right ids."
                )
            mode = bus.read_byte(servo_id, Register.MODE)
            if mode != Mode.POSITION:
                raise SystemExit(
                    f"{name} servo id {servo_id} is in {Mode(mode).name} mode. "
                    "Position mode is required; see tools/servo_scan.py."
                )

        jog = ServoJog(bus, args)

        print(__doc__.split("Keys:")[1].split("Examples:")[0])
        for axis in jog.ids:
            below, above = jog.headroom(axis)
            home = jog.home[axis]
            print(
                f"{axis:5s} home {home:4d} counts "
                f"({counts_to_degrees(home):.1f} deg), travel -{below}/+{above}"
            )
            if below < args.max_travel or above < args.max_travel:
                print(
                    f"  WARNING: {axis} home is close to the 0/{COUNTS_PER_REV - 1} "
                    "wrap, so travel is asymmetric. Consider re-indexing the horn "
                    "or setting POSITION_OFFSET before sysid."
                )
        print("\nTorque is OFF. Press t to energise, space to stop, q to quit.\n")

        try:
            with RawTerminal() as term:
                while True:
                    key = term.key(0.02)
                    if key is not None:
                        if key in ("q", "\x1b"):
                            break
                        elif key == " ":
                            jog.stop()
                        elif key == "t":
                            jog.set_torque(not jog.torque)
                        elif key == "a":
                            jog.nudge("roll", -jog.step)
                        elif key == "d":
                            jog.nudge("roll", +jog.step)
                        elif key == "s":
                            jog.nudge("pitch", -jog.step)
                        elif key == "w":
                            jog.nudge("pitch", +jog.step)
                        elif key == "[":
                            jog.step = max(1, jog.step // 2)
                            jog.message = f"step {jog.step}"
                        elif key == "]":
                            jog.step = min(100, jog.step * 2)
                            jog.message = f"step {jog.step}"
                        elif key == "h":
                            jog.go_home()
                        elif key == "z":
                            jog.set_home()

                    jog.check_load()
                    sys.stdout.write("\r" + jog.status_line(pose_reader())[:200] + "  ")
                    sys.stdout.flush()
        except STSError as exc:
            print(f"\nBus error: {exc}")
        finally:
            for servo_id in (args.roll_id, args.pitch_id):
                try:
                    bus.torque_disable(servo_id)
                except STSError:
                    pass
            print("\nTorque released on both servos.")


if __name__ == "__main__":
    main()
