"""Interactive view of the compiled maze -- tilt it by hand and watch the ball.

    python3 -m sim.view_maze
    python3 -m sim.view_maze --flat        # bare plate, no walls or holes

Controls:
    arrow keys   tilt the board (hold to slew; release to hold the angle)
    space        level the board
    R            put the ball back at the start
    R (fallen)   also happens automatically two seconds after a fall

The tilt slews at the measured hardware rate (6.9 deg/s roll, 8.2 pitch) and is
clamped to the +-4 degree command limit, so what you can do here is what the rig
can do. Commanding a step change would let you flick the board through 8 degrees
in a millisecond, which launches the ball about 10 mm into the air -- possible in
MuJoCo, not on the bench.
"""
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from .board_state import BoardState, TiltDriver
from .mjcf_builder import (build_mjcf, flat_board_layout, layout_to_model,
                           load_layout, load_parameters)

ARROW_LEFT, ARROW_RIGHT, ARROW_DOWN, ARROW_UP = 263, 262, 264, 265
KEY_SPACE, KEY_R = 32, 82


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--flat", action="store_true",
                        help="bare plate: no walls, holes or pads")
    args = parser.parse_args()

    layout = flat_board_layout() if args.flat else load_layout()
    params = load_parameters()
    model = mujoco.MjModel.from_xml_string(build_mjcf(layout, params))
    data = mujoco.MjData(model)
    state = BoardState(model)
    driver = TiltDriver(state, model.opt.timestep,
                        (params["actuator.roll.max_rate"],
                         params["actuator.pitch.max_rate"]))

    W, H = layout["board_width"], layout["board_height"]
    start = layout_to_model(*layout["start_planned"], layout)
    limit = params["actuator.max_tilt"]
    floor_bottom = -params["maze.floor_thickness"]

    def reset_ball() -> None:
        state.set_ball(data, *start)

    mujoco.mj_forward(model, data)
    reset_ball()

    target = np.zeros(2)
    # Roughly two command quanta per press: the rig's floor is 40 counts, about
    # 0.19 degrees, and anything finer is commanding noise.
    nudge = float(np.radians(0.40))

    def on_key(key: int) -> None:
        if key == ARROW_UP:
            target[0] = min(limit, target[0] + nudge)
        elif key == ARROW_DOWN:
            target[0] = max(-limit, target[0] - nudge)
        elif key == ARROW_RIGHT:
            target[1] = min(limit, target[1] + nudge)
        elif key == ARROW_LEFT:
            target[1] = max(-limit, target[1] - nudge)
        elif key == KEY_SPACE:
            target[:] = 0.0
        elif key == KEY_R:
            reset_ball()

    print(__doc__)
    print(f"board {W * 1000:.0f} x {H * 1000:.0f} mm, {model.ngeom} geoms, "
          f"tilt limit +-{np.degrees(limit):.1f} deg")

    fallen_at: float | None = None
    with mujoco.viewer.launch_passive(model, data,
                                      key_callback=on_key) as viewer:
        viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = (
            90.0, -60.0, 0.55)
        last_print = 0.0
        while viewer.is_running():
            tick = time.perf_counter()
            for _ in range(int(round(1.0 / 60.0 / model.opt.timestep))):
                driver.step(data, target[0], target[1])
                mujoco.mj_step(model, data)

            x, y, z = state.ball_board(data)
            if z < floor_bottom:
                if fallen_at is None:
                    fallen_at = time.perf_counter()
                    print("  ball fell through a hole -- resetting in 2 s")
                elif time.perf_counter() - fallen_at > 2.0:
                    reset_ball()
                    fallen_at = None
            else:
                fallen_at = None

            if tick - last_print > 0.5:
                alpha, beta = state.tilt(data)
                print(f"\r  tilt {np.degrees(alpha):+6.2f} {np.degrees(beta):+6.2f} deg   "
                      f"ball {(x + W / 2) * 1000:6.1f} {(y + H / 2) * 1000:6.1f} mm",
                      end="", flush=True)
                last_print = tick

            viewer.sync()
            time.sleep(max(0.0, 1.0 / 60.0 - (time.perf_counter() - tick)))


if __name__ == "__main__":
    main()
