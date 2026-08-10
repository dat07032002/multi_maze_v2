"""Mirror symmetries of the board, and the limit of what they buy.

Gravity is vertical, the plate is rectangular, and both hinges pass through its
centre, so reflecting about either board axis maps a valid **ball-on-plate**
transition to another valid one. Reflecting in x sends ``x -> W - x``,
``vx -> -vx`` and ``beta -> -beta``; in y it sends ``y -> H - y``,
``vy -> -vy`` and ``alpha -> -alpha``. Applying both gives four transitions for
the price of one.

**Correction to the milestone plan.** That plan claimed this is an exact
augmentation for the *maze* task, on the grounds that the policy never observes
maze geometry, only the route lookahead, which mirrors along with everything
else. The observation part is right; the conclusion is not. The **reward**
observes geometry -- ``wall_contact``, ``hole_proximity`` and the local
clearance that normalises cross-track all read the real walls and holes, and
those are not mirror-symmetric. A mirrored transition therefore carries a reward
and a termination that belong to a board which does not exist, and feeding it to
the critic would be teaching it a maze nobody is playing.

So the honest split:

* **Exact, and used**, for the flat-plate task, which has no walls or holes.
* **Not valid** as replay augmentation for the maze.
* The valid maze version is to mirror the *layout* -- walls, holes, route, pads
  -- and treat the result as a second, equally real maze. That is not free
  replay data, it is training on more boards, and it belongs with the
  multi-maze work rather than here.

The transforms live here regardless, because the flat-plate use is real and
because ``mirror_layout`` is the seed of that later work.
"""
from __future__ import annotations

import numpy as np

from contract import policy_contract as pc

# Observation layout, from contract.policy_contract.observation:
#   0:2   ball xy / board size
#   2:4   velocity / reference speed
#   4:6   (alpha, beta) / max tilt
#   6:12  three past actions, each (alpha, beta)
#   12:22 five lookahead points, each ball-relative (dx, dy)
#   22:27 five lookahead clearances, one scalar per point
_BALL = slice(0, 2)
_VELOCITY = slice(2, 4)
_ANGLES = slice(4, 6)
_HISTORY = slice(6, 6 + 2 * pc.ACTION_HISTORY)
# Only the (dx, dy) block flips. Clearance is a scalar per point, invariant
# under reflection and in the same point order, so it passes through untouched
# -- extending this slice to OBSERVATION_SIZE would wrongly negate it.
_LOOKAHEAD = slice(6 + 2 * pc.ACTION_HISTORY,
                   6 + 2 * pc.ACTION_HISTORY + 2 * pc.LOOKAHEAD_COUNT)


def _flip(observation: np.ndarray, axis: int) -> np.ndarray:
    """Reflect an observation about board axis 0 (x) or 1 (y)."""
    out = np.array(observation, dtype=np.float64, copy=True)
    other = 1 - axis

    # Position is normalised to [0, 1], so the reflection is 1 - value.
    out[_BALL][axis] = 1.0 - out[_BALL][axis]
    out[_VELOCITY][axis] = -out[_VELOCITY][axis]
    # beta (index 1 of the angle pair) drives x; alpha (index 0) drives y.
    out[_ANGLES][other] = -out[_ANGLES][other]
    out[_HISTORY][other::2] = -out[_HISTORY][other::2]
    out[_LOOKAHEAD][axis::2] = -out[_LOOKAHEAD][axis::2]
    return out.astype(observation.dtype if hasattr(observation, "dtype")
                      else np.float32)


def _flip_action(action: np.ndarray, axis: int) -> np.ndarray:
    out = np.array(action, dtype=np.float64, copy=True)
    out[1 - axis] = -out[1 - axis]
    return out.astype(np.float32)


def mirror_x(observation, action):
    return _flip(np.asarray(observation), 0), _flip_action(np.asarray(action), 0)


def mirror_y(observation, action):
    return _flip(np.asarray(observation), 1), _flip_action(np.asarray(action), 1)


def augment(observation, action):
    """The four reflections, identity first.

    Valid for the flat-plate task only -- see the module docstring.
    """
    observation = np.asarray(observation)
    action = np.asarray(action)
    both = mirror_y(*mirror_x(observation, action))
    return [(observation, action), mirror_x(observation, action),
            mirror_y(observation, action), both]


def mirror_layout(layout: dict, axis: int = 0) -> dict:
    """Reflect an entire maze -- walls, holes, route, pads -- about one axis.

    The valid way to use the board's symmetry on the maze task: the result is a
    real, different, equally playable board rather than a relabelling of this
    one. Kept here because it is the natural first step of the multi-maze work.
    """
    W, H = layout["board_width"], layout["board_height"]
    span = W if axis == 0 else H
    mirrored = dict(layout)

    def flip(value):
        return span - value

    if axis == 0:
        mirrored["walls_h"] = [[flip(hi), flip(lo), y]
                               for lo, hi, y in layout["walls_h"]]
        mirrored["walls_v"] = [[lo, hi, flip(x)]
                               for lo, hi, x in layout["walls_v"]]
    else:
        mirrored["walls_h"] = [[lo, hi, flip(y)]
                               for lo, hi, y in layout["walls_h"]]
        mirrored["walls_v"] = [[flip(hi), flip(lo), x]
                               for lo, hi, x in layout["walls_v"]]

    def flip_point(point):
        return [flip(point[0]), point[1]] if axis == 0 else [point[0], flip(point[1])]

    mirrored["holes"] = [flip_point(h) for h in layout["holes"]]
    mirrored["waypoints"] = [flip_point(p) for p in layout["waypoints"]]
    mirrored["start_planned"] = flip_point(layout["start_planned"])
    mirrored["goal_planned"] = flip_point(layout["goal_planned"])
    mirrored["tag_pads"] = [
        dict(pad, centre_mm=[span * 1000.0 - pad["centre_mm"][0], pad["centre_mm"][1]]
             if axis == 0 else
             [pad["centre_mm"][0], span * 1000.0 - pad["centre_mm"][1]])
        for pad in layout.get("tag_pads", [])
    ]
    return mirrored
