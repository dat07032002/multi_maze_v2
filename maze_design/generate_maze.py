"""Design a printable TAG maze: one valid route, wrong turns punished by holes.

Physical sizes drive the mechanics:

  ball 12 mm, hole 15 mm, corridor 20 mm clear
    A hole centred in a normal corridor leaves 2.5 mm each side -> the corridor
    is BLOCKED. That is how a wrong turn is closed off.
    A widened section is two corridors merged (43 mm). A hole centred there
    leaves 14 mm each side, so the ball must leave the centreline and commit
    to a side. That is a DODGE.

Two things learned the hard way, both caught by running the route planner
rather than by looking at the render:

  - A blocking hole only 60% into the decoy cell sits 13.8 mm from the route
    centreline against a 13.5 mm requirement, i.e. 0.3 mm from closing the main
    route as well. They now go 95% in.
  - A dodge hole offset toward the pocket is a wall, not a dodge: it leaves
    8.5 mm on the route side, and the detour needs more along-route room than
    one widened cell provides. Centred is the only geometry that works with a
    single widened cell.

No border walls are emitted: the insert drops into the frame, which provides
containment, and signed_ball_clearance models the board boundary anyway.
"""
from __future__ import annotations

import json
import math
import random
from collections import deque

BOARD_W, BOARD_H = 0.259, 0.229
WALL_T = 0.003
WALL_H = 0.008          # 8 mm: above the ball centre (6 mm), so it stays contained
BALL_R = 0.006
HOLE_R = 0.0075

COLS, ROWS = 11, 9
# The grid fills the board edge to edge. It used to be a fixed 23 mm pitch
# centred on the board, which left 3 mm unused at the sides and 11 mm at the
# top and bottom -- wasted play area and a tighter corner for the 18 mm tags.
# Filling exactly makes the pitch different on each axis, so corridors are
# 20.5 mm across and 22.4 mm tall. Both clear the 12 mm ball comfortably.
PITCH_X = BOARD_W / COLS            # 23.5 mm -> 20.5 mm corridor
PITCH_Y = BOARD_H / ROWS            # 25.4 mm -> 22.4 mm corridor
X0 = Y0 = 0.0

# The four corner cells carry the 18 mm AprilTags, printed as part of the maze.
# A tag plus its quiet zone needs a 22 mm square and the board margin outside
# the grid is only 3 x 11 mm, so each tag has to occupy a whole corner cell.
TAG_CELLS = {(0, 0), (COLS - 1, 0), (0, ROWS - 1), (COLS - 1, ROWS - 1)}

# The auto-reload mechanism delivers the ball at the middle of the long
# (259 mm) wall, so the route has to begin there. Five cells clear of either
# top corner, it does not compete with the corner tags -- a corner start would
# have, which is why this position matters.
START_CELL = (COLS // 2, ROWS - 1)

# The central block. A route that skirts it leaves most of the board unused,
# so seeds are scored on how much of this region they actually traverse.
MIDDLE = {(i, j) for i in range(3, 8) for j in range(2, 7)}


def centre(cell):
    i, j = cell
    return (X0 + (i + 0.5) * PITCH_X, Y0 + (j + 0.5) * PITCH_Y)


def neighbours(cell):
    i, j = cell
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n = (i + di, j + dj)
        if 0 <= n[0] < COLS and 0 <= n[1] < ROWS and n not in TAG_CELLS:
            yield n


def cradle_walls():
    """Edges kept sealed so the L-cradle's arms each span two cells.

    The cradle from a single leaf cell is one cell on a side, which reads as a
    small notch. Forbidding these two edges outright -- rather than deleting
    them after carving, which would orphan whatever lies beyond -- extends the
    bottom arm under the exit cell and carries the right arm one cell further
    down, so the two runs join into one long L.
    """
    i, j = START_CELL
    return {
        frozenset(((i - 1, j), (i - 1, j - 1))),   # bottom arm, one cell left
        frozenset(((i, j - 1), (i + 1, j - 1))),   # right arm, one cell down
    }


def carve(rng):
    """Spanning tree over the playable cells, with the start cell as a leaf.

    The reload cell is carved LAST and given exactly one opening. Its other two
    interior sides then become walls, and since it sits on the top row the
    board frame closes the third -- an L-shaped cradle that holds the ball
    where the reload mechanism drops it instead of letting it wander off before
    the run starts. Carving it as an ordinary cell would have given it up to
    three openings.
    """
    FORBIDDEN = cradle_walls()
    cells = [c for c in ((i, j) for i in range(COLS) for j in range(ROWS))
             if c not in TAG_CELLS and c != START_CELL]
    start = cells[0]
    passages, seen, stack = set(), {start}, [start]
    while stack:
        cell = stack[-1]
        options = [n for n in neighbours(cell)
                   if n not in seen and n != START_CELL
                   and frozenset((cell, n)) not in FORBIDDEN]
        if not options:
            stack.pop()
            continue
        nxt = rng.choice(options)
        passages.add(frozenset((cell, nxt)))
        seen.add(nxt)
        stack.append(nxt)

    # Attach the reload cell by a single edge: one way in, one way out.
    #
    # Prefer a sideways exit. Leaving downward would wall the cell left and
    # right -- two parallel walls, a channel rather than a cradle. A sideways
    # exit leaves the bottom wall plus one side wall, which are perpendicular:
    # the L that actually cups the ball.
    # Exit LEFT specifically, so the cradle walls land on the bottom and the
    # RIGHT -- the mirror of the first version, which opened rightward.
    exits = [n for n in neighbours(START_CELL) if n in seen]
    sideways = [n for n in exits
                if n[1] == START_CELL[1] and n[0] < START_CELL[0]]
    if sideways:
        passages.add(frozenset((START_CELL, sideways[0])))
    elif exits:
        passages.add(frozenset((START_CELL, rng.choice(exits))))
    return passages


def bfs(passages, start):
    prev, order = {start: None}, [start]
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for n in neighbours(cell):
            if n not in prev and frozenset((cell, n)) in passages:
                prev[n] = cell
                order.append(n)
                queue.append(n)
    return prev, order


def path_between(passages, start, goal):
    prev, _ = bfs(passages, start)
    path, cell = [], goal
    while cell is not None:
        path.append(cell)
        cell = prev[cell]
    return path[::-1]


def longest_path(passages):
    """The longest route that begins at the reload point.

    Our own rule -- take the deepest path in the spanning tree -- but anchored,
    not free. Unanchored it put the start wherever the tree happened to be
    deepest; forcing both endpoints to opposite corners (the way the repo's
    layouts are built) gave the unique tree path between them, which is short
    and drove badly. Fixing only the start keeps the route as long as the maze
    allows while still beginning where the ball is actually delivered.
    """
    adjacency = {}
    for passage in passages:
        a, b = tuple(passage)
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    previous = {START_CELL: None}
    order = [START_CELL]
    queue = deque([START_CELL])
    while queue:
        cell = queue.popleft()
        for nxt in adjacency.get(cell, ()):
            if nxt not in previous:
                previous[nxt] = cell
                order.append(nxt)
                queue.append(nxt)

    goal = order[-1]                      # last reached = farthest from start
    route, cell = [], goal
    while cell is not None:
        route.append(cell)
        cell = previous[cell]
    return route[::-1]


def route_clearance(point, line):
    """Distance from a point to the polyline through the route cell centres."""
    best = float("inf")
    px, py = point
    for (ax, ay), (bx, by) in zip(line, line[1:]):
        dx, dy = bx - ax, by - ay
        span = dx * dx + dy * dy
        t = 0.0 if span < 1e-12 else max(
            0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / span))
        best = min(best, math.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return best


def build(seed):
    rng = random.Random(seed)
    passages = carve(rng)
    route = longest_path(passages)
    route_set = set(route)
    route_line = [centre(c) for c in route]

    holes, used = [], set()
    interior = route[2:-2]

    # Dodge holes need a CHAMBER: two route cells long, two wide.
    #
    # In a single 20 mm corridor the ball's centre has only 8 mm of freedom,
    # while a 15 mm hole plus a 12 mm ball is a 27 mm exclusion zone. No offset
    # makes such a hole both a real threat and comfortably passable -- three
    # attempts confirmed it, each failing in play-test at 100%.
    #
    # So merge two consecutive route cells with their neighbours on BOTH sides:
    # 46 mm along the route by 66 mm across. The hole sits on the route centre
    # line, so going straight still drops the ball in, but either side leaves a
    # 13.5 mm band for the ball's centre, with 46 mm of travel in which to
    # drift into it.
    #
    # 13.5 mm is the number that matters. The route-following expert's
    # cross-track error is 4.4 mm median and 15 mm at the 90th percentile, so
    # the 3.75 mm band of a two-wide chamber sat inside its normal wander and
    # it fell in every single episode. A dodge has to be wider than the
    # controller's error, not merely wider than the ball.
    dodges = 0
    for cell in interior:
        if dodges >= 2:
            break
        if any(abs(cell[0] - u[0]) + abs(cell[1] - u[1]) < 4 for u in used):
            continue
        position = route.index(cell)
        if not (0 < position < len(route) - 2):
            continue
        before, cell2, after = route[position - 1], route[position + 1], route[position + 2]
        # cell -> cell2 must continue straight, and be straight on both ends.
        if not (before[0] == after[0] or before[1] == after[1]):
            continue
        along_x = cell[1] == cell2[1]
        offsets = ((0, 1), (0, -1)) if along_x else ((1, 0), (-1, 0))
        pocket = [(c[0] + dx, c[1] + dy) for c in (cell, cell2)
                  for dx, dy in offsets]
        placed = False
        if not any(q in route_set or q in used or q in TAG_CELLS
                   or not (0 <= q[0] < COLS and 0 <= q[1] < ROWS)
                   for q in pocket):
            for base, side in zip((cell, cell, cell2, cell2), pocket):
                passages.add(frozenset((base, side)))
            passages.add(frozenset((pocket[0], pocket[2])))   # open along
            passages.add(frozenset((pocket[1], pocket[3])))
            ax, ay = centre(cell)
            bx, by = centre(cell2)
            holes.append(((ax + bx) / 2.0, (ay + by) / 2.0, "dodge"))
            used.update({cell, cell2, *pocket})
            dodges += 1
            placed = True
        if placed:
            continue

    for cell in interior:
        if len(holes) >= 13:
            break
        # Nothing within two cells of the reload point. A hole beside the
        # cradle threatens the ball before it is even moving, which is not a
        # skill test -- it is a bad spawn.
        if (abs(cell[0] - START_CELL[0]) <= 1
                and abs(cell[1] - START_CELL[1]) <= 1):
            continue
        if any(abs(cell[0] - u[0]) + abs(cell[1] - u[1]) < 2 for u in used):
            continue
        options = [n for n in neighbours(cell)
                   if n not in route_set and n not in used]
        rng.shuffle(options)
        for decoy in options:
            cx, cy = centre(cell)
            dx, dy = centre(decoy)
            spot = (cx + 0.95 * (dx - cx), cy + 0.95 * (dy - cy))
            # Check against the WHOLE route, not just this cell. A decoy one
            # cell off the route here can still sit next to another leg where
            # the route snakes back -- that is how a blocking hole ended up
            # 1.00 mm from the correct path and the play-test drove into it.
            if route_clearance(spot, route_line) < HOLE_R + BALL_R + 0.008:
                continue
            passages.add(frozenset((cell, decoy)))     # open the wrong turn
            holes.append((spot[0], spot[1], "block"))
            used.update({cell, decoy})
            break

    # ---- walls: interior only, no border ---------------------------------
    walls_h, walls_v = [], []
    for i in range(COLS):
        for j in range(ROWS):
            cell = (i, j)
            x_lo, y_lo = X0 + i * PITCH_X, Y0 + j * PITCH_Y
            x_hi, y_hi = x_lo + PITCH_X, y_lo + PITCH_Y
            if i < COLS - 1:
                right = (i + 1, j)
                if (cell in TAG_CELLS or right in TAG_CELLS
                        or frozenset((cell, right)) not in passages):
                    walls_v.append([y_lo, y_hi, x_hi])
            if j < ROWS - 1:
                above = (i, j + 1)
                if (cell in TAG_CELLS or above in TAG_CELLS
                        or frozenset((cell, above)) not in passages):
                    walls_h.append([x_lo, x_hi, y_hi])

    def merge(segments):
        out, groups = [], {}
        for seg in segments:
            groups.setdefault(round(seg[2], 6), []).append(seg)
        for group in groups.values():
            group.sort()
            current = list(group[0])
            for seg in group[1:]:
                if abs(seg[0] - current[1]) < 1e-9:
                    current[1] = seg[1]
                else:
                    out.append(current)
                    current = list(seg)
            out.append(current)
        return out

    waypoints = [list(centre(c)) for c in route]
    layout = {
        "board_width": BOARD_W, "board_height": BOARD_H,
        "ball_radius": BALL_R, "wall_thickness": WALL_T,
        "wall_height": WALL_H,
        "walls_h": merge(walls_h), "walls_v": merge(walls_v),
        "walls_angled": [],
        "holes": [[h[0], h[1]] for h in holes],
        "hole_radii": [HOLE_R] * len(holes),
        "waypoints": waypoints,
        "start_planned": waypoints[0], "goal_planned": waypoints[-1],
        "border_walls_printed": False,
        "_comment": "Border omitted: the frame contains the ball.",
    }
    return layout, {str(k): h[2] for k, h in enumerate(holes)}, route


if __name__ == "__main__":
    value, seed, length, middle = pick_seed()
    print(f"best of 400 seeds: seed {seed}, {length} route cells, "
          f"{middle}/{len(MIDDLE)} middle cells used")
    layout, roles, route = build(seed)
    here = __file__.rsplit("/", 1)[0]
    json.dump(layout, open(f"{here}/maze_v1.json", "w"), indent=1)
    json.dump(roles, open(f"{here}/maze_v1_hole_roles.json", "w"), indent=1)
    span = sum(math.dist(layout["waypoints"][k], layout["waypoints"][k + 1])
               for k in range(len(route) - 1))
    print(f"route {len(route)} cells, {span*1000:.0f} mm")
    print(f"holes {len(layout['holes'])}: "
          f"{sum(1 for v in roles.values() if v == 'block')} blocking, "
          f"{sum(1 for v in roles.values() if v == 'dodge')} dodge")
    print(f"walls {len(layout['walls_h'])}H {len(layout['walls_v'])}V (no border)")
