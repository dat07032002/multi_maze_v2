"""Clearance-weighted A* and line-of-sight route simplification."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .maze_map import CameraMazeMap


@dataclass(frozen=True)
class PlannedRoute:
    points_mm: tuple[tuple[float, float], ...]
    grid_path: tuple[tuple[int, int], ...]
    length_mm: float
    minimum_clearance_mm: float


NEIGHBOURS = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
    (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
)


def nearest_free(occupied: np.ndarray, point: tuple[int, int],
                 max_radius: int = 30) -> tuple[int, int]:
    row, col = point
    if not occupied[row, col]:
        return point
    for radius in range(1, max_radius + 1):
        r0, r1 = max(0, row - radius), min(occupied.shape[0], row + radius + 1)
        c0, c1 = max(0, col - radius), min(occupied.shape[1], col + radius + 1)
        free = np.argwhere(~occupied[r0:r1, c0:c1])
        if len(free):
            absolute = free + np.array([r0, c0])
            distances = np.sum((absolute - np.array([row, col])) ** 2, axis=1)
            return tuple(int(v) for v in absolute[int(np.argmin(distances))])
    raise ValueError(f"no free cell within {max_radius} cells of {point}")


def _astar(maze: CameraMazeMap, start, goal,
           clearance_weight: float) -> list[tuple[int, int]]:
    occupied = maze.occupied
    height, width = occupied.shape
    queue = [(0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {start: 0.0}
    while queue:
        _, current = heapq.heappop(queue)
        if current == goal:
            break
        row, col = current
        for dr, dc, distance in NEIGHBOURS:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if occupied[nr, nc]:
                continue
            if dr and dc and (occupied[row + dr, col]
                              or occupied[row, col + dc]):
                continue  # do not cut diagonally through obstacle corners
            clearance = float(maze.clearance_mm[nr, nc])
            multiplier = 1.0 + clearance_weight * math.exp(-clearance / 5.0)
            candidate = cost[current] + distance * multiplier
            neighbour = (nr, nc)
            if candidate >= cost.get(neighbour, math.inf):
                continue
            cost[neighbour] = candidate
            came_from[neighbour] = current
            heuristic = math.hypot(goal[0] - nr, goal[1] - nc)
            heapq.heappush(queue, (candidate + heuristic, neighbour))
    if goal not in cost:
        raise ValueError("no collision-free route between start and goal")
    path = [goal]
    while path[-1] != start:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def _line_is_free(occupied: np.ndarray, a, b) -> bool:
    line = np.zeros(occupied.shape, dtype=np.uint8)
    cv2.line(line, (a[1], a[0]), (b[1], b[0]), 1, thickness=1,
             lineType=cv2.LINE_8)
    return not bool(np.any(occupied & line.astype(bool)))


def _prune_line_of_sight(path: list[tuple[int, int]],
                         occupied: np.ndarray) -> list[tuple[int, int]]:
    if len(path) <= 2:
        return path
    result = [path[0]]
    anchor = 0
    while anchor < len(path) - 1:
        candidate = len(path) - 1
        while candidate > anchor + 1 and not _line_is_free(
                occupied, path[anchor], path[candidate]):
            candidate -= 1
        result.append(path[candidate])
        anchor = candidate
    return result


def _resample(points: list[tuple[float, float]], spacing_mm: float):
    if len(points) < 2:
        return points
    lengths = [0.0]
    for a, b in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.dist(a, b))
    samples = np.arange(0.0, lengths[-1], spacing_mm).tolist()
    if not samples or samples[-1] != lengths[-1]:
        samples.append(lengths[-1])
    output = []
    segment = 0
    for distance in samples:
        while segment + 1 < len(lengths) - 1 and lengths[segment + 1] < distance:
            segment += 1
        span = lengths[segment + 1] - lengths[segment]
        fraction = 0.0 if span == 0 else (distance - lengths[segment]) / span
        a, b = points[segment], points[segment + 1]
        output.append((a[0] + fraction * (b[0] - a[0]),
                       a[1] + fraction * (b[1] - a[1])))
    return output


def plan_route(
    maze: CameraMazeMap,
    start_mm,
    goal_mm,
    *,
    clearance_weight: float = 2.0,
    waypoint_spacing_mm: float = 5.0,
) -> PlannedRoute:
    if waypoint_spacing_mm <= 0 or clearance_weight < 0:
        raise ValueError("invalid route spacing or clearance weight")
    start = nearest_free(maze.occupied, maze.board_to_grid(start_mm))
    goal = nearest_free(maze.occupied, maze.board_to_grid(goal_mm))
    raw_path = _astar(maze, start, goal, clearance_weight)
    pruned = _prune_line_of_sight(raw_path, maze.occupied)
    control = [maze.grid_to_board(point) for point in pruned]
    points = _resample(control, waypoint_spacing_mm)
    length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
    clearances = [maze.clearance_mm[maze.board_to_grid(point)] for point in points]
    return PlannedRoute(
        points_mm=tuple(points), grid_path=tuple(raw_path),
        length_mm=float(length),
        minimum_clearance_mm=float(min(clearances)),
    )
