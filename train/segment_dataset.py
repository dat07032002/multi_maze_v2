"""Generate a balanced local-maneuver dataset for maze policy training.

The dataset contains route geometries and physically valid initial-condition
specifications, not rendered trajectories.  Authentic slices of the real maze
are used first, valid whole-layout reflections are used second, and procedural
flat-plate routes fill the remaining class imbalance.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from contract import policy_contract as pc
from sim.mjcf_builder import load_layout, load_parameters
from sim.route import Route
from sim.route_segments import RouteSegment, classify_route
from sim.symmetry import mirror_layout

KINDS = ("straight", "gentle_left", "gentle_right",
         "sharp_left", "sharp_right")
SPACING_M = 0.002
CONTEXT_M = 0.030


def _resample(points: np.ndarray, spacing: float = SPACING_M) -> np.ndarray:
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(deltas)]
    count = max(2, int(round(cumulative[-1] / spacing)) + 1)
    samples = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack([
        np.interp(samples, cumulative, points[:, axis]) for axis in (0, 1)
    ])


def _local_turn_degrees(points: np.ndarray,
                        half_window_m: float = 0.006) -> np.ndarray:
    """The same 12 mm signed-turn measurement used for the real route."""
    tangents = np.gradient(points, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True),
                           1e-12)
    spacing = float(np.mean(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    span = max(1, int(round(half_window_m / spacing)))
    indices = np.arange(len(points))
    before = tangents[np.maximum(0, indices - span)]
    after = tangents[np.minimum(len(points) - 1, indices + span)]
    cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    dot = np.sum(before * after, axis=1)
    values = np.degrees(np.arctan2(cross, dot))
    return np.convolve(np.pad(values, 1, mode="edge"),
                       np.ones(3) / 3.0, mode="valid")


def _placed_on_board(points: np.ndarray, rng: np.random.Generator,
                     width: float, height: float,
                     margin: float = 0.012) -> np.ndarray:
    """Rotate and translate a local path into a valid board footprint."""
    centred = points - points[0]
    for _ in range(500):
        angle = float(rng.uniform(-math.pi, math.pi))
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        candidate = centred @ rotation.T
        lo, hi = candidate.min(axis=0), candidate.max(axis=0)
        room = np.array([width, height]) - 2.0 * margin - (hi - lo)
        if np.any(room < 0.0):
            continue
        translation = margin - lo + rng.uniform(0.0, room)
        return candidate + translation
    raise RuntimeError("could not place generated segment on the board")


def _authentic_geometry(route: Route, segment: RouteSegment,
                        variant: str, geometry_id: str) -> dict:
    start_s = max(0.0, segment.start_s - CONTEXT_M)
    end_s = min(route.length, segment.end_s + CONTEXT_M)
    start = route.index_at(start_s)
    end = min(len(route.points), route.index_at(end_s) + 1)
    points = route.points[start:end]
    return {
        "id": geometry_id,
        "kind": segment.kind,
        "source": "authentic" if variant == "original"
                  else "mirrored_authentic",
        "layout_variant": variant,
        "source_segment": segment.number,
        "points_m": np.round(points, 6).tolist(),
        "focus_start_m": round(segment.start_s - start_s, 6),
        "focus_end_m": round(segment.end_s - start_s, 6),
        "total_length_m": round(end_s - start_s, 6),
        "peak_local_turn_deg": round(segment.peak_turn_deg, 4),
        "min_clearance_m": round(float(np.min(route.clearance[start:end])), 6),
    }


def _procedural_geometry(kind: str, rng: np.random.Generator,
                         layout: dict, geometry_id: str) -> dict:
    width, height = layout["board_width"], layout["board_height"]
    if kind == "straight":
        length = float(rng.uniform(0.050, 0.155))
        raw = np.array([[0.0, 0.0], [length, 0.0]])
        points = _resample(raw)
        focus_start, focus_end = 0.0, length
        total_heading = peak = 0.0
        radius = None
    else:
        left = kind.endswith("left")
        sharp = kind.startswith("sharp")
        sign = 1.0 if left else -1.0
        for _ in range(2_000):
            if sharp:
                radius = float(rng.uniform(0.008, 0.023))
                total_heading = float(rng.uniform(50.0, 105.0))
            else:
                radius = float(rng.uniform(0.025, 0.065))
                total_heading = float(rng.uniform(20.0, 65.0))
            approach = float(rng.uniform(0.025, 0.055))
            exit_length = float(rng.uniform(0.025, 0.055))
            theta = sign * math.radians(total_heading)
            arc_length = radius * abs(theta)
            if arc_length < 0.014:
                continue
            arc_count = max(8, int(math.ceil(arc_length / 0.001)))
            headings = np.linspace(0.0, theta, arc_count)
            arc = np.column_stack([
                approach + sign * radius * np.sin(headings),
                sign * radius * (1.0 - np.cos(headings)),
            ])
            exit_direction = np.array([math.cos(theta), math.sin(theta)])
            exit_points = arc[-1] + np.linspace(0.001, exit_length,
                                                max(2, int(exit_length / 0.001)))[:, None] \
                * exit_direction
            raw = np.vstack([[[0.0, 0.0], [approach, 0.0]], arc[1:], exit_points])
            points = _resample(raw)
            turns = _local_turn_degrees(points)
            signed_peak = float(turns[np.argmax(np.abs(turns))])
            peak = abs(signed_peak)
            correct_side = signed_peak > 0 if left else signed_peak < 0
            correct_strength = peak >= 30.0 if sharp else 10.0 <= peak < 30.0
            if correct_side and correct_strength:
                focus_start = approach
                focus_end = approach + arc_length
                break
        else:
            raise RuntimeError(f"could not generate a valid {kind} segment")

    points = _placed_on_board(points, rng, width, height)
    clearance = float(rng.uniform(0.002, 0.010))
    result = {
        "id": geometry_id,
        "kind": kind,
        "source": "procedural",
        "layout_variant": "flat_procedural",
        "source_segment": None,
        "points_m": np.round(points, 6).tolist(),
        "focus_start_m": round(float(focus_start), 6),
        "focus_end_m": round(float(focus_end), 6),
        "total_length_m": round(float(np.sum(
            np.linalg.norm(np.diff(points, axis=0), axis=1))), 6),
        "peak_local_turn_deg": round(float(peak), 4),
        "min_clearance_m": round(clearance, 6),
    }
    if radius is not None:
        result["radius_m"] = round(radius, 6)
        result["total_heading_change_deg"] = round(
            sign * total_heading, 4)
    return result


def generate_geometries(per_kind: int = 50, seed: int = 20260807) -> list[dict]:
    """Build an exactly balanced list of maneuver geometries."""
    if per_kind < 1:
        raise ValueError("per_kind must be positive")
    rng = np.random.default_rng(seed)
    layout = load_layout()
    variants = [
        ("original", Route(layout=layout)),
        ("mirror_x", Route(layout=mirror_layout(layout, axis=0))),
    ]
    candidates: dict[str, list[tuple[Route, RouteSegment, str]]] = {
        kind: [] for kind in KINDS
    }
    for variant, route in variants:
        _, segments = classify_route(route)
        for segment in segments:
            # Very short straight classifications are merely shoulders between
            # nearby turns; they are poor stand-alone straight exercises.
            if segment.kind == "straight" and segment.length < 0.012:
                continue
            candidates[segment.kind].append((route, segment, variant))

    geometries = []
    counters = Counter()
    for kind in KINDS:
        for route, segment, variant in candidates[kind][:per_kind]:
            counters[kind] += 1
            geometry_id = f"{kind}-{counters[kind]:03d}"
            geometries.append(_authentic_geometry(
                route, segment, variant, geometry_id))
        while counters[kind] < per_kind:
            counters[kind] += 1
            geometry_id = f"{kind}-{counters[kind]:03d}"
            geometries.append(_procedural_geometry(
                kind, rng, layout, geometry_id))
    return geometries


def generate_initial_conditions(geometries: list[dict], per_geometry: int = 20,
                                seed: int = 20260807) -> list[dict]:
    """Generate varied but corridor-safe starts for every geometry."""
    if per_geometry < 1:
        raise ValueError("per_geometry must be positive")
    rng = np.random.default_rng(seed + 1)
    params = load_parameters()
    width, height = load_layout()["board_width"], load_layout()["board_height"]
    board_extent = np.array([width, height])
    episodes = []
    offset_levels = np.linspace(-0.5, 0.5, 5)
    speed_levels = np.linspace(0.004, 0.044, 5)
    randomization_levels = (0.0, 0.0, 0.1, 0.1, 0.25)

    for geometry in geometries:
        points = np.asarray(geometry["points_m"], dtype=float)
        tangent = points[min(3, len(points) - 1)] - points[0]
        tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
        normal = np.array([-tangent[1], tangent[0]])
        clearance = float(geometry["min_clearance_m"])
        max_offset = max(0.00025, 0.5 * clearance)

        for index in range(per_geometry):
            offset = float(offset_levels[index % len(offset_levels)] *
                           (2.0 * max_offset))
            position = points[0] + offset * normal
            if np.any(position <= params["ball.radius"]) or np.any(
                    position >= board_extent - params["ball.radius"]):
                position = points[0]
                offset = 0.0
            forward_speed = float(speed_levels[(index // 4) % len(speed_levels)])
            lateral_speed = float(rng.uniform(-0.010, 0.010))
            velocity = forward_speed * tangent + lateral_speed * normal
            angles = rng.uniform(-math.radians(0.5), math.radians(0.5), size=2)
            action = np.asarray(pc.angles_to_action(
                angles[0], angles[1], params["actuator.max_tilt"]), dtype=float)
            episodes.append({
                "id": f"{geometry['id']}-episode-{index + 1:02d}",
                "geometry_id": geometry["id"],
                "kind": geometry["kind"],
                "initial_position_m": np.round(position, 6).tolist(),
                "initial_velocity_m_s": np.round(velocity, 6).tolist(),
                "initial_lateral_offset_m": round(offset, 6),
                "initial_board_angles_rad": np.round(angles, 7).tolist(),
                "initial_action_history": np.round(
                    np.tile(action, (pc.ACTION_HISTORY, 1)), 6).tolist(),
                "randomization_scale": randomization_levels[
                    index % len(randomization_levels)],
                "sensor_seed": int(rng.integers(0, 2**31 - 1)),
                "physics_seed": int(rng.integers(0, 2**31 - 1)),
            })
    return episodes


def generate_dataset(per_kind: int = 50, per_geometry: int = 20,
                     seed: int = 20260807) -> dict:
    geometries = generate_geometries(per_kind=per_kind, seed=seed)
    episodes = generate_initial_conditions(
        geometries, per_geometry=per_geometry, seed=seed)
    return {
        "schema": "balanced_local_maneuvers_v1",
        "seed": seed,
        "spacing_m": SPACING_M,
        "classification": {
            "turn_window_m": 0.012,
            "straight_below_deg": 10.0,
            "sharp_at_or_above_deg": 30.0,
        },
        "geometry_target_per_kind": per_kind,
        "initial_conditions_per_geometry": per_geometry,
        "geometries": geometries,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-kind", type=int, default=50)
    parser.add_argument("--conditions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--out", default="artifacts/local_segments/balanced_dataset.json")
    args = parser.parse_args()

    dataset = generate_dataset(args.per_kind, args.conditions, args.seed)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dataset, indent=2), encoding="utf-8")

    geometry_counts = Counter(g["kind"] for g in dataset["geometries"])
    source_counts = Counter(g["source"] for g in dataset["geometries"])
    episode_counts = Counter(e["kind"] for e in dataset["episodes"])
    print(f"saved {len(dataset['geometries']):,} geometries and "
          f"{len(dataset['episodes']):,} episode specifications to "
          f"{destination.resolve()}")
    print("geometry classes:", dict(geometry_counts))
    print("geometry sources:", dict(source_counts))
    print("episode classes:", dict(episode_counts))


if __name__ == "__main__":
    main()
