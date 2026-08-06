"""Board tag geometry: where each tag sits on the tilting plate.

This is the highest-risk input in the estimator. A wrong tag position corrupts
the recovered pose in a way reprojection residuals cannot reveal -- the solve
happily fits whatever geometry it is given, and reports a small residual while
being wrong. The only defence is measuring carefully and validating the numbers
for self-consistency, which is what this module does.

Frame convention:
  origin at the plate's lower-left corner, +x right, +y up, z = 0 at the plate
  surface. Board is 256 x 226 mm. The maze layouts in `maze_design/` use the
  same frame, so wall, hole, and waypoint coordinates need no conversion.

Note z = 0 is the PLAYING SURFACE, not the bottom of the printed part. The maze
prints 11 mm tall over a 3 mm floor, so the corner tag pads top out 8 mm above
this origin and a tag glued there sits at z = 11 mm.

Each tag is stored by the lower-left of its square OUTER BLACK footprint, its
size, and an optional in-plane rotation about its centre. This keeps x/y tied
to caliper-friendly footprint edges even when a tag is turned by 90 degrees.
Corner order matches cv2.aruco: [top-left, top-right, bottom-right,
bottom-left] in the tag's own frame, verified against a rendered reference.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

BOARD_WIDTH_M = 0.256
BOARD_HEIGHT_M = 0.226


@dataclass(frozen=True)
class Tag:
    """One tag: id, square-footprint origin, size, height, and rotation."""

    tag_id: int
    x_m: float
    y_m: float
    size_m: float
    z_m: float = 0.0          # non-zero when mounted on a riser
    label: str = ""
    rotation_deg: float = 0.0  # counter-clockwise in the board XY frame

    def corners(self) -> np.ndarray:
        """Four outer corners in cv2.aruco order, board frame, metres."""
        x, y, s, z = self.x_m, self.y_m, self.size_m, self.z_m
        corners = np.array(
            [
                [x, y + s, z],          # 0 top-left
                [x + s, y + s, z],      # 1 top-right
                [x + s, y, z],          # 2 bottom-right
                [x, y, z],              # 3 bottom-left
            ],
            dtype=np.float64,
        )
        if self.rotation_deg:
            angle = math.radians(self.rotation_deg)
            c, s_angle = math.cos(angle), math.sin(angle)
            rotation = np.array([[c, -s_angle], [s_angle, c]])
            centre = np.array([x + s / 2, y + s / 2])
            corners[:, :2] = (corners[:, :2] - centre) @ rotation.T + centre
        return corners.astype(np.float32)

    def centre(self) -> np.ndarray:
        return np.array(
            [self.x_m + self.size_m / 2, self.y_m + self.size_m / 2, self.z_m],
            dtype=np.float32,
        )


class BoardGeometry:
    """The set of tags mounted on the plate, with validation."""

    def __init__(self, tags: Iterable[Tag], *, board_width_m: float = BOARD_WIDTH_M,
                 board_height_m: float = BOARD_HEIGHT_M, source: str = ""):
        self.tags: Dict[int, Tag] = {t.tag_id: t for t in tags}
        self.board_width_m = float(board_width_m)
        self.board_height_m = float(board_height_m)
        self.source = source
        if len(self.tags) < 3:
            raise ValueError(
                f"Need at least 3 tags to solve a pose, got {len(self.tags)}"
            )

    # ---- construction ----------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "BoardGeometry":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        tags = [
            Tag(
                tag_id=int(entry["id"]),
                x_m=float(entry["x_m"]),
                y_m=float(entry["y_m"]),
                size_m=float(entry["size_m"]),
                z_m=float(entry.get("z_m", 0.0)),
                label=str(entry.get("label", "")),
                rotation_deg=float(entry.get("rotation_deg", 0.0)),
            )
            for entry in data["tags"]
        ]
        return cls(
            tags,
            board_width_m=float(data.get("board_width_m", BOARD_WIDTH_M)),
            board_height_m=float(data.get("board_height_m", BOARD_HEIGHT_M)),
            source=str(path),
        )

    # ---- lookup ----------------------------------------------------------
    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.tags))

    def object_points(self, ids: Iterable[int]) -> np.ndarray:
        """Stacked 3D corners for the given ids, in the order supplied."""
        return np.concatenate([self.tags[int(i)].corners() for i in ids], axis=0)

    def is_planar(self, tolerance_m: float = 1e-4) -> bool:
        heights = [t.z_m for t in self.tags.values()]
        return (max(heights) - min(heights)) <= tolerance_m

    def baseline_m(self) -> float:
        """Largest distance between tag centres: what sets tilt precision."""
        centres = [t.centre() for t in self.tags.values()]
        return max(
            (float(np.linalg.norm(a - b)) for i, a in enumerate(centres)
             for b in centres[i + 1:]),
            default=0.0,
        )

    # ---- validation ------------------------------------------------------
    def validate(self) -> list[str]:
        """Return warnings. Empty list means nothing looked suspicious.

        These are sanity checks on measured numbers, not proof of correctness.
        A consistent but wrongly-measured layout passes all of them.
        """
        warnings: list[str] = []

        for tag in self.tags.values():
            if not (0.0 <= tag.x_m <= self.board_width_m):
                warnings.append(
                    f"tag {tag.tag_id}: x {tag.x_m*1000:.1f} mm is outside the "
                    f"board (0 to {self.board_width_m*1000:.0f} mm)")
            if not (0.0 <= tag.y_m <= self.board_height_m):
                warnings.append(
                    f"tag {tag.tag_id}: y {tag.y_m*1000:.1f} mm is outside the "
                    f"board (0 to {self.board_height_m*1000:.0f} mm)")
            if not (0.005 <= tag.size_m <= 0.060):
                warnings.append(
                    f"tag {tag.tag_id}: size {tag.size_m*1000:.1f} mm is "
                    "implausible for this board")

        sizes = [t.size_m for t in self.tags.values()]
        if max(sizes) - min(sizes) > 1e-4:
            warnings.append(
                f"tag sizes differ by {(max(sizes)-min(sizes))*1000:.2f} mm; "
                "expected identical prints")

        # Overlapping tags means a transcription error somewhere.
        items = list(self.tags.values())
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                if (abs(a.x_m - b.x_m) < max(a.size_m, b.size_m)
                        and abs(a.y_m - b.y_m) < max(a.size_m, b.size_m)):
                    warnings.append(
                        f"tags {a.tag_id} and {b.tag_id} overlap; check the "
                        "measured positions")

        baseline = self.baseline_m()
        if baseline < 0.5 * min(self.board_width_m, self.board_height_m):
            warnings.append(
                f"largest tag separation is only {baseline*1000:.0f} mm; tilt "
                "precision scales with this, so spread them further apart")

        if self.is_planar():
            warnings.append(
                "all tags are coplanar: out-of-plane tilt is ill-conditioned "
                "(~0.2-0.5 deg). Raising two on risers improves it markedly.")

        return warnings

    def describe(self) -> str:
        lines = [
            f"board {self.board_width_m*1000:.0f} x {self.board_height_m*1000:.0f} mm"
            f"   {len(self.tags)} tags   source {self.source or 'in-memory'}",
            f"largest separation {self.baseline_m()*1000:.1f} mm"
            f"   {'coplanar' if self.is_planar() else 'non-coplanar'}",
            "",
            "  id   x (mm)   y (mm)   z (mm)   size (mm)   rot (deg)   label",
        ]
        for tag_id in self.ids:
            t = self.tags[tag_id]
            lines.append(
                f"  {t.tag_id:2d}   {t.x_m*1000:6.1f}   {t.y_m*1000:6.1f}   "
                f"{t.z_m*1000:6.1f}   {t.size_m*1000:7.2f}   "
                f"{t.rotation_deg:8.1f}   {t.label}")
        return "\n".join(lines)


def template(size_mm: float = 18.0, inset_mm: float = 6.0) -> dict:
    """A starting board_tags.json with tags inset from each corner.

    These are NOMINAL positions to be replaced by caliper measurements. They
    exist so the file has the right shape, not so they can be trusted.
    """
    size = size_mm / 1000.0
    inset = inset_mm / 1000.0
    right = BOARD_WIDTH_M - inset - size
    top = BOARD_HEIGHT_M - inset - size
    placements = [
        (0, inset, inset, "bottom-left"),
        (1, right, inset, "bottom-right"),
        (2, right, top, "top-right"),
        (3, inset, top, "top-left"),
    ]
    return {
        "_comment": (
            "NOMINAL values. Replace x_m/y_m/size_m with caliper measurements "
            "of each tag's lower-left OUTER BLACK footprint in the plate frame "
            "(origin lower-left, +x right, +y up). Set z_m if a tag sits on a "
            "riser. Wrong values here corrupt the pose invisibly."
        ),
        "measured": False,
        "board_width_m": BOARD_WIDTH_M,
        "board_height_m": BOARD_HEIGHT_M,
        "family": "tag36h11",
        "tags": [
            {"id": i, "x_m": round(x, 6), "y_m": round(y, 6),
             "size_m": size, "z_m": 0.0, "rotation_deg": 0.0,
             "label": label}
            for i, x, y, label in placements
        ],
    }
