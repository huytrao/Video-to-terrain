"""scene.json — the single source of truth for a reconstructed 3D environment.

Every exporter (GLB, MJCF, USD, .blend) reads this document and nothing else.
Every geometric number carries a ``source`` tag so Stage J can refuse a scene
whose room shell was invented rather than measured.

Schema is versioned; readers must check ``schema_version`` before trusting keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "1.0"

# Where a number came from. Stage J treats these very differently:
#   measured  - solved from observed geometry (plane fit, depth, GT pose)
#   derived   - computed from measured values (extent = max - min)
#   assumed   - a constant nobody observed.  Never legal in room geometry.
Source = Literal["measured", "derived", "assumed"]

VALID_SOURCES = ("measured", "derived", "assumed")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Plane:
    """Oriented plane ax + by + cz + d = 0, normal (a, b, c) unit-length."""

    coeffs: list[float]
    source: Source
    inlier_count: int = 0
    rms_residual_m: float | None = None

    def normal(self) -> list[float]:
        return list(self.coeffs[:3])


@dataclass
class Floor:
    plane: Plane
    polygon_xy: list[list[float]]  # CCW footprint in the scene frame
    source: Source


@dataclass
class Ceiling:
    height_m: float  # distance above the floor plane
    plane: Plane | None
    source: Source


@dataclass
class Wall:
    plane: Plane
    footprint_xy: list[list[float]]  # the two endpoints of the wall segment
    z_range_m: list[float]  # [z_bottom, z_top] in the scene frame
    source: Source


@dataclass
class Room:
    floor: Floor
    ceiling: Ceiling
    walls: list[Wall] = field(default_factory=list)

    def height_m(self) -> float:
        return self.ceiling.height_m

    def footprint_extent_m(self) -> tuple[float, float]:
        """The room's own length and width, independent of world orientation.

        A footprint is a polygon, and its dimensions are edge lengths. Taking
        the span along the world axes instead inflates every room that is not
        aligned to them — which is every real room — and would fail a correct
        reconstruction purely for being rotated.
        """
        poly = self.floor.polygon_xy
        if len(poly) == 4:
            edges = [
                float(
                    ((poly[i][0] - poly[(i + 1) % 4][0]) ** 2
                     + (poly[i][1] - poly[(i + 1) % 4][1]) ** 2) ** 0.5
                )
                for i in range(4)
            ]
            a, b = max(edges[0], edges[1]), min(edges[0], edges[1])
            return (a, b)

        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        return (max(xs) - min(xs), max(ys) - min(ys))


@dataclass
class OrientedBox:
    """Object box. ``size`` is full extent along the box's own axes, in metres."""

    center: list[float]
    size: list[float]
    rotation: list[list[float]]  # 3x3, columns are the box axes in scene frame
    source: Source

    def volume_m3(self) -> float:
        return float(self.size[0] * self.size[1] * self.size[2])


@dataclass
class SceneObject:
    id: str
    category: str
    box: OrientedBox
    confidence: float = 1.0
    mesh_path: str | None = None
    material: dict[str, Any] | None = None
    support_object_id: str | None = None  # what it rests on, for physics


@dataclass
class Camera:
    id: str
    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    position: list[float]
    rotation: list[list[float]]  # world-from-camera
    source: Source


@dataclass
class SceneDocument:
    scene_id: str
    room: Room
    objects: list[SceneObject] = field(default_factory=list)
    cameras: list[Camera] = field(default_factory=list)
    assets: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    units: str = "meters"
    up_axis: str = "z"

    # ---- io -------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["provenance"] = {"written_at": _utcnow(), **self.provenance}
        return d

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @staticmethod
    def read(path: str | Path) -> "SceneDocument":
        raw = json.loads(Path(path).read_text())
        return SceneDocument.from_dict(raw)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "SceneDocument":
        version = raw.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"scene.json schema_version {version!r}, this reader expects {SCHEMA_VERSION!r}"
            )

        def plane(p: dict[str, Any] | None) -> Plane | None:
            if p is None:
                return None
            return Plane(
                coeffs=p["coeffs"],
                source=p["source"],
                inlier_count=p.get("inlier_count", 0),
                rms_residual_m=p.get("rms_residual_m"),
            )

        r = raw["room"]
        room = Room(
            floor=Floor(
                plane=plane(r["floor"]["plane"]),
                polygon_xy=r["floor"]["polygon_xy"],
                source=r["floor"]["source"],
            ),
            ceiling=Ceiling(
                height_m=r["ceiling"]["height_m"],
                plane=plane(r["ceiling"].get("plane")),
                source=r["ceiling"]["source"],
            ),
            walls=[
                Wall(
                    plane=plane(w["plane"]),
                    footprint_xy=w["footprint_xy"],
                    z_range_m=w["z_range_m"],
                    source=w["source"],
                )
                for w in r.get("walls", [])
            ],
        )

        objects = [
            SceneObject(
                id=o["id"],
                category=o["category"],
                box=OrientedBox(
                    center=o["box"]["center"],
                    size=o["box"]["size"],
                    rotation=o["box"]["rotation"],
                    source=o["box"]["source"],
                ),
                confidence=o.get("confidence", 1.0),
                mesh_path=o.get("mesh_path"),
                material=o.get("material"),
                support_object_id=o.get("support_object_id"),
            )
            for o in raw.get("objects", [])
        ]

        cameras = [Camera(**c) for c in raw.get("cameras", [])]

        return SceneDocument(
            scene_id=raw["scene_id"],
            room=room,
            objects=objects,
            cameras=cameras,
            assets=raw.get("assets", {}),
            provenance=raw.get("provenance", {}),
            units=raw.get("units", "meters"),
            up_axis=raw.get("up_axis", "z"),
        )

    # ---- structural checks ---------------------------------------------

    def structural_errors(self) -> list[str]:
        """Schema-level problems, independent of any ground truth."""
        errs: list[str] = []
        if self.units != "meters":
            errs.append(f"units must be 'meters', got {self.units!r}")
        if self.up_axis != "z":
            errs.append(f"up_axis must be 'z', got {self.up_axis!r}")

        for label, src in self._sources():
            if src not in VALID_SOURCES:
                errs.append(f"{label}: invalid source {src!r}")

        seen: set[str] = set()
        for o in self.objects:
            if o.id in seen:
                errs.append(f"duplicate object id {o.id!r}")
            seen.add(o.id)
            if any(s <= 0 for s in o.box.size):
                errs.append(f"{o.id}: non-positive box size {o.box.size}")

        ids = {o.id for o in self.objects}
        for o in self.objects:
            if o.support_object_id and o.support_object_id not in ids:
                errs.append(f"{o.id}: support_object_id {o.support_object_id!r} not in scene")

        return errs

    def _sources(self) -> list[tuple[str, str]]:
        out = [
            ("room.floor", self.room.floor.source),
            ("room.floor.plane", self.room.floor.plane.source),
            ("room.ceiling", self.room.ceiling.source),
        ]
        if self.room.ceiling.plane is not None:
            out.append(("room.ceiling.plane", self.room.ceiling.plane.source))
        for i, w in enumerate(self.room.walls):
            out.append((f"room.walls[{i}]", w.source))
            out.append((f"room.walls[{i}].plane", w.plane.source))
        for o in self.objects:
            out.append((f"objects[{o.id}].box", o.box.source))
        return out

    def assumed_room_values(self) -> list[str]:
        """Room-shell numbers nobody observed — the hardcoded-2.0m-wall detector."""
        return [
            label
            for label, src in self._sources()
            if src == "assumed" and label.startswith("room.")
        ]
