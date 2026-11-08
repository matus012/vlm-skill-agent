"""Room polygons + doorways from map_knowledge.json and point -> room name. Pure (no ROS); the adapters' where() and
the G1 router use it. map_knowledge.json holds rooms only (sim/INTERFACES.md section 2): no objects, no states."""
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAP_KNOWLEDGE = ROOT / "sim" / "worlds" / "clinic" / "generated" / "map_knowledge.json"
Polygon = list[tuple[float, float]]


def load_map(path: Path = MAP_KNOWLEDGE) -> dict[str, Any]:
    """map_knowledge.json, checked to hold nothing beyond world, layout_sha256 and rooms {polygon, doorway}."""
    m = json.loads(Path(path).read_text())
    assert set(m) == {"world", "layout_sha256", "rooms"}, f"{path}: keys {sorted(m)}"
    for name, r in m["rooms"].items():
        assert set(r) <= {"polygon", "doorway"}, f"{path}: room {name} keys {sorted(r)}"
    return m


def load_rooms(path: Path = MAP_KNOWLEDGE) -> dict[str, Polygon]:
    """{room name: polygon [(x, y), ...]} (map frame == world frame), hall included."""
    return {name: [(float(x), float(y)) for x, y in r["polygon"]] for name, r in load_map(path)["rooms"].items()}


def load_doors(path: Path = MAP_KNOWLEDGE) -> dict[str, tuple[float, float]]:
    """{room name: doorway centre (x, y)} for every room with a doorway (all but the hall)."""
    return {name: (float(r["doorway"]["center"][0]), float(r["doorway"]["center"][1]))
            for name, r in load_map(path)["rooms"].items() if "doorway" in r}


def _inside(x: float, y: float, poly: Polygon) -> bool:
    """Even-odd ray cast (half-open edges: a point on a shared edge belongs to exactly one side)."""
    hit = False
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            hit = not hit
    return hit


def _seg_dist(x: float, y: float, a: tuple[float, float], b: tuple[float, float]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((x - a[0]) * dx + (y - a[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(x - a[0] - t * dx, y - a[1] - t * dy)


def _dist(x: float, y: float, poly: Polygon) -> float:
    return min(_seg_dist(x, y, a, b) for a, b in zip(poly, poly[1:] + poly[:1]))


def room_of(x: float, y: float, rooms: dict[str, Polygon]) -> str:
    """The room whose polygon holds (x, y); a point in no room -> the room with the nearest polygon edge."""
    for name, poly in rooms.items():
        if _inside(x, y, poly):
            return name
    return min(rooms, key=lambda n: _dist(x, y, rooms[n]))
