"""Clinic layout spacing (task 002, design ruling 2026-11-04): countable objects read as clearly separate in the place
photos, sit wholly inside one photo wedge, and are not seen through a doorway from another place. 2D only, computed
from truth.json + the generator's part library + places.yaml. Run: uv run pytest tests/test_layout_spacing.py"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import pytest
import yaml

CLINIC = Path(__file__).resolve().parents[1] / "sim" / "worlds" / "clinic"
sys.path.insert(0, str(CLINIC))
import generate  # noqa: E402

TRUTH = json.loads((CLINIC / "generated" / "truth.json").read_text())
LAYOUT = yaml.safe_load((CLINIC / "layout.yaml").read_text())
PLACES = yaml.safe_load((CLINIC / "places.yaml").read_text())["places"]
COUNTABLE = [o for o in TRUTH["objects"] if o["countable"]]

MIN_GAP_M = 0.5        # design ruling: >= 0.5 m of visible floor between any two countable objects
MIN_SEAM_DEG = 5.0     # design ruling: footprint >= 5 deg from the nearest photo seam (seams at place yaw + 45 + k*90)
# Camera can be anywhere within G1's 0.30 m arrival tolerance of the place, plus the base->camera offset (0.0596 m,
# 002 rehearsal photos.json camera_offset_proof): 0.36 m.
CAM_DISC_M = 0.36
MIN_PLACE_CLEARANCE_M = 0.6  # reviewer 002-W2: hall place >= 0.6 m from walls / door frames for Nav2


def footprint(o: dict) -> list[tuple[float, float]]:
    """World-frame corners (ccw) of the object's 2D bounding box over all its parts."""
    xs, ys = [], []
    for kind, size, (px, py, _), _ in generate.parts_for(o["class"], o["state"]):
        hx, hy = (size[0] / 2, size[1] / 2) if kind == "box" else ((size[0], size[0]) if kind == "cyl" else (size, size))
        xs += [px - hx, px + hx]
        ys += [py - hy, py + hy]
    x, y, yaw = o["pose"][0], o["pose"][1], o["pose"][5]
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * u - s * v, y + s * u + c * v) for u, v in
            ((min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys)))]


def _seg_point_dist(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def gap(p: list[tuple[float, float]], q: list[tuple[float, float]]) -> float:
    """Edge-to-edge distance of two non-overlapping convex polygons."""
    edges = lambda poly: list(zip(poly, poly[1:] + poly[:1]))  # noqa: E731
    return min(min(_seg_point_dist(v, a, b) for v in p for a, b in edges(q)),
               min(_seg_point_dist(v, a, b) for v in q for a, b in edges(p)))


def rel_deg(place: dict, pt: tuple[float, float]) -> float:
    """Bearing of pt from the place, relative to the place yaw, in (-180, 180]."""
    a = math.degrees(math.atan2(pt[1] - place["y"], pt[0] - place["x"]) - place["yaw"])
    return (a + 180) % 360 - 180


def wedge(place: dict, o: dict) -> tuple[set[int], float]:
    """(photo indices the footprint corners fall in, min corner distance to a seam in deg)."""
    rel = [rel_deg(place, c) for c in footprint(o)]
    ks = {round(r / 90) % 4 for r in rel}
    return ks, min(45 - abs(r - 90 * round(r / 90)) for r in rel)


def angular_interval(place: dict, o: dict) -> tuple[float, float]:
    rel = [rel_deg(place, c) for c in footprint(o)]
    centre = rel[0]
    rel = [centre + ((r - centre + 180) % 360 - 180) for r in rel]
    return min(rel), max(rel)


def same_room_pairs() -> list[tuple[dict, dict]]:
    return [(a, b) for a, b in itertools.combinations(COUNTABLE, 2) if a["room"] == b["room"]]


def test_pairs_exist() -> None:
    assert {o["class"] for o in COUNTABLE} == {"bed", "chair"} and len(same_room_pairs()) >= 2


@pytest.mark.parametrize("a,b", same_room_pairs(), ids=lambda o: o["id"])
def test_gap_at_least_half_metre(a: dict, b: dict) -> None:
    g = gap(footprint(a), footprint(b))
    assert g >= MIN_GAP_M, f"{a['id']}-{b['id']}: {g:.3f} m"


@pytest.mark.parametrize("a,b", same_room_pairs(), ids=lambda o: o["id"])
def test_gap_not_hidden_by_perspective(a: dict, b: dict) -> None:
    """From the room place, the two footprints' bearing intervals are disjoint: a vertical sight plane through the
    camera passes between them, so the floor in the gap is visible from the near edge to the wall."""
    place = PLACES[a["room"]]
    (a0, a1), (b0, b1) = angular_interval(place, a), angular_interval(place, b)
    sep = max(b0 - a1, a0 - b1)
    assert sep > 0, f"{a['id']}-{b['id']}: bearing intervals overlap by {-sep:.2f} deg"


@pytest.mark.parametrize("o", COUNTABLE, ids=lambda o: o["id"])
def test_inside_one_photo_wedge(o: dict) -> None:
    ks, margin = wedge(PLACES[o["room"]], o)
    assert len(ks) == 1, f"{o['id']} spans photos {sorted(ks)}"
    assert margin >= MIN_SEAM_DEG, f"{o['id']}: {margin:.2f} deg from a seam"


# ------------------------------------------------------------------------------------------------- G4 sightlines
WALLS = [((lo, line), (hi, line)) if axis == "y" else ((line, lo), (line, hi))
         for axis, line, lo, hi in generate.wall_segments(LAYOUT["rooms"])]


def _cross(o: tuple, a: tuple, b: tuple) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _intersects(p: tuple, q: tuple, a: tuple, b: tuple) -> bool:
    """Closed segment intersection (touching a wall end counts as blocked; walls are 0.1 m thick in reality)."""
    d1, d2, d3, d4 = _cross(a, b, p), _cross(a, b, q), _cross(p, q, a), _cross(p, q, b)
    if ((d1 > 0) != (d2 > 0) or d1 == 0 or d2 == 0) and ((d3 > 0) != (d4 > 0) or d3 == 0 or d4 == 0):
        return (min(p[0], q[0]) <= max(a[0], b[0]) and min(a[0], b[0]) <= max(p[0], q[0])
                and min(p[1], q[1]) <= max(a[1], b[1]) and min(a[1], b[1]) <= max(p[1], q[1]))
    return False


def visible_points(place: dict, o: dict) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """(camera, footprint point) pairs with a wall-free 2D line of sight. Conservative: walls are centre lines, no
    object occludes, every bearing is in some photo; the camera ranges over a CAM_DISC_M disc around the place."""
    cams = [(place["x"], place["y"])] + [(place["x"] + r * math.cos(a), place["y"] + r * math.sin(a))
                                         for r in (CAM_DISC_M / 2, CAM_DISC_M) for a in (i * math.pi / 8 for i in range(16))]
    poly = footprint(o)
    pts = [(a[0] + t / 10 * (b[0] - a[0]), a[1] + t / 10 * (b[1] - a[1])) for a, b in zip(poly, poly[1:] + poly[:1]) for t in range(10)]
    return [(c, p) for c in cams for p in pts if not any(_intersects(c, p, a, b) for a, b in WALLS)]


@pytest.mark.parametrize("room", [r for r in PLACES if r != "hall"])
def test_no_foreign_countable_in_sight(room: str) -> None:
    seen = {o["id"]: len(v) for o in COUNTABLE if o["room"] != room and (v := visible_points(PLACES[room], o))}
    assert not seen, f"{room} place sees {seen} (camera, point) sight lines"


def test_hall_place_sees_no_countable() -> None:
    """The hall place (east half, 002-W2 reviewer ruling) sees no bed or chair through any doorway; from (0, 0)
    the ward beds showed through the ward doorway (rehearsal hall k=1: bed_blue_1 2241 px, bed_blue_2 1998 px)."""
    seen = {o["id"]: len(v) for o in COUNTABLE if (v := visible_points(PLACES["hall"], o))}
    assert not seen, f"hall place sees {seen} (camera, point) sight lines"


def place_clearance(name: str) -> float:
    """Distance from a place point to the nearest wall face or door frame."""
    p = (PLACES[name]["x"], PLACES[name]["y"])
    return min(_seg_point_dist(p, a, b) for a, b in WALLS) - LAYOUT["wall"]["thickness"] / 2


def test_hall_place_clear_of_walls() -> None:
    """Hall place only: the ward place sits 0.45 m from its doorway wall by design (bed row inside one photo wedge)."""
    assert place_clearance("hall") >= MIN_PLACE_CLEARANCE_M, f"hall: {place_clearance('hall'):.2f} m"
