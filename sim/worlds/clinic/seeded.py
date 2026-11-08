"""Phase 5 seeded layouts (agent/INTERFACES.md section 10, task 008). Harness side: never imported by agent/.

    placements(seed) -> {object_id: (x, y, yaw)}     the movables' poses for a layout seed (SeedRejected if a rule breaks)
    truth_for(seed, world) -> dict                   that world's truth.json with the movables moved, new layout_sha256
    count_photos(seed, room, cls, world) -> {id: k}  which look_around photo (0..3) holds each instance (ray-caster)

Movables: toolbox_red_1 (room uniform over the rooms that have a toolbox zone: office, reception, storage; since
a recorded deviation the ward has none) and chair_orange_1..4 (each chair's room uniform over the 4 rooms and independent:
0-4 per room, total 4). Everything else stays where layout.yaml puts it.

Draw (one pass, no redraw): rooms first, then per object (toolbox, chair 1..4) one of its room's zones (uniform over the
zones, so every photo wedge that has a zone is equally likely) and a uniform point in it. Pose: a chair faces the
place, the toolbox turns its long side across the line of sight. Every rule is then checked (`problems`); any breach
raises SeedRejected: a seed is used as drawn or not at all.

Zones (ZONES: per room and class, at most one centre rectangle per photo wedge) are derived from the geometry by
`python3 sim/worlds/clinic/seeded.py --derive-zones`: the single-object rules (`alone_problems`) on a GRID_M grid of
centres, then per wedge the largest rectangle of passing cells. tests/test_seeded_layouts.py re-checks a grid of every
zone, so a rejection comes from the movables among themselves (gap, one hiding another, the path to the place).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

CLINIC = Path(__file__).resolve().parent
ROOT = CLINIC.parents[2]
sys.path.insert(0, str(CLINIC))
sys.path.insert(0, str(ROOT / "probes"))
sys.path.insert(0, str(ROOT))
import generate  # noqa: E402
import raycast_seg  # noqa: E402
from agent.truth import BLOCKED_ROOMS, WORLD_DIRS, world_truth_path  # noqa: E402

LAYOUT = yaml.safe_load((CLINIC / "layout.yaml").read_text())
PLACES: dict[str, dict] = yaml.safe_load((CLINIC / "places.yaml").read_text())["places"]
ROOMS = ("ward", "office", "reception", "storage")  # rooms with a place that a movable may go to (not the hall)
TOOLBOX = "toolbox_red_1"
CHAIRS = ("chair_orange_1", "chair_orange_2", "chair_orange_3", "chair_orange_4")
MOVABLES = (TOOLBOX, *CHAIRS)
PLACEMENT_WORLDS = ("closed", "cabinet_open")  # the worlds the placement rules ray-cast in (toolbox_clutter; 008)
# + the Phase 6 blocked variants (INTERFACES section 11): truth_for / photos only, never a placement rule
WORLDS = (*PLACEMENT_WORLDS, *(f"blocked_{r}" for r in BLOCKED_ROOMS))
TARGET_CLASSES = ("toolbox", "chair", "bed", "cabinet")  # design notes: Task targets (G9): never hidden by a movable

# Pinned numbers (sources in the comments; none is new)
MIN_GAP_M = 0.5        # 2026-11-04 (design notes a recorded deviation): >= 0.5 m floor gap between countable objects
MIN_SEAM_DEG = 5.0     # 2026-11-04 (tests/test_layout_spacing.py): footprint >= 5 deg from a photo seam
CAM_DISC_M = 0.36      # tests/test_layout_spacing.py: G1 0.30 m place bound + 0.0596 m base->camera offset
PLACE_BOUND_M = 0.30   # G1 / sim/INTERFACES.md section 5: robot within 0.30 m of the place at the photos
ROBOT_RADIUS_M = 0.189  # turtlebot4_navigation nav2.yaml footprint (octagon, circumradius 0.189 m)
INFLATION_M = 0.45     # turtlebot4_navigation nav2.yaml costmap inflation_radius
NAV_XY_TOL_M = 0.10    # sim/INTERFACES.md section 5: Nav2 xy_goal_tolerance (the arrival error the robot leaves)
CAM_BACK_M, CAM_Z_M = 0.0596, 0.2388  # 002 photos.json: camera 0.0596 m behind base_link, GT camera z 0.2388 m
CLEAR_M = PLACE_BOUND_M + ROBOT_RADIUS_M  # movable footprint vs the place point (robot anywhere in the G1 bound)
TOOLBOX_MIN_DIST_M = 1.9  # a recorded deviation (live G9 failed for close-ups): toolbox centre >= 1.9 m from the place
#                           (centre, as the deviation's data: 005's storage toolbox, named 3/3, is 2.11 m centre / 1.89 m edge)
SILHOUETTE_PAD_PX = 20    # a recorded deviation: the toolbox's image box + 20 px holds no furniture / other movable pixel
GRID_M = 0.05          # reachability / zone-derivation grid cell (a computation step, not a rule)

# Centre rectangles [x0, y0, x1, y1] per room and class; output of --derive-zones (module docstring), pasted verbatim.
ZONES: dict[str, dict[str, list[list[float]]]] = {
    "ward": {
        "toolbox": [],
        "chair": [[-0.55, 1.95, -0.35, 2.2], [-4.65, 1.85, -4.5, 1.85]],
    },
    "office": {
        "toolbox": [[4.4, 2.8, 4.55, 2.85], [2.3, 1.85, 2.85, 1.9]],
        "chair": [[4.2, 2.75, 4.25, 4.85], [1.1, 5.9, 1.35, 6.1], [2.15, 1.9, 3.55, 2.1]],
    },
    "reception": {
        "toolbox": [[-0.6, -3.3, -0.35, -3.1], [-4.5, -2.7, -4.3, -2.5], [-3.45, -6.1, -1.55, -5.3]],
        "chair": [[-0.8, -3.7, -0.4, -2.75], [-1.7, -1.9, -1.65, -1.9], [-4.6, -2.9, -4.15, -2.45], [-3.55, -6.1, -1.45, -5.1]],
    },
    "storage": {
        "toolbox": [[0.45, -4.9, 0.6, -3.05]],
        "chair": [[2.15, -2.2, 2.5, -1.85], [0.4, -4.95, 0.8, -2.85], [1.05, -6.1, 1.05, -6.1]],
    },
}


class SeedRejected(ValueError):
    """The seed's draw breaks a placement rule; the seed is never used. `.reasons` lists every broken rule."""

    def __init__(self, seed: int, reasons: list[str]) -> None:
        super().__init__(f"seed {seed}: " + "; ".join(reasons))
        self.reasons = reasons


# ------------------------------------------------------------------------------------------------------- geometry
def _part_box(cls: str, state: str | None) -> tuple[float, float, float, float, float]:
    """(xmin, ymin, xmax, ymax, ztop) of a class's parts in its local frame."""
    xs, ys, zs = [], [], []
    for kind, size, (px, py, pz), _ in generate.parts_for(cls, state):
        hx, hy, hz = ((size[0] / 2, size[1] / 2, size[2] / 2) if kind == "box" else
                      (size[0], size[0], size[1] / 2) if kind == "cyl" else (size, size, size))
        xs += [px - hx, px + hx]
        ys += [py - hy, py + hy]
        zs.append(pz + hz)
    return min(xs), min(ys), max(xs), max(ys), max(zs)


def footprint(cls: str, state: str | None, x: float, y: float, yaw: float) -> list[tuple[float, float]]:
    """World-frame corners (ccw) of the class's 2D bounding box at pose (x, y, yaw) (as tests/test_layout_spacing.py)."""
    x0, y0, x1, y1, _ = _part_box(cls, state)
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * u - s * v, y + s * u + c * v) for u, v in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]


def _seg_point_dist(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def _edges(poly: list) -> list:
    return list(zip(poly, poly[1:] + poly[:1]))


def _inside(p: tuple[float, float], poly: list) -> bool:
    """Point strictly inside a ccw convex polygon."""
    return all((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) > 0 for a, b in _edges(poly))


def _cross(o: tuple, a: tuple, b: tuple) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _intersects(p: tuple, q: tuple, a: tuple, b: tuple) -> bool:
    """Closed segment intersection (as tests/test_layout_spacing.py)."""
    d1, d2, d3, d4 = _cross(a, b, p), _cross(a, b, q), _cross(p, q, a), _cross(p, q, b)
    if ((d1 > 0) != (d2 > 0) or d1 == 0 or d2 == 0) and ((d3 > 0) != (d4 > 0) or d3 == 0 or d4 == 0):
        return (min(p[0], q[0]) <= max(a[0], b[0]) and min(a[0], b[0]) <= max(p[0], q[0])
                and min(p[1], q[1]) <= max(a[1], b[1]) and min(a[1], b[1]) <= max(p[1], q[1]))
    return False


def gap(p: list, q: list) -> float:
    """Edge-to-edge distance of two convex polygons (0 if they overlap)."""
    if any(_inside(v, q) for v in p) or any(_inside(v, p) for v in q) or any(
            _intersects(a, b, c, d) for a, b in _edges(p) for c, d in _edges(q)):
        return 0.0
    return min(min(_seg_point_dist(v, a, b) for v in p for a, b in _edges(q)),
               min(_seg_point_dist(v, a, b) for v in q for a, b in _edges(p)))


def rel_deg(place: dict, pt: tuple[float, float]) -> float:
    """Bearing of pt from the place relative to the place yaw, in (-180, 180]."""
    a = math.degrees(math.atan2(pt[1] - place["y"], pt[0] - place["x"]) - place["yaw"])
    return (a + 180) % 360 - 180


def angular_interval(place: dict, poly: list) -> tuple[float, float]:
    rel = [rel_deg(place, c) for c in poly]
    rel = [rel[0] + ((r - rel[0] + 180) % 360 - 180) for r in rel]
    return min(rel), max(rel)


def _samples(poly: list, n: int = 10) -> list[tuple[float, float]]:
    return [(a[0] + t / n * (b[0] - a[0]), a[1] + t / n * (b[1] - a[1])) for a, b in _edges(poly) for t in range(n)]


WALLS = [((lo, line), (hi, line)) if axis == "y" else ((line, lo), (line, hi))
         for axis, line, lo, hi in generate.wall_segments(LAYOUT["rooms"])]


def _cams(place: dict) -> list[tuple[float, float]]:
    """Camera points over the CAM_DISC_M disc around a place (as tests/test_layout_spacing.py visible_points)."""
    return [(place["x"], place["y"])] + [(place["x"] + r * math.cos(a), place["y"] + r * math.sin(a))
                                         for r in (CAM_DISC_M / 2, CAM_DISC_M) for a in (i * math.pi / 8 for i in range(16))]


def _cross_np(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (b[..., 0] - o[..., 0])


def sight_lines(cams: list, pts: list) -> int:
    """Number of (camera, point) segments that cross no wall (`_intersects` over every pair, vectorised)."""
    c, q = np.array(cams, float), np.array(pts, float)
    p = np.repeat(c, len(q), axis=0)[:, None, :]
    q = np.tile(q, (len(c), 1))[:, None, :]
    w = np.array(WALLS, float)
    a, b = w[None, :, 0, :], w[None, :, 1, :]
    d1, d2, d3, d4 = _cross_np(a, b, p), _cross_np(a, b, q), _cross_np(p, q, a), _cross_np(p, q, b)
    cut = (((d1 > 0) != (d2 > 0)) | (d1 == 0) | (d2 == 0)) & (((d3 > 0) != (d4 > 0)) | (d3 == 0) | (d4 == 0))
    for i in (0, 1):
        cut &= (np.minimum(p[..., i], q[..., i]) <= np.maximum(a[..., i], b[..., i])) & (
            np.minimum(a[..., i], b[..., i]) <= np.maximum(p[..., i], q[..., i]))
    return int((~cut.any(axis=1)).sum())


def doorway_point(room: str) -> tuple[float, float]:
    axis, line, lo, hi = generate.doorway_of(LAYOUT["rooms"][room])
    return ((lo + hi) / 2, line) if axis == "y" else (line, (lo + hi) / 2)


@lru_cache(maxsize=None)
def fixed_objects() -> tuple[dict, ...]:
    """Every non-movable object of layout.yaml with its footprint; the cabinet's is the union of its closed and open
    states (one zone set serves both worlds)."""
    out = []
    for o in LAYOUT["objects"]:
        if o["id"] in MOVABLES:
            continue
        x, y, yaw = o["pose"]
        states = ("closed", "open") if o["class"] == "cabinet" else (o["state"],)
        boxes = [_part_box(o["class"], s) for s in states]
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        c, s = math.cos(yaw), math.sin(yaw)
        poly = [(x + c * u - s * v, y + s * u + c * v) for u, v in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        out.append({"id": o["id"], "class": o["class"], "room": o["room"], "poly": poly})
    return tuple(out)


CLASS_OF = {o["id"]: o["class"] for o in LAYOUT["objects"]}


def near_limit_m(cls: str) -> float:
    """Closest a movable's footprint may come to the place: its top edge stays below the cropped frame's top row
    seen from the camera at the place, plus the Nav2 arrival error NAV_XY_TOL_M."""
    cam = raycast_seg.CAM
    f = (cam["camera"]["width"] / 2) / math.tan(cam["camera"]["hfov_rad"] / 2)
    up = ((cam["camera"]["height"] - 1) / 2 - cam["self_crop"]["top_px"]) / f  # tan of the top row's elevation
    return (_part_box(cls, None)[4] - CAM_Z_M) / up + NAV_XY_TOL_M


def pose_for(cls: str, room: str, x: float, y: float) -> tuple[float, float, float]:
    """A chair faces the place (its seat front is local +x); the toolbox's long side (local x) is across the sight line."""
    p = PLACES[room]
    b = math.atan2(p["y"] - y, p["x"] - x)
    yaw = b if cls == "chair" else b + math.pi / 2
    return x, y, round(math.atan2(math.sin(yaw), math.cos(yaw)), 4)


# ---------------------------------------------------------------------------------------------------------- rules
def object_problems(cls: str, room: str, pose: tuple[float, float, float]) -> list[str]:
    """Single-object rules (empty = ok): inside the room; >= MIN_GAP_M from the room's fixed target-class objects (beds,
    cabinet: both cabinet states) and not touching other furniture; >= CLEAR_M from the place point and far enough that
    its top is in the cropped frame; wholly inside one photo wedge >= MIN_SEAM_DEG from the seams; no fixed furniture
    between it and the place; no wall-free sight line from any other place (hall included, CAM_DISC_M discs)."""
    poly = footprint(cls, None, *pose)
    place = PLACES[room]
    pp = (place["x"], place["y"])
    bad = []
    x0, y0, x1, y1 = LAYOUT["rooms"][room]["rect"]
    t = LAYOUT["wall"]["thickness"] / 2
    if not all(x0 + t <= x <= x1 - t and y0 + t <= y <= y1 - t for x, y in poly):
        bad.append("outside the room")
    for o in fixed_objects():
        if o["room"] == room and (g := gap(poly, o["poly"])) < (MIN_GAP_M if o["class"] in TARGET_CLASSES else 1e-9):
            bad.append(f"{g:.2f} m from {o['id']}")
    if min(_seg_point_dist(pp, a, b) for a, b in _edges(poly)) < max(CLEAR_M, near_limit_m(cls)) or _inside(pp, poly):
        bad.append("too near the place (robot footprint / top cut off)")
    if cls == "toolbox" and math.hypot(pose[0] - pp[0], pose[1] - pp[1]) < TOOLBOX_MIN_DIST_M:
        bad.append(f"toolbox nearer than {TOOLBOX_MIN_DIST_M} m to the place")
    rel = [rel_deg(place, c) for c in poly]
    if len({round(r / 90) % 4 for r in rel}) != 1 or min(45 - abs(r - 90 * round(r / 90)) for r in rel) < MIN_SEAM_DEG:
        bad.append("on a photo seam")
    furniture = [o for o in fixed_objects() if o["room"] == room]
    if any(_intersects(pp, q, a, b) for q in _samples(poly) for o in furniture for a, b in _edges(o["poly"])):
        bad.append("fixed furniture between it and the place")
    for name, other in PLACES.items():
        if name != room and sight_lines(_cams(other), _samples(poly)):
            bad.append(f"seen from the {name} place")
    return bad


def pair_problems(a: tuple[str, str, list], b: tuple[str, str, list]) -> list[str]:
    """(id, room, footprint) pairs in the same room: >= MIN_GAP_M floor gap, disjoint bearing intervals from the place
    (neither hides the other or the floor gap between them)."""
    if a[1] != b[1]:
        return []
    bad = []
    if (g := gap(a[2], b[2])) < MIN_GAP_M:
        bad.append(f"{a[0]}-{b[0]}: gap {g:.2f} m")
    (a0, a1), (b0, b1) = angular_interval(PLACES[a[1]], a[2]), angular_interval(PLACES[b[1]], b[2])
    if max(b0 - a1, a0 - b1) <= 0:
        bad.append(f"{a[0]}-{b[0]}: overlap seen from the place")
    return bad


def _dist_to_poly(px: np.ndarray, py: np.ndarray, poly: list) -> np.ndarray:
    """Distance of grid points to a convex polygon (0 inside)."""
    d = np.full(px.shape, np.inf)
    inside = np.ones(px.shape, bool)
    for (ax, ay), (bx, by) in _edges(poly):
        dx, dy = bx - ax, by - ay
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy), 0, 1)
        d = np.minimum(d, np.hypot(px - ax - t * dx, py - ay - t * dy))
        inside &= dx * (py - ay) - dy * (px - ax) > 0
    return np.where(inside, 0.0, d)


@lru_cache(maxsize=None)
def _static_free(room: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(grid x, grid y, free) over the room: free = robot centre >= ROBOT_RADIUS_M from walls and fixed furniture
    (the static world is Nav2-proven: Phase 1 G1)."""
    x0, y0, x1, y1 = LAYOUT["rooms"][room]["rect"]
    t = LAYOUT["wall"]["thickness"] / 2
    gx, gy = np.meshgrid(np.arange(x0 + GRID_M / 2, x1, GRID_M), np.arange(y0 + GRID_M / 2, y1, GRID_M))
    free = np.minimum.reduce([gx - x0 - t, x1 - t - gx, gy - y0 - t, y1 - t - gy]) >= ROBOT_RADIUS_M
    door = doorway_point(room)
    free |= np.hypot(gx - door[0], gy - door[1]) <= LAYOUT["rooms"][room]["doorway"]["width"] / 2 - ROBOT_RADIUS_M
    for o in fixed_objects():
        if o["room"] == room:
            free &= _dist_to_poly(gx, gy, o["poly"]) >= ROBOT_RADIUS_M
    return gx, gy, free


def reachable(room: str, polys: list[list]) -> bool:
    """The place is reachable from the doorway on a path that keeps >= INFLATION_M from every movable (no movable puts
    inflated cost on it) and >= ROBOT_RADIUS_M from walls and fixed furniture (4-connected flood fill on GRID_M)."""
    gx, gy, free = _static_free(room)
    free = free.copy()
    for poly in polys:
        free &= _dist_to_poly(gx, gy, poly) >= INFLATION_M
    door, p = doorway_point(room), PLACES[room]
    reach = free & (np.hypot(gx - door[0], gy - door[1]) <= GRID_M)
    goal = np.unravel_index(np.argmin(np.hypot(gx - p["x"], gy - p["y"])), gx.shape)
    while not reach[goal]:
        grown = reach.copy()
        grown[1:] |= reach[:-1]
        grown[:-1] |= reach[1:]
        grown[:, 1:] |= reach[:, :-1]
        grown[:, :-1] |= reach[:, 1:]
        grown &= free
        if (grown == reach).all():
            return False
        reach = grown
    return True


def alone_problems(cls: str, room: str, pose: tuple[float, float, float]) -> list[str]:
    """Rules a movable must pass with no other movable in the world: object_problems, pair rules against the room's
    fixed target-class objects (beds, cabinet), and the place still reachable."""
    poly = footprint(cls, None, *pose)
    bad = object_problems(cls, room, pose)
    for o in fixed_objects():
        if o["class"] in TARGET_CLASSES:
            bad += pair_problems((cls, room, poly), (o["id"], o["room"], o["poly"]))
    if not reachable(room, [poly]):
        bad.append(f"{room}: place unreachable from the doorway")
    if cls == "toolbox" and not bad:  # ray-cast only once the 2D rules hold (cost)
        bad += toolbox_clutter({TOOLBOX: pose}, room)
    return bad


def toolbox_clutter(poses: dict[str, tuple[float, float, float]], room: str) -> list[str]:
    """Deviation 17 rule 2, both worlds: in the ray-cast photo at the room's place that holds the most toolbox pixels,
    the toolbox's pixel bounding box grown by SILHOUETTE_PAD_PX on each side holds no pixel of any other object
    (fixed furniture or another movable in `poses`; walls, floor, ceiling, trim are fine). Empty = ok."""
    bad = []
    for world in PLACEMENT_WORLDS:
        labels = _labels(world)
        name = {v: k for k, v in labels.items()}
        imgs = photos(poses, world, room)
        k = max(range(4), key=lambda i: int((imgs[i] == labels[TOOLBOX]).sum()))
        rows, cols = np.nonzero(imgs[k] == labels[TOOLBOX])
        if not len(rows):
            bad.append(f"{world}: toolbox not in any photo at the {room} place")
            continue
        pad = SILHOUETTE_PAD_PX
        win = imgs[k][max(0, rows.min() - pad):rows.max() + pad + 1, max(0, cols.min() - pad):cols.max() + pad + 1]
        others = sorted(name[int(v)] for v in np.unique(win) if 0 < v < 200 and v != labels[TOOLBOX])
        if others:
            bad.append(f"{world}: {', '.join(others)} inside the toolbox's padded box in photo k={k}")
    return bad


def problems(rooms: dict[str, str], poses: dict[str, tuple[float, float, float]]) -> list[str]:
    """Every rule over the placed movables (the keys of `poses`; 008: all of them, Phase 8 realize.py: a subset)
    (empty = ok): alone_problems per movable, pair rules between movables, the place reachable with all of the room's
    movables in it, the toolbox's padded box (if the toolbox is placed)."""
    placed = [oid for oid in MOVABLES if oid in poses]
    bad = [f"{oid}: {p}" for oid in placed for p in alone_problems(CLASS_OF[oid], rooms[oid], poses[oid])]
    items = [(oid, rooms[oid], footprint(CLASS_OF[oid], None, *poses[oid])) for oid in placed]
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            bad += pair_problems(a, b)
    for room in ROOMS:
        if not reachable(room, [poly for oid, r, poly in items if r == room]):
            bad.append(f"{room}: place unreachable from the doorway")
    if not bad and TOOLBOX in poses:  # the other movables in the toolbox's padded box (alone_problems: the furniture)
        bad += [f"{TOOLBOX}: {b}" for b in toolbox_clutter(poses, rooms[TOOLBOX])]
    return bad


# ----------------------------------------------------------------------------------------------------------- draw
def draw(seed: int) -> tuple[dict[str, str], dict[str, tuple[float, float, float]]]:
    """(room per movable, pose per movable) as drawn from the seed, before any rule check."""
    rng = random.Random(seed)
    rooms = {oid: rng.choice([r for r in ROOMS if ZONES[r][CLASS_OF[oid]]]) for oid in MOVABLES}
    poses = {}
    for oid in MOVABLES:
        cls = CLASS_OF[oid]
        x0, y0, x1, y1 = rng.choice(ZONES[rooms[oid]][cls])
        poses[oid] = pose_for(cls, rooms[oid], round(rng.uniform(x0, x1), 3), round(rng.uniform(y0, y1), 3))
    return rooms, poses


def placements(seed: int) -> dict[str, tuple[float, float, float]]:
    """{object_id: (x, y, yaw)} for every movable; SeedRejected if the draw breaks any rule."""
    rooms, poses = draw(seed)
    bad = problems(rooms, poses)
    if bad:
        raise SeedRejected(seed, bad)
    return poses


def room_of(x: float, y: float) -> str:
    for name in ROOMS:
        x0, y0, x1, y1 = LAYOUT["rooms"][name]["rect"]
        if x0 < x < x1 and y0 < y < y1:
            return name
    raise ValueError(f"({x}, {y}) is in no room")


def truth_for(seed: int, world: str = "closed") -> dict:
    """The world's truth.json with the movables' pose and room replaced; layout_sha256 = sha256(world sha + ":" + seed).
    A blocked_<room> world keeps its "blocker" entry (its truth.json = closed's objects + "blocker")."""
    truth = truth_with(placements(seed), world)
    truth["layout_sha256"] = hashlib.sha256(f"{truth['layout_sha256']}:{seed}".encode()).hexdigest()
    return truth


def truth_with(poses: dict[str, tuple[float, float, float]], world: str) -> dict:
    """The world's truth.json with these objects' pose and room replaced (layout_sha256 untouched)."""
    truth = json.loads(world_truth_path(world).read_text())
    for o in truth["objects"]:
        if o["id"] in poses:
            x, y, yaw = poses[o["id"]]
            o["pose"] = [round(x, 4), round(y, 4), 0.0, 0.0, 0.0, round(yaw, 4)]
            o["room"] = room_of(x, y)
    return truth


# ---------------------------------------------------------------------------------------------------- ray-caster
def camera(place: str, k: int) -> tuple[tuple[float, float, float], float]:
    """(camera xyz, heading) of look_around photo k at a place: base on the place point, camera CAM_BACK_M behind it."""
    p = PLACES[place]
    h = p["yaw"] + k * math.pi / 2
    return (p["x"] - CAM_BACK_M * math.cos(h), p["y"] - CAM_BACK_M * math.sin(h), CAM_Z_M), h


@lru_cache(maxsize=None)
def _labels(world: str) -> dict[str, int]:
    return {o["id"]: o["label"] for o in json.loads(world_truth_path(world).read_text())["objects"]}


@lru_cache(maxsize=None)
def _sdf_prims(world: str) -> list[tuple]:
    return raycast_seg.load_prims(WORLD_DIRS[world] / "clinic.sdf")


def _prims(world: str, poses: tuple) -> list[tuple]:
    """load_prims(sdf, poses) without re-parsing the SDF: the parsed prims with the movables' (x, y, yaw) swapped in
    (tests/test_seeded_layouts.py compares against a full render of load_prims with the overrides)."""
    by_label = {_labels(world)[m]: pose for m, pose in poses}
    return [(lab, by_label[lab][2], (by_label[lab][0], by_label[lab][1], xyz[2]), *rest) if lab in by_label
            else (lab, yaw, xyz, *rest) for lab, yaw, xyz, *rest in _sdf_prims(world)]


@lru_cache(maxsize=None)
def base_render(world: str, place: str, k: int):  # -> (labels, t) of the world without the movables
    movable = {_labels(world)[m] for m in MOVABLES}
    prims = [p for p in raycast_seg.load_prims(WORLD_DIRS[world] / "clinic.sdf") if p[0] not in movable]
    return raycast_seg.render_lt(prims, *camera(place, k))


def photos(poses: dict[str, tuple[float, float, float]], world: str, place: str) -> list:
    """The 4 label images (cropped frame) at a place with the movables in `poses` (the others left out): the cached
    movable-free render
    with the movables ray-cast on top (nearer hit wins), equal to a full render of the overridden SDF."""
    shown = {_labels(world)[m] for m in poses}  # movables not in `poses` are left out
    prims = [p for p in _prims(world, tuple(sorted(poses.items()))) if p[0] in shown]
    out = []
    for k in range(4):
        lab, t = base_render(world, place, k)
        mlab, mt = raycast_seg.render_lt(prims, *camera(place, k))
        near = mt < t
        img = lab.copy()
        img[near] = mlab[near]
        out.append(img)
    return out


def count_photos(seed: int, room: str, cls: str, world: str = "closed") -> dict[str, int]:
    """{object_id: photo index k (0..3, photo number k + 1)} for every `cls` instance in `room` under the seed: the
    look_around photo at the room's place where the instance has the most pixels."""
    truth = truth_for(seed, world)
    imgs = photos(placements(seed), world, room)
    return {o["id"]: max(range(4), key=lambda k: int((imgs[k] == o["label"]).sum()))
            for o in truth["objects"] if o["class"] == cls and o["room"] == room}


# ---------------------------------------------------------------------------------------------- zone derivation
def _cell_ok(args: tuple[str, str, float, float]) -> bool:
    cls, room, x, y = args
    return not alone_problems(cls, room, pose_for(cls, room, x, y))


def _largest_rect(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(r0, r1, c0, c1) inclusive of the largest all-True axis-aligned rectangle (histogram method), None if none."""
    best, out = 0, None
    h = np.zeros(mask.shape[1], int)
    for r in range(mask.shape[0]):
        h = np.where(mask[r], h + 1, 0)
        for c0 in range(mask.shape[1]):
            m = h[c0]
            for c1 in range(c0, mask.shape[1]):
                m = min(m, h[c1])
                if m == 0:
                    break
                if m * (c1 - c0 + 1) > best:
                    best, out = m * (c1 - c0 + 1), (r - m + 1, r, c0, c1)
    return out


def derive_zones() -> dict[str, dict[str, list[list[float]]]]:
    """Per room and class: per photo wedge k the largest rectangle of GRID_M centres that pass alone_problems."""
    from multiprocessing import Pool
    zones: dict[str, dict[str, list[list[float]]]] = {}
    for room in ROOMS:
        x0, y0, x1, y1 = LAYOUT["rooms"][room]["rect"]
        xs = np.round(np.arange(x0 + GRID_M, x1 - GRID_M / 2, GRID_M), 3)
        ys = np.round(np.arange(y0 + GRID_M, y1 - GRID_M / 2, GRID_M), 3)
        wedge = np.array([[round(rel_deg(PLACES[room], (x, y)) / 90) % 4 for x in xs] for y in ys])
        for cls in ("toolbox", "chair"):
            with Pool() as pool:
                ok = np.array(pool.map(_cell_ok, [(cls, room, float(x), float(y)) for y in ys for x in xs]))
            ok = ok.reshape(len(ys), len(xs))
            rects = []
            for k in range(4):
                r = _largest_rect(ok & (wedge == k))
                if r is not None:
                    rects.append([float(xs[r[2]]), float(ys[r[0]]), float(xs[r[3]]), float(ys[r[1]])])
            zones.setdefault(room, {})[cls] = rects
    return zones


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--derive-zones", action="store_true", help="print the ZONES literal derived from the geometry")
    if ap.parse_args().derive_zones:
        print(json.dumps(derive_zones()))
