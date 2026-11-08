"""Blocked-doorway option for the clinic world (task 008 side lane, Phase 6 prep). Off-sim: pure layout arithmetic.

`blocker(layout, room)` returns the spec of one extra static model: a yellow box that fills the room's doorway, so the
room is unreachable for the robot (a plain box would be driven around; this one spans the whole gap and is embedded
0.02 m into each jamb). generate.py emits it with `--blocked <room>`; without that option its output is unchanged.

Design: width = doorway width + 0.04 (jambs), depth 0.30 m from the room-side wall face (0.05 m past the wall line) out
into the hall (so it never enters a room and stays clear of the Phase 5 movable zones, which reach the walls; checked in
tests/test_blocked_doorway.py), height 1.2 m (cuts the
lidar plane at 0.19 m, stays under the 2.1 m lintel; a barrier, not a wall, so it reads as an obstacle in a photo).
Class `blocker`, colour yellow, gz label 220 (unused: objects 1..199, structure 240..244), id `blocker_<room>`.
It is not in truth.json["objects"] (scorers and readers see the same objects as before); truth.json gets a top-level
"blocker" entry instead. map_knowledge.json is unchanged: the map does not know the box.

Phase 6 use: one world variant per blocked room (`generate.py <layout> --blocked storage --out <dir>`), composed with a
Phase 5 seed as usual (the box lies outside every seeded.ZONES footprint). The static map (SLAM / raster) stays the
unblocked one: Nav2's obstacle layer sees the box live, so navigate_to(<blocked room>) fails with `failed: blocked`
once the planner finds no path, and the task's answer lies in another room.
"""
from __future__ import annotations

LABEL = 220
CLASS = "blocker"
COLOUR = "yellow"
JAMB_M, DEPTH_M, HEIGHT_M, WALL_HALF_M = 0.04, 0.30, 1.2, 0.05  # wall thickness 0.1 (layout.yaml)


def blocker(layout: dict, room: str) -> dict:
    """{id, class, colour, room, label, pose [x, y, 0, 0, 0, 0], size [sx, sy, sz]} of the box in `room`'s doorway."""
    r = layout["rooms"][room]
    d = r.get("doorway")
    if d is None:
        raise ValueError(f"room {room} has no doorway to block")
    x0, y0, x1, y1 = r["rect"]
    w = d["width"] + JAMB_M
    shift = DEPTH_M / 2 - WALL_HALF_M  # box centre vs the wall line, towards the hall (away from the room)
    if d["wall"] in ("south", "north"):
        s_ = -1.0 if d["wall"] == "south" else 1.0
        pose, size = [d["center"], (y0 if d["wall"] == "south" else y1) + s_ * shift], [w, DEPTH_M, HEIGHT_M]
    else:
        s_ = -1.0 if d["wall"] == "west" else 1.0
        pose, size = [(x0 if d["wall"] == "west" else x1) + s_ * shift, d["center"]], [DEPTH_M, w, HEIGHT_M]
    return {"id": f"blocker_{room}", "class": CLASS, "colour": COLOUR, "room": room, "label": LABEL,
            "pose": [*pose, 0.0, 0.0, 0.0, 0.0], "size": size}


def parts(b: dict) -> list[tuple]:
    """generate.py part list (local frame, floor at z = 0)."""
    return [("box", tuple(b["size"]), (0.0, 0.0, b["size"][2] / 2), b["colour"])]
