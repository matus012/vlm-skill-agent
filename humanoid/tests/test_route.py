"""G1 adapter router (geometry.route): legs go through doorway centres and never cut a wall."""
from __future__ import annotations

import sys
from itertools import permutations

import pytest
import yaml

from agent.world import load_doors, load_rooms
from humanoid.geometry import route
from humanoid.room import CLINIC_LAYOUT

CLINIC = CLINIC_LAYOUT.parent
sys.path.insert(0, str(CLINIC))
import generate  # noqa: E402

ROOMS, DOORS = load_rooms(), load_doors()  # map_knowledge.json, as the adapter
PLACES = {n: (p["x"], p["y"]) for n, p in yaml.safe_load((CLINIC / "places.yaml").read_text())["places"].items()}
POINTS = {**PLACES, "spawn": (0.0, 0.0)}
CLEAR_M = 0.3  # G1 half-width (~0.25 m shoulders) + margin, from the wall face


def _hits_wall(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Does segment a-b pass within CLEAR_M of any wall segment (generate.wall_segments, doorways cut out)?"""
    rooms = yaml.safe_load(CLINIC_LAYOUT.read_text())["rooms"]
    half_t = yaml.safe_load(CLINIC_LAYOUT.read_text())["wall"]["thickness"] / 2
    for axis, line, lo, hi in generate.wall_segments(rooms):
        k = 1 if axis == "y" else 0  # coordinate fixed along the wall line
        for s in range(101):  # sample the leg every 1 %
            p = (a[0] + (b[0] - a[0]) * s / 100, a[1] + (b[1] - a[1]) * s / 100)
            if abs(p[k] - line) < half_t + CLEAR_M and lo - CLEAR_M < p[1 - k] < hi + CLEAR_M:
                return True
    return False


def test_hall_to_storage_goes_through_its_door() -> None:
    assert route(0.0, 0.0, 2.5, -3.9, ROOMS, DOORS) == [(3.6, -0.9), (3.6, -2.1), (2.5, -3.9)]
    assert route(0.0, 0.0, 2.75, -0.1, ROOMS, DOORS) == [(2.75, -0.1)]


@pytest.mark.parametrize("a,b", list(permutations(POINTS, 2)))
def test_no_leg_cuts_a_wall(a: str, b: str) -> None:
    pts = [POINTS[a], *route(*POINTS[a], *POINTS[b], ROOMS, DOORS)]
    assert pts[-1] == POINTS[b]
    for p, q in zip(pts, pts[1:]):
        assert not _hits_wall(p, q), (a, b, p, q)
