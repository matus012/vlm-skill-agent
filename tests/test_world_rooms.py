"""agent/world.py: room polygons from map_knowledge.json, point -> room (task 003 adapter where(); 005 N8)."""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.world import load_rooms, room_of  # noqa: E402

MAP = ROOT / "sim/worlds/clinic/generated/map_knowledge.json"
PLACES = ROOT / "sim/worlds/clinic/places.yaml"


@pytest.fixture(scope="module")
def rooms() -> dict:
    return load_rooms(MAP)


def test_all_rooms_loaded(rooms: dict) -> None:
    assert set(rooms) == {"hall", "ward", "office", "reception", "storage"}
    assert all(len(p) >= 3 for p in rooms.values())


def test_every_place_maps_to_its_own_room(rooms: dict) -> None:
    places = yaml.safe_load(PLACES.read_text())["places"]
    assert set(places) == set(rooms)
    for name, p in places.items():
        assert room_of(p["x"], p["y"], rooms) == name, name


@pytest.mark.parametrize("xy,room", [((0.0, 0.0), "hall"), ((-4.9, 6.4), "ward"), ((4.9, 6.4), "office"),
                                     ((-0.1, -6.4), "reception"), ((0.1, -6.4), "storage"),
                                     ((1.2, 1.6), "office"), ((-1.2, 1.4), "hall")])
def test_interior_points(rooms: dict, xy: tuple, room: str) -> None:
    assert room_of(*xy, rooms) == room


@pytest.mark.parametrize("xy,room", [((-5.3, 4.0), "ward"), ((6.0, 0.2), "hall"), ((2.0, -7.5), "storage"),
                                     ((-9.0, -9.0), "reception"), ((5.5, 7.0), "office")])
def test_outside_points_go_to_nearest_room(rooms: dict, xy: tuple, room: str) -> None:
    assert room_of(*xy, rooms) == room


def test_shared_edge_belongs_to_exactly_one_room(rooms: dict) -> None:
    # a point on the hall/office wall line: assigned, and the answer is one of the two rooms sharing the edge
    assert room_of(2.5, 1.5, rooms) in {"hall", "office"}
