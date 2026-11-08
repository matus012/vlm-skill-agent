"""Phase 5 seeded layouts (task 008, agent/INTERFACES.md section 10): sim/worlds/clinic/seeded.py, probes/vet_seeds.py
and the ray-caster's per-model pose overrides. Run: uv run pytest tests/test_seeded_layouts.py"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
sys.path.insert(0, str(ROOT / "sim" / "worlds" / "clinic"))
import raycast_seg as R  # noqa: E402
import seeded as S  # noqa: E402
import vet_seeds as V  # noqa: E402

SEEDS = json.loads((ROOT / "tasks" / "phase5_seeds.json").read_text())
OK = SEEDS["ok"]
LAYOUT_POSES = {o["id"]: (o["pose"][0], o["pose"][1], o["pose"][5])
                for o in json.loads(S.world_truth_path("closed").read_text())["objects"] if o["id"] in S.MOVABLES}


def _try(seed: int) -> dict | None:
    try:
        return S.placements(seed)
    except S.SeedRejected:
        return None


# ------------------------------------------------------------------------------------------------- determinism
def test_placements_deterministic_in_and_across_processes() -> None:
    here = {s: _try(s) for s in range(25)}
    assert here == {s: _try(s) for s in range(25)}
    assert any(here.values()) and not all(here.values())  # both outcomes occur in the range
    code = ("import json, sys; sys.path.insert(0, 'sim/worlds/clinic'); import seeded as S\n"
            "out = {}\nfor s in range(25):\n    try: out[s] = S.placements(s)\n    except S.SeedRejected: out[s] = None\n"
            "print(json.dumps(out))")
    other = json.loads(subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                                      check=True).stdout)
    assert {int(k): (v and {o: tuple(p) for o, p in v.items()}) for k, v in other.items()} == here


def test_placements_cover_every_movable_and_draw_is_seed_local() -> None:
    seed = OK[0]["seed"]
    a = S.placements(seed)
    import random
    random.seed(12345)  # the global RNG must not matter
    assert S.placements(seed) == a and set(a) == set(S.MOVABLES)


# ------------------------------------------------------------------------------------- the vetted list (json)
def test_json_has_enough_ok_seeds_and_a_consistent_summary() -> None:
    sm = SEEDS["summary"]
    assert sm["ok"] == len(OK) >= 200 and sm["rejected"] == len(SEEDS["rejected"])
    assert sm["vetted"] == SEEDS["seed_range"][1] - SEEDS["seed_range"][0]
    assert sorted(e["seed"] for e in OK + SEEDS["rejected"]) == list(range(*SEEDS["seed_range"]))
    assert SEEDS["zones"] == S.ZONES  # the list was vetted with the zones in the code


def test_every_ok_seed_holds_every_rule() -> None:
    for e in OK:
        poses = S.placements(e["seed"])  # raises SeedRejected if any rule breaks
        assert {o: list(p) for o, p in poses.items()} == e["poses"], e["seed"]
        rooms = {o: S.room_of(*p[:2]) for o, p in poses.items()}
        assert rooms == e["rooms"]  # placements ran problems() (rules + toolbox silhouette) without a breach
        assert e["chairs_per_room"] == {r: sum(rooms[c] == r for c in S.CHAIRS) for r in S.ROOMS}
        assert sum(e["chairs_per_room"].values()) == 4 and e["toolbox_room"] == rooms[S.TOOLBOX]
        assert {r: sorted(m) for r, m in e["chair_photos"].items()} == {
            r: sorted(c for c in S.CHAIRS if rooms[c] == r) for r in S.ROOMS if e["chairs_per_room"][r]}
        assert e["spans_2_photos"] == {r: len(set(m.values())) >= 2 for r, m in e["chair_photos"].items()}


def test_rejected_seeds_raise_with_the_listed_reasons() -> None:
    for e in SEEDS["rejected"]:
        if e["reasons"][0].startswith("rule: "):
            with pytest.raises(S.SeedRejected) as ex:
                S.placements(e["seed"])
            assert ["rule: " + r for r in ex.value.reasons] == e["reasons"]
        else:  # rules pass, a ray-cast gate failed: never used either
            assert _try(e["seed"]) is not None


def test_distribution_needs_of_task_008() -> None:
    """Toolbox in every room, rooms with 0 chairs, varying counts, and chairs of one room over >= 2 photos."""
    sm = SEEDS["summary"]
    assert set(sm["toolbox_room"]) == {r for r in S.ROOMS if S.ZONES[r]["toolbox"]}
    for r in S.ROOMS:
        assert "0" in sm["chairs_per_room_histogram"][r] and len(sm["chairs_per_room_histogram"][r]) >= 3
    assert sm["seeds_with_a_room_spanning_2_photos"] >= len(OK) // 2


# ------------------------------------------------------------------------------------------------ rejection
def test_planted_bad_draw_raises_seed_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    rooms, poses = S.draw(OK[0]["seed"])
    poses = dict(poses)
    poses["chair_orange_2"] = poses["chair_orange_1"]  # two chairs on one spot
    rooms["chair_orange_2"] = rooms["chair_orange_1"]
    monkeypatch.setattr(S, "draw", lambda seed: (rooms, poses))
    with pytest.raises(S.SeedRejected) as ex:
        S.placements(OK[0]["seed"])
    assert any("chair_orange_1-chair_orange_2: gap 0.00 m" in r for r in ex.value.reasons)


@pytest.mark.parametrize("room,cls", [(r, c) for r in S.ROOMS for c in ("toolbox", "chair")])
def test_zones_pass_every_single_object_rule(room: str, cls: str) -> None:
    """Every GRID_M grid point (the derivation's cells, edges included) plus 10 random points of each zone."""
    rng = np.random.default_rng(0)
    for x0, y0, x1, y1 in S.ZONES[room][cls]:
        pts = [(x, y) for x in np.linspace(x0, x1, round((x1 - x0) / S.GRID_M) + 1)
               for y in np.linspace(y0, y1, round((y1 - y0) / S.GRID_M) + 1)]
        pts += list(zip(rng.uniform(x0, x1, 10), rng.uniform(y0, y1, 10)))
        for x, y in pts:
            assert S.alone_problems(cls, room, S.pose_for(cls, room, float(x), float(y))) == [], (room, cls, x, y)


def test_the_frozen_layout_breaks_the_reachability_rule_nowhere() -> None:
    for room in S.ROOMS:
        assert S.reachable(room, [])


# ------------------------------------------------------------------------------------------------ truth_for
@pytest.mark.parametrize("world", ["closed", "cabinet_open"])
def test_truth_for_changes_only_the_movables_and_the_sha(world: str) -> None:
    seed = OK[0]["seed"]
    base = json.loads(S.world_truth_path(world).read_text())
    t = S.truth_for(seed, world)
    assert t["layout_sha256"] == hashlib.sha256(f"{base['layout_sha256']}:{seed}".encode()).hexdigest()
    assert {k: v for k, v in t.items() if k not in ("objects", "layout_sha256")} == {
        k: v for k, v in base.items() if k not in ("objects", "layout_sha256")}
    poses = S.placements(seed)
    assert [o["id"] for o in t["objects"]] == [o["id"] for o in base["objects"]]
    for o, b in zip(t["objects"], base["objects"]):
        if o["id"] not in S.MOVABLES:
            assert o == b
            continue
        assert {k: v for k, v in o.items() if k not in ("pose", "room")} == {
            k: v for k, v in b.items() if k not in ("pose", "room")}
        x, y, yaw = poses[o["id"]]
        assert o["pose"] == [round(x, 4), round(y, 4), 0.0, 0.0, 0.0, round(yaw, 4)]
        assert o["room"] == OK[0]["rooms"][o["id"]]
    assert S.truth_for(seed, world) == t and S.truth_for(OK[1]["seed"], world)["layout_sha256"] != t["layout_sha256"]


# ------------------------------------------------------------------------------------- ray-caster overrides
def test_override_with_the_layout_poses_is_the_unmoved_world() -> None:
    sdf = S.WORLD_DIRS["closed"] / "clinic.sdf"
    assert R.load_prims(sdf, LAYOUT_POSES) == R.load_prims(sdf)
    moved = R.load_prims(sdf, {"toolbox_red_1": (-1.0, 2.5, 0.3)})
    assert moved != R.load_prims(sdf) and len(moved) == len(R.load_prims(sdf))
    with pytest.raises(KeyError):
        R.load_prims(sdf, {"no_such_model": (0.0, 0.0, 0.0)})


@pytest.mark.parametrize("place", ["reception", "storage"])
def test_composited_photos_equal_a_full_render(place: str) -> None:
    """seeded.photos (movable-free render + movables on top) == a full ray-cast of the overridden SDF, pixel for
    pixel: with the layout poses (the frozen world's photos) and with a seed's poses."""
    sdf = S.WORLD_DIRS["closed"] / "clinic.sdf"
    for poses in (LAYOUT_POSES, S.placements(OK[0]["seed"])):
        full = R.load_prims(sdf, poses)
        for k, img in enumerate(S.photos(poses, "closed", place)):
            assert np.array_equal(img, R.render(full, *S.camera(place, k))), (place, k)


@pytest.mark.parametrize("place,k", [("storage", 2), ("hall", 1), ("ward", 0)])
def test_window_culling_renders_exactly_the_unculled_image(place: str, k: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """render_lt tests each primitive only on its projected window; pixel-equal to testing every ray on every
    primitive (the pre-008 renderer)."""
    prims = R.load_prims()
    win = R.render_lt(prims, *S.camera(place, k), stride=2)
    monkeypatch.setattr(R, "_window", lambda prim, o_w, yaw, stride, shape: (0, shape[0], 0, shape[1]))
    ref = R.render_lt(prims, *S.camera(place, k), stride=2)
    assert np.array_equal(win[0], ref[0]) and np.array_equal(win[1], ref[1]) and (win[0] > 0).all()


# ------------------------------------------------------------------------------------------------- vetting
def test_vetting_rejects_a_chair_on_a_photo_seam() -> None:
    poses = S.placements(OK[0]["seed"])
    p = S.PLACES["storage"]
    poses["chair_orange_1"] = S.pose_for("chair", "storage", p["x"] + 2 * math.cos(math.pi / 4),
                                         p["y"] + 2 * math.sin(math.pi / 4))  # bearing 45 deg: the 0/90 seam
    bad, _ = V.vet_poses(poses)
    assert any(r.startswith("rule: chair_orange_1: on a photo seam") for r in bad)
    assert any(r.startswith("closed G3 chair_orange_1: in 2 photos") for r in bad)  # the ray-caster sees it too


def test_vetting_rejects_a_chair_seen_through_a_doorway() -> None:
    poses = S.placements(OK[0]["seed"])
    poses["chair_orange_1"] = S.pose_for("chair", "reception", -4.6, -2.0)  # in the hall place's view (W2 search)
    bad, _ = V.vet_poses(poses)
    assert any(r.startswith("rule: chair_orange_1: seen from the hall place") for r in bad)
    assert any(r.startswith("closed G4 chair_orange_1 (reception)") and r.endswith("at the hall place") for r in bad)


def test_vetting_passes_an_ok_seed_and_count_photos_matches_the_json() -> None:
    e = next(e for e in OK if e["any_room_spans_2_photos"])
    bad, chair_photos = V.vet_poses(S.placements(e["seed"]))
    assert bad == [] and chair_photos == e["chair_photos"]
    for room, m in e["chair_photos"].items():
        assert S.count_photos(e["seed"], room, "chair") == m
    assert S.count_photos(e["seed"], "ward", "bed") == {"bed_blue_1": 1, "bed_blue_2": 1, "bed_blue_3": 1}


# ------------------------------------------------------------------------- a recorded deviation: toolbox naming rules
def test_planted_toolbox_in_front_of_the_office_desk_is_rejected() -> None:
    """2.5, 5.2: between the office place and the desk (the live G9 'red cabinet on the table' photos)."""
    pose = S.pose_for("toolbox", "office", 2.5, 5.2)
    clutter = S.toolbox_clutter({S.TOOLBOX: pose}, "office")
    assert len(clutter) == 2 and all("desk_wood_1" in c for c in clutter)  # both worlds
    poses = S.placements(OK[0]["seed"])
    poses[S.TOOLBOX] = pose
    bad, _ = V.vet_poses(poses)
    assert any(r.startswith("rule: toolbox_red_1: ") for r in bad)


def test_planted_toolbox_against_furniture_at_a_legal_distance_is_rejected_by_the_silhouette_rule() -> None:
    """4.4, 3.0 in the office: >= 1.9 m and every 2D rule holds; the bookshelf is inside the padded image box."""
    pose = S.pose_for("toolbox", "office", 4.4, 3.0)
    assert math.hypot(4.4 - S.PLACES["office"]["x"], 3.0 - S.PLACES["office"]["y"]) >= S.TOOLBOX_MIN_DIST_M
    assert S.object_problems("toolbox", "office", pose) == []
    bad = S.alone_problems("toolbox", "office", pose)
    assert bad and all("bookshelf_wood_1 inside the toolbox's padded box" in b for b in bad)


def test_planted_toolbox_at_1_5_m_is_rejected() -> None:
    p = S.PLACES["office"]
    pose = S.pose_for("toolbox", "office", p["x"] + 1.5, p["y"])
    assert "toolbox nearer than 1.9 m to the place" in S.object_problems("toolbox", "office", pose)


def test_other_movables_count_as_clutter_too() -> None:
    e = next(e for e in OK if e["chairs_per_room"][e["toolbox_room"]] == 0)
    poses = S.placements(e["seed"])
    assert S.toolbox_clutter(poses, e["toolbox_room"]) == []
    tx, ty, _ = poses[S.TOOLBOX]
    p = S.PLACES[e["toolbox_room"]]
    d = math.hypot(tx - p["x"], ty - p["y"])
    ux, uy = -(ty - p["y"]) / d, (tx - p["x"]) / d  # across the sight line
    hits = []
    for side in (1, -1):  # a chair 0.75 m beside the toolbox (edges ~3 cm apart: inside the 20 px pad at ~2 m)
        poses["chair_orange_1"] = S.pose_for("chair", e["toolbox_room"], tx + side * 0.75 * ux, ty + side * 0.75 * uy)
        hits += S.toolbox_clutter(poses, e["toolbox_room"])
    assert any("chair_orange_1 inside the toolbox's padded box" in c for c in hits)


def test_every_ok_seed_keeps_the_toolbox_out_of_the_ward_and_far_from_the_place() -> None:
    assert S.ZONES["ward"]["toolbox"] == [] and SEEDS["summary"]["toolbox_room"].get("ward") is None
    for e in OK:
        x, y, _ = e["poses"][S.TOOLBOX]
        p = S.PLACES[e["toolbox_room"]]
        assert math.hypot(x - p["x"], y - p["y"]) >= S.TOOLBOX_MIN_DIST_M
