"""Phase 8 layout specs (agent/INTERFACES.md section 12, task 011). Harness / truth side: never imported by agent/ or
the robot adapters (agent/truth.py imports it lazily for the scorer, like seeded.py).

    realize(spec) -> {"world", "poses", "parked", "notes"}   where every spec object goes (deterministic)
    truth_for_spec(spec) -> dict                              the world's truth.json, poses replaced, parked removed
    spec_sha256(spec) -> str                                  sha256 of the canonical spec JSON
    park_pose(oid) -> (x, y, yaw)                             the fixed pose of a parked object, outside the building

Spec (gen_phase8.render_pass layout): {"seed": int, "objects": {id: {"room": r[, "state": s]}}, "removed": [ids]}; other
keys (index, tasks) are ignored. Objects: toolbox_red_1, cabinet_blue_1, bed_blue_1..3, chair_orange_1..4.
  world   = cabinet_open if the cabinet's state is open, else closed;
  toolbox -> its spec room on 008's zones and rules (seeded.problems), else parked (the ward has no zone);
  chairs  -> the whole kept row in its spec room on 008's zones and rules, toolbox placed first and kept fixed; no fit
             in DRAWS draws -> the whole row is parked;
  beds / cabinet stay at their frozen poses if their spec room is the frozen one (ward / storage), else parked;
  removed beds / chairs are parked.
The 008 rules are checked with the frozen beds and cabinet in place even when they are parked (seeded.fixed_objects;
stricter, never looser). The ray-caster G2-G4 check of a realized spec is probes/vet_seeds.check_spec.
RNG: random.Random(int(sha256("realize:" + str(seed)))). Toolbox draws first (up to DRAWS), then chair-row draws.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import seeded as S  # noqa: E402

TOOLBOX, CABINET = S.TOOLBOX, "cabinet_blue_1"
BEDS = ("bed_blue_1", "bed_blue_2", "bed_blue_3")
CHAIRS = S.CHAIRS
SPEC_OBJECTS = (TOOLBOX, CABINET, *BEDS, *CHAIRS)  # every object a spec places or parks (parking slot k = index)
FIXED_ROOM = {CABINET: "storage", **{b: "ward" for b in BEDS}}  # beds / cabinet never move inside the building
DRAWS = 200  # task 011 piece R brief: bounded draws per object group before parking
PARK_X0, PARK_DX, PARK_Y = 40.0, 3.0, 40.0  # task 011 piece R brief: outside the building, far from it and the dock (20, 20)
Pose = tuple[float, float, float]


def canonical(spec: dict[str, Any]) -> dict[str, Any]:
    """The spec's own keys (seed, objects, removed), checked: every spec object exactly once, kept or removed."""
    c = {"seed": int(spec["seed"]), "objects": {k: dict(v) for k, v in spec["objects"].items()},
         "removed": list(spec["removed"])}
    ids = [*c["objects"], *c["removed"]]
    assert sorted(ids) == sorted(SPEC_OBJECTS), f"spec objects {sorted(ids)} != {sorted(SPEC_OBJECTS)}"
    assert TOOLBOX in c["objects"] and CABINET in c["objects"], "toolbox and cabinet are never removed"
    assert c["objects"][CABINET]["state"] in ("open", "closed"), c["objects"][CABINET]
    assert all(v["room"] in S.ROOMS for v in c["objects"].values()), c["objects"]
    return c


def spec_json(spec: dict[str, Any]) -> str:
    return json.dumps(canonical(spec), sort_keys=True, separators=(",", ":"))


def spec_sha256(spec: dict[str, Any]) -> str:
    return hashlib.sha256(spec_json(spec).encode()).hexdigest()


def _frozen(world: str = "closed") -> dict[str, dict]:
    return {o["id"]: o for o in json.loads(S.world_truth_path(world).read_text())["objects"]}


def park_pose(oid: str) -> Pose:
    """Slot k = SPEC_OBJECTS index: (40 + 3 k, 40), yaw as in the frozen truth."""
    return PARK_X0 + PARK_DX * SPEC_OBJECTS.index(oid), PARK_Y, _frozen()[oid]["pose"][5]


def _draw(rng: random.Random, oid: str, room: str) -> Pose:
    """One 008-style draw (seeded.draw): a uniform zone of the room, a uniform point in it, seeded.pose_for's yaw."""
    cls = S.CLASS_OF[oid]
    x0, y0, x1, y1 = rng.choice(S.ZONES[room][cls])
    return S.pose_for(cls, room, round(rng.uniform(x0, x1), 3), round(rng.uniform(y0, y1), 3))


def _row_pairs_bad(row: dict[str, Pose], room: str) -> bool:
    """Cheap pre-check (a subset of seeded.problems' pair rules): any chair pair of the row breaks a pair rule."""
    items = [(oid, room, S.footprint("chair", None, *p)) for oid, p in row.items()]
    return any(S.pair_problems(a, b) for i, a in enumerate(items) for b in items[i + 1:])


def realize(spec: dict[str, Any]) -> dict[str, Any]:
    """{"world", "poses": {id: (x, y, yaw)} of every placed object, "parked": [ids] in SPEC_OBJECTS order, "notes"}."""
    return copy.deepcopy(_realize(spec_json(spec)))


@lru_cache(maxsize=None)
def _realize(spec_text: str) -> dict[str, Any]:
    c = json.loads(spec_text)
    objs = c["objects"]
    rng = random.Random(int(hashlib.sha256(f"realize:{c['seed']}".encode()).hexdigest(), 16))
    world = "cabinet_open" if objs[CABINET]["state"] == "open" else "closed"
    poses: dict[str, Pose] = {}
    rooms: dict[str, str] = {}
    notes: list[str] = []
    frozen = _frozen(world)
    for oid in (CABINET, *BEDS):
        if oid in objs and objs[oid]["room"] == FIXED_ROOM[oid]:
            p = frozen[oid]["pose"]
            poses[oid] = (p[0], p[1], p[5])
        elif oid in objs:
            notes.append(f"{oid}: spec room {objs[oid]['room']} is not its fixed room {FIXED_ROOM[oid]}: parked")
    room = objs[TOOLBOX]["room"]
    if not S.ZONES[room]["toolbox"]:
        notes.append(f"{TOOLBOX}: no toolbox zone in the {room}: parked")
    else:
        for _ in range(DRAWS):
            p = _draw(rng, TOOLBOX, room)
            if not S.alone_problems("toolbox", room, p):  # alone rules + reachability + padded box
                poses[TOOLBOX], rooms[TOOLBOX] = p, room
                break
        else:
            notes.append(f"{TOOLBOX}: no draw in the {room} passed the rules in {DRAWS} draws: parked")
    kept = [ch for ch in CHAIRS if ch in objs]
    if kept:
        room = objs[kept[0]]["room"]
        assert all(objs[ch]["room"] == room for ch in kept), f"chair row in several rooms: {objs}"
        movable = {oid: poses[oid] for oid in rooms}  # the toolbox, if placed
        for k in range(DRAWS):
            row = {ch: _draw(rng, ch, room) for ch in kept}
            if _row_pairs_bad(row, room):
                continue
            if not S.problems({**rooms, **{ch: room for ch in kept}}, {**movable, **row}):
                poses.update(row)
                notes.append(f"chair row of {len(kept)} in the {room}: fit at draw {k + 1}")
                break
        else:
            notes.append(f"chair row of {len(kept)} in the {room}: no fit in {DRAWS} draws: parked")
    notes += [f"{oid}: removed: parked" for oid in c["removed"]]
    parked = [oid for oid in SPEC_OBJECTS if oid not in poses]
    return {"world": world, "poses": {oid: poses[oid] for oid in SPEC_OBJECTS if oid in poses}, "parked": parked,
            "notes": notes}


def truth_for_spec(spec: dict[str, Any]) -> dict:
    """The realized world's truth.json with the placed objects' pose and room replaced and the parked objects removed;
    layout_sha256 = sha256(world layout sha + ":spec:" + canonical spec JSON)."""
    r = realize(spec)
    truth = S.truth_with(r["poses"], r["world"])
    truth["objects"] = [o for o in truth["objects"] if o["id"] not in r["parked"]]
    truth["layout_sha256"] = hashlib.sha256(f"{truth['layout_sha256']}:spec:{spec_json(spec)}".encode()).hexdigest()
    return truth


def spec_photos(poses: dict[str, Pose], world: str, place: str) -> list:
    """The 4 label images at a place with exactly the spec objects in `poses` (beds / cabinet included, parked ones
    left out): seeded.photos' composite over a render without any spec object."""
    shown = {S._labels(world)[m] for m in poses}
    prims = [p for p in S._prims(world, tuple(sorted(poses.items()))) if p[0] in shown]
    out = []
    for k in range(4):
        lab, t = _base_render(world, place, k)
        mlab, mt = S.raycast_seg.render_lt(prims, *S.camera(place, k))
        near = mt < t
        img = lab.copy()
        img[near] = mlab[near]
        out.append(img)
    return out


@lru_cache(maxsize=None)
def _base_render(world: str, place: str, k: int) -> tuple:
    """(labels, t) of the world without any spec object."""
    spec = {S._labels(world)[m] for m in SPEC_OBJECTS}
    return S.raycast_seg.render_lt([p for p in S._sdf_prims(world) if p[0] not in spec], *S.camera(place, k))
