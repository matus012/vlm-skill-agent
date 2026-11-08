#!/usr/bin/env python3
"""H-P3 readable on the G1 (task 012 design section 5): G9 naming on G1 look_around photos of vetted DEV layouts.

    python humanoid/probes/g1_spot.py pick --pairs pairs.json --out D [--exclude 1,2]  # dev seeds still needed (SIM_WORLD)
    python humanoid/probes/g1_spot.py live --seeds 1,2 --out D             # live, one sim (env SIM_WORLD)
    python humanoid/probes/g1_spot.py select --pairs pairs.json --out D    # -> D/g9_selection.json, D/g9_seeds.json
    python humanoid/probes/g1_spot.py verdict --pairs pairs.json --out D   # -> D/hp3.json (after g9_readable)

Seeds: the pool of humanoid/probes/hp1_reach.dev_seeds() (ok seeds of tasks/phase5_seeds.json that filter rule c keeps
out of every test list, no counted seed) minus 008's failed spot seeds (pick_phase6.EXCLUDED: 693), highest first.
Groups: every movable pair (toolbox / chair, room) of pairs.json (humanoid/probes/g1_pairs.py, the test list's target
pairs) as <cls>@<room>, plus the fixed targets bed (ward), cabinet:closed, cabinet:open (storage). Every group needs
N_PHOTOS photos from N_PHOTOS DIFFERENT seeds' look_arounds at the group's room place (a fixed target photographed in
3 seed runs = 3 separate look_arounds, each with its own pose).
pick: the seeds SIM_WORLD's groups still need: per group (closed: the movables, bed, cabinet:closed; cabinet_open:
cabinet:open) the seeds with a candidate photo already in D count, then pool seeds not yet run in D (nor --exclude)
(a D/<seed> dir = tried, photos or not) are drawn in pool order while the group has < N_PHOTOS: a movable group draws only a seed that places that class in
that room (phase5_seeds.json record), a fixed one any seed. Empty output = nothing left to run (or the pool is out).
The plan alternates pick / live until pick is empty; a seed runs in one world only (D/<seed> is per seed).
live: per seed G1Scene.place of the world's frozen layout with the movables at seeded.placements(seed), D/<seed>/truth.json = seeded.truth_for,
look_around photos at the 4 room places (hp2_photos.take_place) in D/<seed>/photos/<room>/; D/spot_<partition>_<seed0>.json.
select: per group the candidates (movable: spot_check.g9_candidates UNCHANGED; fixed: the same rule on the fixed
object, state from the seed's truth.json) through spot_check.g9_pick UNCHANGED (an unused seed first) -> D/g9_selection.json
(read by `probes/g9_readable.py --selection`; fixed candidates carry class + state, so the cabinet gets the state call)
and D/g9_seeds.json {group: seeds}; exit 0 iff each group has N_PHOTOS photos from N_PHOTOS different seeds.
verdict: each test pair -> its g9_readable group in D/g9_sel (movable: <cls>@<room>; bed: bed; cabinet: cabinet:<state>,
both states when the pair has none) -> D/hp3.json with the group's seeds, pass iff every group passes (names >= 2/3,
state 3/3) on N_PHOTOS different seeds.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "probes"), str(ROOT / "sim/worlds/clinic")]
import phase1_gates as pg  # noqa: E402
import pick_phase6  # noqa: E402
import seeded as S  # noqa: E402
import spot_check as sc  # noqa: E402
import g9_readable as g9  # noqa: E402
from g9_readable import N_PHOTOS  # noqa: E402

MOVABLE = ("toolbox", "chair")
FIXED = {"closed": [("bed", "ward", None), ("cabinet", "storage", "closed")],
         "cabinet_open": [("cabinet", "storage", "open")]}  # (class, room, state) of the fixed targets per world


def pool() -> list[int]:
    from humanoid.probes.hp1_reach import dev_seeds
    return sorted(dev_seeds() - pick_phase6.EXCLUDED, reverse=True)


def movable_pairs(pairs: dict) -> list[tuple[str, str]]:
    return sorted({(p["class"], p["room"]) for p in pairs["pairs"] if p["class"] in MOVABLE})


def has(e: dict, cls: str, room: str) -> bool:
    return e["toolbox_room"] == room if cls == "toolbox" else e["chairs_per_room"].get(room, 0) > 0


def group(cls: str, room: str, state: str | None) -> str:
    return f"{cls}@{room}" if cls in MOVABLE else f"{cls}:{state}" if state else cls


def groups(pairs: dict, world: str | None = None) -> list[tuple[str, str, str | None]]:
    """(class, room, state) of the groups `world` photographs (None: every world's)."""
    mov = [(c, r, None) for c, r in movable_pairs(pairs)] if world in ("closed", None) else []
    return [*FIXED.get(world, []), *mov] if world else [*FIXED["closed"], *FIXED["cabinet_open"], *mov]


def run_seeds(out: Path) -> list[str]:
    return sorted((p.parent.name for p in out.glob("*/truth.json")), key=int)


def candidates(out: Path, seeds: list[str], cls: str, room: str, state: str | None) -> list[dict]:
    """Movable: spot_check.g9_candidates unchanged. Fixed: the same rule (one per photo at the room's place, the
    object's px >= MIN_PX) on the fixed object of that class + state; the candidate carries class and state."""
    if cls in MOVABLE:
        return sc.g9_candidates(out, seeds, cls, room)
    place, cands = S.PLACES[room], []
    for seed in seeds:
        truth = json.loads((out / seed / "truth.json").read_text())
        objs = [o for o in truth["objects"] if o["class"] == cls and o["room"] == room and o.get("state") == state]
        pdir = out / seed / "photos" / room
        if not objs or not (pdir / "photos.json").exists():
            continue
        meta, imgs = sc.pc.load_place(pdir)
        for p, img in zip(meta["photos"], imgs):
            px, o = max(((int((img == o["label"]).sum()), o) for o in objs), key=lambda t: t[0])
            if px >= sc.MIN_PX:
                cands.append({"group": group(cls, room, state), "class": cls, "instance": o["id"], "pixels": px,
                              "state": state, "seed": int(seed), "place": room, "k": p["k"],
                              "photo": f"{seed}/{room}/{p['k']}",
                              "distance_m": round(math.dist(o["pose"][:2], (place["x"], place["y"])), 3),
                              "rgb": f"{seed}/photos/{room}/{p['rgb']}", "seg": f"{seed}/photos/{room}/{p['seg']}"})
    return cands


def pick(pairs: dict, world: str, seeds: list[int], ok: dict[int, dict], out: Path,
         exclude: frozenset[int] = frozenset()) -> list[int]:
    """Pool seeds (in order) still needed by `world`'s groups, given the seeds already run in `out` (module doc)."""
    done = run_seeds(out)
    skip = {int(d.name) for d in out.glob("*") if d.name.isdigit()} | set(exclude)  # tried, even if placing failed
    chosen: list[int] = []
    for cls, room, state in groups(pairs, world):
        fits = (lambda s: has(ok[s], cls, room)) if cls in MOVABLE else (lambda s: True)
        n = len({c["seed"] for c in candidates(out, done, cls, room, state)})
        n += sum(fits(c) for c in chosen)
        for s in seeds:
            if n >= N_PHOTOS:
                break
            if s not in skip and s not in chosen and fits(s):
                chosen.append(s)
                n += 1
    return chosen


def live(seeds: list[int], out: Path) -> int:
    from adapters.robots import make
    from adapters.run_task import sim_world
    from humanoid.probes.hp2_photos import SelfView, frozen_want, scene, take_place
    import map_world_check as mwc
    world = sim_world()
    bad = sorted(set(seeds) - set(pool()))
    if bad:
        raise SystemExit(f"not vetted dev seeds (never used): {bad}")
    places = mwc.load_places(pg.PLACES_YAML)
    sjs = scene()
    lut = np.asarray(sjs["label_of_body"], np.uint8)
    part, res = out / f"spot_{os.environ['GZ_PARTITION']}_{seeds[0]}.json", {}
    rb = make("g1")
    place = rb.run_kw["place_objects"]
    try:
        sv = SelfView(rb.ad.drive, sjs["robot_body_ids"])
        for seed in seeds:
            sdir = out / str(seed)
            sdir.mkdir(parents=True)
            log: dict = {}
            why = place({**frozen_want(world), **sc.want_for(seed, world)}, log)
            rec = {"world": world, "place": why, "teleport": log}
            if why is None:
                (sdir / "truth.json").write_text(json.dumps(S.truth_for(seed, world), indent=1))
                rec["at_place"] = {r: take_place(rb.ad, rb.gt_pose, r, places[r], sdir / "photos", lut, sv)["at_place"]
                                   for r in S.ROOMS}
            res[str(seed)] = rec
            print(f"seed {seed} ({world}): placed {why is None}, at place {rec.get('at_place')}", flush=True)
            part.write_text(json.dumps({"seeds": res}, indent=1))
    finally:
        rb.close()
    return 0 if all(r["place"] is None and all(r["at_place"].values()) for r in res.values()) else 1


def select(pairs: dict, out: Path) -> int:
    seeds = run_seeds(out)
    sel = {group(*g): sc.g9_pick(candidates(out, seeds, *g)) for g in groups(pairs)}
    used = {g: sorted({c["seed"] for c in s}) for g, s in sel.items()}
    (out / "g9_selection.json").write_text(json.dumps(sel, indent=1))
    (out / "g9_seeds.json").write_text(json.dumps(used, indent=1))
    for g, s in sel.items():
        print(f"{g:18} {len(s)}/{N_PHOTOS} seeds {used[g]}  " + "  ".join(f"{c['photo']} ({c['pixels']} px)" for c in s))
    return 0 if all(len(s) >= N_PHOTOS and len(used[g]) >= N_PHOTOS for g, s in sel.items()) else 1


def groups_for(p: dict) -> list[str]:
    """g9_readable groups (in D/g9_sel) of a test pair."""
    if p["class"] == "cabinet" and not p["state"]:
        return [group("cabinet", p["room"], s) for s in ("open", "closed")]
    return [group(p["class"], p["room"], p["state"])]


def verdict(pairs: dict, out: Path, rehearsal: bool) -> int:
    f, sf = out / "g9_sel" / "g9_summary.json", out / "g9_seeds.json"
    summ = json.loads(f.read_text())["groups"] if f.exists() else {}
    used = json.loads(sf.read_text()) if sf.exists() else {}
    rows = []
    for p in pairs["pairs"]:
        gs = groups_for(p)
        got = [summ.get(g) for g in gs]
        rows.append({**{k: p[k] for k in ("class", "room", "state", "n")}, "groups": gs, "results": got,
                     "seeds": {g: used.get(g, []) for g in gs},
                     "pass": all(r is not None and r["pass"] and len(used.get(g, [])) >= N_PHOTOS
                                 for g, r in zip(gs, got))})
    v = pg.write(out, "hp3", {"criterion": "every target pair of the test list named in >= 2/3 G1 photos from 3 "
                                           "different dev seeds' look_arounds; cabinet state 3/3 (probes/g9_readable.py)",
                              "pairs": rows,
                              "pass": bool(rows) and all(r["pass"] for r in rows)}, rehearsal)
    return 0 if v["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pick", "live", "select", "verdict"])
    ap.add_argument("--pairs", type=Path)
    ap.add_argument("--seeds")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--rehearsal", action="store_true")
    a = ap.parse_args()
    if a.cmd == "live":
        return live([int(s) for s in a.seeds.split(",")], a.out)
    pairs = json.loads(a.pairs.read_text())
    if a.cmd == "pick":
        ex = frozenset(int(s) for s in a.exclude.split(",") if s)
        print(",".join(map(str, pick(pairs, os.environ.get("SIM_WORLD", "closed"), pool(), sc.ok_seeds(), a.out, ex))))
        return 0
    return select(pairs, a.out) if a.cmd == "select" else verdict(pairs, a.out, a.rehearsal)


if __name__ == "__main__":
    raise SystemExit(main())
