#!/usr/bin/env python3
"""Phase 5 live spot check of seeded layouts vs the ray-caster, and the G9 photo selection for new placements (task 008).

    python3 probes/spot_check.py --seeds 698,694 --out DIR    # live: one running sim, env SIM_WORLD = its world
    python3 probes/spot_check.py --merge DIR                  # DIR/spot_<world>.json parts -> DIR/spot.json (no sim)
    python3 probes/spot_check.py --g9-select DIR              # DIR/spot.json + photos -> DIR/g9_selection.json (no sim)

Live, per seed (each must be an ok seed of tasks/phase5_seeds.json, else refused): the seed's layout is applied with
adapters/run_task.py's teleport() (gz set_pose + read-back, <= 2 cm / 1 deg; z and gz model name from the world's frozen
truth, as run_task.run_one), seeded.truth_for(seed, world) goes to DIR/<seed>/truth.json, probes/phase1_gates.py takes
the photos at all 5 places (gate_photos: navigate + photos.py) and runs Phase 1's G2 / G3 / G4 / G7 on those frames
against that truth (gate_checks; JSONs tagged rehearsal: they are not Phase 1 verdicts). Then the real frames are
compared with the ray-caster's prediction (seeded.photos at the nominal camera, the code vet_seeds.py vetted with):
  presence  per place and photo k: the set of truth objects with >= G2_MIN_PX px (live vs ray-cast);
  photo     per room: the photo (argmax px, None if never seen) of every countable object and movable of the room;
  check     every Phase 1 check (photos at place, G2, G3, G4, G7) passes live (the ray-caster passed them all);
  vetting   the ray-cast chair -> photo map equals the one vet_seeds.py wrote (code drift guard).
agree = no diff. A seed whose teleport or photos fail is recorded with a diff (agree false); the next seed still runs.
Writes DIR/spot_<world>.json after every seed; nothing is restored (the next run teleports anyway).
spot.json = {"seeds": {seed: {"world", "checks": {gate: pass}, "agree", "diffs": [...]}}, "agree_all", "n_seeds"}.

G9 selection (task 008 "G9 for new placements"): per new (class, room) pair, N_PHOTOS photos at the room's place from
the spot check's agreeing seeds, each showing a movable of that class with >= MIN_PX px (one candidate per photo: the
instance with most px). Greedy pick: an unused seed first, then the largest spread in object-to-place distance from the
photos already picked, then more px. Output {"<class>@<room>": [photo, ...]} with rgb/seg paths relative to DIR, read
by `probes/g9_readable.py --selection`. Exit 0 iff every pair has N_PHOTOS photos.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "probes"))
sys.path.insert(0, str(ROOT / "sim" / "worlds" / "clinic"))
import phase1_checks as pc  # noqa: E402
import seeded as S  # noqa: E402
from g9_readable import MIN_PX, N_PHOTOS  # noqa: E402

SEEDS_JSON = ROOT / "tasks" / "phase5_seeds.json"
# task 008 "G9 for new placements": the (class, room) pairs Phases 1-4 never photographed (toolbox was in storage,
# chairs in reception). Deviation 17 (design review): the toolbox never goes to the ward any more; the chair pairs passed 3/3 on the
# old layouts and are re-checked here on the new ones (their photos come with the spot check at no extra sim time).
NEW_PAIRS = (("toolbox", "office"), ("toolbox", "reception"),
             ("chair", "ward"), ("chair", "office"), ("chair", "storage"))

Images = dict[str, list[np.ndarray]]  # place -> the 4 cropped label images


def ok_seeds(path: Path = SEEDS_JSON) -> dict[int, dict]:
    return {e["seed"]: e for e in json.loads(path.read_text())["ok"]}


def want_for(seed: int, world: str) -> dict[str, tuple[float, float, float, float]]:
    """{gz model name: (x, y, z, yaw)} of the seed's movables, z from the world's frozen truth (as run_task.run_one)."""
    frozen = {o["id"]: o for o in json.loads(S.world_truth_path(world).read_text())["objects"]}
    return {frozen[oid]["model_name"]: (x, y, frozen[oid]["pose"][2], yaw) for oid, (x, y, yaw) in S.placements(seed).items()}


def predicted(seed: int, world: str) -> Images:
    poses = S.placements(seed)
    return {p: S.photos(poses, world, p) for p in S.PLACES}


def photo_of(imgs: list[np.ndarray], label: int) -> int | None:
    px = pc.pixel_counts(imgs, label)
    return int(np.argmax(px)) if sum(px) else None


def compare(live: Images, pred: Images, truth: dict) -> list[dict]:
    """Presence (>= G2_MIN_PX) per place and photo, and the photo of each countable object / movable of a room."""
    diffs: list[dict] = []
    for place, pimgs in pred.items():
        if place not in live:
            diffs.append({"kind": "no_photos", "place": place})
            continue
        for k, (li, pi) in enumerate(zip(live[place], pimgs)):
            assert li.shape == pi.shape, f"{place} k={k}: live {li.shape} vs ray-cast {pi.shape}"
            for o in truth["objects"]:
                lp, pp = int((li == o["label"]).sum()), int((pi == o["label"]).sum())
                if (lp >= pc.G2_MIN_PX) != (pp >= pc.G2_MIN_PX):
                    diffs.append({"kind": "presence", "place": place, "k": k, "object": o["id"],
                                  "live_px": lp, "raycast_px": pp})
        for o in truth["objects"]:
            if o["room"] == place and (o["countable"] or o["id"] in S.MOVABLES):
                lk, pk = photo_of(live[place], o["label"]), photo_of(pimgs, o["label"])
                if lk != pk:
                    diffs.append({"kind": "photo", "room": place, "object": o["id"], "live_k": lk, "raycast_k": pk})
    return diffs


def check_diffs(checks: dict[str, bool]) -> list[dict]:
    return [{"kind": "check", "gate": g, "live": False, "raycast": True} for g, ok in checks.items() if not ok]


def vetting_diffs(pred: Images, truth: dict, vetted: dict) -> list[dict]:
    """The ray-cast chair -> photo map vs tasks/phase5_seeds.json's (both from seeded.photos at the nominal camera)."""
    now: dict[str, dict[str, int | None]] = {}
    for o in truth["objects"]:
        if o["class"] == "chair" and o["room"] in S.ROOMS:
            now.setdefault(o["room"], {})[o["id"]] = photo_of(pred[o["room"]], o["label"])
    return [] if now == vetted["chair_photos"] else [{"kind": "vetting", "raycast": now, "vetted": vetted["chair_photos"]}]


def record(world: str, checks: dict[str, bool], diffs: list[dict]) -> dict:
    return {"world": world, "checks": checks, "agree": not diffs, "diffs": diffs}


def spot_one(robot: Any, places: dict, seed: int, world: str, out: Path, vetted: dict) -> dict:  # noqa: ANN401
    """Live: teleport, truth, photos, Phase 1 checks, comparison; one seed's spot.json record."""
    import phase1_gates as pg
    from adapters.run_task import teleport
    sdir = out / str(seed)
    sdir.mkdir(parents=True)
    try:
        why = teleport(want_for(seed, world))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:  # the read-back died (as run_task)
        why = f"read-back {type(e).__name__}: {e}"
    if why is not None:
        return record(world, {}, [{"kind": "teleport", "why": why}])
    truth = S.truth_for(seed, world)
    (sdir / "truth.json").write_text(json.dumps(truth, indent=1))
    checks = {"photos": bool(pg.gate_photos(robot, places, sdir, True)["pass"])}
    checks.update({r["gate"]: bool(r["pass"]) for r in pg.gate_checks(sdir, truth, places, True)})
    live = {p: pc.load_place(sdir / "photos" / p)[1] for p in S.PLACES if (sdir / "photos" / p / "photos.json").exists()}
    pred = predicted(seed, world)
    return record(world, checks, check_diffs(checks) + compare(live, pred, truth) + vetting_diffs(pred, truth, vetted))


def run_live(seeds: list[int], out: Path) -> int:
    import map_world_check as mwc
    from adapters.run_task import sim_world
    world = sim_world()
    vetted = ok_seeds()
    bad = [s for s in seeds if s not in vetted]
    if bad:
        raise SystemExit(f"not ok seeds in {SEEDS_JSON.name} (never used): {bad}")
    places = mwc.load_places(ROOT / "sim/worlds/clinic/places.yaml")
    robot = mwc.make_robot()
    part = out / f"spot_{world}.json"
    res: dict[str, dict] = {}
    for seed in seeds:
        try:
            res[str(seed)] = spot_one(robot, places, seed, world, out, vetted[seed])
        except Exception as e:  # one seed's crash (photos.py, gz read) is recorded; the sim's other seeds still run
            res[str(seed)] = record(world, {}, [{"kind": "error", "why": f"{type(e).__name__}: {e}"[-2000:]}])
        print(f"seed {seed} ({world}): {'AGREE' if res[str(seed)]['agree'] else 'DIFF'} "
              f"{len(res[str(seed)]['diffs'])} diffs", flush=True)
        part.write_text(json.dumps({"seeds": res}, indent=1))
    return 0 if all(r["agree"] for r in res.values()) else 1


def merge(out: Path) -> dict:
    seeds: dict[str, dict] = {}
    for p in sorted(out.glob("spot_*.json")):
        part = json.loads(p.read_text())["seeds"]
        assert not set(part) & set(seeds), f"{p}: seeds {sorted(set(part) & set(seeds))} in two parts"
        seeds.update(part)
    res = {"seeds": seeds, "agree_all": bool(seeds) and all(r["agree"] for r in seeds.values()), "n_seeds": len(seeds)}
    (out / "spot.json").write_text(json.dumps(res, indent=1))
    print(f"spot: {sum(r['agree'] for r in seeds.values())}/{len(seeds)} seeds agree; agree_all {res['agree_all']}")
    return res


# ------------------------------------------------------------------------------------------------------ G9 select
def g9_candidates(out: Path, seeds: list[str], cls: str, room: str) -> list[dict]:
    """One candidate per photo at the room's place holding a movable of `cls` in `room` with >= MIN_PX px."""
    place = S.PLACES[room]
    cands = []
    for seed in seeds:
        truth = json.loads((out / seed / "truth.json").read_text())
        objs = [o for o in truth["objects"] if o["class"] == cls and o["room"] == room and o["id"] in S.MOVABLES]
        pdir = out / seed / "photos" / room
        if not objs or not (pdir / "photos.json").exists():
            continue
        meta, imgs = pc.load_place(pdir)
        for p, img in zip(meta["photos"], imgs):
            px, o = max(((int((img == o["label"]).sum()), o) for o in objs), key=lambda t: t[0])
            if px >= MIN_PX:
                cands.append({"group": f"{cls}@{room}", "class": cls, "instance": o["id"], "pixels": px,
                              "state": None, "seed": int(seed), "place": room, "k": p["k"],
                              "photo": f"{seed}/{room}/{p['k']}",
                              "distance_m": round(math.dist(o["pose"][:2], (place["x"], place["y"])), 3),
                              "rgb": f"{seed}/photos/{room}/{p['rgb']}", "seg": f"{seed}/photos/{room}/{p['seg']}"})
    return cands


def g9_pick(cands: list[dict], n: int = N_PHOTOS) -> list[dict]:
    """Greedy: an unused seed first, then the largest distance spread from the picked photos, then more px."""
    pool, chosen = sorted(cands, key=lambda c: c["photo"]), []
    while pool and len(chosen) < n:
        seeds = {c["seed"] for c in chosen}
        best = min(pool, key=lambda c: (c["seed"] in seeds,
                                        -min((abs(c["distance_m"] - x["distance_m"]) for x in chosen), default=0.0),
                                        -c["pixels"], c["photo"]))
        chosen.append(best)
        pool.remove(best)
    return chosen


def g9_select(out: Path) -> int:
    spot = json.loads((out / "spot.json").read_text())
    seeds = sorted((s for s, r in spot["seeds"].items() if r["agree"]), key=int)
    sel = {f"{c}@{r}": g9_pick(g9_candidates(out, seeds, c, r)) for c, r in NEW_PAIRS}
    (out / "g9_selection.json").write_text(json.dumps(sel, indent=1))
    for g, s in sel.items():
        print(f"{g:18} {len(s)}/{N_PHOTOS}  " + "  ".join(f"{c['photo']} ({c['pixels']} px, {c['distance_m']} m)" for c in s))
    short = [g for g, s in sel.items() if len(s) < N_PHOTOS]
    print("g9 selection " + ("OK" if not short else f"SHORT: {short}") + f" (from agreeing seeds {seeds})")
    return 0 if not short else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--seeds", help="comma list of layout seeds (live, needs --out)")
    m.add_argument("--merge", type=Path, metavar="DIR")
    m.add_argument("--g9-select", type=Path, metavar="DIR")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.merge:
        return 0 if merge(a.merge)["agree_all"] else 1
    if a.g9_select:
        return g9_select(a.g9_select)
    if a.out is None:
        ap.error("--seeds needs --out")
    return run_live([int(s) for s in a.seeds.split(",")], a.out)


if __name__ == "__main__":
    raise SystemExit(main())
