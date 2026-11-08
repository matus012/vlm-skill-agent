"""Vet Phase 5 layout seeds off-sim with the ray-caster (task 008, agent/INTERFACES.md section 10).

    python3 probes/vet_seeds.py --seeds 0 1000            # writes tasks/phase5_seeds.json

Per seed: sim/worlds/clinic/seeded.placements (every geometric rule; SeedRejected = rejected), then per world (closed,
cabinet_open) the 4 look_around photos at every place (probes/raycast_seg.py, movables at the seed's poses, camera at
the nominal pose: base on the place point) and Phase 1's gates (probes/phase1_checks.py thresholds):
  G2  every object of a room >= G2_MIN_PX in one of that room's photos;
  G3  every countable object of a room >= G3_ONE_PHOTO_PCT % of its pixels in one photo, and every movable visible in
      exactly one of its room's photos (wholly inside it);
  G4  no countable object or movable of another room reaches G4_MAX_LEAK_PX in any photo of a place (hall included).
The output lists ok seeds (with their poses, rooms and chair -> photo maps) apart from rejected ones (with reasons);
a rejected seed is never used. Deterministic: same range -> byte-identical file.

Phase 6 precondition P3 (task 009, INTERFACES section 11): given seeds in given (blocked) worlds, the same G2/G3/G4 at
every REACHABLE place (with --skip-blocked the blocked room's place is left out), ray-cast in that world's clinic.sdf
so the blocker box (label 220, not an object: never checked itself) occludes rays:

    python3 probes/vet_seeds.py --worlds blocked_storage,blocked_ward --check-seeds 3,17,42 --skip-blocked --out FILE

writes {"records": [{seed, world, ok, skipped_place, places: {place: [breaches]}, blocker_px: {place: px}}], ...}
and exits 1 if any record is not ok.

Phase 8 filter rule (b) (task 011, INTERFACES section 12): check_spec(spec) -> (ok, breaches), the same G2/G3/G4 at
every place for a layout spec realized by sim/worlds/clinic/realize.py (parked objects absent from every photo).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
sys.path.insert(0, str(ROOT / "sim" / "worlds" / "clinic"))
import phase1_checks as pc  # noqa: E402
import realize as RZ  # noqa: E402
import seeded as S  # noqa: E402

OUT = ROOT / "tasks" / "phase5_seeds.json"
WORLDS = ("closed", "cabinet_open")


def raycast_problems(poses: dict[str, tuple[float, float, float]], world: str) -> tuple[list[str], dict]:
    """(G2/G3/G4 breaches, {room: {chair id: photo k}}) for the movables at `poses` in one world."""
    truth = S.truth_with(poses, world)
    bad: list[str] = []
    chair_photos: dict[str, dict[str, int]] = {}
    for place in S.PLACES:
        bad += place_problems(S.photos(poses, world, place), world, place, truth, chair_photos)
    return bad, chair_photos


def place_problems(imgs: list, world: str, place: str, truth: dict, chair_photos: dict[str, dict[str, int]]) -> list[str]:
    """G2/G3/G4 breaches in the 4 photos at one place (truth["objects"] only: the blocker is not an object); fills
    chair_photos."""
    bad: list[str] = []
    room = place if place in S.ROOMS else None
    for o in truth["objects"]:
        px = pc.pixel_counts(imgs, o["label"])
        if o["room"] == room:
            if max(px) < pc.G2_MIN_PX:
                bad.append(f"{world} G2 {o['id']}: max {max(px)} px at {place}")
            if o["countable"] and not pc.counted_once_ok(imgs, o["label"]):
                bad.append(f"{world} G3 {o['id']}: px per photo {px} at {place}")
            if o["id"] in S.MOVABLES and sum(v > 0 for v in px) != 1:
                bad.append(f"{world} G3 {o['id']}: in {sum(v > 0 for v in px)} photos {px} at {place}")
            if o["class"] == "chair":
                chair_photos.setdefault(room, {})[o["id"]] = int(np.argmax(px))
        elif (o["countable"] or o["id"] in S.MOVABLES) and max(px) >= pc.G4_MAX_LEAK_PX:
            bad.append(f"{world} G4 {o['id']} ({o['room']}): {max(px)} px at the {place} place")
    return bad


def check_world(seed: int, world: str, skip_blocked: bool) -> dict:
    """P3 record of one (seed, world): G2/G3/G4 per place, the blocked room's place left out if skip_blocked."""
    try:
        poses = S.placements(seed)
    except S.SeedRejected as e:
        return {"seed": seed, "world": world, "ok": False, "reasons": ["rule: " + r for r in e.reasons]}
    truth = S.truth_with(poses, world)
    blocker = truth.get("blocker")
    skip = blocker["room"] if skip_blocked and blocker else None
    places, blocker_px = {}, {}
    for place in S.PLACES:
        if place == skip:
            continue
        imgs = S.photos(poses, world, place)
        places[place] = place_problems(imgs, world, place, truth, {})
        if blocker:
            blocker_px[place] = sum(pc.pixel_counts(imgs, blocker["label"]))
    return {"seed": seed, "world": world, "ok": not any(places.values()), "skipped_place": skip,
            "places": places, "blocker_px": blocker_px}


def check_spec(spec: dict) -> tuple[bool, list[str]]:
    """Phase 8 filter rule (b) (INTERFACES section 12): (ok, G2/G3/G4 breaches at every place) of the realized spec in
    its world, the photos holding exactly the placed spec objects (parked ones are outside every view)."""
    r = RZ.realize(spec)
    truth = RZ.truth_for_spec(spec)
    bad = [b for place in S.PLACES
           for b in place_problems(RZ.spec_photos(r["poses"], r["world"], place), r["world"], place, truth, {})]
    return not bad, bad


def vet_poses(poses: dict[str, tuple[float, float, float]]) -> tuple[list[str], dict]:
    """(every breach, {room: {chair id: photo k}}) of a full placement: the geometric rules (seeded.problems), then the
    ray-cast G2/G3/G4 in both worlds."""
    rooms = {oid: S.room_of(*poses[oid][:2]) for oid in S.MOVABLES}
    bad = ["rule: " + r for r in S.problems(rooms, poses)]
    maps = {}
    for world in WORLDS:
        b, maps[world] = raycast_problems(poses, world)
        bad += b
    if maps["closed"] != maps["cabinet_open"]:
        bad.append(f"chair photos differ between worlds: {maps}")
    return bad, maps["closed"]


def vet(seed: int) -> dict:
    """One seed's record: ok, reasons, and for an ok seed its layout and per-room chair -> photo map."""
    try:
        poses = S.placements(seed)
    except S.SeedRejected as e:
        return {"ok": False, "reasons": ["rule: " + r for r in e.reasons]}
    bad, chair_photos = vet_poses(poses)
    if bad:
        return {"ok": False, "reasons": bad}
    rooms = {oid: S.room_of(*poses[oid][:2]) for oid in S.MOVABLES}
    spans = {r: len(set(m.values())) >= 2 for r, m in chair_photos.items()}
    return {"ok": True, "toolbox_room": rooms[S.TOOLBOX],
            "chairs_per_room": {r: sum(rooms[c] == r for c in S.CHAIRS) for r in S.ROOMS},
            "chair_photos": chair_photos, "spans_2_photos": spans, "any_room_spans_2_photos": any(spans.values()),
            "poses": {oid: list(poses[oid]) for oid in S.MOVABLES}, "rooms": rooms}


def summary(ok: list[dict], rejected: list[dict]) -> dict:
    kinds = Counter()
    for reasons in (r["reasons"] for r in rejected):
        kinds.update({"rule: " + next((k for k in ("gap", "overlap", "unreachable") if k in r), "other")
                      if r.startswith("rule") else " ".join(r.split()[:2]) for r in reasons})
    return {"vetted": len(ok) + len(rejected), "ok": len(ok), "rejected": len(rejected),
            "rejected_seeds_by_reason": dict(sorted(kinds.items())),
            "toolbox_room": dict(sorted(Counter(e["toolbox_room"] for e in ok).items())),
            "chairs_per_room_histogram": {r: dict(sorted(Counter(e["chairs_per_room"][r] for e in ok).items()))
                                          for r in S.ROOMS},
            "seeds_with_a_room_spanning_2_photos": sum(e["any_room_spans_2_photos"] for e in ok)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs=2, metavar=("FIRST", "STOP"), help="range(FIRST, STOP)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--jobs", type=int, default=None, help="worker processes (default: all cores)")
    ap.add_argument("--worlds", help="P3: comma-separated worlds (e.g. blocked_storage,blocked_ward)")
    ap.add_argument("--check-seeds", help="P3: comma-separated seeds to check in every --worlds world")
    ap.add_argument("--skip-blocked", action="store_true", help="P3: leave out the blocked room's place")
    a = ap.parse_args()
    if a.worlds:
        if not a.check_seeds or a.out == OUT:
            ap.error("--worlds needs --check-seeds and an --out other than tasks/phase5_seeds.json")
        jobs = [(int(s), w, a.skip_blocked) for w in a.worlds.split(",") for s in a.check_seeds.split(",")]
        with Pool(a.jobs) as pool:
            recs = pool.starmap(check_world, jobs)
        bad = [(r["seed"], r["world"]) for r in recs if not r["ok"]]
        a.out.write_text(json.dumps({"about": "P3 (INTERFACES section 11): G2/G3/G4 at the reachable places",
                                     "skip_blocked": a.skip_blocked, "records": recs, "not_ok": bad},
                                    indent=1, sort_keys=True) + "\n")
        print(f"{len(recs) - len(bad)}/{len(recs)} ok; not ok: {bad}")
        sys.exit(1 if bad else 0)
    if not a.seeds:
        ap.error("--seeds FIRST STOP is required (or --worlds for P3)")
    seeds = list(range(*a.seeds))
    with Pool(a.jobs) as pool:
        recs = pool.map(vet, seeds, chunksize=4)
    ok = [{"seed": s, **r} for s, r in zip(seeds, recs) if r["ok"]]
    rejected = [{"seed": s, "reasons": r["reasons"]} for s, r in zip(seeds, recs) if not r["ok"]]
    out = {"about": __doc__.strip().splitlines()[0], "seed_range": a.seeds, "worlds": list(WORLDS),
           "photo_index": "k = 0..3: look_around photo k at place yaw + k*90 deg (photo number k + 1)",
           "thresholds": {"G2_MIN_PX": pc.G2_MIN_PX, "G3_ONE_PHOTO_PCT": pc.G3_ONE_PHOTO_PCT,
                          "G4_MAX_LEAK_PX": pc.G4_MAX_LEAK_PX, "MIN_GAP_M": S.MIN_GAP_M, "MIN_SEAM_DEG": S.MIN_SEAM_DEG},
           "zones": S.ZONES, "summary": summary(ok, rejected), "ok": ok, "rejected": rejected}
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
