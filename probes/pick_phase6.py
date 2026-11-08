#!/usr/bin/env python3
"""Task 009 run lists by rule: writes phase6_counted.yaml (20) and phase6_dev.yaml (10) to --out. No sim, no VLM, never
reads a run result: the inputs are tasks/phase5_seeds.json (the vetted pool), the excluded seeds below, and --p3 (the
P3 checker's JSON, probes/vet_seeds.py --worlds ...).

    python3 probes/pick_phase6.py --p3 FILE --out DIR

--p3 FILE: a JSON list of records, or a dict whose "records" (vet_seeds.py) or "results" key holds that list; every
record needs at least {seed, world, ok}. A blocked-world entry may only use a (seed, world) whose record has ok true;
guard entries (closed / cabinet_open) need only the phase5_seeds.json ok record.

Rule (written into each list's header):
1. Slots = the committed 009 slot plan (SLOTS_COUNTED / SLOTS_DEV below: task, start, world, answer room, seed
   condition). No substitution: a slot no seed can fill stops the pick, naming the slot. A slot with alternative
   conditions (Count "prefer several photos") takes the first alternative some pool seed meets.
2. Pool = phase5_seeds.json ok seeds minus: the layout seeds of tasks/phase5_counted.yaml + phase5_dev.yaml, every
   `--seeds` of hpc/plans/p5_spot.sh, the first spot set (FIRST_SPOT) and 693.
3. Layout seeds: the pool in the order random.Random("115-009-layout").sample(sorted(pool), len(pool)); the slots of
   both lists are filled fewest-candidates-first (ties: counted before dev, then run order), each taking the first
   unused seed in that order that meets its condition (and, in a blocked world, is P3 ok there).
4. Conditions read the phase5_seeds.json record; every picked seed is then checked against
   seeded.truth_for(seed, "closed") (toolbox room, chairs per room) and Count against seeded.count_photos in the entry's
   world (one / several photos); a mismatch stops the pick.
5. Sampling seeds: random.Random("115-009-counted" / "115-009-dev").sample(range(1, 2**31 - 1), n) in run order.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from probes import pick_phase5 as p5  # noqa: E402

FIRST_SPOT = {672, 673, 675, 676, 689, 690, 691, 692, 694, 698}  # 008's first spot set (before p5_spot.sh)
EXCLUDED = {693}  # carried from 008
P5_LISTS = (ROOT / "tasks" / "phase5_counted.yaml", ROOT / "tasks" / "phase5_dev.yaml")
# The 3-job x 2-sim layout: [closed, blocked_storage], [closed, blocked_reception], [cabinet_open, blocked_ward].
SPLIT = "3:2"
SHARD_WORLDS = {(0, 1): "blocked_storage", (1, 1): "blocked_reception", (2, 0): "cabinet_open", (2, 1): "blocked_ward"}
Slot = dict[str, Any]
Cond = tuple  # ("toolbox", room) | ("chairs", room, counts, several | None) | ("any",)


def s(task: str, start: str, world: str, answer: str, *alts: Cond, com: str = "") -> Slot:
    return {"task": task, "start": start, "world": world, "answer": answer, "alts": list(alts) or [("any",)],
            "com": com}


def tb(room: str) -> Cond:
    return ("toolbox", room)


def ch(room: str, counts: tuple[int, ...], several: bool | None) -> Cond:
    return ("chairs", room, counts, several)


F2SR, F2RS = "find-toolbox-storage-reception", "find-toolbox-reception-storage"
CSR, CRS = "cond-toolbox-storage-reception", "cond-toolbox-reception-storage"
BS, BR, BW = "blocked_storage", "blocked_reception", "blocked_ward"
OPEN, N13 = "state-cabinet-open", (1, 2, 3)
OFFICE_PREF = (ch("office", N13, True), ch("office", N13, False))  # office chairs 1-3, several photos preferred

# The committed 009 slot plan, in run order. Blocked: never start in the blocked room, answer never in it.
SLOTS_COUNTED = [
    s(F2SR, "hall", BS, "reception", tb("reception"), com="first room blocked"),
    s(F2SR, "office", BS, "reception", tb("reception"), com="first room blocked"),
    s(F2RS, "hall", BR, "storage", tb("storage"), com="first room blocked"),
    s(F2RS, "ward", BR, "storage", tb("storage"), com="first room blocked"),
    s(CSR, "hall", BS, "reception", tb("reception"), com="first room wrong"),
    s(CSR, "ward", BS, "reception", tb("reception"), com="first room wrong"),
    s(CRS, "hall", BR, "storage", tb("storage"), com="first room wrong"),
    s(CRS, "office", BR, "storage", tb("storage"), com="first room wrong"),
    s("count-chairs-office", "office", BW, "office", *OFFICE_PREF),
    s(OPEN, "storage", BW, "storage"),  # cabinet closed -> "no"
    # guard (Phase 5 style): starts hall 3, answer room 2, other 5
    s("find-toolbox", "hall", "closed", "reception", tb("reception")),
    s("find-toolbox", "office", "closed", "storage", tb("storage")),
    s("find-beds", "reception", "closed", "ward"),
    s(OPEN, "ward", "cabinet_open", "storage"),  # yes
    s(OPEN, "storage", "closed", "storage"),  # no (the unblocked twin of the blocked_ward check)
    s("count-chairs", "hall", "closed", "reception", ch("reception", (3,), True)),
    s("count-chairs-storage", "ward", "closed", "storage", ch("storage", (2,), True)),
    s("count-chairs-ward", "ward", "cabinet_open", "ward", ch("ward", (1,), False)),
    s(CRS, "office", "closed", "reception", tb("reception"), com="first room right"),
    s("cond-beds-ward", "hall", "closed", "ward", com="first room wrong"),
]
SLOTS_DEV = [
    s(F2SR, "ward", BS, "reception", tb("reception"), com="first room blocked"),
    s(F2RS, "hall", BR, "storage", tb("storage"), com="first room blocked"),
    s(CSR, "hall", BS, "reception", tb("reception"), com="first room wrong"),
    s(CRS, "office", BR, "storage", tb("storage"), com="first room wrong"),
    s("count-chairs-office", "hall", BW, "office", ch("office", N13, None)),
    s(OPEN, "reception", BW, "storage"),
    s("find-toolbox", "ward", "closed", "storage", tb("storage")),
    s("count-chairs-storage", "hall", "closed", "storage", ch("storage", N13, None)),
    s(OPEN, "office", "cabinet_open", "storage"),
    s("count-chairs", "ward", "cabinet_open", "reception", ch("reception", N13, None)),
]


def excluded_seeds() -> set[int]:
    lists = {e["layout_seed"] for p in P5_LISTS for e in yaml.safe_load(p.read_text())}
    return lists | p5.spot_seeds() | FIRST_SPOT | EXCLUDED


def pool() -> dict[int, dict[str, Any]]:
    ex = excluded_seeds()
    return {k: v for k, v in p5.ok_seeds().items() if k not in ex}


def read_p3(path: Path) -> set[tuple[int, str]]:
    """The (seed, world) pairs whose P3 record is ok (format: module doc)."""
    d = json.loads(Path(path).read_text())
    recs = d if isinstance(d, list) else d.get("records", d.get("results"))
    assert isinstance(recs, list), f"{path}: no record list (want a list, or a dict with 'records' / 'results')"
    for r in recs:
        assert {"seed", "world", "ok"} <= set(r), f"{path}: record without seed / world / ok: {r}"
    return {(int(r["seed"]), r["world"]) for r in recs if r["ok"] is True}


def meets(c: Cond, rec: dict[str, Any]) -> bool:
    if c[0] == "toolbox":
        return rec["toolbox_room"] == c[1]
    if c[0] == "chairs":
        _, room, counts, sev = c
        return rec["chairs_per_room"].get(room, 0) in counts and sev in (None, rec["spans_2_photos"].get(room, False))
    return True


def name(k: str, i: int, sl: Slot) -> str:
    return f"{'counted' if k == 'c' else 'dev'} slot {i + 1} ({sl['task']}, {sl['world']}, start {sl['start']})"


@functools.cache
def truth_closed(seed: int) -> dict[str, Any]:
    from sim.worlds.clinic import seeded
    return seeded.truth_for(seed, "closed")


def several_photos(seed: int, room: str, world: str) -> bool:
    from sim.worlds.clinic import seeded
    return len(set(seeded.count_photos(seed, room, "chair", world).values())) >= 2


def verify(seed: int, sl: Slot, rec: dict[str, Any], label: str) -> None:
    """Rule 4: the record the condition read agrees with truth_for / count_photos."""
    rooms = {o["id"]: o["room"] for o in truth_closed(seed)["objects"]}
    chairs = {r: sum(1 for k, v in rooms.items() if k.startswith("chair_orange_") and v == r) for r in p5.COUNT}
    assert rooms["toolbox_red_1"] == rec["toolbox_room"], f"{label}: seed {seed} toolbox room != truth_for"
    assert all(chairs[r] == rec["chairs_per_room"].get(r, 0) for r in chairs), f"{label}: seed {seed} chairs != truth_for"
    for c in sl["alts"]:
        if c[0] == "chairs" and meets(c, rec):
            sev = several_photos(seed, c[1], sl["world"])
            assert sev == rec["spans_2_photos"].get(c[1], False), f"{label}: seed {seed} photo split != count_photos"


def pick(p3_ok: set[tuple[int, str]]) -> dict[str, list[tuple[dict[str, Any], str]]]:
    pl = pool()
    order = random.Random("115-009-layout").sample(sorted(pl), len(pl))
    slots = {"c": SLOTS_COUNTED, "d": SLOTS_DEV}
    cands, chosen = {}, {}
    for k, ls in slots.items():
        for i, sl in enumerate(ls):
            blocked = sl["world"].startswith("blocked_")
            for c in sl["alts"]:
                xs = [x for x in order if meets(c, pl[x]) and (not blocked or (x, sl["world"]) in p3_ok)]
                if xs:
                    cands[k, i], chosen[k, i] = xs, c
                    break
            else:
                sys.exit(f"pick_phase6: cannot fill {name(k, i, sl)}: no pool seed meets {sl['alts']}"
                         + (" and is P3 ok" if blocked else ""))
    used: set[int] = set()
    seed_of: dict[tuple[str, int], int] = {}
    for key in sorted(cands, key=lambda t: (len(cands[t]), t[0] != "c", t[1])):
        free = [x for x in cands[key] if x not in used]
        if not free:
            sys.exit(f"pick_phase6: cannot fill {name(*key, slots[key[0]][key[1]])}: its {len(cands[key])} "
                     "candidate seeds are all taken by other slots")
        seed_of[key] = free[0]
        used.add(free[0])
    out = {}
    for k, nm in (("c", "counted"), ("d", "dev")):
        ss = random.Random(f"115-009-{nm}").sample(range(1, 2**31 - 1), len(slots[k]))
        rows = []
        for i, sl in enumerate(slots[k]):
            x = seed_of[k, i]
            verify(x, sl, pl[x], name(k, i, sl))
            c = chosen[k, i]
            com = (("several photos" if pl[x]["spans_2_photos"].get(c[1]) else "one photo") if c[0] == "chairs"
                   else sl["com"])
            rows.append(({"run_id": f"p6{k}{i + 1:02d}", "task_id": sl["task"], "start_place": sl["start"],
                          "seed": ss[i], "layout_seed": x, "world": sl["world"]}, com))
        out[nm] = rows
    return out


def line(e: dict[str, Any], com: str) -> str:
    return (f"- {{run_id: {e['run_id']}, task_id: {e['task_id'] + ',':31s} start_place: {e['start_place'] + ',':10s} "
            f"seed: {e['seed']:10d}, layout_seed: {e['layout_seed']:3d}, world: {e['world']}}}"
            + (f"  # {com}" if com else ""))


def split_args() -> str:
    return f"--split {SPLIT} " + " ".join(f"--shard-world {j}:{i}={w}" for (j, i), w in sorted(SHARD_WORLDS.items()))


HEAD = {"counted": """\
# Task 009 counted set: 20 runs = 10 blocked (world blocked_<room>: answer never in the blocked room, no start in it)
# + 10 guard (closed / cabinet_open, Phase 5 style) on seeded layouts, each used once, written before any Phase 6 VLM
# output. Entry {run_id, task_id, start_place, seed, layout_seed, world}. task_id resolves in tasks/dev.yaml, then
# tasks/phase5_tasks.yaml, then tasks/phase6_tasks.yaml. True answers are never written here: agent/truth.run_truth
# derives them from truth_for(layout_seed, world). Checked by tests/test_phase6_lists.py. Line comments: Count one /
# several photos; Conditional first room right / wrong; two-room Find first room blocked.
# Selection rule: written by probes/pick_phase6.py (module doc = the rule) from the committed 009 slot plan.
""", "dev": """\
# Task 009 dev set: 10 runs for dev blocks on seeded layouts. Never counted. Format and selection rule as
# tasks/phase6_counted.yaml (probes/pick_phase6.py; one layout-seed draw over both lists).
"""}


def write(out_dir: Path, p3: Path) -> dict[str, Path]:
    lists = pick(read_p3(p3))
    p3r = p3.resolve()
    shown = p3r.relative_to(ROOT) if p3r.is_relative_to(ROOT) else p3r
    sha = hashlib.sha256(p3.read_bytes()).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for nm, rows in lists.items():
        path = out_dir / f"phase6_{nm}.yaml"
        path.write_text(HEAD[nm] + f"# p3: {shown} sha256 {sha}\n# excluded seeds: "
                        f"{' '.join(map(str, sorted(excluded_seeds())))}\n# phase3_run.py {split_args()}\n"
                        + "".join(line(e, c) + "\n" for e, c in rows))
        paths[nm] = path
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3", type=Path, required=True, help="P3 checker JSON (module doc)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    for p in write(a.out, a.p3).values():
        print(p)
    print(f"phase3_run.py {split_args()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
