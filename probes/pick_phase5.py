#!/usr/bin/env python3
"""Task 008 run lists by rule (a recorded deviation re-pick): writes phase5_counted.yaml and phase5_dev.yaml to --out. No sim,
no VLM, never reads a run result: the only inputs are tasks/phase5_seeds.json (the vetted pool), the seed list of
hpc/plans/p5_spot.sh (excluded) and --toolbox-rooms (the rooms where the toolbox passed G9).

    python3 probes/pick_phase5.py --toolbox-rooms office,reception,storage --out DIR

Rule (written into each list's header):
1. Slots = the committed 2026-11-05 slot plan (SLOTS_COUNTED / SLOTS_DEV below: task, start, world, seed condition).
2. A Find / Conditional slot the pool cannot fill (its toolbox room not in --toolbox-rooms, or no ok seed meets its
   condition, e.g. all 4 chairs in one room) is substituted, in run order, keeping its start and world. Cap per type =
   floor(0.4 x the type's runs) answers per room (task 008: no room > 40 %). Room tally = the type's kept slots.
   - Find: find-toolbox in the toolbox room with the fewest answers so far, if below the cap (ties: office, reception,
     storage); else find-beds-where (the beds, ward).
   - Conditional, same branch (first room right / wrong): a toolbox prompt whose first (right) / second (wrong) room is
     the toolbox room with the fewest answers so far, if below the cap (ties as Find; prompt = the first such in
     dev.yaml then phase5_tasks.yaml order); else beds: cond-beds-ward-office (right) / cond-beds-ward (wrong).
3. Layout seeds: the ok seeds in the order random.Random("115-008-layout").sample(sorted(ok seeds), len(ok seeds)),
   p5_spot.sh's seeds dropped; the slots of both lists are filled fewest-candidates-first (ties: counted before dev,
   then run order), each taking the first unused seed in that order that meets its condition (beds slots: any seed).
4. Sampling seeds: random.Random("115-008-counted" / "115-008-dev").sample(range(1, 2**31 - 1), n) in run order.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.tasks import all_tasks  # noqa: E402

SEEDS_JSON = ROOT / "tasks" / "phase5_seeds.json"
SPOT_PLAN = ROOT / "hpc" / "plans" / "p5_spot.sh"
ROOM_ORDER = ("office", "reception", "storage")  # tie order (the toolbox rooms of the vetted pool)
CAP_SHARE = 0.4  # task 008 Counted sets: no room > 40 % of a type's answers
O, C = "state-cabinet-open", "state-cabinet-closed"
COUNT = {"office": "count-chairs-office", "reception": "count-chairs", "storage": "count-chairs-storage",
         "ward": "count-chairs-ward"}
Slot = dict[str, Any]


def s(task: str, start: str, *, world: str | None = None, **cond: Any) -> Slot:
    """cond: toolbox=room | chairs=(room, n, several) | all_chairs=True | nothing (any seed)."""
    return {"task": task, "start": start, "world": world, **cond}


def find(room: str | None, start: str) -> Slot:
    return s("find-toolbox", start, toolbox=room) if room else s("find-chairs-plain", start, all_chairs=True)


def cond(task: str, right: bool, start: str) -> Slot:
    return {"task": task, "start": start, "world": None, "right": right}


def count(room: str, n: int, several: bool, start: str) -> Slot:
    return s(COUNT[room], start, chairs=(room, n, several))


# The committed slot plan (tasks/phase5_*.yaml of commit e5c2cf5), in run order.
SLOTS_COUNTED = [
    s(O, "hall"), s(O, "ward"), s(O, "reception"), s(O, "hall", world="cabinet_open"),
    s(O, "office", world="cabinet_open"), s(O, "storage", world="cabinet_open"), s(C, "ward"), s(C, "office"),
    s(C, "hall", world="cabinet_open"), s(C, "reception", world="cabinet_open"),
    count("office", 3, True, "hall"), count("office", 2, True, "reception"), count("office", 1, False, "storage"),
    count("reception", 3, True, "ward"), count("reception", 2, True, "hall"), count("reception", 1, False, "reception"),
    count("storage", 2, False, "office"), count("storage", 1, False, "hall"), count("ward", 2, True, "ward"),
    count("ward", 1, False, "office"),
    cond("cond-toolbox", True, "hall"), cond("cond-toolbox-office", True, "ward"), cond("cond-toolbox-ward", True, "hall"),
    cond("cond-toolbox-reception", True, "storage"), cond("cond-toolbox-office-ward", True, "office"),
    cond("cond-toolbox", False, "storage"), cond("cond-toolbox-office", False, "office"),
    cond("cond-toolbox-ward", False, "hall"), cond("cond-toolbox-storage-ward", False, "reception"),
    cond("cond-toolbox-reception-storage", False, "reception"),
    find("storage", "hall"), find("storage", "office"), find("office", "hall"), find("office", "ward"),
    find("reception", "storage"), find("reception", "office"), find("reception", "reception"), find("ward", "reception"),
    find("ward", "ward"), find(None, "hall"),
]
SLOTS_DEV = [
    s(O, "office"), s(O, "ward", world="cabinet_open"), s(C, "hall"),
    count("office", 2, True, "hall"), count("ward", 1, False, "reception"), count("storage", 3, True, "hall"),
    cond("cond-toolbox-ward", True, "hall"), cond("cond-toolbox-reception-storage", False, "office"),
    cond("cond-toolbox-office-ward", True, "reception"),
    find("ward", "hall"), find("office", "storage"), find("reception", "ward"),
]


def ok_seeds() -> dict[int, dict[str, Any]]:
    return {e["seed"]: e for e in json.loads(SEEDS_JSON.read_text())["ok"] if e.get("ok") is True}


def spot_seeds() -> set[int]:
    """The live spot-check / G9 seeds: every `--seeds a,b,...` of hpc/plans/p5_spot.sh."""
    return {int(x) for m in re.findall(r"--seeds ([\d,]+)", SPOT_PLAN.read_text()) for x in m.split(",")}


def toolbox_conds() -> list[tuple[str, str, str]]:
    """(task id, first room, second room) of every toolbox conditional, in dev.yaml then phase5_tasks.yaml order."""
    return [(t.id, t.target["if_not_in"], t.target["else_room"]) for t in all_tasks()
            if t.kind == "conditional" and t.target.get("object") == "toolbox_red_1"]


def resolve_cond(slot: Slot) -> Slot:
    """A conditional slot's answer room = the toolbox's room (first room if right, else second)."""
    first, second = next((a, b) for t, a, b in toolbox_conds() if t == slot["task"])
    return {**slot, "toolbox": first if slot["right"] else second}


def condition(slot: Slot) -> Callable[[dict[str, Any]], bool]:
    if "toolbox" in slot:
        return lambda e: e["toolbox_room"] == slot["toolbox"]
    if "chairs" in slot:
        room, n, sev = slot["chairs"]
        return lambda e: e["chairs_per_room"].get(room, 0) == n and e["spans_2_photos"].get(room, False) == sev
    if slot.get("all_chairs"):
        return lambda e: max(e["chairs_per_room"].values()) == 4
    return lambda e: True


def answer_room(slot: Slot) -> str:
    return slot.get("toolbox") or slot.get("answer") or "office"  # all_chairs: the room is per seed (never kept here)


def substitute(slots: list[Slot], rooms: list[str], pool: dict[int, dict[str, Any]]) -> list[Slot]:
    """Rule 2: replace the Find / Conditional slots the pool cannot fill."""
    slots = [resolve_cond(x) if "right" in x else x for x in slots]
    out = list(slots)
    for kind, is_kind in (("find", lambda x: x["task"].startswith("find-")), ("conditional", lambda x: "right" in x)):
        idx = [i for i, x in enumerate(slots) if is_kind(x)]
        cap = int(CAP_SHARE * len(idx))
        bad = [i for i in idx if (slots[i].get("toolbox") not in rooms and "toolbox" in slots[i])
               or not any(condition(slots[i])(e) for e in pool.values())]
        tally = Counter(answer_room(slots[i]) for i in idx if i not in bad)
        for i in bad:
            room = min((r for r in ROOM_ORDER if r in rooms and tally[r] < cap),
                       key=lambda r: (tally[r], ROOM_ORDER.index(r)), default=None)
            x = slots[i]
            if kind == "find":
                new = s("find-toolbox", x["start"], toolbox=room) if room else {**s("find-beds-where", x["start"]),
                                                                                "answer": "ward"}
            elif room:
                task = next(t for t, a, b in toolbox_conds() if (a if x["right"] else b) == room)
                new = {**cond(task, x["right"], x["start"]), "toolbox": room}
            else:
                new = {**cond("cond-beds-ward-office" if x["right"] else "cond-beds-ward", x["right"], x["start"]),
                       "answer": "ward"}
            tally[answer_room(new)] += 1
            out[i] = new
    return out


def pick(rooms: list[str]) -> dict[str, list[dict[str, Any]]]:
    pool = ok_seeds()
    lists = {"c": substitute(SLOTS_COUNTED, rooms, pool), "d": substitute(SLOTS_DEV, rooms, pool)}
    spot = spot_seeds()
    order = [x for x in random.Random("115-008-layout").sample(sorted(pool), len(pool)) if x not in spot]
    cands = {(k, i): [x for x in order if condition(sl)(pool[x])] for k, ls in lists.items() for i, sl in enumerate(ls)}
    used: set[int] = set()
    seed_of: dict[tuple[str, int], int] = {}
    for key in sorted(cands, key=lambda t: (len(cands[t]), t[0] != "c", t[1])):
        seed_of[key] = next(x for x in cands[key] if x not in used)
        used.add(seed_of[key])
    out = {}
    for k, name in (("c", "counted"), ("d", "dev")):
        ss = random.Random(f"115-008-{name}").sample(range(1, 2**31 - 1), len(lists[k]))
        rows = []
        for i, sl in enumerate(lists[k]):
            e = {"run_id": f"p5{k}{i + 1:02d}", "task_id": sl["task"], "start_place": sl["start"], "seed": ss[i],
                 "layout_seed": seed_of[(k, i)], **({"world": sl["world"]} if sl["world"] else {})}
            rec = pool[e["layout_seed"]]
            com = ("several photos" if rec["spans_2_photos"][sl["chairs"][0]] else "one photo") if "chairs" in sl \
                else ("first room right" if sl["right"] else "first room wrong") if "right" in sl else ""
            rows.append((e, com))
        out[name] = rows
    return out


def line(e: dict[str, Any], com: str) -> str:
    w = f", world: {e['world']}" if "world" in e else ""
    return (f"- {{run_id: {e['run_id']}, task_id: {e['task_id'] + ',':31s} start_place: {e['start_place'] + ',':10s} "
            f"seed: {e['seed']:10d}, layout_seed: {e['layout_seed']:3d}{w}}}" + (f"  # {com}" if com else ""))


HEAD = {"counted": """\
# Task 008 counted set: 40 runs = 10 per type (check_state, count, conditional, find) on seeded layouts, each used once,
# written before any Phase 5 VLM output. Entry {run_id, task_id, start_place, seed, layout_seed} (+ `world:
# cabinet_open`; absent = closed). task_id resolves in tasks/dev.yaml, then tasks/phase5_tasks.yaml. True answers are
# never written here: agent/truth.run_truth derives them from truth_for(layout_seed, world).
# Checked by tests/test_phase5_lists.py; gates: probes/phase5_gates.py. Line comments: Count one / several photos
# (phase5_seeds.json spans_2_photos of the counted room); Conditional first room right / wrong.
# Selection rule: written by probes/pick_phase5.py (module doc = the rule) from the committed 2026-11-05 slot plan,
# re-picked once after a recorded deviation (design review: new toolbox placement rule, re-vetted pool, toolbox rooms = G9-passing rooms).
""", "dev": """\
# Task 008 dev set: 12 runs for dev blocks (prompt / packet debugging) on seeded layouts. Never counted. Format and
# selection rule as tasks/phase5_counted.yaml (probes/pick_phase5.py; one layout-seed draw over both lists).
"""}


def write(out_dir: Path, rooms: list[str]) -> dict[str, Path]:
    lists, spot = pick(rooms), sorted(spot_seeds())
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, rows in lists.items():
        p = out_dir / f"phase5_{name}.yaml"
        p.write_text(HEAD[name] + f"# toolbox rooms: {', '.join(rooms)}\n# spot-check seeds excluded: "
                     f"{' '.join(map(str, spot))}\n" + "".join(line(e, c) + "\n" for e, c in rows))
        paths[name] = p
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--toolbox-rooms", required=True, help="comma list: the rooms where the toolbox passed G9")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rooms = a.toolbox_rooms.split(",")
    assert set(rooms) <= set(ROOM_ORDER) and len(rooms) >= 2, f"toolbox rooms {rooms}: need >= 2 of {ROOM_ORDER}"
    for p in write(a.out, rooms).values():
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
