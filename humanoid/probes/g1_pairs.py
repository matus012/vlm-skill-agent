#!/usr/bin/env python3
"""H-P3 step 1 (task 012 design section 5): the (class, room[, state]) target pairs of the test list, AGGREGATES ONLY.

    /opt/venv/bin/python humanoid/probes/g1_pairs.py --out pairs.json [--list tasks/phase8_pass2.yaml]

Read by script, never shown: per entry, the task's target objects (probes/build_phase8.target_ids) with their spec
room; state only for check_state (the cabinet's state). Printed and written: the list sha256, the entry count and each
pair with its count. Never a prompt, seed, run id or anything per entry. By construction (filter rule a) the pairs are a
subset of build_phase8.G9_ROOMS plus chair rooms; a pair outside that is reported (`outside_g9_rooms`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from probes import build_phase8 as bp  # noqa: E402

LIST = ROOT / "tasks/phase8_pass2.yaml"


def tasks_of(list_path: Path) -> Path:
    return list_path.with_name(list_path.stem + "_tasks.yaml")


def pairs(entries: list[dict], tasks: dict[str, dict]) -> Counter:
    c: Counter = Counter()
    for e in entries:
        t, spec = tasks[e["task_id"]], e["layout_spec"]
        for oid in bp.target_ids(t, spec):
            o = spec["objects"][oid]
            c[(bp.cls_of(oid), o["room"], o.get("state") if t["kind"] == "check_state" else None)] += 1
    return c


def outside(c: Counter) -> int:
    return sum(n for (cls, room, _), n in c.items() if cls != "chair" and room not in bp.G9_ROOMS.get(cls, set()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--list", type=Path, default=LIST)
    a = ap.parse_args()
    entries = yaml.safe_load(a.list.read_text())
    tasks = {t["id"]: t for t in yaml.safe_load(tasks_of(a.list).read_text())}
    c = pairs(entries, tasks)
    res = {"list": a.list.name, "list_sha256": bp.sha256(a.list), "entries": len(entries),
           "pairs": [{"class": k[0], "room": k[1], "state": k[2], "n": n} for k, n in sorted(c.items(), key=str)],
           "outside_g9_rooms": outside(c)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"{a.list.name} sha256 {res['list_sha256'][:12]}: {len(entries)} entries, {len(c)} target pairs, "
          f"{res['outside_g9_rooms']} target objects outside G9_ROOMS + chair rooms")
    for p in res["pairs"]:
        print(f"  {p['class']:8s} {p['room']:10s} {p['state'] or '-':7s} {p['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
