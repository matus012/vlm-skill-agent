#!/usr/bin/env python3
"""R1 instrument checks on the G1 (task 012 design sections 4, 8 R1, 9). No bar unless stated; one sim each (in_sim).

    /opt/venv/bin/python humanoid/probes/r1_checks.py mocap   --out D   # G1Scene.place read-back, synthetic specs
    /opt/venv/bin/python humanoid/probes/r1_checks.py drift   --out D --places hall,ward
    /opt/venv/bin/python humanoid/probes/r1_checks.py blocked --out D   # SIM_WORLD blocked_<room>, 3 runtime runs
    /opt/venv/bin/python humanoid/probes/r1_checks.py logs    --run RUN   # node log advance (FastDDS hang check)

mocap: every spec of tasks/phase8_dev.yaml + phase8_regress.yaml realizing in SIM_WORLD, placed with the robot's placer
  (run_task.layout_plan targets, as a run); records why (None = every body + robot_z landed), the place log and the GT
  robot pose after it (the hall reset); then the frozen layout again. D/r1_mocap_<world>.json, pass iff all landed.
drift: per place, go_to, then TURNS raw Drive.turn_to (no recentre: "before") and, after a fresh go_to, TURNS adapter
  turn_to (recentre on: "after"), headings place yaw + k * 90 deg; GT xy offset from the pre-turn point after every turn
  and the net offset per 4-turn set. D/r1_drift_<places>.json.
blocked: probes/phase6_pre.py's P1 rule (p1_script, p1_problems, unchanged) for BLOCKED_TRIES runs from the first
  reachable starts: navigate_to(<blocked room>) -> 'failed: blocked' with nav_sim_s <= 120 s, GT not inside, then
  navigate_to(hall) -> 'reached'; a fall is a crash, which fails. D/r1_blocked.json, pass iff all pass.
logs: per s<i>/launch*.log the "sim <t> s" lines' wall stamps (rclpy log prefix): count, max wall gap, restarts.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "probes")]

TURNS = 20         # design section 8 R1: 20 turns x 5 places, before / after recentre
BLOCKED_TRIES = 3  # design section 4: 3 tries of navigate_to storage in blocked_storage
LISTS = ("tasks/phase8_dev.yaml", "tasks/phase8_regress.yaml")


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def mocap(rb: Any, out: Path) -> dict:  # noqa: ANN401
    import yaml

    import phase1_gates as pg
    from adapters.run_task import layout_plan, sim_world
    from agent.truth import load_truth, world_truth_path
    sys.path.insert(0, str(ROOT / "sim/worlds/clinic"))
    import realize as rz
    world = sim_world()
    frozen = load_truth(world_truth_path(world))
    specs = {}
    for f in LISTS:
        for e in yaml.safe_load((ROOT / f).read_text()):
            if e["world"] == world:
                specs.setdefault(rz.spec_sha256(e["layout_spec"]), e["layout_spec"])
    recs = []
    for sha, spec in specs.items():
        want = layout_plan(world, frozen, None, spec)[0]
        log: dict = {}
        why = rb.run_kw["place_objects"](want, log)
        gt = rb.gt_pose()
        recs.append({"spec_sha256": sha[:12], "bodies": len(want), "why": why, "log": log, "gt_after": list(gt),
                     "robot_from_hall_m": math.dist(gt[:2], (0.0, 0.0))})
        print(f"mocap {sha[:12]} {world}: {'landed' if why is None else why}", flush=True)
    from humanoid.probes.hp2_photos import frozen_want
    restore = rb.run_kw["place_objects"](frozen_want(world), {})  # the frozen world again for what runs next
    return pg.write(out, f"r1_mocap_{world}", {"world": world, "lists": LISTS, "specs": len(recs), "placements": recs,
                                               "frozen_restore": restore,
                                               "pass": bool(recs) and all(r["why"] is None for r in recs) and restore is None},
                    True)


def drift(rb: Any, out: Path, names: list[str]) -> dict:  # noqa: ANN401
    import map_world_check as mwc
    import phase1_gates as pg
    places = mwc.load_places(pg.PLACES_YAML)
    ad, res = rb.ad, {}
    for name in names:
        x, y, yaw = places[name]
        res[name] = {}
        for mode, turn in (("raw", lambda t: ad.drive.turn_to(t, timeout_s=30.0)), ("adapter", ad.turn_to)):
            go = ad.go_to(x, y, yaw, pg.G1_MAX_SIM_S)
            g0, rows = rb.gt_pose(), []
            for k in range(TURNS):
                turn(wrap(yaw + k * math.pi / 2))
                g = rb.gt_pose()
                rows.append({"k": k, "gt": list(g), "offset_m": math.dist(g[:2], g0[:2])})
            sets = [math.dist(rows[i + 3]["gt"][:2], (rows[i - 1]["gt"] if i else list(g0))[:2]) for i in range(0, TURNS, 4)]
            res[name][mode] = {"go_to": go.status, "gt_start": list(g0), "turns": rows, "net_drift_m_per_set": sets,
                               "max_offset_m": max(r["offset_m"] for r in rows), "max_set_drift_m": max(sets)}
            print(f"drift {name} {mode}: max offset {res[name][mode]['max_offset_m']:.3f} m, "
                  f"max 4-turn set {max(sets):.3f} m", flush=True)
    return pg.write(out, "r1_drift_" + "-".join(names), {"turns_per_mode": TURNS, "places": res, "pass": None}, True)


def blocked(rb: Any, out: Path) -> dict:  # noqa: ANN401
    import map_world_check as mwc
    import phase1_gates as pg1
    import phase2_gates as pg2
    import phase6_pre as p6
    from adapters.run_task import run_one, sim_world
    from agent.planners import ScriptedPlanner
    from agent.truth import load_tasks, load_truth, world_truth_path
    world = sim_world()
    truth = load_truth(world_truth_path(world))
    room = p6.blocked_room(truth, world)
    rect, places = p6.room_rect(truth, room), mwc.load_places(pg1.PLACES_YAML)
    task = {t.id: t for t in load_tasks()}[p6.P1_TASK]
    dirs = []
    for k, start in enumerate(p6.p1_starts(p6.reachable(places, room))[:BLOCKED_TRIES]):
        d = out / "blocked" / f"{k:02d}-{start}"
        run_one(rb.ad, rb.gt_pose, task, ScriptedPlanner(p6.p1_script(room), p6.P1_ID), d, start_place=start,
                extra_meta=rb.scene_meta or None, **rb.run_kw)
        dirs.append(d)
    v = p6.p1_verdict(dirs, world, room, rect)
    v["pass"] = len(v["tries"]) == BLOCKED_TRIES and all(t["pass"] for t in v["tries"])
    return pg2.write(out, "r1_blocked", {**v, "tries_wanted": BLOCKED_TRIES}, True)


LOG_RE = re.compile(r"\[(\d+\.\d+)\] \[g1_sim\]: sim ([0-9.]+) s, RTF ([0-9.]+)")


def log_advance(text: str) -> dict:
    rows = [(float(w), float(t), float(r)) for w, t, r in LOG_RE.findall(text)]
    gaps = [b[0] - a[0] for a, b in zip(rows, rows[1:])]
    return {"lines": len(rows), "max_wall_gap_s": max(gaps, default=None),
            "last_sim_s": rows[-1][1] if rows else None, "last_rtf": rows[-1][2] if rows else None}


def logs(run: Path) -> int:
    res = {}
    for s in sorted(run.glob("s[0-9]*")):
        res[s.name] = {p.name: log_advance(p.read_text(errors="replace")) for p in sorted(s.glob("launch*.log"))}
        res[s.name]["restarts"] = len((s / "restarts.txt").read_text().splitlines()) if (s / "restarts.txt").exists() else 0
    (run / "r1_logs.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["mocap", "drift", "blocked", "logs"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--places", default="hall,ward,office,reception,storage")
    ap.add_argument("--run", type=Path)
    a = ap.parse_args()
    if a.cmd == "logs":
        return logs(a.run)
    from adapters.robots import make
    rb = make("g1")
    try:
        if a.cmd == "mocap":
            r = mocap(rb, a.out)
        elif a.cmd == "drift":
            r = drift(rb, a.out, a.places.split(","))
        else:
            r = blocked(rb, a.out)
    finally:
        rb.close()
    return 0 if r["pass"] is not False else 1


if __name__ == "__main__":
    raise SystemExit(main())
