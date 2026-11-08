#!/usr/bin/env python3
"""H-P1 reach on the G1 (task 012 design section 5; design notes a recorded deviation). Port of probes/phase1_gates.py
gate_g1 for one room per sim; GT from G1SimTruth, moves through G1Adapter.go_to (adapters.robots.make("g1")).

    /opt/venv/bin/python humanoid/probes/hp1_reach.py --room R --out D [--layout-seed S] [--collision C]  # live, one sim
    /opt/venv/bin/python humanoid/probes/hp1_reach.py --merge D --seeds S1,S2,S3            # no sim -> D/hp1.json

Live: 10 counted tries into R, starts cycling the other places hall first (phase1_gates g1_starts / g1_plan_step,
imported unchanged), positioning moves uncounted. First G1Scene.place of the frozen closed layout, with --layout-seed S
the movables at seeded.placements(S) (a
vetted DEV seed: an ok seed of tasks/phase5_seeds.json that build_phase8 filter rule c keeps out of every test list,
never a counted Phase 5 seed; anything else is refused). A try passes iff reached, GT <= 0.30 m of the place, <= 120 s
sim, valid start (GT <= 0.30 m of its from-place), min base z >= FALL_Z_M over its own pose history (no fall), not
stalled, and no collision. --collision (recorded as collision_def; design ruling, the default may flip):
  contact (default)  any robot-clinic MuJoCo contact during the try: contacts_mocap + contacts_static > 0 (g1_node's
                     /g1/contacts counters, humanoid/contacts.py: floor excluded, counted per physics step);
  contact_mocap      a placed movable touched: contacts_mocap > 0;
  clearance          the old proxy (layouts only): the pelvis path within CONTACT_M of a placed movable's footprint.
Every try reports all of them: contacts_mocap, contacts_static, contacts_seen (bodies in the messages' `last` lists),
clearance_m (min mj_geomDistance pelvis -> non-floor geom over the try, no bar) and footprint_clearance_m. A fall
resets the robot through the placer (hall spawn) and the tries go on (it is a fail, never re-counted).
Evidence per try: every /g1/odom sample of the move (G1Trace's gt_history.jsonl) appended to
D/hp1_<tag>_<room>_history.jsonl; the try records [i0, i1) into it. Part: D/hp1_<tag>_<room>.json, tag closed|seed<S>.
Merge: closed + every --seeds layout, 4 rooms each; pass iff every set is 40/40.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "probes"), str(ROOT / "sim/worlds/clinic")]
import map_world_check as mwc  # noqa: E402
import phase1_gates as pg  # noqa: E402
import seeded as S  # noqa: E402
from phase1_gates import G1_MAX_ERR_M, G1_MAX_SIM_S, G1_TRIES, ROOMS, g1_plan_step, g1_starts  # noqa: E402

CONTACT_M = 0.35  # design section 5 leg clearance: a footprint + 0.35 m is where the G1's body is (no contact sensor)
MAX_MOVES = 200   # as gate_g1's recovery assert


def dev_seeds() -> set[int]:
    """Ok seeds of tasks/phase5_seeds.json that filter rule c keeps out of every Phase 8 list, minus counted ones."""
    import yaml
    from probes import build_phase8 as bp
    ok = {e["seed"] for e in json.loads((ROOT / "tasks/phase5_seeds.json").read_text())["ok"]}
    counted = {e.get("layout_seed") for e in yaml.safe_load((ROOT / "tasks/phase5_counted.yaml").read_text())}
    return (ok & bp.used_seeds()) - counted


def footprints(seed: int | None) -> list[list[tuple[float, float]]]:
    if seed is None:
        return []
    return [S.footprint(S.CLASS_OF[oid], None, x, y, yaw) for oid, (x, y, yaw) in S.placements(seed).items()]


def point_poly(p: tuple[float, float], poly: list[tuple[float, float]]) -> float:
    return 0.0 if S._inside(p, poly) else min(S._seg_point_dist(p, a, b) for a, b in S._edges(poly))


def clearance(rows: list[dict], foots: list) -> float | None:
    """Closest approach (m) of the pelvis path to any footprint; None without footprints or samples."""
    d = [point_poly((r["x"], r["y"]), f) for r in rows for f in foots]
    return min(d) if d else None


class ContactWatch:
    """Own rclpy context + node + executor thread on /g1/contacts (as G1SimTruth); keeps every message."""

    def __init__(self) -> None:
        import threading

        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from std_msgs.msg import String
        self.ctx = rclpy.Context()
        self.ctx.init()
        self.node = Node("g1_contact_watch", context=self.ctx)
        self.msgs: list[dict] = []
        self.node.create_subscription(String, "/g1/contacts", lambda m: self.msgs.append(json.loads(m.data)),
                                      qos_profile_sensor_data)
        self.ex = SingleThreadedExecutor(context=self.ctx)
        self.ex.add_node(self.node)
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _spin(self) -> None:
        while not self.stopping.is_set():
            self.ex.spin_once(timeout_sec=0.1)

    def wait_past(self, t: float) -> None:
        """Until a message with t_sim >= t arrived; raises after ODOM_HANG_S wall (node dead or not publishing)."""
        import time

        from humanoid.drive import ODOM_HANG_S
        w0 = time.monotonic()
        while not self.msgs or self.msgs[-1]["t_sim"] < t:
            assert time.monotonic() - w0 < ODOM_HANG_S, f"no /g1/contacts at t_sim >= {t} in {ODOM_HANG_S} s wall"
            time.sleep(0.02)

    def close(self) -> None:
        self.stopping.set()
        self.thread.join()
        self.ex.shutdown()
        self.node.destroy_node()
        self.ctx.try_shutdown()


def contact_delta(msgs: list[dict], t0: float, t1: float) -> dict[str, Any]:
    """Counters' delta and min clearance over (t0, t1] from /g1/contacts messages: base = the last message with
    t_sim <= t0, end = the first with t_sim >= t1 (so up to one 0.1 s message period of overlap at each end)."""
    base = [x for x in msgs if x["t_sim"] <= t0][-1]
    end = next(x for x in msgs if x["t_sim"] >= t1)
    win = [x for x in msgs if base["t_sim"] < x["t_sim"] <= end["t_sim"]]
    cl = [x["clearance_m"] for x in win if x["clearance_m"] is not None]
    seen = sorted({c["other_body"] for x in win for c in x["last"] if base["t_sim"] < c["t_sim"] <= end["t_sim"]})
    return {"contacts_mocap": end["n_mocap_total"] - base["n_mocap_total"],
            "contacts_static": end["n_static_total"] - base["n_static_total"], "contacts_seen": seen,
            "clearance_m": min(cl) if cl else None}


COLLISION_DEFS = ("contact", "contact_mocap", "clearance")


def collided(rec: dict[str, Any], how: str) -> bool:
    if how == "contact":
        return rec["contacts_mocap"] + rec["contacts_static"] > 0
    if how == "contact_mocap":
        return rec["contacts_mocap"] > 0
    assert how == "clearance", how
    return rec["footprint_clearance_m"] is not None and rec["footprint_clearance_m"] < CONTACT_M


def try_pass(rec: dict[str, Any], fall_z: float) -> bool:
    return bool(rec["valid_start"] and rec["status"] == "reached" and rec["err_m"] <= G1_MAX_ERR_M
                and rec["sim_s"] <= G1_MAX_SIM_S and rec["min_z"] is not None and rec["min_z"] >= fall_z
                and not rec["stalled"] and not rec["collision"])


def tag_of(seed: int | None) -> str:
    return "closed" if seed is None else f"seed{seed}"


def run_room(rb: Any, room: str, places: dict, out: Path, seed: int | None, rehearsal: bool,  # noqa: ANN401
             watch: ContactWatch, how: str) -> dict:
    from humanoid.drive import FALL_Z_M, RobotFell  # ROS: live only
    from humanoid.probes.hp2_photos import frozen_want
    from probes.spot_check import want_for
    tag = tag_of(seed)
    ad = rb.ad
    place, trace_of = rb.run_kw["place_objects"], rb.run_kw["trace_factory"]
    want = {**frozen_want("closed"), **(want_for(seed, "closed") if seed is not None else {})}
    foots = footprints(seed)
    hist = out / f"hp1_{tag}_{room}_history.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    hist.write_text("")
    n_rows, tries, moves, resets = 0, [], [], []

    def reset(why: str) -> str | None:  # robot back to the hall spawn (and the layout again) through the placer
        log: dict = {}
        res = place(want, log)
        resets.append({"why": why, "place_result": res, "log": log})
        return res
    if (why := reset("layout")) is not None:
        return pg.write(out, f"hp1_{tag}_{room}", {"room": room, "seed": seed, "error": f"placement: {why}",
                                                   "pass": False}, rehearsal)

    def move(target: str, limit: float) -> dict:
        nonlocal n_rows
        d = out / "tries" / tag / room / f"{len(tries) + len(moves):03d}-{target}"
        gt0 = rb.gt_pose()
        watch.wait_past(ad.sim_time())
        t0 = watch.msgs[-1]["t_sim"]  # base = a received /g1/contacts message: a probe's first move may predate any
        #                               message (145170 sim 0: IndexError in contact_delta), so t0 is taken from one
        trace = trace_of(rb.gt_pose, d / "gt_trace.jsonl").start()
        fell = None
        try:
            g = ad.go_to(*places[target], limit)
            r = {"status": g.status, "nav_status": g.nav_status, "sim_s": g.sim_s, "wall_s": g.wall_s}
        except RobotFell as e:
            fell, r = str(e), {"status": "fell", "nav_status": "FELL", "sim_s": math.inf, "wall_s": None}
        finally:
            trace.stop()
        rows = [json.loads(x) for x in (d / "gt_history.jsonl").read_text().splitlines()] \
            if (d / "gt_history.jsonl").exists() else []
        with hist.open("a") as f:
            f.writelines(json.dumps(x) + "\n" for x in rows)
        i0, n_rows = n_rows, n_rows + len(rows)
        gt1 = rb.gt_pose()
        t1 = ad.sim_time()
        watch.wait_past(t1)
        rec = {"to": target, **r, "fell": fell, "gt_start": list(gt0), "gt_end": list(gt1),
               "err_m": math.dist(gt1[:2], places[target][:2]), "history": [i0, n_rows],
               "min_z": min((x["z"] for x in rows), default=None), "stalled": r["nav_status"] == "STALLED",
               "t_sim": [t0, t1], **contact_delta(watch.msgs, t0, t1), "footprint_clearance_m": clearance(rows, foots)}
        return {**rec, "collision": collided(rec, how)}

    need = Counter((s, room) for s in g1_starts(room))
    cur: str | None = None
    while sum(need.values()):
        assert len(moves) + len(tries) < MAX_MOVES, "cannot recover"
        if cur is None:  # start / after a failed move: to the hall, uncounted
            rec = move("hall", mwc.NAV_SIM_LIMIT_S)
            moves.append({"from": None, "recovery": True, **rec})
        else:
            target, counted = g1_plan_step(need, cur)
            rec = {"from": cur, **move(target, G1_MAX_SIM_S if counted else mwc.NAV_SIM_LIMIT_S)}
            if counted:
                need[(cur, target)] -= 1
                rec["start_err_m"] = math.dist(rec["gt_start"][:2], places[cur][:2])
                rec["valid_start"] = rec["start_err_m"] <= G1_MAX_ERR_M
                rec["pass"] = try_pass(rec, FALL_Z_M)
                tries.append(rec)
                print(f"H-P1 {tag} try {len(tries):2d} {cur:9s} -> {target:9s} {rec['status']:8s} {rec['sim_s']:6.1f} s "
                      f"err {rec['err_m']:.3f} m  min z {rec['min_z']}  contacts mocap {rec['contacts_mocap']} "
                      f"static {rec['contacts_static']} {rec['contacts_seen']}  clear {rec['clearance_m']}  "
                      f"{'ok' if rec['pass'] else 'FAIL'}", flush=True)
            else:
                moves.append(rec)
        target = rec["to"]
        if rec["fell"] is not None:  # back on its feet at the hall spawn (not the hall place): recover from there
            reset(f"fell on the way to {target}")
            cur = None
        else:
            cur = target if rec["status"] == "reached" else None
    n = sum(t["pass"] for t in tries)
    return pg.write(out, f"hp1_{tag}_{room}", {
        "room": room, "seed": seed, "tag": tag, "history_file": hist.name, "fall_z_m": FALL_Z_M,
        "collision_def": how, "contact_m": CONTACT_M if how == "clearance" and foots else None, "passed": n, "total": len(tries), "per_room": f"{n}/{G1_TRIES}",
        "falls": sum(t["fell"] is not None for t in tries + moves), "resets": resets,
        "tries": tries, "uncounted_moves": moves, "pass": len(tries) == G1_TRIES and n == G1_TRIES}, rehearsal)


def merge(out: Path, seeds: list[int], rehearsal: bool) -> dict:
    sets, tries, missing, defs = {}, [], [], set()
    for tag in ["closed"] + [tag_of(s) for s in seeds]:
        per = {}
        for room in ROOMS:
            p = out / f"hp1_{tag}_{room}.json"
            if not p.exists():
                missing.append(p.name)
                continue
            part = json.loads(p.read_text())
            per[room] = part.get("per_room", part.get("error"))
            defs.add(part.get("collision_def", "clearance"))
            tries += [{"layout": tag, "room": room, "history_file": part.get("history_file"), **t}
                      for t in part.get("tries", [])]
        n = sum(t["pass"] for t in tries if t["layout"] == tag)
        sets[tag] = {"per_room": per, "passed": n, "pass": len(per) == len(ROOMS) and n == G1_TRIES * len(ROOMS)}
    fails = [t for t in tries if not t["pass"]]
    return pg.write(out, "hp1", {
        "criterion": f"per layout (closed world + {len(seeds)} vetted dev seeds) and room {G1_TRIES} go_to tries "
                     f"(starts cycling the other places, hall first); pass = reached and GT <= {G1_MAX_ERR_M} m and "
                     f"<= {G1_MAX_SIM_S} s sim and valid start and min base z >= fall bound and not stalled and no collision "
                     f"({'/'.join(sorted(defs))}); each layout 40/40",
        "collision_defs": sorted(defs),
        "layouts": sets, "missing_parts": missing, "seeds": seeds,
        "fail_reasons": dict(Counter(k for t in fails for k, bad in (
            ("not_reached", t["status"] != "reached"), ("err", t["err_m"] > G1_MAX_ERR_M), ("fell", t["fell"] is not None),
            ("stalled", t["stalled"]), ("collision", t["collision"]), ("invalid_start", not t["valid_start"]),
            ("sim_s", t["sim_s"] > G1_MAX_SIM_S)) if bad)),
        "tries": tries, "pass": not missing and all(s["pass"] for s in sets.values())}, rehearsal)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", choices=ROOMS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--layout-seed", type=int, default=None)
    ap.add_argument("--merge", type=Path, metavar="D")
    ap.add_argument("--seeds", default="", help="merge: the dev layout seeds")
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--collision", choices=COLLISION_DEFS, default="contact")
    a = ap.parse_args()
    if a.merge:
        return 0 if merge(a.merge, [int(s) for s in a.seeds.split(",") if s], a.rehearsal)["pass"] else 1
    if a.room is None or a.out is None:
        ap.error("live needs --room and --out")
    if a.layout_seed is not None and a.layout_seed not in dev_seeds():
        raise SystemExit(f"seed {a.layout_seed} is not a vetted dev seed (phase5 ok, rule-c excluded, not counted)")
    from adapters.robots import make
    rb = make("g1")
    watch = ContactWatch()
    try:
        r = run_room(rb, a.room, mwc.load_places(pg.PLACES_YAML), a.out, a.layout_seed, a.rehearsal, watch, a.collision)
    finally:
        watch.close()
        rb.close()
    return 0 if r["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
