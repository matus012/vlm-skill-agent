#!/usr/bin/env python3
"""H-P1 contact sanity control (design ruling on "collision" = a MuJoCo contact): a walk deliberately into a chair must
register on /g1/contacts.

    /opt/venv/bin/python humanoid/probes/contact_control.py --out D [--layout-seed 474]   # live, one sim (in_sim)

The seed's movables placed (G1Scene.place, robot at the hall spawn), the chair nearest to any place picked, go_to that
place facing the chair (positioning, avoidance on), then raw /cmd_vel straight ahead (vx = the adapter's recentre_min_v,
no avoidance, no steering) until the chair shows in /g1/contacts, the pelvis stops closing on the chair (drive's stall
rule: STALL_M in STALL_S sim) or the robot falls. pass = contacts_mocap > 0 and the chair among the touched bodies.
Out: D/contact_control_seed<S>.json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from humanoid.probes import hp1_reach as hp1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layout-seed", type=int, default=474)
    ap.add_argument("--rehearsal", action="store_true")
    a = ap.parse_args()
    if a.layout_seed not in hp1.dev_seeds():
        raise SystemExit(f"seed {a.layout_seed} is not a vetted dev seed")
    from adapters.robots import make
    from humanoid.drive import CFG, STALL_M, STALL_S, RobotFell
    from humanoid.probes.hp2_photos import frozen_want
    from probes.spot_check import want_for
    places = hp1.mwc.load_places(hp1.pg.PLACES_YAML)
    seeded = want_for(a.layout_seed, "closed")
    want = {**frozen_want("closed"), **seeded}
    chair, place = min(((c, p) for c in seeded if c.startswith("chair") for p in places),
                       key=lambda cp: math.dist(seeded[cp[0]][:2], places[cp[1]][:2]))
    cx, cy = seeded[chair][:2]
    px, py = places[place][:2]
    v = CFG["adapter"]["recentre_min_v"]
    body = "clinic/" + chair
    rb = make("g1")
    watch = hp1.ContactWatch()
    res: dict = {"seed": a.layout_seed, "chair": chair, "chair_xy": [cx, cy], "from_place": place, "vx": v}
    try:
        ad = rb.ad
        log: dict = {}
        res["place_result"] = rb.run_kw["place_objects"](want, log)
        assert res["place_result"] is None, res["place_result"]
        g = ad.go_to(px, py, math.atan2(cy - py, cx - px), hp1.mwc.NAV_SIM_LIMIT_S)
        res["positioning"] = {"status": g.status, "nav_status": g.nav_status, "sim_s": g.sim_s, "gt": list(rb.gt_pose())}
        dr = ad.drive
        t0 = ad.sim_time()
        watch.wait_past(t0)
        best, t_best, why = math.inf, t0, None
        try:
            while why is None:
                dr.send(v, 0.0, 0.0)
                t = dr._tick()  # next odom sample (raises RobotFell below FALL_Z_M)
                dist = math.dist((dr.pose.x, dr.pose.y), (cx, cy))
                if dist < best - STALL_M:
                    best, t_best = dist, t
                if any(c["other_body"] == body and c["t_sim"] > t0 for m in watch.msgs[-3:] for c in m["last"]):
                    why = "contact"
                elif t - t_best > STALL_S:
                    why = "stalled"
        except RobotFell as e:
            why = f"fell: {e}"
        dr.send(0.0, 0.0, 0.0)
        t1 = ad.sim_time()
        watch.wait_past(t1)
        res.update(walk_end=why, walk_sim_s=t1 - t0, gt_end=list(rb.gt_pose()), **hp1.contact_delta(watch.msgs, t0, t1))
        res["pass"] = res["contacts_mocap"] > 0 and body in res["contacts_seen"]
    finally:
        watch.close()
        rb.close()
    print(f"contact control seed {a.layout_seed}: {chair} from {place}: {res['walk_end']}, mocap {res['contacts_mocap']} "
          f"static {res['contacts_static']} seen {res['contacts_seen']} -> {'PASS' if res['pass'] else 'FAIL'}", flush=True)
    return 0 if hp1.pg.write(a.out, f"contact_control_seed{a.layout_seed}", res, a.rehearsal)["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
