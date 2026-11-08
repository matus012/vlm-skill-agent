"""Clinic smoke (side lane H, task 003; no gate): through G1Adapter only, walk from the spawn to the hall place, then
to each named room's place, and there take look_around's 4 photos at place yaw + k*90 deg (turn_to + photo, as
agent/skills.py look_around does from a place). Saves RGB, raw label (uint8 truth ids) and a false-colour label PNG
per photo, per-label pixel counts, the per-sample pose history, and sideH-003-clinic-smoke.json to $SIDEH_EVIDENCE.
Needs a running `g1_node.py --clinic sim/worlds/clinic/layout.yaml`.
  SIDEH_EVIDENCE=<dir> humanoid/run_g1.sh humanoid/probes/clinic_smoke.py storage reception
"""
from __future__ import annotations

import json
import math
import sys
import time

import numpy as np
import yaml
from PIL import Image

from humanoid.adapter_g1 import G1Adapter
from humanoid.geometry import wrap
from humanoid.probes.common import EVIDENCE, write_verdict
from humanoid.room import CLINIC_LAYOUT

NAV_TIMEOUT_S = 120.0  # sim seconds, INTERFACES section 1 (the G1 limit)


def false_colour(lab: np.ndarray) -> np.ndarray:
    """Distinct colour per label id for looking at (0 = black); deterministic hash of the id."""
    rng = np.random.default_rng(0).integers(40, 256, size=(256, 3), dtype=np.uint8)
    rng[0] = 0
    return rng[lab]


def main() -> None:
    rooms = sys.argv[1:] or ["storage"]
    places = {n: (p["x"], p["y"], p["yaw"]) for n, p in
              yaml.safe_load((CLINIC_LAYOUT.parent / "places.yaml").read_text())["places"].items()}
    objects = json.loads((CLINIC_LAYOUT.parent / "generated/truth.json").read_text())["objects"]  # instrument side
    names = {o["label"]: o["id"] for o in objects}
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    ad = G1Adapter()
    w_start = time.monotonic()
    out: dict = {"spawn_pose": list(ad.pose()), "spawn_room": ad.where(), "legs": []}
    for target in ["hall", *rooms]:
        x, y, yaw = places[target]
        go = ad.go_to(x, y, yaw, NAV_TIMEOUT_S)
        px, py, pyaw = ad.pose()
        leg = {"place": target, "status": go.status, "nav_status": go.nav_status, "sim_s": go.sim_s,
               "wall_s": go.wall_s, "gt_err_m": math.hypot(px - x, py - y),
               "yaw_err_deg": math.degrees(wrap(pyaw - yaw)), "where": ad.where(), "route": ad.last_route}
        if target != "hall":
            leg["photos"] = []
            for k in range(4):
                err = ad.turn_to(wrap(yaw + k * math.pi / 2))
                f = ad.photo()
                stem = f"{target}_{k + 1}"
                f.rgb.save(EVIDENCE / f"{stem}_rgb.png")
                Image.fromarray(f.seg).save(EVIDENCE / f"{stem}_label.png")
                Image.fromarray(false_colour(f.seg)).save(EVIDENCE / f"{stem}_label_vis.png")
                ids, px_n = np.unique(f.seg, return_counts=True)
                counts = {names.get(int(i), str(int(i))): int(n) for i, n in zip(ids, px_n)}
                room_objs = {o["id"]: counts.get(o["id"], 0) for o in objects if o["room"] == target}
                leg["photos"].append({"n": k + 1, "turn_err_deg": math.degrees(err), "meta": f.meta,
                                      "seg_dtype": str(f.seg.dtype), "rgb_size": list(f.rgb.size),
                                      "room_object_px": room_objs, "all_label_px": counts})
        out["legs"].append(leg)
        print(json.dumps({k: v for k, v in leg.items() if k not in ("route", "photos")}), flush=True)
    ad.stop()
    hist = ad.drive.pose.history
    (EVIDENCE / "sideH-003-pose-history.jsonl").write_text("".join(json.dumps(h) + "\n" for h in hist))
    out.update({"min_base_z": ad.drive.pose.min_z, "total_wall_s": time.monotonic() - w_start,
                "total_sim_s": hist[-1][0] - hist[0][0], "pose_samples": len(hist),
                "note": "pose() is the sim ground-truth pelvis pose (the G1 has no localization), so gt_err_m is the "
                        "pelvis distance to the place point; photos are looked at by the reader (Read tool), not here"})
    write_verdict("003-clinic-smoke", out)
    ad.close()


if __name__ == "__main__":
    main()
