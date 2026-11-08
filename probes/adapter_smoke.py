#!/usr/bin/env python3
"""Live smoke of adapters/turtlebot4.py TB4Adapter (task 003, worker A). Re-runnable.

    source sim/env.sh; python3 probes/adapter_smoke.py --out DIR [--photo-place ward] [--wall 1.45,-1.5]

With the clinic sim + AMCL + Nav2 running (sim/launch_clinic.sh localization:=true nav2:=true), holding the gz lock.
1. go_to(place x, y, yaw, 120 s sim) for each of the 5 places; GT position / yaw error (Gazebo), sim / wall time; where().
2. At --photo-place: turn_to(place yaw + k*90 deg) + photo() for k = 0..3; crop shape == (480 - top - bottom, 640);
   labels in the cropped seg vs truth.json objects of that room (and of other rooms).
3. go_to(--wall: a point inside a wall) -> expect "blocked" within the limit; then go_to(hall) -> expect "reached".
Writes DIR/adapter_smoke.json and DIR/photos/<k>_rgb.png, <k>_seg.png.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.turtlebot4 import SimTruth, TB4Adapter  # noqa: E402

PLACES = ROOT / "sim/worlds/clinic/places.yaml"
TRUTH = ROOT / "sim/worlds/clinic/generated/truth.json"
LIMIT_S = 120.0  # contract section 1: go_to sim limit (G1)
ORDER = ["hall", "office", "reception", "storage", "ward"]


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def go(ad: TB4Adapter, truth: SimTruth, name: str, x: float, y: float, yaw: float | None) -> dict:
    r = ad.go_to(x, y, yaw, LIMIT_S)
    gt = truth.gt_pose()
    rec = {"target": [x, y, yaw], "status": r.status, "nav_status": r.nav_status, "sim_s": r.sim_s, "wall_s": r.wall_s,
           "gt_pose": list(gt), "gt_err_m": math.dist(gt[:2], (x, y)),
           "gt_yaw_err_deg": None if yaw is None else math.degrees(wrap(gt[2] - yaw)),
           "loc_pose": list(ad.pose()), "where": ad.where()}
    print(f"{name:10s} {r.status:8s} {r.nav_status:10s} sim {r.sim_s:6.1f} s  wall {r.wall_s:6.1f} s  GT err "
          f"{rec['gt_err_m']:.3f} m  yaw {rec['gt_yaw_err_deg'] if yaw is None else round(rec['gt_yaw_err_deg'], 2)} deg"
          f"  where {rec['where']}", flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--photo-place", default="ward")
    ap.add_argument("--wall", default="1.45,-1.5", help="x,y inside a wall (default: hall/storage wall, x 0..2.9 middle)")
    a = ap.parse_args()
    places = yaml.safe_load(PLACES.read_text())["places"]
    objs = json.loads(TRUTH.read_text())["objects"]
    (a.out / "photos").mkdir(parents=True, exist_ok=True)
    res: dict = {"limit_sim_s": LIMIT_S, "places": {}, "photos": [], "blocked": {}}
    w0 = time.monotonic()
    ad, truth = TB4Adapter(), SimTruth()
    try:
        res["start"] = {"loc_pose": list(ad.pose()), "gt_pose": list(truth.gt_pose()), "where": ad.where(),
                        "sim_time": ad.sim_time()}
        for name in ORDER:
            p = places[name]
            res["places"][name] = go(ad, truth, name, p["x"], p["y"], p["yaw"])

        pp = places[a.photo_place]
        if ad.where() != a.photo_place:
            res["places"][a.photo_place + "_again"] = go(ad, truth, a.photo_place, pp["x"], pp["y"], pp["yaw"])
        room_labels = {o["label"]: o["id"] for o in objs if o["room"] == a.photo_place}
        other_labels = {o["label"]: o["id"] for o in objs if o["room"] != a.photo_place}
        for k in range(4):
            target = wrap(pp["yaw"] + k * math.pi / 2)
            t0 = time.monotonic()
            err = ad.turn_to(target)
            f = ad.photo()
            gt = truth.gt_pose()
            labs, cnt = np.unique(f.seg, return_counts=True)
            px = dict(zip(labs.tolist(), cnt.tolist()))
            rec = {"k": k, "target_deg": math.degrees(target), "turn_err_loc_deg": math.degrees(err),
                   "gt_yaw_err_deg": math.degrees(wrap(gt[2] - target)), "rgb_size_wh": list(f.rgb.size),
                   "seg_shape": list(f.seg.shape), "seg_dtype": str(f.seg.dtype), "meta": f.meta,
                   "room_object_px": {room_labels[l]: n for l, n in px.items() if l in room_labels},
                   "other_room_object_px": {other_labels[l]: n for l, n in px.items() if l in other_labels},
                   "wall_s": time.monotonic() - t0}
            rec["shape_ok"] = rec["seg_shape"] == [480 - f.meta["crop"]["top_px"] - f.meta["crop"]["bottom_px"], 640] \
                and rec["rgb_size_wh"] == [640, rec["seg_shape"][0]]
            f.rgb.save(a.out / "photos" / f"{k}_rgb.png")
            cv2.imwrite(str(a.out / "photos" / f"{k}_seg.png"), f.seg)
            res["photos"].append(rec)
            print(f"photo {k}: target {rec['target_deg']:7.1f} deg  loc err {rec['turn_err_loc_deg']:+.3f}  GT err "
                  f"{rec['gt_yaw_err_deg']:+.3f} deg  shape {rec['seg_shape']} ok {rec['shape_ok']}  room objs "
                  f"{rec['room_object_px']}  other {rec['other_room_object_px']}  {rec['wall_s']:.1f} s wall", flush=True)

        wx, wy = (float(v) for v in a.wall.split(","))
        res["blocked"]["wall"] = go(ad, truth, "wall", wx, wy, 0.0)
        res["blocked"]["hall_after"] = go(ad, truth, "hall", places["hall"]["x"], places["hall"]["y"], places["hall"]["yaw"])
        ad.stop()
        res["stop_ok"] = True
    finally:
        ad.close()
    pl = res["places"]
    res["summary"] = {
        "all_places_reached": all(pl[n]["status"] == "reached" for n in ORDER),
        "where_matches_place": all(pl[n]["where"] == n for n in ORDER),
        "max_gt_err_m": max(pl[n]["gt_err_m"] for n in ORDER),
        "photos_shape_ok": all(p["shape_ok"] for p in res["photos"]),
        "photos_with_room_objects": sum(bool(p["room_object_px"]) for p in res["photos"]),
        "room_objects_seen": sorted({i for p in res["photos"] for i in p["room_object_px"]}),
        "wall_blocked": res["blocked"]["wall"]["status"] == "blocked",
        "wall_sim_s": res["blocked"]["wall"]["sim_s"], "wall_wall_s": res["blocked"]["wall"]["wall_s"],
        "hall_after_reached": res["blocked"]["hall_after"]["status"] == "reached",
        "total_wall_s": time.monotonic() - w0}
    (a.out / "adapter_smoke.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["summary"], indent=2))
    s = res["summary"]
    ok = s["all_places_reached"] and s["where_matches_place"] and s["photos_shape_ok"] and s["wall_blocked"] \
        and s["hall_after_reached"]
    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
