#!/usr/bin/env python3
"""Task 002 gate runner: one command per gate, each writes `<out>/<gate>.json` (verdict + numbers) and its raw data.

Sim gates need the clinic sim with AMCL + Nav2 on the saved map (sim/launch_clinic.sh localization:=true nav2:=true)
and `source sim/env.sh` with the sim's ROS_DOMAIN_ID / GZ_PARTITION:
    python3 probes/phase1_gates.py g5      --out DIR [--rehearsal]   # truth.json regen + live truth_check
    python3 probes/phase1_gates.py g8      --out DIR [--rehearsal]   # map = world (map_world_check.py) at all 5 places
    python3 probes/phase1_gates.py photos  --out DIR [--rehearsal]   # navigate + photos.py at the 5 places
    python3 probes/phase1_gates.py checks  --out DIR [--rehearsal]   # G2 / G3 / G4 / G7 from DIR/photos (no sim)
    python3 probes/phase1_gates.py g6      --out DIR [--rehearsal]   # contact sheets from DIR/photos (no sim; look!)
    python3 probes/phase1_gates.py g1      --out DIR [--rehearsal] [--rooms ward,office]  # 40 NavigateToPose tries
        (only the 4-room run writes g1.json and can pass G1; a --rooms subset writes g1_<rooms>.json, pass false)
    python3 probes/phase1_gates.py all     --out DIR [--rehearsal]   # all of the above, in this order
    python3 probes/phase1_gates.py g9photos --out DIR --variant closed [--truth T] [--poses name:x,y,yawdeg ...]
`--rehearsal` tags every JSON as a rehearsal (not the counted run).

Thresholds are task 002's gate text (002-phase1-clinic-world.md -> Gates); G2/G3/G4/G7 numbers live in phase1_checks.py.
map -> world: identity, proven by G8 (T_WORLD_FROM_MAP below; change it only with a measured G8 transform).
"""
from __future__ import annotations

import argparse
import datetime as dt
import filecmp
import json
import math
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import map_world_check as mwc
import phase1_checks as pc

ROOT = Path(__file__).resolve().parents[1]
CLINIC = ROOT / "sim/worlds/clinic"
PLACES_YAML = CLINIC / "places.yaml"
ROOMS = ["ward", "office", "reception", "storage"]
PLACES = ROOMS + ["hall"]
G1_TRIES, G1_MAX_ERR_M, G1_MAX_SIM_S = 10, 0.30, 120.0  # task 002 G1
G1_TOTAL = G1_TRIES * len(ROOMS)  # 40 / 40
PHOTO_MAX_ERR_M = G1_MAX_ERR_M  # photos count only with GT within G1's bound of the place point (design review B3)
G9_MIN_PX, G9_PHOTOS = 1000, 3  # task 002 G9
T_WORLD_FROM_MAP = (0.0, 0.0, 0.0)  # G8: identity (002-M map_world_check, rehearsal + counted run)


def write(out: Path, name: str, verdict: dict, rehearsal: bool) -> dict:
    verdict = {"gate": name, "rehearsal": rehearsal, "written_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
               **verdict}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(verdict, indent=2, default=float))
    print(f"{name}: {'PASS' if verdict.get('pass') else ('LOOK' if verdict.get('pass') is None else 'FAIL')}", flush=True)
    return verdict


def to_world(p: mwc.Pose2) -> mwc.Pose2:
    return mwc.compose(T_WORLD_FROM_MAP, p)


# ------------------------------------------------------------------------------------------------------------- G1
def g1_starts(room: str) -> list[str]:
    """Start place of try k for `room`: cycling through the other places, hall first."""
    others = ["hall"] + [r for r in ROOMS if r != room]
    return [others[k % len(others)] for k in range(G1_TRIES)]


def g1_plan_step(need: Counter, cur: str) -> tuple[str, bool]:
    """Next move from `cur`: (target, counted). Counted = a still-needed (cur -> room) try; else reposition to the
    place with the most remaining tries out of it."""
    cands = [r for r in ROOMS if need[(cur, r)] > 0]
    if cands:
        return max(cands, key=lambda r: (need[(cur, r)], -ROOMS.index(r))), True
    out_of = {p: sum(n for (s, _), n in need.items() if s == p) for p in PLACES if p != cur}
    return max(out_of, key=lambda p: (out_of[p], -PLACES.index(p))), False


def gate_g1(robot, places: dict, out: Path, rehearsal: bool, rooms: list[str]) -> dict:  # noqa: ANN001
    """`rooms` subset: split the 40 tries over lock stretches (each half keeps its own start cycle). Only the 4-room
    run writes g1.json and can pass; a subset writes g1_<rooms>.json with pass false (no merge)."""
    need = Counter((s, r) for r in rooms for s in g1_starts(r))
    tries, moves = [], []
    r0 = robot.navigate(places["hall"])
    moves.append({"to": "hall", **r0})
    cur = "hall" if r0["status"] == robot.SUCCEEDED else None
    while sum(need.values()):
        if cur is None:  # after a failed move: recover to the hall, uncounted
            r = robot.navigate(places["hall"])
            moves.append({"to": "hall", "recovery": True, **r})
            cur = "hall" if r["status"] == robot.SUCCEEDED else None
            assert len(moves) < 200, "cannot recover"
            continue
        target, counted = g1_plan_step(need, cur)
        gt0, _ = mwc.gt_pose("clinic")
        r = robot.navigate(places[target], sim_limit_s=G1_MAX_SIM_S if counted else mwc.NAV_SIM_LIMIT_S)
        robot.spin_for_sim(0.5)
        gt1, _ = mwc.gt_pose("clinic")
        goal_w = to_world(places[target])
        err = math.dist(gt1[:2], goal_w[:2])
        start_err = math.dist(gt0[:2], to_world(places[cur])[:2])
        rec = {"from": cur, "to": target, **r, "gt_start": gt0, "gt_end": gt1, "goal_world": goal_w, "err_m": err,
               "start_err_m": start_err}
        if counted:
            need[(cur, target)] -= 1
            rec["valid_start"] = start_err <= G1_MAX_ERR_M  # a try counts only if it really started at `from`
            rec["pass"] = bool(rec["valid_start"] and r["status"] == robot.SUCCEEDED and err <= G1_MAX_ERR_M
                               and r["sim_s"] <= G1_MAX_SIM_S)
            tries.append(rec)
            print(f"G1 try {len(tries):2d} {cur:9s} -> {target:9s} {r['status_name']:10s} {r['sim_s']:6.1f} s sim  "
                  f"rtf {r.get('rtf') or 0:.2f}  start {start_err:.3f} m  err {err:.3f} m  "
                  f"{'ok' if rec['pass'] else 'FAIL'}", flush=True)
        else:
            moves.append(rec)
        cur = target if r["status"] == robot.SUCCEEDED else None
    per_room = {room: f"{sum(t['pass'] for t in tries if t['to'] == room)}/{G1_TRIES}" for room in rooms}
    n = sum(t["pass"] for t in tries)
    full = rooms == ROOMS
    return write(out, "g1" if full else "g1_" + "-".join(rooms), {
        "criterion": f"per room {G1_TRIES} NavigateToPose tries, starts cycling through the other places (hall first); "
                     f"pass = SUCCEEDED and GT within {G1_MAX_ERR_M} m of the place point (map->world {T_WORLD_FROM_MAP}) "
                     f"and <= {G1_MAX_SIM_S} s sim time; gate = 40/40",
        "passed": n, "total": len(tries), "per_room": per_room,
        "rooms": rooms, "starts_per_room": {room: dict(Counter(t["from"] for t in tries if t["to"] == room)) for room in rooms},
        "worst_err_m": max(t["err_m"] for t in tries), "worst_sim_s": max(t["sim_s"] for t in tries),
        "worst_start_err_m": max(t["start_err_m"] for t in tries),
        "invalid_start_tries": sum(not t["valid_start"] for t in tries),
        "verdict_scope": "full 4-room run (the G1 verdict)" if full else "room subset: never the G1 verdict, pass false",
        "tries": tries, "uncounted_moves": moves,
        "pass": full and len(tries) == G1_TOTAL and n == G1_TOTAL}, rehearsal)


# ------------------------------------------------------------------------------------------------------ photos
def take_photos(robot, name: str, pose: mwc.Pose2, out: Path) -> dict:  # noqa: ANN001
    """Navigate to a map pose (one retry), then photos.py at headings pose.yaw + 0/90/180/270 -- only if the last
    navigation SUCCEEDED and the GT pose right before the photos is within PHOTO_MAX_ERR_M of the place point."""
    navs = [robot.navigate(pose)]
    if navs[-1]["status"] != robot.SUCCEEDED:
        navs.append(robot.navigate(pose))
    robot.spin_for_sim(0.5)
    gt, _ = mwc.gt_pose("clinic")
    err = math.dist(gt[:2], to_world(pose)[:2])
    rec = {"pose_map": pose, "nav": navs, "gt_at_start": gt, "err_to_pose_m": err,
           "at_place": bool(navs[-1]["status"] == robot.SUCCEEDED and err <= PHOTO_MAX_ERR_M)}
    if not rec["at_place"]:
        print(f"photos {name}: NOT at the place (nav {navs[-1]['status_name']}, GT err {err:.3f} m): no photos", flush=True)
        return rec
    cmd = [sys.executable, str(ROOT / "probes/photos.py"), "--place", name, "--out", str(out),
           "--yaw0", f"{math.degrees(pose[2]):.4f}"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print(p.stdout.strip(), flush=True)
    assert p.returncode == 0, p.stderr[-3000:]
    return {**rec, "photos_stdout": p.stdout}


def gate_photos(robot, places: dict, out: Path, rehearsal: bool) -> dict:  # noqa: ANN001
    runs = {name: take_photos(robot, name, places[name], out / "photos") for name in PLACES}
    return write(out, "photos", {
        "criterion": f"per place: last navigation SUCCEEDED and GT within {PHOTO_MAX_ERR_M} m of the place point right "
                     "before the photos (else no photos)",
        "places": places, "err_to_pose_m": {n: r["err_to_pose_m"] for n, r in runs.items()},
        "at_place": {n: r["at_place"] for n, r in runs.items()}, "runs": runs,
        "pass": all(r["at_place"] for r in runs.values())}, rehearsal)


# ------------------------------------------------------------------------------------------ G2 / G3 / G4 / G7
def photo_problems(out: Path, places: dict) -> dict[str, list[str]]:
    """Per place, why its photos cannot be used (empty = usable): not at the place per the photos step, no photos,
    or not exactly 4 photos at place yaw + 0/90/180/270."""
    summary = out / "photos.json"
    at = json.loads(summary.read_text()).get("at_place", {}) if summary.exists() else {}
    bad = {}
    for name in PLACES:
        meta_path = out / "photos" / name / "photos.json"
        if at.get(name) is not True:
            bad[name] = [f"photo step: not at the place (at_place={at.get(name)}, {summary})"]
        elif not meta_path.exists():
            bad[name] = [f"no {meta_path}"]
        else:
            bad[name] = pc.photo_set_problems(json.loads(meta_path.read_text()), math.degrees(places[name][2]))
    return bad


def gate_checks(out: Path, truth: dict, places: dict, rehearsal: bool) -> list[dict]:
    bad = {n: b for n, b in photo_problems(out, places).items() if b}
    if bad:  # refuse: no G2 / G3 / G4 / G7 number from photos that do not prove the place
        return [write(out, g, {"refused": bad, "pass": False}, rehearsal) for g in ("g2", "g3", "g4", "g7")]
    res = {name: pc.check_place(out / "photos" / name, truth, None if name == "hall" else name,
                                math.degrees(places[name][2])) for name in PLACES}
    rooms = {r: res[r] for r in ROOMS}
    g2_px = {r: v["g2_max_px"] for r, v in rooms.items()}
    yaw = {p: v["yaw_err_deg"] for p, v in res.items()}
    g2 = write(out, "g2", {
        "criterion": f"each room object >= {pc.G2_MIN_PX} px in >= 1 of 4 photos; every photo's GT yaw within "
                     f"+-{pc.G2_MAX_YAW_DEG} deg of its target (all 5 places)",
        "max_px_per_object": g2_px, "min_px": min(min(v.values()) for v in g2_px.values()),
        "min_px_object": min(((o, n) for v in g2_px.values() for o, n in v.items()), key=lambda x: x[1])[0],
        "yaw_err_deg": yaw, "worst_abs_yaw_err_deg": max(abs(e) for v in yaw.values() for e in v),
        "pass": all(v["g2_pass"] for v in rooms.values()) and all(v["yaw_pass"] for v in res.values())}, rehearsal)
    g3f = {r: v["g3_fraction"] for r, v in rooms.items()}
    fr = [f for v in g3f.values() for f in v.values()]
    g3 = write(out, "g3", {"criterion": f"countable object: >= {pc.G3_ONE_PHOTO_PCT} % of its visible px in one photo",
                           "fraction": g3f, "worst_fraction": None if None in fr else min(fr),
                           "pass": all(v["g3_pass"] for v in rooms.values())}, rehearsal)
    g4l = {r: v["g4_leak_px"] for r, v in rooms.items()}
    g4 = write(out, "g4", {"criterion": f"no other-room countable object >= {pc.G4_MAX_LEAK_PX} px in any photo",
                           "leak_px": g4l, "worst_leak_px": max(n for v in g4l.values() for n in v.values()),
                           "pass": all(v["g4_pass"] for v in rooms.values())}, rehearsal)
    unl = {p: v["g7_unlabelled_frac"] for p, v in res.items()}
    neg = {}
    for p in PLACES:
        meta, _ = pc.load_place(out / "photos" / p)
        for ph in meta["photos"]:
            full = cv2.imread(str(out / "photos" / p / ph["seg_full"]), cv2.IMREAD_UNCHANGED)
            neg[f"{p}/{ph['k']}"] = {"unlabelled_frac": pc.unlabelled_fraction(full), "check_passes": pc.unlabelled_ok(full)}
    neg_ok = any(not v["check_passes"] for v in neg.values())
    g7 = write(out, "g7", {
        "criterion": f"label 0 <= {pc.G7_MAX_UNLABELLED_PERMILLE / 10} % of every cropped photo at all 5 places; "
                     "negative control: the same check fails on uncropped frames",
        "unlabelled_frac": unl, "worst_unlabelled_frac": max(f for v in unl.values() for f in v),
        "negative_control_uncropped": neg, "negative_control_fails_on_n_frames": sum(not v["check_passes"] for v in neg.values()),
        "negative_control_ok": neg_ok, "pass": all(v["g7_pass"] for v in res.values()) and neg_ok}, rehearsal)
    return [g2, g3, g4, g7]


# ------------------------------------------------------------------------------------------------------------ G6
def contact_sheet(place_dir: Path) -> np.ndarray:
    meta = json.loads((place_dir / "photos.json").read_text())
    tiles = []
    for ph in meta["photos"]:
        img = cv2.imread(str(place_dir / ph["rgb"]))
        cv2.putText(img, f"{meta['place']} k={ph['k']} target {ph['target_yaw_deg']:.1f} deg  GT err "
                    f"{ph['gt_yaw_error_deg']:+.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, f"{meta['place']} k={ph['k']} target {ph['target_yaw_deg']:.1f} deg  GT err "
                    f"{ph['gt_yaw_error_deg']:+.2f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    return np.vstack([np.hstack(tiles[0:2]), np.hstack(tiles[2:4])])


def gate_g6(out: Path, rehearsal: bool) -> dict:
    sheets = {}
    for room in ROOMS:
        path = out / f"g6_{room}.png"
        cv2.imwrite(str(path), contact_sheet(out / "photos" / room))
        sheets[room] = str(path)
    return write(out, "g6", {"criterion": "one 4-photo contact sheet per room, checked by eye (look-at-the-artifact)",
                             "sheets": sheets, "pass": None}, rehearsal)


# ------------------------------------------------------------------------------------------------------------ G5
def gate_g5(out: Path, rehearsal: bool) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(CLINIC / "generate.py"), str(CLINIC / "layout.yaml"), "--out", tmp],
                       check=True, capture_output=True, text=True)
        same = {f: filecmp.cmp(Path(tmp) / f, CLINIC / "generated" / f, shallow=False) for f in ("truth.json", "clinic.sdf")}
        live = Path(tmp) / "truth_check.json"
        p = subprocess.run([sys.executable, str(ROOT / "probes/truth_check.py"), "--json", str(live)],
                           capture_output=True, text=True, timeout=120)
        rec = json.loads(live.read_text()) if live.exists() else {}
    return write(out, "g5", {"criterion": "truth.json (and clinic.sdf) regenerate byte-identically; every object in the "
                                          "running sim at its pose (<= 0.01 m), no extra models; standard_dock (exempt) "
                                          "parked at (20, 20) +- 0.05 m; the sim loaded the committed clinic.sdf (sha256 match)",
                             "byte_identical": same, "truth_check_exit": p.returncode, "truth_check_stdout": p.stdout,
                             "dock_pose": rec.get("dock_pose"), "sim_world_dir": rec.get("sim_world_dir"),
                             "sim_clinic_sdf_sha256": rec.get("sim_clinic_sdf_sha256"),
                             "sim_world_file": rec.get("sim_world_file"),
                             "expected_world_file": rec.get("expected_world_file"),
                             "world_matches": rec.get("world_matches"),
                             "pass": all(same.values()) and p.returncode == 0 and rec.get("world_matches") is True},
                 rehearsal)


# ------------------------------------------------------------------------------------------------------------ G8
def gate_g8(robot, places: dict, out: Path, rehearsal: bool, map_yaml: Path = ROOT / "sim/maps/clinic.yaml",  # noqa: ANN001
            world_dir: Path = mwc.WORLD_DIR) -> dict:
    """Movables masked iff the map is the 008 movables-free map (mwc.default_mask); g8.json records both verdicts."""
    res = mwc.run(robot, map_yaml, places, PLACES, "clinic", mwc.default_mask(map_yaml), world_dir)
    return write(out, "g8", res, rehearsal)


# ------------------------------------------------------------------------------------------------------------ G9
def g9_summary(photo_root: Path, truth: dict) -> dict:
    """Per object class: every cropped photo with its class pixel count; the best G9_PHOTOS photos with >= G9_MIN_PX."""
    by_class: dict[str, list[int]] = {}
    for o in truth["objects"]:
        by_class.setdefault(o["class"], []).append(o["label"])
    rows = []
    for pj in sorted(photo_root.glob("*/photos.json")):
        meta, labels = pc.load_place(pj.parent)
        for ph, lab in zip(meta["photos"], labels):
            rows.append((f"{pj.parent.name}/{ph['rgb']}", {c: int(np.isin(lab, ls).sum()) for c, ls in by_class.items()}))
    out = {}
    for c in sorted(by_class):
        ok = sorted(((name, px[c]) for name, px in rows if px[c] >= G9_MIN_PX), key=lambda x: -x[1])
        out[c] = {"n_photos_ge_min": len(ok), "best": ok[:G9_PHOTOS], "enough": len(ok) >= G9_PHOTOS}
    return out


def gate_g9photos(robot, poses: dict, out: Path, truth_path: Path, variant: str, rehearsal: bool) -> dict:  # noqa: ANN001
    vdir = out / variant
    runs = {name: take_photos(robot, name, pose, vdir) for name, pose in poses.items()}
    shutil.copy(truth_path, vdir / "truth.json")
    summary = g9_summary(vdir, json.loads(truth_path.read_text()))
    return write(vdir, "g9photos", {"variant": variant, "truth": str(truth_path), "poses": poses, "runs": runs,
                                    "criterion": f">= {G9_MIN_PX} px in >= {G9_PHOTOS} different cropped photos per class",
                                    "per_class": summary, "pass": all(v["enough"] for v in summary.values())}, rehearsal)


def parse_poses(items: list[str], places: dict) -> dict:
    poses = {}
    for it in items:
        name, xyz = it.split(":")
        if xyz in places:
            poses[name] = places[xyz]
        else:
            x, y, yaw_deg = (float(v) for v in xyz.split(","))
            poses[name] = (x, y, math.radians(yaw_deg))
    return poses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["g1", "g5", "g6", "g8", "photos", "checks", "all", "g9photos"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--truth", type=Path, default=CLINIC / "generated/truth.json")
    ap.add_argument("--variant", default="closed")
    ap.add_argument("--rooms", default=",".join(ROOMS), help="g1: room subset (comma list)")
    ap.add_argument("--map", type=Path, default=ROOT / "sim/maps/clinic.yaml",
                    help="g8: the map the sim navigates on (008: sim/maps/clinic_static.yaml, the movables-free SLAM map)")
    ap.add_argument("--poses", nargs="*", default=None, help="g9photos: name:x,y,yawdeg or name:<place>")
    a = ap.parse_args()
    places = mwc.load_places(PLACES_YAML)
    truth = json.loads(a.truth.read_text())
    sim = a.gate in ("g1", "g8", "photos", "all", "g9photos")
    robot = mwc.make_robot() if sim else None
    results = []
    if a.gate in ("g5", "all"):
        results.append(gate_g5(a.out, a.rehearsal))
    if a.gate in ("g8", "all"):
        results.append(gate_g8(robot, places, a.out, a.rehearsal, a.map, a.truth.parent))  # world dir of --truth
    if a.gate in ("photos", "all"):
        results.append(gate_photos(robot, places, a.out, a.rehearsal))
    if a.gate in ("checks", "all"):
        results += gate_checks(a.out, truth, places, a.rehearsal)
    if a.gate in ("g6", "all"):
        results.append(gate_g6(a.out, a.rehearsal))
    if a.gate in ("g1", "all"):
        results.append(gate_g1(robot, places, a.out, a.rehearsal, a.rooms.split(",")))
    if a.gate == "g9photos":
        poses = parse_poses(a.poses, places) if a.poses else {n: places[n] for n in PLACES}
        results.append(gate_g9photos(robot, poses, a.out, a.truth, a.variant, a.rehearsal))
    return 0 if all(r.get("pass") is not False for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
