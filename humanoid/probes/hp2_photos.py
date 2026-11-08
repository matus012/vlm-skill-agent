#!/usr/bin/env python3
"""H-P2 photos on the G1 (task 012 design section 5; design notes a recorded deviation). Port of probes/phase1_gates.py
take_photos / gate_photos: at each of the 5 places of the closed world, go_to (one retry), then look_around's 4 x
(turn_to + photo) at place yaw + 0/90/180/270 through G1Adapter (adapters.robots.make("g1")).

    /opt/venv/bin/python humanoid/probes/hp2_photos.py --out D [--rehearsal]       # live, one closed-world sim

First the frozen closed layout is placed (G1Scene.place of truth.json's spec-object poses; an earlier probe on the node
may have moved them). Writes the probes/photos.py dir format (D/photos/<place>/photos.json, <k>_rgb.png, uint8 <k>_seg.png, <k>_seg_full.png;
D/photos.json with at_place), then runs phase1_gates.gate_checks UNCHANGED (G2 / G3 / G4 / G7 -> D/g2..g7.json):
  - GT yaw per photo = the pelvis yaw (/g1/odom; the node publishes no camera pose); crop 0 / 0 (self_crop).
  - <k>_seg_full.png = the G7 negative control (a recorded deviation + addendum): the frame of /head_rgb_pitch/segmentation (the
    head mount pitched 47.6 deg down; the level frames hold 0 robot px) through scene.json label_of_body with the G1's own body ids -> label 0) with the
    most robot pixels among the frames since the previous photo (walk in + turn). gate_checks' "fails on uncropped
    frames" control thus reads "fails on self-view frames"; robot pixels per frame are recorded.
  - per 4-turn set: net GT drift (xy at the last photo vs before the first turn): evidence for the recentre, no bar.
D/hp2.json: the four verdicts + drift + the negative control numbers; pass iff G2, G3, G4 and G7 pass.
take_place() is shared with humanoid/probes/g1_spot.py (H-P3).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "probes"), str(ROOT / "sim/worlds/clinic")]
import map_world_check as mwc  # noqa: E402
import phase1_checks as pc  # noqa: E402
import phase1_gates as pg  # noqa: E402
import realize as rz  # noqa: E402
import seeded as S  # noqa: E402

PHOTO_MAX_ERR_M = pg.PHOTO_MAX_ERR_M


def wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def scene() -> dict:
    return json.loads((Path(os.environ["G1_SCENES"]) / os.environ["GZ_PARTITION"] / "scene.json").read_text())


def frozen_want(world: str) -> dict[str, tuple[float, float, float, float]]:
    """Every spec object at its pose in the world's frozen truth.json (the placer's form): the world as its truth says,
    whatever an earlier placement on this node left."""
    objs = json.loads(S.world_truth_path(world).read_text())["objects"]
    return {o["model_name"]: (o["pose"][0], o["pose"][1], o["pose"][2], o["pose"][5]) for o in objs
            if o["id"] in rz.SPEC_OBJECTS}


class SelfView:
    """Keeps the raw body-id segmentation frame with the most robot pixels since the last take() (on the adapter's
    Drive node, which every go_to / turn_to / photo spins). `topic` = /head_rgb_pitch/segmentation: the G7 control
    frames (same mount pitched 47.6 deg down, a recorded deviation addendum); `fracs` = robot px / 307200 of every frame
    seen since the last take()."""

    def __init__(self, node: Any, robot_ids: list[int], topic: str = "/head_rgb_pitch/segmentation") -> None:  # noqa: ANN401
        from sensor_msgs.msg import Image

        from humanoid.probes.common import image_to_array
        self.ids, self.best, self.frames, self.fracs = np.asarray(robot_ids), None, 0, []

        def on_seg(m: Image) -> None:
            seg = image_to_array(m)
            n = int(np.isin(seg, self.ids).sum())
            self.frames += 1
            self.fracs.append(n / seg.size)
            if self.best is None or n > self.best[0]:
                self.best = (n, seg.copy())
        node.create_subscription(Image, topic, on_seg, 2)

    def take(self) -> tuple[int, int, np.ndarray | None]:
        best, frames, self.best, self.frames, self.fracs = self.best, self.frames, None, 0, []
        return (best[0], frames, best[1]) if best else (0, frames, None)


def take_place(ad: Any, gt_pose: Any, name: str, pose: tuple[float, float, float], out: Path,  # noqa: ANN401
               lut: np.ndarray, sv: SelfView) -> dict:
    """go_to the place (one retry), then 4 x (turn_to, photo) at place yaw + k * 90 deg, photos.py dir format."""
    import cv2
    navs = [ad.go_to(*pose, pg.G1_MAX_SIM_S)]
    if navs[-1].status != "reached":
        navs.append(ad.go_to(*pose, pg.G1_MAX_SIM_S))
    gt = gt_pose()
    err = math.dist(gt[:2], pose[:2])
    rec = {"pose_map": list(pose), "nav": [dataclasses.asdict(n) for n in navs], "gt_at_start": list(gt), "err_to_pose_m": err,
           "at_place": bool(navs[-1].status == "reached" and err <= PHOTO_MAX_ERR_M)}
    if not rec["at_place"]:
        print(f"photos {name}: NOT at the place ({navs[-1].nav_status}, GT err {err:.3f} m): no photos", flush=True)
        return rec
    pdir = out / name
    pdir.mkdir(parents=True, exist_ok=True)
    photos = []
    for k, h in enumerate(pc.HEADINGS_DEG):
        target = wrap(pose[2] + math.radians(h))
        settled = ad.turn_to(target)
        fr = ad.photo()
        g = gt_pose()
        fracs = list(sv.fracs)
        robot_px, n_frames, selfv = sv.take()
        assert selfv is not None, "no /head_rgb_pitch/segmentation frame (node without --pitch-control?)"
        full = lut[selfv]
        cv2.imwrite(str(pdir / f"{k}_rgb.png"), cv2.cvtColor(np.asarray(fr.rgb), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(pdir / f"{k}_seg.png"), np.asarray(fr.seg, np.uint8))
        cv2.imwrite(str(pdir / f"{k}_seg_full.png"), full.astype(np.uint8))
        photos.append({"k": k, "heading_rel_deg": h, "target_yaw_rad": target, "target_yaw_deg": math.degrees(target),
                       "gt_pose": list(g), "gt_yaw_source": "pelvis (/g1/odom)",
                       "gt_yaw_error_deg": math.degrees(wrap(g[2] - target)),
                       "adapter_yaw_error_deg": math.degrees(settled), "frame_meta": fr.meta,
                       "selfview_frames": n_frames, "selfview_robot_px": robot_px,
                       "control_robot_frac_max": max(fracs), "control_robot_frac_mean": sum(fracs) / len(fracs),
                       "rgb": f"{k}_rgb.png", "seg": f"{k}_seg.png", "seg_full": f"{k}_seg_full.png"})
        print(f"photo {name} {k}: target {math.degrees(target):8.2f} deg  GT err {photos[-1]['gt_yaw_error_deg']:+.3f} "
              f"deg  self-view robot px {robot_px}", flush=True)
    drift = math.dist(photos[-1]["gt_pose"][:2], gt[:2])
    (pdir / "photos.json").write_text(json.dumps({
        "place": name, "robot": "g1", "crop": {"top_px": 0, "bottom_px": 0},
        "label_rule": "scene.json label_of_body of the MuJoCo body id (0 = robot / world)",
        "seg_full_rule": "G7 negative control: pitched-camera frame (robot ids -> 0) with most robot px since the last photo",
        "net_drift_m": drift, "photos": photos}, indent=1))
    return {**rec, "net_drift_m": drift}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rehearsal", action="store_true")
    a = ap.parse_args()
    from adapters.robots import make
    from agent.truth import load_truth, world_truth_path
    places = mwc.load_places(pg.PLACES_YAML)
    sc = scene()
    lut = np.asarray(sc["label_of_body"], np.uint8)
    rb = make("g1")
    try:
        log: dict = {}
        if (why := rb.run_kw["place_objects"](frozen_want("closed"), log)) is not None:
            raise SystemExit(f"H-P2: the frozen closed layout did not land: {why}")
        sv = SelfView(rb.ad.drive, sc["robot_body_ids"])
        runs = {n: take_place(rb.ad, rb.gt_pose, n, places[n], a.out / "photos", lut, sv) for n in pg.PLACES}
    finally:
        rb.close()
    pg.write(a.out, "photos", {"criterion": f"per place: go_to reached and GT within {PHOTO_MAX_ERR_M} m right before "
                                            "the photos (else no photos)", "places": places,
                               "at_place": {n: r["at_place"] for n, r in runs.items()}, "runs": runs,
                               "pass": all(r["at_place"] for r in runs.values())}, a.rehearsal)
    g2, g3, g4, g7 = pg.gate_checks(a.out, load_truth(world_truth_path("closed")), places, a.rehearsal)
    drift = {n: r.get("net_drift_m") for n, r in runs.items()}
    neg = g7.get("negative_control_uncropped", {})
    ph = {n: json.loads((a.out / "photos" / n / "photos.json").read_text())["photos"] for n in pg.PLACES if runs[n]["at_place"]}
    control = {n: {"robot_frac_max": max(p["control_robot_frac_max"] for p in v),
                   "robot_frac_mean": sum(p["control_robot_frac_mean"] for p in v) / len(v),
                   "g7_fails_on": sum(not neg[f"{n}/{p['k']}"]["check_passes"] for p in v), "frames": len(v)}
               for n, v in ph.items()}
    res = pg.write(a.out, "hp2", {
        "criterion": "G2 >= 1000 px per room object in >= 1 photo, yaw +-1.5 deg; G3 >= 90 %; G4 < 200 px; G7 <= 0.1 % "
                     "unlabelled and the same check fails on >= 1 pitched-camera control frame (robot ids -> 0); phase1_gates.gate_checks",
        "verdicts": {g["gate"]: g["pass"] for g in (g2, g3, g4, g7)},
        "negative_control_fails_on_n_frames": g7.get("negative_control_fails_on_n_frames"),
        "negative_control_frames": len(neg),
        "selfview_robot_px_max": max((p["selfview_robot_px"] for v in ph.values() for p in v), default=None),
        "control_per_place": control,
        "control_robot_frac_max": max((c["robot_frac_max"] for c in control.values()), default=None),
        "control_g7_verdict": "FAIL" if g7.get("negative_control_ok") else "PASS (control not met)",
        "net_drift_m_per_set": drift, "net_drift_m_max": max((d for d in drift.values() if d is not None), default=None),
        "pass": all(g["pass"] for g in (g2, g3, g4, g7))}, a.rehearsal)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
