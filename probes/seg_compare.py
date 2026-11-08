"""006 gate M2 (hpc/PERUN_SIM.md section 4): segmentation of this sim vs the local machine reference photos at the same GT poses.

    python probes/seg_compare.py --out DIR [--ref DIR] [--world clinic]

Run inside one sim with that sim's env (ROS_DOMAIN_ID, GZ_PARTITION) and no Nav2 goal active. For every reference
photo (`--ref`/<place>/photos.json, 5 places x 4 headings, the local machine world v1; v2 changed one colour only, labels equal):
- teleport: target base = recorded GT camera pose (-) camera offset, where the offset (model frame == base_link ->
  `oakd_rgb_camera_frame`) is measured live once from Gazebo pose/info (GT base + GT camera), never assumed; the robot
  model is moved with `gz service /world/<w>/set_pose` (this sim's GZ_PARTITION, from the env).
- settle (/odom still, probes/photos.py rule), read the GT camera pose (1), take frames stamped after it, read it
  again (2). Counted only if both reads are within POS_TOL_M / ROT_TOL_DEG of the reference; else teleport once more,
  then the frame is recorded as not counted. Read 1 -> read 2 = drift after the teleport.
- frames: the probes/photos.py code path (same topics, seg bridge, depth fusion, crop); agreement = fraction of equal
  label pixels between this cropped label frame and the reference `<k>_seg.png`.
Also: live camera_info (probes/camera_check.py numbers) and sha256 of clinic.sdf / truth.json vs WORLD_TAG.
Writes DIR/m2.json, DIR/<place>/<k>_{seg,rgb}.png, DIR/m2_contact_sheet.png. The robot is put back where it started.
"""
import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from sensor_msgs.msg import CameraInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
import camera_check as cc  # noqa: E402
import photos  # noqa: E402
from photos import Pose3, compose, qrot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = Path.home() / ".cache/vlm_skill_agent/runs/2026-11-04/002-counted/photos"
WORLD_TAG = "phase1-world-frozen-v2"
WORLD_FILES = ("sim/worlds/clinic/generated/clinic.sdf", "sim/worlds/clinic/generated/truth.json")
# gate numbers: hpc/PERUN_SIM.md section 4 / task 006 M2
POS_TOL_M, ROT_TOL_DEG = 0.005, 0.2
AGREE_MIN = 0.98
ATTEMPTS = 2  # teleport, then retry once


def inv(p: Pose3) -> Pose3:
    qc = p[1] * np.array([-1.0, -1.0, -1.0, 1.0])
    return -qrot(qc, p[0]), qc


def offset_of(base: Pose3, cam: Pose3) -> Pose3:
    """Camera pose in the base (model) frame from two world poses."""
    return compose(inv(base), cam)


def base_for_camera(cam_target: Pose3, offset: Pose3) -> Pose3:
    """World base pose that puts the camera at `cam_target`."""
    return compose(cam_target, inv(offset))


def pose_err(a: Pose3, b: Pose3) -> tuple[float, float]:
    """(position error m, rotation angle deg) between two world poses."""
    d = float(abs(np.dot(a[1] / np.linalg.norm(a[1]), b[1] / np.linalg.norm(b[1]))))
    return float(np.linalg.norm(a[0] - b[0])), 2 * math.degrees(math.acos(min(1.0, d)))


def pose_ok(err: tuple[float, float]) -> bool:
    return err[0] <= POS_TOL_M and err[1] <= ROT_TOL_DEG


def agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of pixels with equal labels; frames of different shape are refused."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return float((a == b).mean())


def camera_info_ok(width: int, height: int, fx: float) -> bool:
    return width == cc.WIDTH and height == cc.HEIGHT and abs(fx - cc.FX_EXPECTED) <= cc.FX_TOL


def world_sha(repo: Path = ROOT, tag: str = WORLD_TAG, files: tuple[str, ...] = WORLD_FILES) -> dict[str, object]:
    out: dict[str, object] = {"tag": tag, "files": {}}
    for f in files:
        disk = hashlib.sha256((repo / f).read_bytes()).hexdigest()
        tagged = hashlib.sha256(subprocess.run(["git", "-C", str(repo), "show", f"{tag}:{f}"], check=True,
                                               capture_output=True).stdout).hexdigest()
        out["files"][f] = {"disk": disk, "tag": tagged, "equal": disk == tagged}
    out["pass"] = all(v["equal"] for v in out["files"].values())
    return out


def frame_record(errs: list[tuple[float, float]], agree: float | None) -> dict[str, object]:
    """errs: GT camera error vs reference for every read of the final attempt (read 1, read 2)."""
    counted = bool(errs) and all(pose_ok(e) for e in errs)
    return {"pos_err_m": max(e[0] for e in errs), "rot_err_deg": max(e[1] for e in errs), "counted": counted,
            "agreement": agree, "pass": bool(counted and agree is not None and agree >= AGREE_MIN)}


def verdict(frames: list[dict], cam_ok: bool, sha_ok: bool, n_expected: int) -> dict[str, object]:
    n_pass = sum(f["pass"] for f in frames)
    return {"frames_expected": n_expected, "frames": len(frames), "frames_counted": sum(f["counted"] for f in frames),
            "frames_pass": n_pass, "camera_info_pass": cam_ok, "world_sha_pass": sha_ok,
            "pass": bool(cam_ok and sha_ok and len(frames) == n_expected and n_pass == n_expected)}


def jsonable(o: object) -> object:
    """json.dumps default: numpy scalars -> Python (np.bool_ from comparisons on numpy floats killed the m2.json write in
    job 143786 after all 20 frames were measured)."""
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def colourise(lab: np.ndarray) -> np.ndarray:
    lut = np.random.default_rng(115).integers(40, 256, (256, 3), dtype=np.uint8)
    lut[0] = 0
    return lut[lab]


def contact_sheet(rows: list[tuple[str, np.ndarray, np.ndarray | None, float | None]], path: Path) -> None:
    """One row per frame: reference | this sim | differing pixels (white), half size, with the agreement."""
    tiles = []
    for name, ref, live, agree in rows:
        live_c = colourise(live) if live is not None else np.zeros((*ref.shape, 3), np.uint8)
        diff = np.zeros_like(live_c) if live is None else np.repeat(((ref != live) * 255).astype(np.uint8)[..., None], 3, 2)
        row = cv2.resize(np.hstack([colourise(ref), live_c, diff]), None, fx=0.5, fy=0.5,
                         interpolation=cv2.INTER_NEAREST)
        txt = f"{name}  agreement {agree:.4f}" if agree is not None else f"{name}  NOT COUNTED"
        cv2.putText(row, txt, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(row)
    head = np.zeros((30, tiles[0].shape[1], 3), np.uint8)
    cv2.putText(head, "reference (local) | this sim | differing pixels", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2)
    cv2.imwrite(str(path), cv2.cvtColor(np.vstack([head, *tiles]), cv2.COLOR_RGB2BGR))


def pose_dict(p: Pose3) -> dict[str, list[float]]:
    return {"xyz": p[0].tolist(), "quat_xyzw": p[1].tolist()}


def set_pose(world: str, model: str, p: Pose3) -> None:
    (x, y, z), (qx, qy, qz, qw) = p
    req = (f'name: "{model}", position: {{x: {x:.9f}, y: {y:.9f}, z: {z:.9f}}}, '
           f'orientation: {{x: {qx:.12f}, y: {qy:.12f}, z: {qz:.12f}, w: {qw:.12f}}}')
    out = subprocess.run(["gz", "service", "-s", f"/world/{world}/set_pose", "--reqtype", "gz.msgs.Pose",
                          "--reptype", "gz.msgs.Boolean", "--timeout", "5000", "--req", req],
                         check=True, capture_output=True, text=True, timeout=20).stdout
    assert "data: true" in out, f"set_pose {model} refused: {out!r}"


def model_and_camera(world: str) -> tuple[Pose3, Pose3, float]:
    """(GT model == base_link world pose, GT camera world pose, sim time), one pose/info message."""
    poses, t = photos.gz_poses(world)
    base = poses[photos.MODEL]
    return base, compose(base, poses[photos.CAM_LINK]), t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ref", type=Path, default=REF)
    ap.add_argument("--world", default="clinic")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(photos.ROBOT_CFG.read_text())
    top, bottom = cfg["self_crop"]["top_px"], cfg["self_crop"]["bottom_px"]
    sha = world_sha()  # fail fast on a world mismatch / missing tag before any teleport
    refs = []
    for pj in sorted(args.ref.glob("*/photos.json")):
        meta = json.loads(pj.read_text())
        assert meta["crop"]["top_px"] == top and meta["crop"]["bottom_px"] == bottom, f"{pj}: crop differs from config"
        refs += [(meta["place"], ph, pj.parent) for ph in meta["photos"]]

    bridge = photos.start_seg_bridge()
    rclpy.init()
    node = photos.Camera("odom")  # GT only; 'odom' = no AMCL client
    info: list[CameraInfo] = []
    node.create_subscription(CameraInfo, f"{cc.NS}/camera_info", lambda m: info.append(m), 5)
    frames, sheet = [], []
    try:
        node.spin_until(lambda: node.tf.can_transform("base_link", photos.CAM_LINK, photos.Time()), 60.0, "TF")
        node.frames_after(0.0)
        node.spin_until(lambda: bool(info), 30.0, "camera_info")
        ci = info[-1]
        cam_info = {"width": ci.width, "height": ci.height, "fx": ci.k[0], "fy": ci.k[4],
                    "pass": camera_info_ok(ci.width, ci.height, ci.k[0])}
        home_base, home_cam, _ = model_and_camera(args.world)
        offset = offset_of(home_base, home_cam)  # measured live, once
        offset_proof = photos.check_offset(node, offset)
        for place, ph, ref_dir in refs:
            target = (np.array(ph["gt_camera_pose_world"]["xyz"]), np.array(ph["gt_camera_pose_world"]["quat_xyzw"]))
            base_target = base_for_camera(target, offset)
            tries = []
            for _ in range(ATTEMPTS):
                set_pose(args.world, photos.MODEL, base_target)
                node.last.pop("odom", None)
                node.settle()
                _, cam1, t1 = model_and_camera(args.world)
                fr = node.frames_after(t1)
                _, cam2, t2 = model_and_camera(args.world)
                errs = [pose_err(cam1, target), pose_err(cam2, target)]
                tries.append({"read1": pose_dict(cam1), "read2": pose_dict(cam2), "gt_sim_s": [t1, t2],
                              "err_read1": errs[0], "err_read2": errs[1], "drift": pose_err(cam1, cam2)})
                if all(pose_ok(e) for e in errs):
                    break
            ref = cv2.imread(str(ref_dir / ph["seg"]), cv2.IMREAD_UNCHANGED)
            lab, n_hidden = photos.fused_labels(photos.to_array(fr["seg"]), photos.to_array(fr["depth"]))
            lab_c, rgb_c = photos.crop(lab, top, bottom), photos.crop(photos.to_array(fr["rgb"]), top, bottom)
            agree = agreement(ref, lab_c) if all(pose_ok(e) for e in errs) else None
            rec = frame_record(errs, agree)
            (args.out / place).mkdir(exist_ok=True)
            cv2.imwrite(str(args.out / place / f"{ph['k']}_seg.png"), lab_c)
            cv2.imwrite(str(args.out / place / f"{ph['k']}_rgb.png"), cv2.cvtColor(rgb_c, cv2.COLOR_RGB2BGR))
            frames.append({"place": place, "k": ph["k"], "reference_pose": pose_dict(target),
                           "base_target": pose_dict(base_target), "attempts": tries, "seg_px_zeroed_by_depth": n_hidden,
                           "frame_stamps_sim_s": {k: photos.stamp(m) for k, m in fr.items()}, **rec})
            sheet.append((f"{place}/{ph['k']}", ref, lab_c if rec["counted"] else None, agree))
            print(f"{place}/{ph['k']}: pos {rec['pos_err_m'] * 1000:.2f} mm rot {rec['rot_err_deg']:.3f} deg "
                  f"drift {tries[-1]['drift'][0] * 1000:.2f} mm attempts {len(tries)} agreement {agree}", flush=True)
        set_pose(args.world, photos.MODEL, home_base)
        node.last.pop("odom", None)
        node.settle()
        back, _, _ = model_and_camera(args.world)
        restore = {"home": pose_dict(home_base), "after": pose_dict(back), "err": pose_err(back, home_base)}
    finally:
        node.destroy_node()
        rclpy.shutdown()
        os.killpg(bridge.pid, signal.SIGTERM)
        bridge.wait(10)
    res = {"gate": "006 M2", "tolerances": {"pos_m": POS_TOL_M, "rot_deg": ROT_TOL_DEG, "agreement_min": AGREE_MIN},
           "env": {k: os.environ.get(k) for k in ("ROS_DOMAIN_ID", "GZ_PARTITION", "SLURM_JOB_ID")},
           "ref": str(args.ref), "crop": {"top_px": top, "bottom_px": bottom}, "camera_info": cam_info,
           "world_sha": sha, "camera_offset_live": pose_dict(offset), "camera_offset_proof": offset_proof,
           "restore": restore, "frames": frames,
           **verdict(frames, cam_info["pass"], bool(sha["pass"]), len(refs))}
    (args.out / "m2.json").write_text(json.dumps(res, indent=2, default=jsonable))
    contact_sheet(sheet, args.out / "m2_contact_sheet.png")
    print("M2:", "PASS" if res["pass"] else "FAIL", f"({res['frames_pass']}/{len(refs)} frames)", flush=True)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
