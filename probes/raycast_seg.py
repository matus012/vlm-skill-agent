"""CPU ray-cast of the generated clinic world into a segmentation label image (no sim).

    python3 probes/raycast_seg.py --validate DIR   # compare with a photos.py run (per-label px of every photo)

Reads clinic.sdf (every visual is a box / vertical cylinder / sphere with zero visual rotation, model yaw only) and
renders the label id per pixel for a pinhole camera (config/robots/turtlebot4.yaml: 640x480, HFOV 90 deg, then the
self-crop rows). Pitch / roll of the camera are ignored (measured < 0.3 deg). Used to design the layout spacing
(task 002 W2) before the sim settles G2/G3/G4.
"""
import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SDF = ROOT / "sim/worlds/clinic/generated/clinic.sdf"
CAM = yaml.safe_load((ROOT / "config/robots/turtlebot4.yaml").read_text())


def load_prims(sdf: Path = SDF, poses: dict[str, tuple[float, float, float]] | None = None) -> list[tuple]:
    """[(label, yaw, model_xyz, kind, size, local_xyz)] for every visual of every labelled model.
    poses: per-model override {model name: (x, y, yaw)} of the SDF pose (z kept); unknown names raise KeyError."""
    prims = []
    poses = dict(poses or {})
    for m in ET.parse(sdf).getroot().iter("model"):
        lab = m.find(".//plugin/label")
        if lab is None:
            continue
        mx, my, mz, _, _, myaw = (float(v) for v in m.find("pose").text.split())
        if m.get("name") in poses:
            mx, my, myaw = poses.pop(m.get("name"))
        for vis in m.iter("visual"):
            p = [float(v) for v in vis.find("pose").text.split()]
            assert p[3:] == [0, 0, 0], "visual rotation not supported"
            g = vis.find("geometry")[0]
            if g.tag == "box":
                size = tuple(float(v) for v in g.find("size").text.split())
            elif g.tag == "cylinder":
                size = (float(g.find("radius").text), float(g.find("length").text))
            else:
                size = float(g.find("radius").text)
            prims.append((int(lab.text), myaw, (mx, my, mz), g.tag, size, tuple(p[:3])))
    if poses:
        raise KeyError(f"pose overrides for models not in {sdf}: {sorted(poses)}")
    return prims


def rays(stride: int = 1) -> np.ndarray:
    """Unit-free camera-frame directions (x fwd, y left, z up) of the cropped image, shape (H, W, 3)."""
    w, h = CAM["camera"]["width"], CAM["camera"]["height"]
    f = (w / 2) / math.tan(CAM["camera"]["hfov_rad"] / 2)
    top, bot = CAM["self_crop"]["top_px"], h - CAM["self_crop"]["bottom_px"]
    v, u = np.mgrid[top:bot:stride, 0:w:stride].astype(np.float64)
    return np.stack([np.full_like(u, f), -(u - (w - 1) / 2), -(v - (h - 1) / 2)], axis=-1)


def _window(prim: tuple, o_w: np.ndarray, yaw: float, stride: int, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """(r0, r1, c0, c1) ray-grid window that holds every ray able to hit the primitive (its world AABB projected), or
    None if no ray can. Conservative: a window too large changes nothing, every ray is still tested exactly."""
    _, myaw, mxyz, kind, size, lxyz = prim
    hx, hy, hz = (np.asarray(size) / 2 if kind == "box" else (size[0], size[0], size[1] / 2) if kind == "cylinder"
                  else (size, size, size))
    cm, sm = math.cos(myaw), math.sin(myaw)
    cx, cy = mxyz[0] + cm * lxyz[0] - sm * lxyz[1], mxyz[1] + sm * lxyz[0] + cm * lxyz[1]
    ex, ey = abs(cm) * hx + abs(sm) * hy, abs(sm) * hx + abs(cm) * hy
    corners = np.array([[cx + i * ex, cy + j * ey, mxyz[2] + lxyz[2] + k * hz] for i in (-1, 1) for j in (-1, 1)
                        for k in (-1, 1)]) - o_w
    c, s = math.cos(yaw), math.sin(yaw)
    fwd = c * corners[:, 0] + s * corners[:, 1]
    if (fwd <= 0).all():
        return None
    h, w = shape
    if (fwd <= 1e-6).any():
        return 0, h, 0, w
    f = (CAM["camera"]["width"] / 2) / math.tan(CAM["camera"]["hfov_rad"] / 2)
    u = (CAM["camera"]["width"] - 1) / 2 - f * (-s * corners[:, 0] + c * corners[:, 1]) / fwd
    v = (CAM["camera"]["height"] - 1) / 2 - f * corners[:, 2] / fwd - CAM["self_crop"]["top_px"]
    r0, r1 = max(0, math.floor(v.min() / stride)), min(h, math.floor(v.max() / stride) + 2)
    c0, c1 = max(0, math.floor(u.min() / stride)), min(w, math.floor(u.max() / stride) + 2)
    return (r0, r1, c0, c1) if r0 < r1 and c0 < c1 else None


def render_lt(prims: list[tuple], xyz: tuple[float, float, float], yaw: float, stride: int = 1
              ) -> tuple[np.ndarray, np.ndarray]:
    """(label image uint8 with 0 = no hit, ray parameter t of the hit or inf) from camera position xyz, heading yaw.
    t is along the unnormalised ray of rays(stride), so two renders with the same camera compare by t."""
    d_cam = rays(stride)
    shape = d_cam.shape[:2]
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.stack([c * d_cam[..., 0] - s * d_cam[..., 1], s * d_cam[..., 0] + c * d_cam[..., 1], d_cam[..., 2]], axis=-1)
    best = np.full(shape, np.inf)
    out = np.zeros(shape, np.uint8)
    o_w = np.asarray(xyz, float)
    for prim in prims:
        win = _window(prim, o_w, yaw, stride, shape)
        if win is None:
            continue
        r0, r1, c0, c1 = win
        lab, myaw, mxyz, kind, size, lxyz = prim
        dw = d[r0:r1, c0:c1].reshape(-1, 3)
        cm, sm = math.cos(myaw), math.sin(myaw)
        rel = o_w - np.asarray(mxyz)
        o = np.array([cm * rel[0] + sm * rel[1], -sm * rel[0] + cm * rel[1], rel[2]]) - np.asarray(lxyz)
        dl = np.stack([cm * dw[:, 0] + sm * dw[:, 1], -sm * dw[:, 0] + cm * dw[:, 1], dw[:, 2]], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            if kind == "box":
                half = np.asarray(size) / 2
                t1, t2 = (-half - o) / dl, (half - o) / dl
                tn = np.nanmax(np.minimum(t1, t2), axis=1)
                tf = np.nanmin(np.maximum(t1, t2), axis=1)
                t = np.where((tn <= tf) & (tf > 0), np.maximum(tn, 0), np.inf)
            elif kind == "cylinder":
                r, hl = size[0], size[1] / 2
                a = dl[:, 0] ** 2 + dl[:, 1] ** 2
                b = 2 * (o[0] * dl[:, 0] + o[1] * dl[:, 1])
                cc = o[0] ** 2 + o[1] ** 2 - r * r
                disc = b * b - 4 * a * cc
                sq = np.sqrt(np.maximum(disc, 0))
                ts0, ts1 = (-b - sq) / (2 * a), (-b + sq) / (2 * a)
                tz0, tz1 = (-hl - o[2]) / dl[:, 2], (hl - o[2]) / dl[:, 2]
                tn = np.maximum(ts0, np.minimum(tz0, tz1))
                tf = np.minimum(ts1, np.maximum(tz0, tz1))
                t = np.where((disc >= 0) & (tn <= tf) & (tf > 0), np.maximum(tn, 0), np.inf)
            else:
                r = size
                a = (dl ** 2).sum(1)
                b = 2 * (dl @ o)
                cc = o @ o - r * r
                disc = b * b - 4 * a * cc
                t0 = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
                t = np.where((disc >= 0) & (t0 > 0), t0, np.inf)
        t = t.reshape(r1 - r0, c1 - c0)
        sb, so = best[r0:r1, c0:c1], out[r0:r1, c0:c1]
        hit = t < sb
        sb[hit] = t[hit]
        so[hit] = lab
    return out, best


def render(prims: list[tuple], xyz: tuple[float, float, float], yaw: float, stride: int = 1) -> np.ndarray:
    """Label image (uint8, 0 = no hit) seen from camera position xyz with heading yaw."""
    return render_lt(prims, xyz, yaw, stride)[0]


def counts(img: np.ndarray, stride: int = 1) -> dict[int, int]:
    u, c = np.unique(img, return_counts=True)
    return {int(k): int(v) * stride * stride for k, v in zip(u, c) if 0 < k < 200}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=Path, required=True, help="photos.py output dir with <place>/photos.json")
    a = ap.parse_args()
    import cv2
    prims = load_prims()
    for pj in sorted(a.validate.glob("*/photos.json")):
        meta = json.loads(pj.read_text())
        for k, ph in enumerate(meta["photos"]):
            pose = ph["gt_camera_pose_world"]
            sim = cv2.imread(str(pj.parent / ph["seg"]), cv2.IMREAD_UNCHANGED)
            ray = render(prims, tuple(pose["xyz"]), pose["yaw_rad"])
            agree = float((sim == ray).mean())
            print(f"{meta['place']:9s} k={k} pixel-agreement {agree:.4f}  sim {counts(sim)}  ray {counts(ray)}")


if __name__ == "__main__":
    main()
