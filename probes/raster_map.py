#!/usr/bin/env python3
"""Task 008 static map cross-check (INTERFACES section 10 "Map"): rasterise the generated clinic SDF as the 2D lidar sees it, without
the movables, into a Nav2 occupancy map. Offline, no sim, byte-identical output for the same SDF.

    python3 probes/raster_map.py [--sdf sim/worlds/clinic/generated/clinic.sdf] [--out sim/maps/clinic_static_raster]
    python3 probes/raster_map.py --compare sim/maps/clinic_static.yaml   # a SLAM map vs the raster (JSON report)

Role (2026-11-05): a cross-check of the Phase 5 SLAM map (sim/maps/clinic_static.*), not the Nav2 map itself.

Geometry: every visual of every labelled model, loaded by raycast_seg.load_prims (boxes / vertical cylinders / spheres,
model yaw only). The generator writes each collision identical to its visual (same pose + geometry, checked on the
frozen v2 SDF), so visual (what gz-sensors' gpu_lidar renders) and collision give the same slice.
Scan plane: z = LIDAR_Z, the rplidar_link origin of the TB4 standard URDF (xacro of ros_ws turtlebot4_description:
base_link -> shell_link +0.0942 m -> rplidar_link +0.098715 m; Create3's base_link is on the ground), one horizontal
ray ring (vertical samples 1, angle 0). A prim is in the map if the plane cuts it; its slice (rectangle / disc) marks
every cell it overlaps with positive area (thin legs still mark their cell, like a lidar return would).
Movables (toolbox_red_1, chair_orange_1..4, INTERFACES section 10) are left out; Nav2's obstacle layer sees them live.
Frame: origin ORIGIN is a whole number of cells from the world origin, map yaw 0, so cell (i, j) covers world
[ox + i * res, ox + (i + 1) * res) exactly as map_server reads it: map frame == world frame with no offset (G8).
Values (trinary, map_saver's): occupied 0, free 254 = the non-occupied cells 4-connected to the hall place (the
reachable interior at scan height; rooms count only through their doorways), unknown 205 = everything else (outside the
building, never observed; same as the 002 SLAM map).
"""
from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import yaml

from raycast_seg import load_prims

ROOT = Path(__file__).resolve().parents[1]
SDF = ROOT / "sim/worlds/clinic/generated/clinic.sdf"
PLACES = ROOT / "sim/worlds/clinic/places.yaml"
MOVABLES = ("toolbox_red_1", "chair_orange_1", "chair_orange_2", "chair_orange_3", "chair_orange_4")
LIDAR_Z = 0.0942 + 0.098715  # m above the floor, see docstring
RES = 0.05                   # = sim/maps/clinic.yaml
ORIGIN = (-5.5, -7.0)        # whole cells: building is x +-5.05, y +-6.55 (walls 0.1 m)
SIZE = (220, 280)            # cells (w, h): x -5.5..5.5, y -7.0..7.0
OCC, FREE, UNKNOWN = 0, 254, 205
EPS = 1e-9                   # a footprint edge lying on a cell edge does not mark the neighbour


def slices(prims: list[tuple], z: float) -> list[tuple]:
    """Cut of each prim by the plane at z: ('box', cx, cy, yaw, hx, hy) or ('disc', cx, cy, r)."""
    out = []
    for _, myaw, (mx, my, mz), kind, size, (lx, ly, lz) in prims:
        c, s = math.cos(myaw), math.sin(myaw)
        cx, cy, dz = mx + c * lx - s * ly, my + s * lx + c * ly, z - (mz + lz)
        if kind == "box" and abs(dz) < size[2] / 2:
            out.append(("box", cx, cy, myaw, size[0] / 2, size[1] / 2))
        elif kind == "cylinder" and abs(dz) < size[1] / 2:
            out.append(("disc", cx, cy, size[0]))
        elif kind == "sphere" and abs(dz) < size:
            out.append(("disc", cx, cy, math.sqrt(size * size - dz * dz)))
    return out


def occupancy(cuts: list[tuple], origin: tuple[float, float] = ORIGIN, size: tuple[int, int] = SIZE,
              res: float = RES) -> np.ndarray:
    """bool (h, w), row 0 = max y (image order): cells of the grid (origin, size, res) overlapped by any cut."""
    w, h = size
    x0 = origin[0] + res * np.arange(w)  # cell min corners
    y0 = origin[1] + res * np.arange(h)
    X0, Y0 = np.meshgrid(x0, y0)
    occ = np.zeros((h, w), bool)
    for cut in cuts:
        if cut[0] == "disc":
            _, cx, cy, r = cut
            dx = np.maximum(np.maximum(X0 - cx, cx - (X0 + res)), 0)
            dy = np.maximum(np.maximum(Y0 - cy, cy - (Y0 + res)), 0)
            hit = dx * dx + dy * dy < (r - EPS) ** 2
        else:  # separating axes: x, y (cell) and the box's own u, v
            _, cx, cy, yaw, hx, hy = cut
            c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
            ex, ey = c * hx + s * hy, s * hx + c * hy  # box half-extent along x / y
            mx, my = X0 + res / 2 - cx, Y0 + res / 2 - cy  # cell centre rel. box centre
            hit = (np.abs(mx) < ex + res / 2 - EPS) & (np.abs(my) < ey + res / 2 - EPS)
            cu, su = math.cos(yaw), math.sin(yaw)
            cell_r = res / 2 * (c + s)  # cell half-extent projected on u (and on v)
            hit &= np.abs(cu * mx + su * my) < hx + cell_r - EPS
            hit &= np.abs(-su * mx + cu * my) < hy + cell_r - EPS
        occ |= hit
    return occ[::-1]


def cell_of(x: float, y: float, origin: tuple[float, float] = ORIGIN, size: tuple[int, int] = SIZE,
            res: float = RES) -> tuple[int, int]:
    """(row, col) of the cell holding world point (x, y)."""
    return size[1] - 1 - math.floor((y - origin[1]) / res), math.floor((x - origin[0]) / res)


def build(sdf: Path = SDF, exclude: tuple[str, ...] = MOVABLES, origin: tuple[float, float] = ORIGIN,
          size: tuple[int, int] = SIZE, res: float = RES) -> tuple[np.ndarray, np.ndarray]:
    """(uint8 trinary image, bool occupancy) of the SDF minus `exclude` on the grid (origin, size, res)."""
    names = {int(m.find(".//plugin/label").text): m.get("name") for m in ET.parse(sdf).getroot().iter("model")
             if m.find(".//plugin/label") is not None}
    prims = [p for p in load_prims(sdf) if names[p[0]] not in exclude]
    occ = occupancy(slices(prims, LIDAR_Z), origin, size, res)
    hall = yaml.safe_load(PLACES.read_text())["places"]["hall"]
    seed = cell_of(hall["x"], hall["y"], origin, size, res)
    assert not occ[seed], "hall place is occupied"
    mask = np.zeros((size[1] + 2, size[0] + 2), np.uint8)
    cv2.floodFill(occ.astype(np.uint8), mask, (seed[1], seed[0]), 1, flags=4 | cv2.FLOODFILL_MASK_ONLY | (1 << 8))
    img = np.full(occ.shape, UNKNOWN, np.uint8)
    img[mask[1:-1, 1:-1] == 1] = FREE
    img[occ] = OCC
    return img, occ


def clusters(mask: np.ndarray, origin: tuple[float, float], res: float) -> list[dict]:
    """8-connected components of `mask`: world centroid (m) and size (cells), largest first."""
    n, _, stats, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    h = mask.shape[0]
    out = [{"xy": [round(origin[0] + (cen[i][0] + 0.5) * res, 3), round(origin[1] + (h - 1 - cen[i][1] + 0.5) * res, 3)],
            "cells": int(stats[i, cv2.CC_STAT_AREA])} for i in range(1, n)]
    return sorted(out, key=lambda c: (-c["cells"], c["xy"]))


def compare(slam_yaml: Path, sdf: Path = SDF) -> dict:
    """Occupied-cell agreement of a SLAM map (any origin / extent, yaw 0) with the raster, on the SLAM map's own grid
    (the SDF slice is re-rasterised there, so no resampling between misaligned grids). Raster side = its lidar-visible
    cells (occupied, 4-adjacent to reachable free); SLAM side = cells <= (1 - occupied_thresh) * 255. "Within 1 cell" =
    the 3x3 neighbourhood. Overlap = the SLAM map's extent."""
    meta = yaml.safe_load(slam_yaml.read_text())
    assert abs(float(meta["origin"][2])) < 1e-9, "rotated map origin not supported"
    slam = cv2.imread(str(slam_yaml.parent / meta["image"]), cv2.IMREAD_GRAYSCALE)
    origin, res = (float(meta["origin"][0]), float(meta["origin"][1])), float(meta["resolution"])
    img, _ = build(sdf, MOVABLES, origin, (slam.shape[1], slam.shape[0]), res)
    free = img == FREE
    near_free = free | np.roll(free, 1, 0) | np.roll(free, -1, 0) | np.roll(free, 1, 1) | np.roll(free, -1, 1)
    ras = (img == OCC) & near_free
    so = slam <= int(round((1.0 - float(meta["occupied_thresh"])) * 255))
    k = np.ones((3, 3), np.uint8)
    ras_only = ras & ~cv2.dilate(so.astype(np.uint8), k).astype(bool)
    slam_only = so & ~cv2.dilate((img == OCC).astype(np.uint8), k).astype(bool)
    return {"slam_map": str(slam_yaml), "grid": {"origin": list(origin), "res": res, "w": slam.shape[1], "h": slam.shape[0]},
            "lidar_z_m": LIDAR_Z, "raster_visible_cells": int(ras.sum()), "slam_occupied_cells": int(so.sum()),
            "raster_with_slam_within_1": float(1 - ras_only.sum() / ras.sum()),
            "slam_with_raster_within_1": float(1 - slam_only.sum() / so.sum()),
            "raster_only_clusters": clusters(ras_only, origin, res), "slam_only_clusters": clusters(slam_only, origin, res)}


def scan_endpoints(occ: np.ndarray, grid: tuple, x: float, y: float, yaw: float) -> np.ndarray:
    """Simulated rplidar scan for offline checks: 640 beams, 0.164..12 m, from 0.04 m behind base_link (URDF), ray
    march in 0.005 m steps over a bool occupancy `occ` on grid ((ox, oy), (w, h), res); first-hit world xy (n, 2)."""
    (ox, oy), (w, h), res = grid
    lx, ly = x - 0.04 * math.cos(yaw), y - 0.04 * math.sin(yaw)
    a = np.linspace(-math.pi, math.pi, 640, endpoint=False) + yaw
    t = np.arange(0.164, 12.0, 0.005)
    X, Y = lx + np.cos(a)[:, None] * t, ly + np.sin(a)[:, None] * t
    r = np.clip(h - 1 - np.floor((Y - oy) / res).astype(int), 0, h - 1)  # past the first hit: clipped, never used
    hit = occ[r, np.clip(np.floor((X - ox) / res).astype(int), 0, w - 1)]
    f = hit.argmax(1)
    return np.stack([X[np.arange(640), f], Y[np.arange(640), f]], 1)[hit.any(1)]


def write(img: np.ndarray, out: Path, sdf: Path) -> None:
    out.with_suffix(".pgm").write_bytes(f"P5\n{img.shape[1]} {img.shape[0]}\n255\n".encode() + img.tobytes())
    out.with_suffix(".yaml").write_text(
        f"# probes/raster_map.py (task 008): {sdf.relative_to(ROOT)} minus {', '.join(MOVABLES)}, sliced at z = "
        f"{LIDAR_Z:.6f} m. map frame == world frame.\n"
        f"image: {out.name}.pgm\nmode: trinary\nresolution: {RES:.3f}\norigin: [{ORIGIN[0]:.3f}, {ORIGIN[1]:.3f}, 0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf", type=Path, default=SDF)
    ap.add_argument("--out", type=Path, default=ROOT / "sim/maps/clinic_static_raster")
    ap.add_argument("--compare", type=Path, help="SLAM map yaml: print the agreement report (JSON), write nothing")
    a = ap.parse_args()
    if a.compare:
        print(json.dumps(compare(a.compare.resolve(), a.sdf.resolve()), indent=1))
        return
    img, _ = build(a.sdf.resolve())
    write(img, a.out.resolve(), a.sdf.resolve())
    print(f"{a.out}.pgm {img.shape[1]}x{img.shape[0]}: occupied {(img == OCC).sum()} free {(img == FREE).sum()} "
          f"unknown {(img == UNKNOWN).sum()}")


if __name__ == "__main__":
    main()
