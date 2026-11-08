"""Phase 1 gate-check helpers (task 002 G2, G3, G4, G7): pure functions over photos.py output + truth.json, no sim.

Inputs (sim/INTERFACES.md sections 2 and 4): `<dir>/<place>/photos.json`, `<k>_seg.png` (cropped, label id per
pixel, 0 = background or the robot itself), `truth.json` (objects with `label`, `room`, `countable`).
Thresholds are task 002's gate text, written as integer comparisons so no float rounding sits on a boundary.
"""
import json
from pathlib import Path

import cv2
import numpy as np

# the Phase 1 task's gates (not included)
G2_MIN_PX = 1000          # each object of the room: >= 1000 seg px in at least one of the 4 photos
G2_MAX_YAW_DEG = 1.5      # each photo's GT yaw within +-1.5 deg of its target
G3_ONE_PHOTO_PCT = 90     # countable object: >= 90 % of its visible px (summed over the 4 photos) in one photo
G4_MAX_LEAK_PX = 200      # countable object of another room / hall: < 200 px in every photo (fails at >= 200)
G7_MAX_UNLABELLED_PERMILLE = 1  # unlabelled (label 0) <= 0.1 % of the cropped frame
HEADINGS_DEG = (0, 90, 180, 270)  # the 4 photos of a place, relative to the place yaw (places.yaml)
TARGET_TOL_DEG = 1e-3  # target yaw vs place yaw + heading: photos.py gets --yaw0 rounded to 4 decimals (not a gate number)


def load_place(place_dir: Path) -> tuple[dict, list[np.ndarray]]:
    meta = json.loads((place_dir / "photos.json").read_text())
    labels = []
    for p in meta["photos"]:
        img = cv2.imread(str(place_dir / p["seg"]), cv2.IMREAD_UNCHANGED)
        assert img is not None and img.ndim == 2 and img.dtype == np.uint8, place_dir / p["seg"]
        labels.append(img)
    return meta, labels


def pixel_counts(labels: list[np.ndarray], label: int) -> list[int]:
    return [int((lab == label).sum()) for lab in labels]


def max_pixels(labels: list[np.ndarray], label: int) -> int:
    return max(pixel_counts(labels, label))


def counted_once_fraction(labels: list[np.ndarray], label: int) -> float | None:
    """Share of the object's visible pixels (summed over photos) that lie in its best photo; None if never visible."""
    c = pixel_counts(labels, label)
    return max(c) / sum(c) if sum(c) else None


def counted_once_ok(labels: list[np.ndarray], label: int) -> bool:
    c = pixel_counts(labels, label)
    return sum(c) > 0 and 100 * max(c) >= G3_ONE_PHOTO_PCT * sum(c)


def unlabelled_fraction(label_img: np.ndarray) -> float:
    return float((label_img == 0).sum() / label_img.size)


def unlabelled_ok(label_img: np.ndarray) -> bool:
    return 1000 * int((label_img == 0).sum()) <= G7_MAX_UNLABELLED_PERMILLE * label_img.size


def yaw_ok(err_deg: float) -> bool:
    return abs(err_deg) <= G2_MAX_YAW_DEG


def photo_set_problems(meta: dict, place_yaw_deg: float) -> list[str]:
    """Empty iff the place has exactly 4 photos, at headings 0/90/180/270 (in order) relative to the place yaw."""
    ph = meta["photos"]
    bad = [] if len(ph) == len(HEADINGS_DEG) else [f"{len(ph)} photos, need {len(HEADINGS_DEG)}"]
    for p, h in zip(ph, HEADINGS_DEG):
        off = (p["target_yaw_deg"] - place_yaw_deg - h + 180.0) % 360.0 - 180.0
        if p["heading_rel_deg"] != h or abs(off) > TARGET_TOL_DEG:
            bad.append(f"photo k={p['k']}: heading {p['heading_rel_deg']}, target {p['target_yaw_deg']:.4f} deg; "
                       f"need heading {h} = place yaw {place_yaw_deg:.4f} + {h}")
    return bad


def check_place(place_dir: Path, truth: dict, room: str | None, place_yaw_deg: float) -> dict:
    """All per-place gate numbers. `room` = the room this place covers (None for the hall: G7 and yaw only)."""
    meta, labels = load_place(place_dir)
    bad = photo_set_problems(meta, place_yaw_deg)
    assert not bad, f"{place_dir}: {bad}"
    yaw_errs = [p["gt_yaw_error_deg"] for p in meta["photos"]]
    res: dict = {
        "place": meta["place"], "room": room,
        "g7_unlabelled_frac": [unlabelled_fraction(lab) for lab in labels],
        "g7_pass": all(unlabelled_ok(lab) for lab in labels),
        "yaw_err_deg": yaw_errs, "yaw_pass": all(yaw_ok(e) for e in yaw_errs),
    }
    if room is None:
        return res
    mine = [o for o in truth["objects"] if o["room"] == room]
    foreign = [o for o in truth["objects"] if o["room"] != room and o["countable"]]
    res["g2_max_px"] = {o["id"]: max_pixels(labels, o["label"]) for o in mine}
    res["g2_pass"] = all(v >= G2_MIN_PX for v in res["g2_max_px"].values()) and res["yaw_pass"]
    res["g3_fraction"] = {o["id"]: counted_once_fraction(labels, o["label"]) for o in mine if o["countable"]}
    res["g3_pass"] = all(counted_once_ok(labels, o["label"]) for o in mine if o["countable"])
    res["g4_leak_px"] = {o["id"]: max_pixels(labels, o["label"]) for o in foreign}
    res["g4_pass"] = all(v < G4_MAX_LEAK_PX for v in res["g4_leak_px"].values())
    return res
