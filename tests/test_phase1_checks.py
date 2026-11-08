"""probes/phase1_checks.py on synthetic label images (no sim). Boundaries sit exactly on the task 002 gate numbers."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))
import phase1_checks as pc  # noqa: E402

H, W = 100, 100  # 10 000 px synthetic frame


def frame(fill: int = 240, **objs: int) -> np.ndarray:
    """Frame of `fill` with n pixels of each label, written row-major from the top-left (labels as 'l<id>=n')."""
    img = np.full(H * W, fill, np.uint8)
    i = 0
    for key, n in objs.items():
        img[i: i + n] = int(key[1:])
        i += n
    return img.reshape(H, W)


# --- G2 coverage: max over photos >= 1000 -----------------------------------------------------------------------
def test_g2_max_over_photos() -> None:
    labs = [frame(l7=10), frame(l7=1000), frame(), frame(l7=500)]
    assert pc.max_pixels(labs, 7) == 1000
    assert pc.max_pixels(labs, 7) >= pc.G2_MIN_PX


def test_g2_boundary() -> None:
    assert pc.max_pixels([frame(l7=999)] * 4, 7) < pc.G2_MIN_PX


def test_object_absent() -> None:
    labs = [frame()] * 4
    assert pc.max_pixels(labs, 9) == 0
    assert pc.counted_once_fraction(labs, 9) is None
    assert not pc.counted_once_ok(labs, 9)  # never visible: G3 cannot hold (G2 fails it too)


# --- G3 counted once ---------------------------------------------------------------------------------------------
def test_g3_split_evenly_fails() -> None:
    labs = [frame(l3=1500), frame(l3=1500), frame(), frame()]
    assert pc.counted_once_fraction(labs, 3) == 0.5
    assert not pc.counted_once_ok(labs, 3)


def test_g3_exactly_90_pct_passes() -> None:
    labs = [frame(l3=900), frame(l3=100), frame(), frame()]
    assert pc.counted_once_ok(labs, 3)


def test_g3_just_under_90_pct_fails() -> None:
    labs = [frame(l3=899), frame(l3=101), frame(), frame()]
    assert not pc.counted_once_ok(labs, 3)


def test_g3_one_photo_only() -> None:
    assert pc.counted_once_ok([frame(), frame(l3=2000), frame(), frame()], 3)


# --- G4 leakage: < 200 px passes, >= 200 fails ------------------------------------------------------------------
@pytest.mark.parametrize("n, ok", [(0, True), (199, True), (200, False), (201, False)])
def test_g4_leak_boundary(n: int, ok: bool) -> None:
    labs = [frame(), frame(l12=n), frame(), frame(l12=n // 2)]
    assert (pc.max_pixels(labs, 12) < pc.G4_MAX_LEAK_PX) is ok


# --- G7 unlabelled <= 0.1 % of the frame -------------------------------------------------------------------------
@pytest.mark.parametrize("n0, ok", [(0, True), (10, True), (11, False), (5000, False)])
def test_g7_boundary(n0: int, ok: bool) -> None:
    img = frame(l0=n0)
    assert pc.unlabelled_ok(img) is ok
    assert pc.unlabelled_fraction(img) == n0 / (H * W)


def test_g7_cropped_vs_uncropped() -> None:
    """Negative control shape: self-view rows at the bottom fail uncropped and pass once cropped off."""
    img = frame()
    img[90:, 40:60] = 0  # 200 px (2 %) robot blob in the bottom 10 rows
    assert not pc.unlabelled_ok(img)
    assert pc.unlabelled_ok(img[: H - 10])


# --- yaw ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("err, ok", [(0.0, True), (1.5, True), (-1.5, True), (1.5001, False), (-2.0, False)])
def test_yaw_boundary(err: float, ok: bool) -> None:
    assert pc.yaw_ok(err) is ok


# --- check_place end to end on files -------------------------------------------------------------------------------
def write_place(d: Path, labs: list[np.ndarray], yaw_errs: list[float]) -> Path:
    p = d / "storage"
    p.mkdir(parents=True)
    photos = []
    for k, (lab, e) in enumerate(zip(labs, yaw_errs)):
        cv2.imwrite(str(p / f"{k}_seg.png"), lab)
        photos.append({"k": k, "seg": f"{k}_seg.png", "gt_yaw_error_deg": e, "heading_rel_deg": 90.0 * k,
                       "target_yaw_deg": 90.0 * k})
    (p / "photos.json").write_text(json.dumps({"place": "storage", "photos": photos}))
    return p


TRUTH = {"objects": [
    {"id": "toolbox_red_1", "room": "storage", "countable": False, "label": 1},
    {"id": "chair_1", "room": "storage", "countable": True, "label": 2},
    {"id": "chair_2", "room": "reception", "countable": True, "label": 3},
    {"id": "plant_1", "room": "reception", "countable": False, "label": 4},
]}


def test_check_place_pass(tmp_path: Path) -> None:
    labs = [frame(l1=1200), frame(l2=1000, l4=5000), frame(l3=199), frame()]
    r = pc.check_place(write_place(tmp_path, labs, [0.1, -0.2, 1.5, 0.0]), TRUTH, "storage", 0.0)
    assert r["g2_max_px"] == {"toolbox_red_1": 1200, "chair_1": 1000}
    assert r["g4_leak_px"] == {"chair_2": 199}  # non-countable foreign object (plant_1) is not leakage
    assert r["g2_pass"] and r["g3_pass"] and r["g4_pass"] and r["g7_pass"] and r["yaw_pass"]


def test_check_place_fail_each(tmp_path: Path) -> None:
    labs = [frame(l1=999, l0=11), frame(l2=600), frame(l2=600, l3=200), frame()]
    r = pc.check_place(write_place(tmp_path, labs, [0.0, 0.0, 1.6, 0.0]), TRUTH, "storage", 0.0)
    assert not r["g2_pass"] and not r["g3_pass"] and not r["g4_pass"] and not r["g7_pass"] and not r["yaw_pass"]
    assert r["g3_fraction"]["chair_1"] == 0.5


def test_check_place_yaw_alone_fails_g2(tmp_path: Path) -> None:
    labs = [frame(l1=5000), frame(l2=5000), frame(), frame()]
    r = pc.check_place(write_place(tmp_path, labs, [0.0, 0.0, 0.0, -1.51]), TRUTH, "storage", 0.0)
    assert r["g3_pass"] and r["g4_pass"] and not r["yaw_pass"] and not r["g2_pass"]


def test_check_place_hall(tmp_path: Path) -> None:
    r = pc.check_place(write_place(tmp_path, [frame()] * 4, [0.0] * 4), TRUTH, None, 0.0)
    assert r["g7_pass"] and r["yaw_pass"] and "g2_pass" not in r


# --- photo set: exactly 4 photos at place yaw + 0/90/180/270 -------------------------------------------------------
def photo_meta(headings: list[float], place_yaw_deg: float) -> dict:
    return {"photos": [{"k": k, "heading_rel_deg": h, "target_yaw_deg": (place_yaw_deg + h + 180) % 360 - 180}
                       for k, h in enumerate(headings)]}


def test_photo_set_ok_at_45_deg_place() -> None:
    assert pc.photo_set_problems(photo_meta([0.0, 90.0, 180.0, 270.0], 45.0), 45.0) == []


@pytest.mark.parametrize("headings", [[], [0.0, 90.0, 180.0], [0.0, 90.0, 180.0, 270.0, 0.0], [0.0, 90.0, 270.0, 180.0]])
def test_photo_set_wrong_count_or_order(headings: list[float]) -> None:
    assert pc.photo_set_problems(photo_meta(headings, 0.0), 0.0)


def test_photo_set_wrong_place_yaw() -> None:
    assert pc.photo_set_problems(photo_meta([0.0, 90.0, 180.0, 270.0], 0.0), 45.0)


def test_check_place_refuses_short_set(tmp_path: Path) -> None:
    """all([]) / all(3 photos) must not pass: a short photos.json is an error, not a clean place."""
    d = write_place(tmp_path, [frame(l1=5000)] * 3, [0.0] * 3)
    with pytest.raises(AssertionError, match="3 photos"):
        pc.check_place(d, TRUTH, "storage", 0.0)
