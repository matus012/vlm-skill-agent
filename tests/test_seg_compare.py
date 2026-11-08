"""probes/seg_compare.py (006 M2) on synthetic data: pose math, read-back check, agreement, world sha, verdict. No sim."""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))
import seg_compare as sc  # noqa: E402
from photos import compose  # noqa: E402

RNG = np.random.default_rng(6)


def rand_pose() -> sc.Pose3:
    q = RNG.normal(size=4)
    return RNG.uniform(-5, 5, 3), q / np.linalg.norm(q)


def rot_z(p: sc.Pose3, deg: float) -> sc.Pose3:
    a = math.radians(deg) / 2
    return p[0], compose((np.zeros(3), np.array([0, 0, math.sin(a), math.cos(a)])), (np.zeros(3), p[1]))[1]


def test_inverse_is_identity() -> None:
    for _ in range(200):
        p = rand_pose()
        e = sc.pose_err(compose(p, sc.inv(p)), (np.zeros(3), np.array([0.0, 0, 0, 1])))
        assert e[0] < 1e-12 and e[1] < 1e-5


def test_round_trip_random_poses() -> None:
    """offset from (base, cam); base for a target camera pose; composing back gives the target to < 1e-9."""
    for _ in range(500):
        base, off_true = rand_pose(), rand_pose()
        cam = compose(base, off_true)
        off = sc.offset_of(base, cam)
        assert np.allclose(off[0], off_true[0], atol=1e-9)
        target = rand_pose()
        tb = sc.base_for_camera(target, off)
        back = compose(tb, off)
        assert np.linalg.norm(back[0] - target[0]) < 1e-9
        assert 1 - abs(float(np.dot(back[1], target[1]))) < 1e-12


def test_pose_err_known_values() -> None:
    p = rand_pose()
    assert sc.pose_err(p, p)[0] == 0 and sc.pose_err(p, p)[1] < 1e-5
    q = (p[0] + np.array([0.003, 0.004, 0.0]), p[1])
    assert sc.pose_err(p, q)[0] == pytest.approx(0.005)
    assert sc.pose_err(p, rot_z(p, 0.15))[1] == pytest.approx(0.15, abs=1e-6)
    assert sc.pose_err(p, (p[0], -p[1]))[1] < 1e-5  # q and -q are the same rotation


def test_tolerance_boundaries() -> None:
    assert sc.pose_ok((0.005, 0.2)) and sc.pose_ok((0.0, 0.0))
    assert not sc.pose_ok((0.0051, 0.0)) and not sc.pose_ok((0.0, 0.201))


@pytest.mark.parametrize("bad", [
    (np.array([0.006, 0, 0]), 0.0),   # 6 mm wrong in the offset translation
    (np.zeros(3), 0.25),              # 0.25 deg wrong in the offset yaw
])
def test_wrong_offset_caught_by_read_back(bad: tuple[np.ndarray, float]) -> None:
    """The sim moves the real robot (true offset); the probe planned with a wrong one -> the GT read-back fails."""
    base = (np.array([1.0, 2.0, -0.005]), np.array([0.0, 0.0, 0.0, 1.0]))
    off_true = (np.array([-0.0596, 0.0, 0.24353]), np.array([0.0, 0.0, 0.0, 1.0]))
    off_wrong = rot_z((off_true[0] + bad[0], off_true[1]), bad[1])
    for _ in range(50):
        target = compose(rand_pose(), off_true)
        planned = sc.base_for_camera(target, off_wrong)
        read_back = compose(planned, off_true)
        assert not sc.pose_ok(sc.pose_err(read_back, target))
        assert sc.pose_ok(sc.pose_err(compose(sc.base_for_camera(target, off_true), off_true), target))


def test_agreement_identical_and_known_shift() -> None:
    a = RNG.integers(0, 256, (377, 640), dtype=np.uint8)
    assert sc.agreement(a, a.copy()) == 1.0
    stripes = np.repeat(np.arange(160, dtype=np.uint8), 4)[None].repeat(377, 0)  # 4-px wide label stripes
    assert sc.agreement(stripes, np.roll(stripes, 1, axis=1)) == 0.75  # every 4th column changes label
    b = a.copy()
    idx = RNG.choice(a.size, 37, replace=False)
    b.flat[idx] = a.flat[idx] + 1
    assert sc.agreement(a, b) == pytest.approx(1 - 37 / a.size, abs=1e-15)


def test_agreement_refuses_shape_mismatch() -> None:
    a = np.zeros((377, 640), np.uint8)
    for b in (np.zeros((480, 640), np.uint8), np.zeros((377, 640, 3), np.uint8), np.zeros((377, 1), np.uint8)):
        with pytest.raises(ValueError, match="shape mismatch"):  # (377, 1) would broadcast silently
            sc.agreement(a, b)


def test_camera_info_ok() -> None:
    fx = sc.cc.FX_EXPECTED  # 320 / tan(1.5708 / 2) = 319.9997
    assert sc.camera_info_ok(640, 480, 320.0) and sc.camera_info_ok(640, 480, fx + 0.499)
    assert sc.camera_info_ok(640, 480, fx - 0.499)
    assert not sc.camera_info_ok(640, 480, fx + 0.501) and not sc.camera_info_ok(640, 480, fx - 0.501)
    assert not sc.camera_info_ok(640, 481, 320.0) and not sc.camera_info_ok(639, 480, 320.0)


def test_world_sha(tmp_path: Path) -> None:
    def git(*a: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *a], check=True,
                       capture_output=True)
    git("init", "-q")
    for f, txt in (("w.sdf", "a"), ("t.json", "{}")):
        (tmp_path / f).write_text(txt)
    git("add", ".")
    git("commit", "-qm", "x")
    git("tag", "frozen")
    assert sc.world_sha(tmp_path, "frozen", ("w.sdf", "t.json"))["pass"]
    (tmp_path / "t.json").write_text('{"x": 1}')
    r = sc.world_sha(tmp_path, "frozen", ("w.sdf", "t.json"))
    assert not r["pass"] and r["files"]["w.sdf"]["equal"] and not r["files"]["t.json"]["equal"]
    with pytest.raises(subprocess.CalledProcessError):
        sc.world_sha(tmp_path, "no-such-tag", ("w.sdf",))


@pytest.mark.skipif(subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "-q", "--verify",
                                    "refs/tags/phase1-world-frozen-v2"], capture_output=True).returncode != 0,
                    reason="needs the git history with tag phase1-world-frozen-v2 (absent in a one-commit snapshot)")
def test_real_repo_world_matches_v2() -> None:
    assert sc.world_sha()["pass"]


def test_frame_record_and_verdict() -> None:
    ok, far = (0.001, 0.05), (0.006, 0.05)
    assert sc.frame_record([ok, ok], 0.98)["pass"]
    assert not sc.frame_record([ok, ok], 0.9799)["pass"]
    r = sc.frame_record([ok, far], None)  # read 2 drifted out of tolerance
    assert not r["counted"] and not r["pass"] and r["pos_err_m"] == 0.006
    good = [sc.frame_record([ok, ok], 0.99) for _ in range(20)]
    assert sc.verdict(good, True, True, 20)["pass"]
    assert not sc.verdict(good, False, True, 20)["pass"]
    assert not sc.verdict(good, True, False, 20)["pass"]
    assert not sc.verdict(good[:19], True, True, 20)["pass"]
    assert not sc.verdict(good[:19] + [r], True, True, 20)["pass"]
    assert sc.verdict(good[:19] + [r], True, True, 20)["frames_counted"] == 19


def test_jsonable_numpy_scalars() -> None:
    import json
    res = {"ok": np.bool_(True), "err": (np.float64(0.001), np.float32(0.5)), "n": [np.int64(3)]}
    assert json.loads(json.dumps(res, default=sc.jsonable)) == {"ok": True, "err": [0.001, 0.5], "n": [3]}
    with pytest.raises(TypeError):
        json.dumps({"x": object()}, default=sc.jsonable)


def test_contact_sheet(tmp_path: Path) -> None:
    import cv2
    a = RNG.integers(0, 256, (377, 640), dtype=np.uint8)
    sc.contact_sheet([("ward/0", a, a, 1.0), ("ward/1", a, None, None)], tmp_path / "s.png")
    img = cv2.imread(str(tmp_path / "s.png"))
    assert img.shape[1] == 960 and img.shape[0] == 30 + 2 * 188
