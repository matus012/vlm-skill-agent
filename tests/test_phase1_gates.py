"""Verdict code of probes/phase1_gates.py (G1 roll-up, photos at the place, G2/G3/G4/G7 roll-ups), the sim-time wait
of map_world_check.navigate and the G5 dock / world-file helpers of truth_check.py. Fakes only: no ROS, no sim."""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))
import map_world_check as mwc  # noqa: E402
import phase1_gates as pg  # noqa: E402
import truth_check as tc  # noqa: E402

# All places at x = 0 so an x offset of 0.30 gives an error of exactly 0.30 (no float rounding on the boundary).
PLACES = {"hall": (0.0, 0.0, 0.0), "ward": (0.0, 10.0, 0.0), "office": (0.0, 20.0, 0.0),
          "reception": (0.0, 30.0, 0.0), "storage": (0.0, 40.0, 0.0)}
NAMES = {4: "SUCCEEDED", 6: "ABORTED", -2: "SIM_TIMEOUT"}


class FakeRobot:
    """Navigation index j (0 = the initial move to the hall). outcomes[j] may set status, sim_s, end_off (x offset of
    the GT read right after move j) and start_off (x offset of the GT read right before move j)."""
    SUCCEEDED = 4

    def __init__(self, outcomes: dict | None = None) -> None:
        self.pos, self.outcomes, self.j, self.reads, self.limits = PLACES["hall"], outcomes or {}, -1, 0, []

    def navigate(self, goal: tuple, sim_limit_s: float = mwc.NAV_SIM_LIMIT_S) -> dict:
        self.j, self.reads = self.j + 1, 0
        self.limits.append(sim_limit_s)
        o = self.outcomes.get(self.j, {})
        st = o.get("status", 4)
        if st == 4:
            self.pos = goal
        sim_s = o.get("sim_s", 30.0)
        return {"status": st, "status_name": NAMES[st], "sim_s": sim_s, "wall_s": 2 * sim_s, "rtf": 0.5}

    def spin_for_sim(self, sim_s: float) -> None:
        pass

    def gt(self, world: str) -> tuple[tuple, float]:
        if self.j >= 1 and self.reads == 0:
            off = self.outcomes.get(self.j, {}).get("end_off", 0.0)
        else:
            off = self.outcomes.get(self.j + 1, {}).get("start_off", 0.0)
        self.reads += 1
        return (self.pos[0] + off, self.pos[1], self.pos[2]), 0.0


def run_g1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcomes: dict | None = None,
           rooms: list[str] = pg.ROOMS) -> tuple[dict, FakeRobot]:
    robot = FakeRobot(outcomes)
    monkeypatch.setattr(pg.mwc, "gt_pose", robot.gt)
    return pg.gate_g1(robot, PLACES, tmp_path, True, rooms), robot


# --------------------------------------------------------------------------------------------------------------- G1
def test_g1_all_good_passes_40_of_40(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    v, robot = run_g1(monkeypatch, tmp_path)
    assert v["pass"] and v["passed"] == v["total"] == 40
    assert v["per_room"] == {r: "10/10" for r in pg.ROOMS} and v["invalid_start_tries"] == 0
    assert (v["tries"][0]["from"], v["tries"][0]["to"], v["tries"][1]["from"]) == ("hall", "ward", "ward")
    assert robot.limits[0] == mwc.NAV_SIM_LIMIT_S and robot.limits[1] == robot.limits[2] == pg.G1_MAX_SIM_S
    assert json.loads((tmp_path / "g1.json").read_text())["pass"] is True


def test_g1_one_aborted_try_is_39_of_40_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    v, _ = run_g1(monkeypatch, tmp_path, {1: {"status": 6}})
    assert not v["tries"][0]["pass"] and v["passed"] == 39 and v["total"] == 40 and not v["pass"]


@pytest.mark.parametrize("off, ok", [(0.300, True), (0.301, False)])
def test_g1_gt_error_boundary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, off: float, ok: bool) -> None:
    v, _ = run_g1(monkeypatch, tmp_path, {1: {"end_off": off}})
    assert v["tries"][0]["err_m"] == pytest.approx(off) and v["tries"][0]["valid_start"]
    assert v["tries"][0]["pass"] is ok and v["pass"] is ok


@pytest.mark.parametrize("sim_s, ok", [(120.0, True), (120.01, False)])
def test_g1_sim_time_boundary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sim_s: float, ok: bool) -> None:
    v, _ = run_g1(monkeypatch, tmp_path, {1: {"sim_s": sim_s}})
    assert v["tries"][0]["pass"] is ok and v["pass"] is ok


def test_g1_timeout_result_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    v, _ = run_g1(monkeypatch, tmp_path, {1: {"status": -2, "sim_s": 120.0}})
    assert v["tries"][0]["status_name"] == "SIM_TIMEOUT" and not v["tries"][0]["pass"] and not v["pass"]


@pytest.mark.parametrize("off, ok", [(0.30, True), (0.31, False)])
def test_g1_start_pose_must_be_at_from_place(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, off: float,
                                             ok: bool) -> None:
    """Try 2 (ward -> office): GT before the goal is `off` from the ward point; the try itself goes fine."""
    v, _ = run_g1(monkeypatch, tmp_path, {2: {"start_off": off}})
    t = v["tries"][1]
    assert t["from"] == "ward" and t["status_name"] == "SUCCEEDED" and t["err_m"] == 0.0
    assert t["valid_start"] is ok and t["pass"] is ok and v["pass"] is ok
    assert v["invalid_start_tries"] == (0 if ok else 1)


def test_g1_room_subset_never_passes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    v, _ = run_g1(monkeypatch, tmp_path, rooms=["ward", "office"])
    assert v["passed"] == v["total"] == 20 and v["pass"] is False
    assert (tmp_path / "g1_ward-office.json").exists() and not (tmp_path / "g1.json").exists()


# ----------------------------------------------------------------------------------------------------------- photos
class PhotoRobot:
    SUCCEEDED = 4

    def __init__(self, fail_navs: dict[str, int] | None = None, off: dict[str, float] | None = None,
                 fail_at_goal: bool = False) -> None:
        self.fail_navs, self.off, self.pos, self.goal_name = fail_navs or {}, off or {}, None, None
        self.fail_at_goal = fail_at_goal  # a failed navigation that still ends on the place point

    def navigate(self, goal: tuple, sim_limit_s: float = mwc.NAV_SIM_LIMIT_S) -> dict:
        self.goal_name = next(n for n, p in PLACES.items() if p == goal)
        if self.fail_navs.get(self.goal_name, 0) > 0:
            self.fail_navs[self.goal_name] -= 1
            self.pos = goal if self.fail_at_goal else self.pos
            return {"status": 6, "status_name": "ABORTED", "sim_s": 10.0}
        self.pos = goal
        return {"status": 4, "status_name": "SUCCEEDED", "sim_s": 10.0}

    def spin_for_sim(self, sim_s: float) -> None:
        pass

    def gt(self, world: str) -> tuple[tuple, float]:
        p = self.pos or (99.0, 99.0, 0.0)
        return (p[0] + self.off.get(self.goal_name, 0.0), p[1], p[2]), 0.0


def run_photos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, robot: PhotoRobot) -> tuple[dict, list[str]]:
    shot: list[str] = []

    class Done:
        returncode, stdout, stderr = 0, "", ""

    def fake_run(cmd: list[str], **kw: object) -> Done:
        shot.append(cmd[cmd.index("--place") + 1])
        return Done()

    monkeypatch.setattr(pg.mwc, "gt_pose", robot.gt)
    monkeypatch.setattr(pg.subprocess, "run", fake_run)
    return pg.gate_photos(robot, PLACES, tmp_path, True), shot


def test_photos_all_at_place(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    v, shot = run_photos(monkeypatch, tmp_path, PhotoRobot(fail_navs={"ward": 1}))  # one retry is fine
    assert v["pass"] and shot == pg.PLACES and all(v["at_place"].values())


def test_photos_two_failed_navs_no_photos(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    v, shot = run_photos(monkeypatch, tmp_path, PhotoRobot(fail_navs={"office": 2}))
    assert not v["pass"] and v["at_place"]["office"] is False and "office" not in shot


def test_photos_need_succeeded_even_at_the_point(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    v, shot = run_photos(monkeypatch, tmp_path, PhotoRobot(fail_navs={"office": 2}, fail_at_goal=True))
    assert v["err_to_pose_m"]["office"] == 0.0 and v["at_place"]["office"] is False and "office" not in shot


@pytest.mark.parametrize("off, ok", [(0.30, True), (0.31, False)])
def test_photos_gt_error_boundary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, off: float, ok: bool) -> None:
    v, shot = run_photos(monkeypatch, tmp_path, PhotoRobot(off={"storage": off}))
    assert v["err_to_pose_m"]["storage"] == pytest.approx(off)
    assert v["at_place"]["storage"] is ok and v["pass"] is ok and ("storage" in shot) is ok


# ------------------------------------------------------------------------------------------- G2 / G3 / G4 / G7
TRUTH = {"objects": [{"id": f"chair_{r}", "room": r, "countable": True, "label": 10 + i} for i, r in enumerate(pg.ROOMS)]}


def photo_dir(tmp_path: Path, yaw_errs: dict[str, list[float]] | None = None, dirty_uncropped: tuple[str, ...] = ("hall",),
              n_photos: int = 4, at_place: dict | None = None) -> Path:
    """Synthetic run folder: every cropped frame fully labelled (G7 clean), each room's chair 2000 px in photo 0 only;
    uncropped frames = cropped plus a 2 % label-0 robot blob at the places in `dirty_uncropped`."""
    out = tmp_path / "run"
    for name, pose in PLACES.items():
        d = out / "photos" / name
        d.mkdir(parents=True)
        yaw0 = math.degrees(pose[2])
        photos = []
        for k in range(n_photos):
            lab = np.full((100, 100), 200, np.uint8)
            if name in pg.ROOMS and k == 0:
                lab.reshape(-1)[:2000] = 10 + pg.ROOMS.index(name)
            full = np.vstack([lab, np.zeros((10, 100), np.uint8) if name in dirty_uncropped else lab[:10]])
            cv2.imwrite(str(d / f"{k}_seg.png"), lab)
            cv2.imwrite(str(d / f"{k}_seg_full.png"), full)
            photos.append({"k": k, "heading_rel_deg": 90.0 * k, "target_yaw_deg": yaw0 + 90.0 * k,
                           "gt_yaw_error_deg": (yaw_errs or {}).get(name, [0.0] * 4)[k],
                           "seg": f"{k}_seg.png", "seg_full": f"{k}_seg_full.png", "rgb": f"{k}_rgb.png"})
        (d / "photos.json").write_text(json.dumps({"place": name, "photos": photos}))
    (out / "photos.json").write_text(json.dumps({"at_place": at_place or {n: True for n in pg.PLACES}}))
    return out


def verdicts(out: Path) -> dict[str, dict]:
    return {v["gate"]: v for v in pg.gate_checks(out, TRUTH, PLACES, True)}


def test_checks_all_pass_with_working_negative_control(tmp_path: Path) -> None:
    v = verdicts(photo_dir(tmp_path))
    assert all(v[g]["pass"] for g in ("g2", "g3", "g4", "g7"))
    assert v["g7"]["negative_control_fails_on_n_frames"] == 4


def test_g7_fails_when_every_uncropped_frame_is_clean(tmp_path: Path) -> None:
    v = verdicts(photo_dir(tmp_path, dirty_uncropped=()))
    assert v["g2"]["pass"] and v["g3"]["pass"] and v["g4"]["pass"]
    assert not v["g7"]["negative_control_ok"] and not v["g7"]["pass"]


@pytest.mark.parametrize("place", ["hall", "office"])
def test_g2_fails_on_one_photo_yaw_1_6_deg(tmp_path: Path, place: str) -> None:
    v = verdicts(photo_dir(tmp_path, yaw_errs={place: [0.0, 0.0, 1.6, 0.0]}))
    assert not v["g2"]["pass"] and v["g2"]["worst_abs_yaw_err_deg"] == 1.6
    assert v["g3"]["pass"] and v["g4"]["pass"] and v["g7"]["pass"]


def assert_refused(v: dict[str, dict], place: str) -> None:
    for g in ("g2", "g3", "g4", "g7"):
        assert v[g]["pass"] is False and list(v[g]["refused"]) == [place]


def test_checks_refuse_missing_place(tmp_path: Path) -> None:
    out = photo_dir(tmp_path)
    for f in (out / "photos" / "reception").iterdir():
        f.unlink()
    (out / "photos" / "reception").rmdir()
    assert_refused(verdicts(out), "reception")


def test_checks_refuse_place_not_reached(tmp_path: Path) -> None:
    out = photo_dir(tmp_path, at_place={**{n: True for n in pg.PLACES}, "ward": False})
    assert_refused(verdicts(out), "ward")


def test_checks_refuse_without_photo_step_summary(tmp_path: Path) -> None:
    out = photo_dir(tmp_path)
    (out / "photos.json").unlink()
    v = verdicts(out)
    assert all(v[g]["pass"] is False and set(v[g]["refused"]) == set(pg.PLACES) for g in v)


@pytest.mark.parametrize("n", [0, 3])
def test_checks_refuse_short_photo_set(tmp_path: Path, n: int) -> None:
    v = verdicts(photo_dir(tmp_path, n_photos=n))
    assert all(v[g]["pass"] is False and set(v[g]["refused"]) == set(pg.PLACES) for g in v)


# ------------------------------------------------------------------------------------- navigate: sim-time wait
class Clock:
    def __init__(self, sim_per_spin: float, wall_per_spin: float, done_at_sim: float = math.inf) -> None:
        self.sim, self.wall, self.ds, self.dw, self.done_at = 0.0, 0.0, sim_per_spin, wall_per_spin, done_at_sim

    def spin(self) -> None:
        self.sim, self.wall = self.sim + self.ds, self.wall + self.dw

    def wait(self, sim_limit_s: float, wall_cap_s: float) -> tuple[str, float, float]:
        return mwc.wait_result(lambda: self.sim >= self.done_at, lambda: self.sim, self.spin, sim_limit_s, wall_cap_s,
                               wall_now=lambda: self.wall)


def test_wait_result_arrives_at_the_limit() -> None:
    assert Clock(1.0, 1.0, done_at_sim=120.0).wait(120.0, 800.0) == ("done", 120.0, 120.0)


def test_wait_sim_timeout_at_low_rtf() -> None:
    """RTF 0.2: 120 s sim = 600 s wall, inside the 800 s hang guard -> a sim timeout, never a wall one."""
    how, sim_s, wall_s = Clock(0.2, 1.0).wait(120.0, mwc.NAV_WALL_CAP_S)
    assert how == "sim_timeout" and 120.0 < sim_s <= 120.2 + 1e-9 and wall_s == pytest.approx(sim_s / 0.2)


def test_wait_wall_hang_when_sim_clock_stalls() -> None:
    how, sim_s, wall_s = Clock(0.0, 10.0).wait(120.0, mwc.NAV_WALL_CAP_S)
    assert how == "wall_hang" and sim_s == 0.0 and wall_s > mwc.NAV_WALL_CAP_S


# ------------------------------------------------------------------------------------- G5 dock + world file
@pytest.mark.parametrize("x, ok", [(20.0, True), (20.04, True), (20.06, False), (0.5, False)])
def test_dock_parked(x: float, ok: bool) -> None:
    pose, bad = tc.dock_check({"standard_dock": [{"position": {"x": x, "y": 20.0}}]})
    assert pose == [x, 20.0, 0.0] and (bad is None) is ok


@pytest.mark.parametrize("n", [0, 2])
def test_dock_missing_or_duplicated(n: int) -> None:
    assert tc.dock_check({"standard_dock": [{"position": {"x": 20.0, "y": 20.0}}] * n})[1]


def test_resolve_world_cwd_then_resource_path(tmp_path: Path) -> None:
    a, b, cwd = tmp_path / "a", tmp_path / "b", tmp_path / "cwd"
    for d in (a, b, cwd):
        d.mkdir()
    (b / "clinic.sdf").write_text("b")
    assert tc.resolve_world("clinic.sdf", cwd, f"{a}:{b}") == b / "clinic.sdf"
    (a / "clinic.sdf").write_text("a")
    assert tc.resolve_world("clinic.sdf", cwd, f"{a}:{b}") == a / "clinic.sdf"
    (cwd / "clinic.sdf").write_text("c")
    assert tc.resolve_world("clinic.sdf", cwd, f"{a}:{b}") == cwd / "clinic.sdf"
    assert tc.resolve_world("nope.sdf", cwd, f"{a}:{b}") is None


# ---------- G5 world file: live /proc format + loaded == committed (design review re-check) ----------
def test_cmd_argv_ruby_retitled_and_plain() -> None:
    retitled = b"gz sim -s -r --headless-rendering --render-engine-server ogre2 -v 3 clinic.sdf" + b"\0" * 40
    plain = b"gz\0sim\0-s\0-r\0clinic.sdf\0"
    for raw in (retitled, plain):
        argv = tc.cmd_argv(raw)
        assert "sim" in argv and "-s" in argv and argv[-1] == "clinic.sdf", raw


def test_world_mismatch(tmp_path: Path) -> None:
    good, other, same = tmp_path / "a.sdf", tmp_path / "b.sdf", tmp_path / "c.sdf"
    good.write_text("<sdf>closed</sdf>"), other.write_text("<sdf>open</sdf>"), same.write_text("<sdf>closed</sdf>")
    assert tc.world_mismatch(same, good) is None
    assert "!=" in tc.world_mismatch(other, good)              # e.g. a sim started from the cabinet-open folder
    assert "not found" in tc.world_mismatch(None, good)
    assert "missing" in tc.world_mismatch(same, tmp_path / "nope.sdf")


@pytest.mark.parametrize("matches, ok", [(True, True), (False, False), (None, False)])
def test_g5_verdict_needs_the_world_file_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, matches, ok: bool) -> None:
    """g5.json carries truth_check's world_matches and fails without it, even if truth_check itself exits 0."""
    import shutil
    import subprocess
    from types import SimpleNamespace

    def fake_run(cmd: list[str], **_: object) -> SimpleNamespace:
        if cmd[1].endswith("generate.py"):  # regeneration: hand back the committed files
            for f in ("truth.json", "clinic.sdf"):
                shutil.copy(pg.CLINIC / "generated" / f, Path(cmd[-1]) / f)
        else:  # truth_check --json PATH
            Path(cmd[-1]).write_text(json.dumps({"dock_pose": [20, 20, 0], "world_matches": matches,
                                                 "expected_world_file": "x/clinic.sdf", "sim_world_file": "x/clinic.sdf"}))
        return SimpleNamespace(returncode=0, stdout="G5-live: PASS\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rec = pg.gate_g5(tmp_path, rehearsal=True)
    assert rec["world_matches"] is matches and rec["expected_world_file"] == "x/clinic.sdf"
    assert rec["pass"] is ok
