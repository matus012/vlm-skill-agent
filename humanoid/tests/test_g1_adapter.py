"""G1Adapter go_to / turn_to logic on a fake Drive (no sim): recentre, stall -> STALLED, whole-call timeout -> TIMEOUT,
yaw None (012 section 4); sync before routing after a layout reset, no recentre out leg into a wall, a blocked go_to
returns standing (p8d11). Mutants: tests/mutants_g1.py."""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import rclpy

import humanoid.adapter_g1 as ag
from agent.world import load_map
from humanoid.drive import Pose

MAP = Path(__file__).resolve().parents[2] / "sim/worlds/clinic/generated/map_knowledge.json"  # rooms + doorways only
HALL_PLACE = (2.75, -0.1)  # hall -> hall: one leg
Pt = tuple[float, float]


class FakeDrive:
    """Legs land exactly on the target after `leg_s` sim; each turn lands on the heading after `turn_s` and then
    shifts the robot by the next entry of `drifts` (none left -> no drift). `fresh` (x, y): where the robot really is
    after a layout reset; the pose shows it only once sync() ran. `wall_ahead_m`: scan return straight ahead."""

    def __init__(self) -> None:
        self.pose = Pose(t=0.0)
        self.leg_s, self.turn_s, self.drifts, self.stall, self.short = 5.0, 5.0, [], False, 0.0
        self.fresh: tuple[float, float] | None = None
        self.wall_ahead_m = math.inf
        self.held = 0.0
        self.goto_calls: list[tuple[float, float, float]] = []

    @property
    def scan(self) -> SimpleNamespace:
        r = [math.inf] * 360
        r[180] = self.wall_ahead_m  # angle_min -pi + 180 deg = straight ahead
        return SimpleNamespace(ranges=r, angle_min=-math.pi, angle_increment=math.radians(1.0), range_min=0.1,
                               range_max=4.0)

    def sync(self) -> None:
        if self.fresh is not None:
            (self.pose.x, self.pose.y), self.fresh = self.fresh, None

    def goto(self, tx: float, ty: float, timeout_s: float, gate_m: float, stop_m: float = 0.15,
             min_v: float = 0.0) -> dict:
        self.goto_calls.append((round(tx, 9), round(ty, 9), stop_m, min_v))
        dt = min(self.leg_s, timeout_s + 0.02)
        self.pose.t += dt
        if not self.stall:
            self.pose.x, self.pose.y = tx - self.short, ty  # a leg may stop `short` of its target (< stop_m)
        return {"target": [tx, ty], "elapsed_s": dt, "timed_out": dt > timeout_s, "stalled": self.stall}

    def turn_to(self, target: float, timeout_s: float) -> dict:
        self.pose.t += self.turn_s
        self.pose.yaw = target
        if self.drifts:
            dx, dy = self.drifts.pop(0)
            self.pose.x, self.pose.y = self.pose.x + dx, self.pose.y + dy
        return {"target_deg": math.degrees(target), "elapsed_s": self.turn_s}

    def hold(self, s: float) -> None:
        self.pose.t += s
        self.held += s

    def send(self, vx: float, vy: float, wz: float) -> None:
        pass


@pytest.fixture
def ad(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ag.G1Adapter:
    scene = tmp_path / "scene.json"
    scene.write_text(json.dumps({"label_of_body": [0, 7]}))
    fake = FakeDrive()
    monkeypatch.setattr(rclpy, "init", lambda: None)
    monkeypatch.setattr(rclpy, "spin_once", lambda node, timeout_sec=None: None)
    monkeypatch.setattr(ag, "Drive", lambda name: fake)
    monkeypatch.setattr(ag, "FrameTap", lambda node: None)
    fake.wait_pose = lambda: None
    return ag.G1Adapter(scene_json=scene, map_path=MAP)


def test_turn_drift_is_recentred(ad: ag.G1Adapter) -> None:
    ad.drive.drifts = [(0.2, 0.0)]
    err = ad.turn_to(math.pi / 2)
    c = ad.calls[-1]
    assert abs(err) < 1e-9
    assert len(c["recentres"]) == 1 and c["recentres"][0]["drift_m"] == pytest.approx(0.2)
    assert c["drift_m"] <= ag.RECENTRE_M
    out = (0.0, round(ag.RECENTRE_OUT_M, 9), 0.15, 0.0)  # out along the target heading, then back to the first turn's xy
    assert ad.drive.goto_calls == [out, (0.0, 0.0, ag.RECENTRE_STOP_M, ag.RECENTRE_MIN_V)]


def test_small_drift_is_left_alone(ad: ag.G1Adapter) -> None:
    ad.drive.drifts = [(0.05, 0.0)]
    ad.turn_to(0.5)
    assert ad.calls[-1]["recentres"] == [] and ad.drive.goto_calls == []


def test_recentre_rounds_are_bounded(ad: ag.G1Adapter) -> None:
    ad.drive.drifts = [(0.2, 0.0)] * 10
    ad.turn_to(1.0)
    assert len(ad.calls[-1]["recentres"]) == ag.RECENTRE_ROUNDS == 2


def test_anchor_survives_look_around(ad: ag.G1Adapter) -> None:
    """Four turns that each drift 0.06 m (under RECENTRE_M alone) still come back: the anchor is the first turn's xy."""
    ad.drive.drifts = [(0.06, 0.0)] * 4 + [(0.0, 0.0)] * 4
    for k in range(4):
        ad.turn_to(k * math.pi / 2)
    assert math.hypot(ad.drive.pose.x, ad.drive.pose.y) <= ag.RECENTRE_M


def test_go_to_anchors_at_the_goal(ad: ag.G1Adapter) -> None:
    ad.drive.drifts = [(0.0, 0.15)]
    r = ad.go_to(*HALL_PLACE, 0.0, 120.0)
    assert (r.status, r.nav_status) == ("reached", "SUCCEEDED")
    assert ad.drive.goto_calls[-1] == (*HALL_PLACE, ag.RECENTRE_STOP_M, ag.RECENTRE_MIN_V)
    assert ad.calls[-1]["turn"]["drift_m"] <= ag.RECENTRE_M


def test_look_around_after_a_short_go_to_ends_at_its_own_start(ad: ag.G1Adapter) -> None:
    """go_to stops 0.12 m short of the goal (inside STOP_TOL_M); the look_around after it must not walk to the goal:
    its 4 turns (each drifting 0.05 m) end within RECENTRE_M of where the step began (hand help SLACK_M 0.10)."""
    ad.drive.short = 0.12
    assert ad.go_to(*HALL_PLACE, 0.0, 120.0).status == "reached"
    ad.drive.short = 0.0
    start = (ad.drive.pose.x, ad.drive.pose.y)
    assert math.dist(start, HALL_PLACE) == pytest.approx(0.12)
    ad.drive.drifts = [(0.0, 0.05)] * 4 + [(0.0, 0.0)] * 8
    for k in range(4):
        ad.turn_to(k * math.pi / 2)
    assert math.dist((ad.drive.pose.x, ad.drive.pose.y), start) <= ag.RECENTRE_M
    assert all(c["anchor"] == list(start) for c in ad.calls[-4:])


def test_stalled_leg_is_blocked(ad: ag.G1Adapter) -> None:
    ad.drive.stall = True
    r = ad.go_to(*HALL_PLACE, 0.0, 120.0)
    assert (r.status, r.nav_status) == ("blocked", "STALLED")
    assert ad.drive.held == ag.SETTLE_S  # returns standing still, so the caller's GT read is the rest pose


def test_whole_call_timeout(ad: ag.G1Adapter) -> None:
    """The leg ends inside the limit but the final turn overruns it -> TIMEOUT, never 'reached'."""
    ad.drive.leg_s, ad.drive.turn_s = 100.0, 30.0
    r = ad.go_to(*HALL_PLACE, 0.0, 120.0)
    assert (r.status, r.nav_status) == ("blocked", "TIMEOUT") and r.sim_s > 120.0


def test_leg_timeout(ad: ag.G1Adapter) -> None:
    ad.drive.leg_s = 200.0
    r = ad.go_to(*HALL_PLACE, 0.0, 120.0)
    assert (r.status, r.nav_status) == ("blocked", "TIMEOUT")


def test_yaw_none_faces_travel(ad: ag.G1Adapter) -> None:
    r = ad.go_to(*HALL_PLACE, None, 120.0)
    assert r.status == "reached"
    assert ad.drive.pose.yaw == pytest.approx(math.atan2(HALL_PLACE[1], HALL_PLACE[0]))


def map_walls() -> list[tuple[Pt, Pt]]:
    """Wall segments of map_knowledge.json: every room polygon edge (the hall's too), each doorway on it cut out."""
    rooms = load_map(MAP)["rooms"].values()
    doors = [r["doorway"] for r in rooms if "doorway" in r]
    walls = []
    for r in rooms:
        for a, b in zip(r["polygon"], r["polygon"][1:] + r["polygon"][:1]):
            k = 0 if a[1] == b[1] else 1  # coordinate that varies along the edge
            lo, hi = sorted((a[k], b[k]))
            cuts = sorted((d["center"][k] - d["width"] / 2, d["center"][k] + d["width"] / 2) for d in doors
                          if abs(d["center"][1 - k] - a[1 - k]) < 1e-9 and lo < d["center"][k] < hi)
            for u, v in zip([lo] + [c[1] for c in cuts], [c[0] for c in cuts] + [hi]):
                p, q = list(a), list(a)
                p[k], q[k] = u, v
                walls.append((tuple(p), tuple(q)))
    return walls


def crosses(p: Pt, q: Pt, a: Pt, b: Pt) -> bool:
    """Do segments p-q and a-b intersect (touching counts)?"""
    def side(o: Pt, s: Pt, t: Pt) -> float:
        return (s[0] - o[0]) * (t[1] - o[1]) - (s[1] - o[1]) * (t[0] - o[0])
    d1, d2, d3, d4 = side(a, b, p), side(a, b, q), side(p, q, a), side(p, q, b)
    return d1 * d2 <= 0 and d3 * d4 <= 0


def test_map_walls_cut_the_doorways() -> None:
    assert crosses((3.6, -0.9), (3.6, -2.1), (0.0, -1.5), (5.0, -1.5))  # the uncut storage wall would block its door
    assert not any(crosses((3.6, -0.9), (3.6, -2.1), *w) for w in map_walls())
    assert any(crosses((0.0, 0.0), (2.5, -3.9), *w) for w in map_walls())  # p8d11's leg: through the storage wall


def test_go_to_routes_from_the_pose_after_a_reset(ad: ag.G1Adapter) -> None:
    """p8d11: the previous run ended at storage; G1Scene.place reset the robot to the hall spawn while nothing spun the
    adapter's node, so the pose still said storage and go_to(storage) walked one straight leg from the spawn into the
    storage wall. go_to must route from where the robot is (sync first): every leg it drives misses every wall."""
    ad.drive.pose.x, ad.drive.pose.y = 2.48, -3.9
    ad.drive.fresh = (0.0, 0.0)
    assert ad.go_to(2.5, -3.9, 0.0, 120.0).status == "reached"
    legs = [(x, y) for x, y, *_ in ad.drive.goto_calls]
    assert legs[:3] == [(3.6, -0.9), (3.6, -2.1), (2.5, -3.9)]
    walls = map_walls()
    for p, q in zip([(0.0, 0.0), *legs], legs):
        assert not any(crosses(p, q, *w) for w in walls), (p, q)


def test_recentre_skips_the_out_leg_into_a_wall(ad: ag.G1Adapter) -> None:
    """The scan sees a wall 0.3 m ahead after the turn: no out leg (it would walk into the wall), only the way back."""
    ad.drive.drifts = [(0.2, 0.0)]
    ad.drive.wall_ahead_m = 0.3
    ad.turn_to(math.pi / 2)
    assert ad.drive.goto_calls == [(0.0, 0.0, ag.RECENTRE_STOP_M, ag.RECENTRE_MIN_V)]
    assert ad.calls[-1]["recentres"][0]["out_leg"] is None
