import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import skills  # noqa: E402
from agent.fake_adapter import PLACES, FakeAdapter  # noqa: E402
from agent.skills import SkillContext  # noqa: E402

ROOMS = ["ward", "reception", "office", "storage"]


def ctx_at(dx: float, loc_yaw: float = 0.5) -> tuple[SkillContext, FakeAdapter]:
    """Robot localized dx metres east of the storage place point (place yaw 0.0), heading loc_yaw."""
    ad = FakeAdapter("storage")
    ad.x, ad.y, ad.yaw = PLACES["storage"][0] + dx, PLACES["storage"][1], loc_yaw
    return SkillContext(ad, PLACES, "find", ROOMS), ad


def targets(ctx: SkillContext) -> list[float]:
    r = skills.execute("look_around", {}, ctx)
    assert [p.number for p in r.photos] == [1, 2, 3, 4] and r.text == "4 photos"
    return [p.frame.meta["heading_target_rad"] for p in r.photos]


@pytest.mark.parametrize("dx", [0.0, 0.30])
def test_look_around_at_a_place_starts_at_place_yaw(dx: float) -> None:
    ctx, _ = ctx_at(dx)
    assert ctx.place_at() == "storage"
    assert targets(ctx) == pytest.approx([0.0, math.pi / 2, math.pi, -math.pi / 2])


def test_look_around_off_place_starts_at_localization_yaw() -> None:
    ctx, _ = ctx_at(0.31, loc_yaw=0.5)
    assert ctx.place_at() is None
    want = [skills.wrap(0.5 + k * math.pi / 2) for k in range(4)]
    assert targets(ctx) == pytest.approx(want)


def test_look_around_start_read_once_before_turning() -> None:
    ctx, ad = ctx_at(0.31, loc_yaw=3.0)  # turning changes the pose yaw; targets must still be 3.0 + k*90deg
    assert targets(ctx) == pytest.approx([skills.wrap(3.0 + k * math.pi / 2) for k in range(4)])
    assert ad.calls.count("turn_to") == 4 and ad.calls.count("photo") == 4


def test_navigate_to_returns_one_photo_numbered_1_after_arrival() -> None:
    ad = FakeAdapter("hall")
    r = skills.execute("navigate_to", {"place": "office"}, SkillContext(ad, PLACES, "find", ROOMS))
    assert r.text == "reached" and [p.number for p in r.photos] == [1]
    assert ad.calls[-3:] == ["go_to", "pose", "photo"] and r.photos[0].frame.meta["loc_pose"][:2] == [2.5, 3.8]
    assert r.detail["loc_dist_m"] == pytest.approx(0.0)


@pytest.mark.parametrize("short, text", [(0.43, "failed: blocked"), (0.31, "failed: blocked"), (0.30, "reached"),
                                         (0.29, "reached")])
def test_navigate_to_reached_only_within_place_bound(short: float, text: str) -> None:
    """INTERFACES section 2: Nav2 SUCCEEDED short of the place (planner tolerance) is not 'reached'."""
    x, y, _ = PLACES["office"]
    ad = FakeAdapter("hall", short=((x, y, short),))
    r = skills.execute("navigate_to", {"place": "office"}, SkillContext(ad, PLACES, "find", ROOMS))
    assert r.text == text and r.detail["nav_status"] == "SUCCEEDED"
    assert r.detail["loc_dist_m"] == pytest.approx(short) and [p.number for p in r.photos] == [1]


def test_report_stops() -> None:
    ad = FakeAdapter("hall")
    r = skills.execute("report", {"answer": "ward", "photo": 1, "note": ""}, SkillContext(ad, PLACES, "find", ROOMS))
    assert r.done and r.text == "reported" and ad.stopped and r.photos == []


def test_load_places_real_file() -> None:
    p = skills.load_places(ROOT / "sim" / "worlds" / "clinic" / "places.yaml")
    assert set(p) == {"hall", "ward", "office", "reception", "storage"}
    assert p["hall"] == pytest.approx((2.75, -0.1, 0.7854))
