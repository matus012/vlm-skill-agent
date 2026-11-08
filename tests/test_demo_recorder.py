"""Pure parts of probes/demo_recorder.py (timeline, speed marking, layout, clock mapping) and adapters/gt_trace.py."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
import demo_recorder as dr  # noqa: E402
from adapters.gt_trace import GtTrace  # noqa: E402

STEPS = [  # 003 counted g1/00-find-toolbox stamps (rounded)
    {"sim_t0": 73.41, "sim_t1": 96.67, "wall_t0": 1000.0, "wall_t1": 1025.6, "planner_wall_s": 0.02},
    {"sim_t0": 96.86, "sim_t1": 151.36, "wall_t0": 1025.8, "wall_t1": 1092.7, "planner_wall_s": 0.0},
    {"sim_t0": 161.55, "sim_t1": 162.39, "wall_t0": 1102.7, "wall_t1": 1103.8, "planner_wall_s": 10.0},
]


def test_timeline_covers_sim_in_order_and_marks_speed() -> None:
    segs = dr.timeline(STEPS)
    assert [s.kind for s in segs] == ["title", "think", "act", "hold", "think", "act", "think", "act", "hold", "end"]
    moving = [s for s in segs if s.sim_b > s.sim_a]
    for a, b in zip(moving, moving[1:]):  # contiguous sim time, no gap, no overlap
        assert b.sim_a == pytest.approx(a.sim_b)
    assert moving[0].sim_a == pytest.approx(73.41 - 0.02) and moving[-1].sim_b == pytest.approx(162.39)
    acts = [s for s in segs if s.kind == "act"]
    assert all(s.speed == pytest.approx(dr.ACT_SPEED) for s in acts)
    think = [s for s in segs if s.kind == "think"]
    assert think[2].show_s == dr.THINK_SHOW_S and think[2].speed == pytest.approx(10.19 / dr.THINK_SHOW_S)
    assert think[1].speed == pytest.approx(1.0)  # a short wait is shown in real time
    for i in range(3):  # every step is on screen for at least the reading time
        assert sum(s.show_s for s in segs if s.step == i and s.kind != "end") >= dr.MIN_STEP_S - 1e-9


def test_speed_labels() -> None:
    segs = dr.timeline(STEPS)
    labels = {s.kind: dr.speed_label(s) for s in segs}
    assert labels["title"] == labels["hold"] == labels["end"] == "PAUSED"
    assert labels["act"] == "x8 sim time"
    assert dr.speed_label(segs[-4]) == "x5 (planner wait 10.2 s)"
    assert dr.speed_label(dr.Segment("think", 0, 0.0, 0.5, 0.5)) == "x1 (planner wait 0.5 s)"


def test_frame_times_monotone_and_complete() -> None:
    segs = dr.timeline(STEPS)
    ft = dr.frame_times(segs, fps=15)
    assert len(ft) == round(sum(s.show_s for s in segs) * 15)
    t = np.array([x for x, _ in ft])
    assert np.all(np.diff(t) >= -1e-9) and t[-1] == pytest.approx(162.39)
    assert ft[0][1].kind == "title" and ft[-1][1].kind == "end"
    act1 = [x for x, s in ft if s.kind == "act" and s.step == 1]  # x8: ~8 sim s per video s
    assert np.median(np.diff(act1)) == pytest.approx(8 / 15, rel=1e-6)


def test_layout_tiles_frame() -> None:
    boxes = dr.layout(1280, 720)
    assert set(boxes) == {"sim", "map", "photos", "text"}
    assert sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in boxes.values()) == 1280 * 720
    assert all((x1 - x0, y1 - y0) == (640, 360) for x0, y0, x1, y1 in boxes.values())
    assert boxes["sim"][:2] == (0, 0) and boxes["text"][2:] == (1280, 720)


def test_wall_to_sim_maps_step_stamps() -> None:
    w = np.array([1000.0, 1012.8, 1025.6, 1025.7, 2000.0])
    t = dr.wall_to_sim(STEPS, w)
    assert t[0] == pytest.approx(73.41) and t[1] == pytest.approx((73.41 + 96.67) / 2)
    assert t[2] == pytest.approx(96.67) and 96.67 < t[3] < 96.86 and t[4] == pytest.approx(162.39)


def test_trim_and_call_str() -> None:
    assert dr.trim(None) is None and dr.trim("a  b\n c") == "a b c"
    long = dr.trim("word " * 200, 50)
    assert len(long) <= 54 and long.endswith("...")
    assert dr.call_str({"name": "navigate_to", "args": {"place": "storage"}}) == 'navigate_to(place="storage")'
    assert dr.call_str({"name": None, "raw": "x"}).startswith("(invalid call)")


def test_gt_trace_writes_samples_and_never_raises(tmp_path: Path) -> None:
    n = {"k": 0}

    def pose() -> tuple[float, float, float]:
        n["k"] += 1
        if n["k"] == 2:
            raise RuntimeError("gz timeout")
        return (1.0, 2.0, 0.5)

    tr = GtTrace(pose, tmp_path / "run" / "gt_trace.jsonl").start()
    time.sleep(1.2)
    summary = tr.stop()
    rows = [json.loads(l) for l in (tmp_path / "run" / "gt_trace.jsonl").read_text().splitlines()]
    assert summary["errors"] == 1 and "gz timeout" in summary["last_error"] and summary["samples"] >= 2
    assert rows[-1] == summary and all({"wall_t", "x", "y", "yaw", "call_s"} <= set(r) for r in rows[:-1])
    assert len(rows) - 1 == summary["samples"]


def test_gt_trace_bad_path_and_hung_call_do_not_raise(tmp_path: Path) -> None:
    (tmp_path / "f").write_text("")
    bad = GtTrace(lambda: (0.0, 0.0, 0.0), tmp_path / "f" / "x" / "gt_trace.jsonl").start()  # parent is a file
    assert bad.stop()["errors"] == 1
    hung = GtTrace(lambda: time.sleep(5) or (0.0, 0.0, 0.0), tmp_path / "gt_trace.jsonl").start()
    t0 = time.time()
    hung.stop()
    assert time.time() - t0 < 1.5
