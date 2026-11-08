"""Pure parts of probes/robot_bodies.py (task 013) and the recorder's speed flags; mujoco-only parts skip in the main venv."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
import robot_bodies as rb  # noqa: E402

URDF = """<robot name="r">
<link name="base"><visual><origin xyz="0 0 0.1" rpy="0 0 1.5707963267948966"/><geometry><mesh filename="package://irobot_create_description/meshes/a.dae" scale="1 1 2"/></geometry></visual></link>
<joint name="j1" type="fixed"><parent link="base"/><child link="mid"/><origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/></joint>
<link name="mid"/>
<joint name="j2" type="continuous"><parent link="mid"/><child link="wheel"/><origin xyz="0 0.5 0" rpy="-1.5707963267948966 0 0"/></joint>
<link name="wheel"><visual><geometry><cylinder length="0.1" radius="0.2"/></geometry><material name="b"><color rgba="0.1 0.1 0.1 1"/></material></visual></link>
</robot>"""


def test_urdf_visuals_compose_zero_joint_transforms() -> None:
    v = {x.link: x for x in rb.urdf_visuals(URDF)}
    assert v["base"].kind == "mesh" and v["base"].scale == (1, 1, 2)
    np.testing.assert_allclose(v["base"].T[:3, 3], [0, 0, 0.1], atol=1e-12)
    # wheel: base -> mid (x+1, yaw 90) -> wheel (local y 0.5 = world -x 0.5 from mid; roll -90)
    T = v["wheel"].T
    np.testing.assert_allclose(T[:3, 3], [1 - 0.5, 0, 0], atol=1e-12)
    np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-12)
    assert v["wheel"].kind == "cylinder" and v["wheel"].size == (0.2, 0.1) and v["wheel"].rgba == (0.1, 0.1, 0.1, 1.0)


def test_rpy_order_is_urdf_rz_ry_rx() -> None:
    T = rb.rpy_xyz_to_mat((0.3, -0.2, 0.9), (0, 0, 0))
    ref = np.array([[math.cos(0.9), -math.sin(0.9), 0], [math.sin(0.9), math.cos(0.9), 0], [0, 0, 1]])
    ry = np.array([[math.cos(-0.2), 0, math.sin(-0.2)], [0, 1, 0], [-math.sin(-0.2), 0, math.cos(-0.2)]])
    rx = np.array([[1, 0, 0], [0, math.cos(0.3), -math.sin(0.3)], [0, math.sin(0.3), math.cos(0.3)]])
    np.testing.assert_allclose(T[:3, :3], ref @ ry @ rx, atol=1e-12)


@pytest.mark.parametrize("rpy", [(0, 0, 1.0), (0.4, -0.3, 2.0), (math.pi, 0, 0), (0, math.pi, 0.2)])
def test_mat_to_quat_roundtrips(rpy: tuple) -> None:
    R = rb.rpy_xyz_to_mat(rpy, (0, 0, 0))[:3, :3]
    w, x, y, z = rb.mat_to_quat(R)
    Q = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    np.testing.assert_allclose(Q, R, atol=1e-9)


def test_resolve_package_uri() -> None:
    assert rb.resolve_package("package://irobot_create_description/meshes/a.dae").name == "a.dae"
    assert rb.resolve_package("package://turtlebot4_description/meshes/shell.dae").parts[-3:] == (
        "turtlebot4_description", "meshes", "shell.dae")


def test_wall_to_tsim_maps_and_clamps() -> None:
    hw, ht = np.array([100.0, 101.0, 103.0]), np.array([10.0, 10.5, 12.5])
    np.testing.assert_allclose(rb.wall_to_tsim(hw, ht, np.array([99.0, 100.5, 102.0, 200.0])), [10.0, 10.25, 11.5, 12.5])


def _write_qpos(path: Path, n: int = 3, nq: int = 9, bad_tail: bool = False) -> None:
    lines = [json.dumps({"model": "m", "nq": nq})]
    for i in range(n):
        q = [float(i)] * nq
        q[3:7] = [1.0, 0.0, 0.0, 0.0]
        lines.append(json.dumps({"t_sim": 10.0 + i, "qpos": q}))
    path.write_text("\n".join(lines) + ("\n{\"t_sim\": 14.0, \"qpo" if bad_tail else "\n"))


def test_read_qpos_skips_cut_last_line_and_checks_nq(tmp_path: Path) -> None:
    _write_qpos(tmp_path / "q.jsonl", bad_tail=True)
    head, t, q = rb.read_qpos(tmp_path / "q.jsonl")
    assert head["nq"] == 9 and t.tolist() == [10.0, 11.0, 12.0] and q.shape == (3, 9)
    (tmp_path / "bad.jsonl").write_text(json.dumps({"nq": 5}) + "\n" + json.dumps({"t_sim": 0, "qpos": [0.0] * 4}) + "\n")
    with pytest.raises(AssertionError):
        rb.read_qpos(tmp_path / "bad.jsonl")


def test_interp_qpos_linear_clamped_and_quat_normalised(tmp_path: Path) -> None:
    _write_qpos(tmp_path / "q.jsonl")
    _, t, q = rb.read_qpos(tmp_path / "q.jsonl")
    mid = rb.interp_qpos(t, q, 10.5)
    assert mid[0] == pytest.approx(0.5) and mid[8] == pytest.approx(0.5)
    assert np.linalg.norm(mid[3:7]) == pytest.approx(1.0)
    assert rb.interp_qpos(t, q, 0.0)[0] == 0.0 and rb.interp_qpos(t, q, 99.0)[0] == 2.0
    q2 = q.copy()
    q2[1, 3:7] = [0.0, 0.0, 0.0, 1.0]  # a quarter-turn-ish jump: still unit length half-way
    assert np.linalg.norm(rb.interp_qpos(t, q2, 10.5)[3:7]) == pytest.approx(1.0)


def test_choose_stance_prefers_keyframe_else_bent_default() -> None:
    q0 = np.zeros(8)
    q, src = rb.choose_stance(np.arange(8.0), q0, {5: -0.5})
    assert src == "keyframe" and q.tolist() == list(range(8))
    q, src = rb.choose_stance(None, q0, {5: -0.5, 6: 1.0})
    assert src == "default stance" and q[5] == -0.5 and q[6] == 1.0 and q[0] == 0.0 and q0[5] == 0.0  # q0 untouched


def test_history_pose_at_interpolates_and_unwraps_yaw(tmp_path: Path) -> None:
    rows = [{"t_sim": 0.0, "wall_t": 5.0, "x": 0.0, "y": 0.0, "z": 0.7, "yaw": 3.0},
            {"t_sim": 1.0, "wall_t": 6.0, "x": 2.0, "y": 1.0, "z": 0.8, "yaw": -3.0}]
    (tmp_path / "h.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    h = rb.read_history(tmp_path / "h.jsonl")
    x, y, z, yaw = rb.history_pose_at(h, 0.5)
    assert (x, y, z) == pytest.approx((1.0, 0.5, 0.75)) and math.cos(yaw) == pytest.approx(-1.0, abs=0.02)  # via pi, not 0


def test_load_tb4_fails_loudly_without_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rb, "TB4_DIR", tmp_path / "nope")
    with pytest.raises(SystemExit, match="build-tb4"):
        rb.load_tb4()


def test_speed_flags_reach_timeline_and_badges() -> None:
    import demo_recorder as dr
    steps = [{"sim_t0": 10.0, "sim_t1": 70.0, "wall_t0": 0.0, "wall_t1": 60.0, "planner_wall_s": 0.0},
             {"sim_t0": 90.0, "sim_t1": 100.0, "wall_t0": 80.0, "wall_t1": 90.0, "planner_wall_s": 0.0}]
    segs = dr.timeline(steps, act_speed=6.0, think_show_s=4.0)
    act = next(s for s in segs if s.kind == "act")
    think = next(s for s in segs if s.kind == "think" and s.step == 1)
    assert dr.speed_label(act) == "x6 sim time" and think.show_s == 4.0 and dr.speed_label(think) == "x5 (planner wait 20.0 s)"
    assert dr.speed_label(dr.Segment("act", 0, 0.0, 13.0, 2.0)) == "x6.5 sim time"
