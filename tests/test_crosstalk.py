"""probes/crosstalk.py (006 M5) verdict logic on faked check results. No sim, no ROS graph."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))
import crosstalk as ct  # noqa: E402

CORE = list(ct.CORE_NODES)
TOOLS = ["/transform_listener_impl_5a1b2c3d4e5f", "/ros_gz_bridge"]


def test_node_check_clean() -> None:
    other = ["/transform_listener_impl_aaaaaaaaaaaa", "/ros_gz_bridge"]
    r = ct.node_check({40: CORE + TOOLS, 41: CORE + other})
    assert r["pass"], r


def test_node_check_duplicate_is_crosstalk() -> None:
    r = ct.node_check({40: CORE + ["/amcl"], 41: CORE})
    assert not r["pass"] and r["domains"]["40"]["duplicated"] == ["/amcl"] and r["domains"]["41"]["pass"]


def test_node_check_doubled_graph() -> None:
    r = ct.node_check({40: CORE + CORE, 41: CORE + CORE})
    assert not r["pass"] and len(r["domains"]["41"]["duplicated"]) == len(CORE)


def test_node_check_missing_core() -> None:
    r = ct.node_check({40: CORE[1:], 41: CORE})
    assert not r["pass"] and r["domains"]["40"]["missing_core"] == [CORE[0]]


def test_node_check_extra_in_one_domain() -> None:
    r = ct.node_check({40: CORE + ["/photos115"], 41: CORE, 42: CORE})
    assert not r["pass"] and r["domains"]["40"]["extra_vs_other_domains"] == ["/photos115"]
    assert r["domains"]["41"]["pass"] and r["domains"]["42"]["pass"]


def pose(x: float, yaw_deg: float = 0.0) -> ct.sc.Pose3:
    a = math.radians(yaw_deg) / 2
    return np.array([x, 0.0, 0.0]), np.array([0.0, 0.0, math.sin(a), math.cos(a)])


def test_gz_check() -> None:
    a0, at = pose(1.0), pose(1.0, 90)
    b = {"p1": pose(5.0)}
    assert ct.gz_check(a0, at, pose(1.001, 90.05), b, {"p1": pose(5.0005)})["pass"]
    r = ct.gz_check(a0, at, pose(1.0, 90), b, {"p1": pose(5.0, 90)})  # B turned with A: cross-talk
    assert not r["pass"] and r["positive_control"]["pass"] and not r["negative"]["pass"]
    r = ct.gz_check(a0, at, a0, b, b)  # A did not move: the teleport never reached A's sim
    assert not r["pass"] and not r["positive_control"]["pass"] and r["negative"]["pass"]
    assert not ct.gz_check(a0, a0, a0, b, b)["pass"]  # target == before: not a distinct pose


def test_ros_check() -> None:
    n = "abc"
    assert ct.ros_check([n] * 10, {"41": []}, n, 1)["pass"]
    r = ct.ros_check([n], {"41": [], "42": [n]}, n, 2)
    assert not r["pass"] and r["positive_control"]["pass"] and r["negative"]["b_received"] == {"41": 0, "42": 1}
    r = ct.ros_check([], {"41": []}, n, 0)
    assert not r["pass"] and not r["positive_control"]["pass"]
    assert not ct.ros_check(["other"], {"41": []}, n, 1)["pass"]  # only the nonce counts


def test_overall() -> None:
    ok, bad = {"pass": True}, {"pass": False}
    assert ct.overall(ok, ok, ok, ok, True)
    for i in range(4):
        args = [ok] * 4
        args[i] = bad
        assert not ct.overall(*args, True)
    assert not ct.overall(ok, ok, ok, ok, False)


def test_parse_sim_and_normalise() -> None:
    assert ct.parse_sim("41:j123s1") == (41, "j123s1")
    assert ct.normalise("/transform_listener_impl_5a1b2c3d4e5f") == "/transform_listener_impl"
    assert ct.normalise("/amcl") == "/amcl"
