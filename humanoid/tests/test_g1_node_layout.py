"""Task 012 P1, node level: a real g1_node (closed world, paced) takes two `layout` sets through G1Scene.place().
/clock stays strictly increasing across both (d.time is never reset), each set echoes on /g1/objects and lands, a set
naming a non-mocap body is refused, and /g1/scan carries the 360-ray fan in frame pelvis_yaw.
Takes the shared GPU lock (gz.lockdir); skipped when another run holds it. Runs its node in its own process group.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data as SENSOR_QOS
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan

from humanoid.scene_g1 import G1Scene
from humanoid.tests.test_scene_mocap import G1_XML, ROOT, agent_truth, layout_plan_fn, specs

LOCK = Path(os.environ.get("BUILD_DIR", Path.home() / "vlm_skill_agent_build")) / "gz.lockdir"
UP_S = 180.0  # task 012 section 3: readiness bound


@pytest.mark.skipif(not G1_XML.exists(), reason="run humanoid/setup.sh first")
def test_layout_sets_keep_clock_monotonic(tmp_path: Path) -> None:
    try:
        LOCK.mkdir()
    except FileExistsError:
        pytest.skip(f"{LOCK} held by another run")
    node_proc = None
    try:
        log = (tmp_path / "node.log").open("w")
        node_proc = subprocess.Popen(
            [str(ROOT / "humanoid/run_g1.sh"), str(ROOT / "humanoid/g1_node.py"), "--world", "closed", "--scene-dir",
             str(tmp_path / "scene"), "--rate", "1.0", "--stats", str(tmp_path / "stats.json")],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        scene = G1Scene()
        tap = rclpy.create_node("clock_tap")
        clocks: list[int] = []
        scans: list[LaserScan] = []
        tap.create_subscription(Clock, "/clock", lambda c: clocks.append(c.clock.sec * 10**9 + c.clock.nanosec),
                                SENSOR_QOS)
        tap.create_subscription(LaserScan, "/g1/scan", scans.append, SENSOR_QOS)
        ex = SingleThreadedExecutor()
        ex.add_node(tap)
        threading.Thread(target=ex.spin, daemon=True).start()
        end = time.monotonic() + UP_S
        while len(clocks) < 100 and time.monotonic() < end:
            assert node_proc.poll() is None, (tmp_path / "node.log").read_text()[-2000:]
            time.sleep(0.5)
        assert len(clocks) >= 100, "g1_node not up"

        closed = [s for w, s in specs() if w == "closed"]
        frozen = json.loads(agent_truth.world_truth_path("closed").read_text())
        times = []
        for spec in closed[:2]:
            want = layout_plan_fn()("closed", frozen, None, spec)[0]
            tp: dict = {}
            t0 = clocks[-1]
            assert scene.place(want, tp) is None
            assert tp["models"] == 9 and tp["sends"] == 1 and tp["retries"] == []
            assert scene.last["t_sim"] * 1e9 > t0 and scene.last["robot_z"] >= 0.5
            times.append(scene.last["t_sim"])
        tp = {}
        why = scene.place({"structure_walls": (0.0, 0.0, 0.0, 0.0)}, tp)
        assert why is not None and "refused" in why and tp["sends"] == 1, why
        time.sleep(1.0)
        n = len(clocks)
        assert n > 300 and all(b > a for a, b in zip(clocks[:n - 1], clocks[1:n])), "clock went backwards"
        assert times[1] > times[0] >= 2.0
        s = scans[-1]
        assert len(s.ranges) == 360 and s.header.frame_id == "pelvis_yaw" and s.range_max == pytest.approx(4.0)
        ex.shutdown()
        scene.close()
    finally:
        if node_proc is not None and node_proc.poll() is None:
            os.killpg(node_proc.pid, signal.SIGTERM)
            node_proc.wait(timeout=30)
        LOCK.rmdir()
    stats = json.loads((tmp_path / "stats.json").read_text())
    print("node stats:", json.dumps({k: stats[k] for k in ("rtf", "layouts", "ms_per_step", "min_base_z")}))
    assert stats["layouts"] == 2
