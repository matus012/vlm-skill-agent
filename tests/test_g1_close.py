"""robots.make("g1") then close() leaves no rclpy thread or context behind: the process exits 0 (PERUN job 145060: every
G1 probe aborted at exit, rc 134 "terminate called without an active exception", about half of the runs on the local machine).
A fake /g1/odom + /g1/scan publisher runs here on a private ROS domain; each child process builds the G1 robot, closes
it twice (idempotent) and exits. Several children, since the abort was intermittent."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data as SENSOR_QOS
from sensor_msgs.msg import LaserScan

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = "66"  # private ROS domain: no project sim uses it (TB4 15+, G1 18, test_g1_truth 64)
CHILDREN = 6
CHILD = """
from adapters.robots import make
from humanoid.sim_truth_g1 import G1SimTruth
rb = make("g1")
rb.gt_pose()
rb.close()
rb.close()
import threading
print("threads", sorted(t.name for t in threading.enumerate()), flush=True)
"""


def test_make_close_exits_zero(tmp_path: Path) -> None:
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "clinic.xml").write_text("<mujoco/>")
    (tmp_path / "p" / "scene.json").write_text(json.dumps({"xml": "clinic.xml", "world": "closed",
                                                          "label_of_body": [0], "mocap_bodies": []}))
    env = {**os.environ, "ROS_DOMAIN_ID": DOMAIN, "ROBOT": "g1", "G1_SCENES": str(tmp_path), "GZ_PARTITION": "p",
           "SIM_WORLD": "closed", "PYTHONPATH": f"{ROOT}:{os.environ.get('PYTHONPATH', '')}"}
    os.environ["ROS_DOMAIN_ID"] = DOMAIN  # the publisher's context reads it at init
    ctx = rclpy.Context()
    ctx.init()
    node = rclpy.create_node("fake_g1_sim", context=ctx)
    pub_odom = node.create_publisher(Odometry, "/g1/odom", SENSOR_QOS)
    pub_scan = node.create_publisher(LaserScan, "/g1/scan", SENSOR_QOS)
    stop = threading.Event()

    def publish() -> None:
        while not stop.is_set():
            o = Odometry()
            o.header.stamp.sec, o.pose.pose.orientation.w, o.pose.pose.position.z = 1, 1.0, 0.75
            pub_odom.publish(o)
            pub_scan.publish(LaserScan(ranges=[2.0] * 8))
            time.sleep(0.02)

    th = threading.Thread(target=publish, daemon=True)
    th.start()
    try:
        runs = [subprocess.run([sys.executable, "-c", CHILD], cwd=ROOT, env=env, capture_output=True, text=True,
                               timeout=120) for _ in range(CHILDREN)]
    finally:
        stop.set()
        th.join(1.0)
        node.destroy_node()
        ctx.try_shutdown()
    bad = [(r.returncode, r.stderr[-300:]) for r in runs if r.returncode != 0]
    assert not bad, bad
    assert all("'g1_sim_truth'" not in r.stdout for r in runs), runs[0].stdout  # the truth spin thread was joined
