"""G1SimTruth (own context + executor thread on /g1/odom, best effort) and G1Trace (GtTrace + gt_history.jsonl)."""
from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path

import pytest
import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data as SENSOR_QOS

import adapters.gt_trace as gt_trace
from humanoid.sim_truth_g1 import G1SimTruth, G1Trace

DOMAIN = "64"  # private ROS domain for the loopback test: no project sim uses it (TB4 15+, G1 18)


def odom(t: float, x: float, z: float = 0.75, yaw: float = 0.3) -> Odometry:
    m = Odometry()
    m.header.stamp.sec, m.header.stamp.nanosec = int(t), int(round((t % 1) * 1e9)) % 1_000_000_000
    m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = x, -x, z
    m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return m


class FakeTruth(G1SimTruth):
    def __init__(self) -> None:  # no ROS: samples are fed by hand
        self.latest, self.sinks, self.sink_errors = None, [], 0


def test_trace_writes_demo_format_and_every_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gt_trace, "PERIOD_S", 0.05)
    truth = FakeTruth()
    truth._on_odom(odom(0.0, 0.0))
    tr = G1Trace(truth.gt_pose, tmp_path / "gt_trace.jsonl").start()
    for i in range(1, 51):
        truth._on_odom(odom(0.02 * i, 0.01 * i))
        time.sleep(0.004)
    s = tr.stop()
    truth._on_odom(odom(2.0, 9.0))  # after stop: not recorded
    hist = [json.loads(line) for line in (tmp_path / "gt_history.jsonl").read_text().splitlines()]
    assert s["history_samples"] == len(hist) == 50 and s["errors"] == 0, s
    assert set(hist[0]) == {"t_sim", "wall_t", "x", "y", "z", "yaw"}
    assert hist[-1]["t_sim"] == pytest.approx(1.0) and hist[-1]["x"] == pytest.approx(0.5)
    assert hist[-1]["yaw"] == pytest.approx(0.3) and hist[-1]["z"] == pytest.approx(0.75)
    lines = [json.loads(line) for line in (tmp_path / "gt_trace.jsonl").read_text().splitlines()]
    assert set(lines[0]) == {"wall_t", "x", "y", "yaw", "call_s"} and "stop" in lines[-1]  # GtTrace format
    assert truth.sinks == []


def test_truth_node_reads_odom_on_its_own_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ROS_DOMAIN_ID", DOMAIN)
    ctx = rclpy.Context()
    ctx.init()
    pub_node = rclpy.create_node("fake_g1_odom", context=ctx)
    pub = pub_node.create_publisher(Odometry, "/g1/odom", SENSOR_QOS)
    stop = threading.Event()

    def publish() -> None:
        k = 0
        while not stop.is_set():
            pub.publish(odom(0.02 * k, 1.5))
            k += 1
            time.sleep(0.02)

    th = threading.Thread(target=publish, daemon=True)
    th.start()
    try:
        truth = G1SimTruth()
        x, y, yaw = truth.gt_pose()
        assert (x, y, yaw) == pytest.approx((1.5, -1.5, 0.3)) and truth.thread.is_alive()
        t1 = truth.sample()["t_sim"]
        time.sleep(0.3)
        assert truth.sample()["t_sim"] > t1  # still updating without anyone spinning it
        truth.close()
        assert not truth.thread.is_alive() and not truth.ctx.ok()  # joined, context down (PERUN 145060 rc 134)
        truth.close()  # idempotent
    finally:
        stop.set()
        th.join(1.0)
        pub_node.destroy_node()
        ctx.try_shutdown()
