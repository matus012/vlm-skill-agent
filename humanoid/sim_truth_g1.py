"""Evaluator-side G1 ground truth (012 design section 2). Never reachable from agent/ and never the adapter's node.

    truth = G1SimTruth()                      # own rclpy context + node + executor thread on /g1/odom (best effort)
    truth.gt_pose()                           # latest odom sample (x, y, yaw), world frame == map frame
    trace = G1Trace(truth.gt_pose, run_dir / "gt_trace.jsonl").start()
    try: ...the run...
    finally: summary = trace.stop()           # GtTrace summary + "history_samples"

G1Trace = adapters.gt_trace.GtTrace (2 Hz gt_trace.jsonl, unchanged format) plus run_dir/gt_history.jsonl with every
/g1/odom sample while it runs: {t_sim, wall_t, x, y, z, yaw}. Like GtTrace, it never raises into the run.
"""
from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import IO

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data as SENSOR_QOS

from adapters.gt_trace import GtTrace
from humanoid.drive import ODOM_HANG_S
from humanoid.geometry import xyzw_to_wxyz, yaw_from_wxyz

Sample = dict[str, float]


class G1SimTruth:
    def __init__(self, name: str = "g1_sim_truth") -> None:
        self.ctx = rclpy.Context()
        self.ctx.init()
        self.node = Node(name, context=self.ctx)
        self.latest: Sample | None = None
        self.sinks: list[Callable[[Sample], None]] = []  # G1Trace history writers (called on the executor thread)
        self.sink_errors = 0
        self.node.create_subscription(Odometry, "/g1/odom", self._on_odom, SENSOR_QOS)
        self.executor = SingleThreadedExecutor(context=self.ctx)
        self.executor.add_node(self.node)
        self.stopping = threading.Event()  # close(): the spin loop exits within one 0.1 s wait, then it is joined
        self.thread = threading.Thread(target=self._spin, name="g1_sim_truth", daemon=True)
        self.thread.start()
        w0 = time.monotonic()
        while self.latest is None:
            if time.monotonic() - w0 > ODOM_HANG_S:
                self.close()
                raise RuntimeError(f"G1SimTruth: no /g1/odom within {ODOM_HANG_S:.0f} s wall")
            time.sleep(0.05)

    def _spin(self) -> None:
        while not self.stopping.is_set():
            self.executor.spin_once(timeout_sec=0.1)

    def _on_odom(self, m: Odometry) -> None:
        p, o = m.pose.pose.position, m.pose.pose.orientation
        s = {"t_sim": m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec, "wall_t": time.time(),
             "x": p.x, "y": p.y, "z": p.z, "yaw": yaw_from_wxyz(xyzw_to_wxyz((o.x, o.y, o.z, o.w)))}
        self.latest = s
        for sink in list(self.sinks):
            try:
                sink(s)
            except Exception:  # noqa: BLE001 (production path: a trace error must not stop the truth thread)
                self.sink_errors += 1

    def sample(self) -> Sample:
        """Latest odom sample; raises when none arrived for ODOM_HANG_S wall (node dead or hung)."""
        s = self.latest
        assert s is not None and time.time() - s["wall_t"] <= ODOM_HANG_S, f"stale /g1/odom: {s}"
        return s

    def gt_pose(self) -> tuple[float, float, float]:
        s = self.sample()
        return s["x"], s["y"], s["yaw"]

    def close(self) -> None:
        """Idempotent, in order: stop the spin loop and join its thread, shut the executor down, destroy the node, shut
        the context down. rclpy's Executor.shutdown() destroys its guard condition without waiting for a thread still
        inside spin(); with that thread never joined, the process aborted at exit about half the time (rc 134,
        "terminate called without an active exception", PERUN job 145060)."""
        if self.stopping.is_set():
            return
        self.stopping.set()
        self.thread.join()
        self.executor.shutdown()
        self.node.destroy_node()
        self.ctx.try_shutdown()


class G1Trace(GtTrace):
    """GtTrace on G1SimTruth.gt_pose (must be that bound method: the history comes from its odom sinks)."""

    def __init__(self, gt_pose: Callable[[], tuple[float, float, float]], path: Path) -> None:
        super().__init__(gt_pose, path)
        self.truth: G1SimTruth = gt_pose.__self__  # type: ignore[attr-defined]
        self.history_path = self.path.parent / "gt_history.jsonl"
        self.history_samples = 0
        self._h: IO[str] | None = None

    def _on_sample(self, s: Sample) -> None:
        self._h.write(json.dumps(s) + "\n")
        self.history_samples += 1

    def start(self) -> G1Trace:
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self._h = self.history_path.open("a")
            self.truth.sinks.append(self._on_sample)
        except Exception as e:  # noqa: BLE001 (production path, as GtTrace.start)
            self.errors, self.last_error = self.errors + 1, repr(e)
        super().start()
        return self

    def stop(self) -> dict[str, int | str | None]:
        try:
            if self._on_sample in self.truth.sinks:
                self.truth.sinks.remove(self._on_sample)
            if self._h is not None:
                self._h.close()
        except Exception as e:  # noqa: BLE001
            self.errors, self.last_error = self.errors + 1, repr(e)
        summary = super().stop()
        summary["history_samples"] = self.history_samples
        return summary
