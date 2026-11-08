"""G1 form of adapters/run_task.teleport() (task 012 section 1): place the spec objects of a layout in a running g1_node.

    G1Scene().place(want, log) -> None | why     want = run_task.layout_plan's {gz model name: (x, y, z, yaw)}

One `layout` parameter set on /g1_sim (rcl_interfaces SetParameters; robot back to the hall spawn + every body's mocap
pose), then the read-back from /g1/objects (transient local) carrying that seq: every body within LAND_M / LAND_DEG and
robot_z >= FALL_Z_M. Same contract and `log` keys (models, sends, retries) as teleport(): a set that times out (no reply)
is sent again, at most SET_RETRIES times, each logged; a refused set is not retried.
Runner side: imports rclpy only (no mujoco, no torch) and no truth module; it only gets the poses to apply.
"""
from __future__ import annotations

import json
import math
import time
from typing import Any

import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

LAND_M, LAND_DEG = 0.02, 1.0  # = adapters/run_task.py LAND_M, LAND_DEG (INTERFACES section 10)
READBACKS, READBACK_WAIT_S = 3, 1.0  # = run_task.READBACKS, READBACK_WAIT_S
SET_RETRIES, SET_WAIT_S = 2, 1.0  # = run_task.SET_POSE_RETRIES, SET_POSE_WAIT_S (a recorded deviation.2)
SET_TIMEOUT_S = 5.0  # = run_task.SET_POSE's gz `--timeout 5000`
SETTLE_S = 2.0  # = g1_node.SETTLE_S: sim seconds at zero command before the node echoes the seq
FALL_Z_M = 0.5  # Phase 9 base-height bar (task 012 section 4: drive.FALL_Z_M)
HALL = (0.0, 0.0, 0.0)  # = humanoid.room.CLINIC_SPAWN (x, y, yaw): the robot is reset there on every layout set
NODE = "/g1_sim"


def check(want: dict[str, tuple[float, float, float, float]], got: dict[str, Any]) -> list[str]:
    """Why each body / the robot missed in one /g1/objects message; [] when all landed."""
    bad = []
    for name, w in want.items():
        if name not in got["bodies"]:
            bad.append(f"{name} not in /g1/objects")
            continue
        g = got["bodies"][name]
        dm, da = math.dist(g[:3], w[:3]), abs(math.degrees(math.remainder(g[3] - w[3], 2 * math.pi)))
        if dm > LAND_M or da > LAND_DEG:
            bad.append(f"{name} at {[round(v, 4) for v in g]}, want {[round(v, 4) for v in w]}: {dm:.3f} m, {da:.2f} deg")
    if got["robot_z"] < FALL_Z_M:
        bad.append(f"robot_z {got['robot_z']:.3f} < {FALL_Z_M} after the reset")
    return bad


class G1Scene:
    """Own rclpy node + executor (spun inside place() only), so it never shares the adapter's node."""

    def __init__(self) -> None:
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node("g1_scene")
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(self.node)
        self.cli = self.node.create_client(SetParameters, f"{NODE}/set_parameters")
        self.last: dict[str, Any] | None = None
        self.node.create_subscription(String, "/g1/objects", lambda m: setattr(self, "last", json.loads(m.data)),
                                      QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def _set(self, text: str) -> tuple[str, str]:
        """('ok' | 'refused' | 'timeout', reason) of one SetParameters call."""
        if not self.cli.wait_for_service(timeout_sec=SET_TIMEOUT_S):
            return "timeout", f"{NODE}/set_parameters not available in {SET_TIMEOUT_S} s"
        val = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=text)
        fut = self.cli.call_async(SetParameters.Request(parameters=[Parameter(name="layout", value=val)]))
        self.ex.spin_until_future_complete(fut, timeout_sec=SET_TIMEOUT_S)
        if not fut.done():
            return "timeout", f"no reply in {SET_TIMEOUT_S} s"
        r = fut.result().results[0]
        return ("ok", "") if r.successful else ("refused", r.reason)

    def place(self, want: dict[str, tuple[float, float, float, float]], log: dict[str, Any] | None = None) -> str | None:
        """Robot to HALL, every want[name] = (x, y, z, yaw) as a mocap pose; None when all landed, else why."""
        log = {} if log is None else log
        log.update(models=len(want), sends=0, retries=[])
        seq = time.time_ns()  # unique across runner restarts, so a stale transient-local echo never matches
        text = json.dumps({"seq": seq, "robot": list(HALL), "bodies": {k: list(v) for k, v in want.items()}})
        for k in range(SET_RETRIES + 1):
            status, why = self._set(text)
            log["sends"] += 1
            if status == "ok":
                break
            if status == "refused" or k == SET_RETRIES:
                return f"set layout: {status}: {why}" + (f" (send {k + 1} of {SET_RETRIES + 1})" if k else "")
            log["retries"].append(f"set layout: send {k + 1} {status}: {why}; sent again")
            time.sleep(SET_WAIT_S)
        bad = [f"no /g1/objects with seq {seq}"]
        for _ in range(READBACKS):  # each read-back waits out the node's settle, then READBACK_WAIT_S
            end = time.monotonic() + SETTLE_S + READBACK_WAIT_S
            while time.monotonic() < end and (self.last is None or self.last["seq"] != seq):
                self.ex.spin_once(timeout_sec=0.05)
            if self.last is not None and self.last["seq"] == seq:  # the echo of this set is final: no re-read
                bad = check(want, self.last)
                return None if not bad else f"read-back seq {seq}: " + "; ".join(bad)
        return f"after {READBACKS} read-backs: " + "; ".join(bad)

    def close(self) -> None:
        """Idempotent; call before the default context is shut down (robots.make closes the adapter last)."""
        if self.node is None:
            return
        self.ex.shutdown()
        self.node.destroy_node()
        self.node = None
