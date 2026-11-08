#!/usr/bin/env python3
"""G1 sim readiness and params (task 012 design section 3); hpc/g1_lib.sh sim_wait_ready / sim_params_check.

    /opt/venv/bin/python    humanoid/probes/ready.py wait --out s<i>/ready.json        # in the sim's env (in_sim)
    /opt/venv-g1/bin/python humanoid/probes/ready.py params --ready s<i>/ready.json --cmdline "<node /proc cmdline>"

wait (bounded READY_S wall, exit 0 iff ready; ready.json either way, the last stdout line says why):
  >= MIN_ODOM /g1/odom samples; a no-op `layout` set on g1_sim (seq NOOP_SEQ, no robot, no bodies)
  echoed on /g1/objects; then the paced RTF over RTF_WINDOW_S sim in RTF_RANGE; base z >= MIN_BASE_Z_M after it; one
  rgb + seg pair with the same stamp; scene.json world == SIM_WORLD. camera_info K is recorded for `params`.
params (exit 0 iff all hold): camera_info fx = fy = 320, cx 319.5, cy 239.5; scene.json label_of_body sha256 == the
  one of a fresh humanoid.room.build_clinic_scene(SIM_WORLD, tmp, $G1_EXT/<g1 xml>) in this image; the node ran with --rate 1.0
  --torch-threads 2. Scene: $G1_SCENES/$GZ_PARTITION/scene.json (the node's --scene-dir).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
READY_S = 180.0                 # design section 3: bounded at 180 s (one restart, then exit 9: g1_job.sbatch)
MIN_ODOM = 50
MIN_BASE_Z_M = 0.6
RTF_WINDOW_S = 10.0
RTF_RANGE = (0.9, 1.1)
CAM_K = {"fx": 320.0, "fy": 320.0, "cx": 319.5, "cy": 239.5}  # humanoid/README.md camera_info
NODE_ARGS = ("--rate 1.0", "--torch-threads 2")
NOOP_SEQ = -1  # G1Scene.place uses seq = time.time_ns(), so this echo is never taken for a placement's


def scene_json() -> Path:
    return Path(os.environ["G1_SCENES"]) / os.environ["GZ_PARTITION"] / "scene.json"


def labels_sha(scene: dict) -> str:
    return hashlib.sha256(json.dumps(scene["label_of_body"]).encode()).hexdigest()


def ready_problems(rec: dict) -> list[str]:
    """Why a wait record is not ready (empty = ready)."""
    why = []
    if rec["odom_samples"] < MIN_ODOM:
        why.append(f"odom_samples {rec['odom_samples']} < {MIN_ODOM}")
    if not rec["layout_echo"]:
        why.append("no /g1/objects echo of the no-op layout")
    if rec["rtf"] is None or not RTF_RANGE[0] <= rec["rtf"] <= RTF_RANGE[1]:
        why.append(f"paced rtf {rec['rtf']} over {RTF_WINDOW_S} s sim not in {list(RTF_RANGE)}")
    if rec["base_z"] is None or rec["base_z"] < MIN_BASE_Z_M:
        why.append(f"base z {rec['base_z']} < {MIN_BASE_Z_M}")
    if not rec["rgb_seg_pair"]:
        why.append("no rgb + seg pair with the same stamp")
    if rec["scene_world"] != rec["sim_world"]:
        why.append(f"scene.json world {rec['scene_world']!r} != SIM_WORLD {rec['sim_world']!r}")
    return why


def params_problems(cam_k: list[float] | None, cmdline: str, labels_now: str, labels_fresh: str) -> list[str]:
    why = []
    if cam_k is None:
        why.append("no camera_info")
    else:
        got = {"fx": cam_k[0], "fy": cam_k[4], "cx": cam_k[2], "cy": cam_k[5]}
        # at the README's own precision (0.1 px): fovy 73.74 deg gives fx = 319.99997
        why += [f"camera_info {k} {got[k]} (want {v})" for k, v in CAM_K.items() if round(got[k], 1) != v]
    if labels_now != labels_fresh:
        why.append(f"scene.json label_of_body sha {labels_now[:12]} != fresh build {labels_fresh[:12]}")
    why += [f"node not run with {a!r}" for a in NODE_ARGS if f" {a} " not in f" {cmdline} "]
    return why


def wait(out: Path) -> int:
    import rclpy
    from nav_msgs.msg import Odometry
    from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
    from rcl_interfaces.srv import SetParameters
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    def st(m) -> float:  # noqa: ANN001
        return m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec

    w0 = time.monotonic()
    rec = {"odom_samples": 0, "layout_echo": False, "rtf": None, "base_z": None, "rgb_seg_pair": False,
           "sim_world": os.environ.get("SIM_WORLD", "closed"), "scene_world": None, "camera_k": None}
    odom: list[tuple[float, float, float]] = []  # (sim t, wall, z)
    stamps: dict[str, set[float]] = {"rgb": set(), "seg": set()}
    rclpy.init()
    node = rclpy.create_node("g1_ready")
    node.create_subscription(Odometry, "/g1/odom", lambda m: odom.append(
        (st(m), time.monotonic(), m.pose.pose.position.z)), qos_profile_sensor_data)
    node.create_subscription(Image, "/head_rgb/image_raw", lambda m: stamps["rgb"].add(st(m)), 2)
    node.create_subscription(Image, "/head_rgb/segmentation", lambda m: stamps["seg"].add(st(m)), 2)
    node.create_subscription(CameraInfo, "/head_rgb/camera_info", lambda m: rec.__setitem__("camera_k", list(m.k)),
                             qos_profile_sensor_data)
    echo: list[dict] = []
    node.create_subscription(String, "/g1/objects", lambda m: echo.append(json.loads(m.data)), QoSProfile(
        depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
    client = node.create_client(SetParameters, "/g1_sim/set_parameters")

    def spin_until(cond) -> bool:  # noqa: ANN001
        while not cond():
            if time.monotonic() - w0 > READY_S:
                return False
            rclpy.spin_once(node, timeout_sec=0.1)
        return True
    try:
        if spin_until(lambda: len(odom) >= MIN_ODOM) and client.wait_for_service(timeout_sec=max(
                0.0, READY_S - (time.monotonic() - w0))):
            layout = {"seq": NOOP_SEQ, "bodies": {}}  # no robot key: nothing moves
            client.call_async(SetParameters.Request(parameters=[Parameter(name="layout", value=ParameterValue(
                type=ParameterType.PARAMETER_STRING, string_value=json.dumps(layout)))]))
            rec["layout_echo"] = spin_until(lambda: any(e.get("seq") == NOOP_SEQ for e in echo))
        if rec["layout_echo"]:
            t0, wa, _ = odom[-1]
            if spin_until(lambda: odom[-1][0] >= t0 + RTF_WINDOW_S):
                t1, wb, z = odom[-1]
                rec["rtf"], rec["base_z"] = (t1 - t0) / (wb - wa), z
        spin_until(lambda: bool(stamps["rgb"] & stamps["seg"]) and rec["camera_k"] is not None)
    finally:
        rec["odom_samples"] = len(odom)
        rec["rgb_seg_pair"] = bool(stamps["rgb"] & stamps["seg"])
        node.destroy_node()
        rclpy.try_shutdown()
    rec["scene_json"] = str(scene_json())
    rec["scene_world"] = json.loads(scene_json().read_text()).get("world") if scene_json().exists() else None
    why = ready_problems(rec)
    rec.update(ready=not why, why=why, wall_s=round(time.monotonic() - w0, 1))
    out.write_text(json.dumps(rec, indent=1))
    print(f"READY rtf {rec['rtf']:.3f} base_z {rec['base_z']:.3f} world {rec['sim_world']} in {rec['wall_s']} s"
          if not why else "NOT READY: " + "; ".join(why), flush=True)
    return 0 if not why else 1


def params(ready: Path, cmdline: str) -> int:
    from humanoid.g1_node import EXT, G1_REL  # MuJoCo + AGILE: /opt/venv-g1 only
    from humanoid.room import build_clinic_scene
    world = os.environ.get("SIM_WORLD", "closed")
    now = labels_sha(json.loads(scene_json().read_text()))
    with tempfile.TemporaryDirectory() as tmp:
        build_clinic_scene(world, Path(tmp), EXT / G1_REL)
        fresh = labels_sha(json.loads((Path(tmp) / "scene.json").read_text()))
    why = params_problems(json.loads(ready.read_text()).get("camera_k"), cmdline, now, fresh)
    print(f"world {world}, labels sha {now[:12]} (fresh {fresh[:12]}), cmdline: {cmdline}")
    print("\n".join(why))
    print("PARAMS OK" if not why else "PARAMS MISMATCH", flush=True)
    return 0 if not why else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("wait").add_argument("--out", type=Path, required=True)
    p = sub.add_parser("params")
    p.add_argument("--ready", type=Path, required=True)
    p.add_argument("--cmdline", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    return wait(a.out) if a.cmd == "wait" else params(a.ready, a.cmdline)


if __name__ == "__main__":
    raise SystemExit(main())
