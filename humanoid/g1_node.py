"""G1 in MuJoCo as a ROS 2 robot: one process owns physics, policy and rendering (all on the main thread).

In:  /cmd_vel (geometry_msgs/Twist; vx, vy, wz clamped by AGILE's CommandManager box, zeroed if stale)
Out: /clock, /g1/odom (nav_msgs/Odometry, ground truth from the free joint, frame world -> pelvis) + TF,
     /head_rgb/image_raw (rgb8), /head_rgb/camera_info, /head_rgb/segmentation (16UC1 body ids, 0 = world),
     /head_rgb_pitch/segmentation (only with --pitch-control: 16UC1 body ids from the same mount pitched 47.6 deg
     down, 2 Hz sim; the H-P2 G7 negative control, never an agent input),
     /g1/scan (sensor_msgs/LaserScan, mj_ray fan from the pelvis, frame pelvis_yaw = pelvis xy + yaw only, 10 Hz sim),
     /g1/objects (std_msgs/String JSON {seq, t_sim, bodies: {gz name: [x, y, z, yaw]}, robot_z}, transient local),
     /g1/contacts (std_msgs/String JSON, best effort, 10 Hz sim: {t_sim, n_mocap_total, n_static_total, clearance_m,
     last: [{robot_body, other_body, t_sim, dist}]}): humanoid/contacts.py robot-clinic contact-steps (floor excluded),
     scanned after every physics step, cumulative since node start; clearance_m = min pelvis -> non-floor geom distance
     (mj_geomDistance, capped at SCAN_MAX_M, null beyond) over the policy steps since the previous message.
Param `layout` (task 012 section 1): JSON {seq, robot: [x, y, yaw] (optional), bodies: {gz name: [x, y, z, yaw]}}. The
callback only queues it; the main thread applies it before the next policy step: robot to the default stance at
(x, y, yaw) with qvel 0 and policy.reset(), mocap poses, mj_forward, then 2 s sim at zero command (/cmd_vel ignored),
then one /g1/objects message with that seq. d.time is never reset (/clock stays monotonic).
The agent never reads /g1/odom, /g1/objects or /g1/scan (brief rule); the waypoint controller, runner and probes do.

Walking: WBC-AGILE `Velocity-G1-v0` (LEAPP ONNX) through AGILE's own sim2mujoco classes, unchanged:
physics 500 Hz, policy 50 Hz (decimation 10), exactly as scripts/sim2mujoco_eval.py runs it.

CPU pinning (PERUN job 145060): AGILE / leapp build their onnxruntime sessions with default thread options, so ORT starts
one intra-op thread per physical core of the whole machine and pins each to its core, ignoring the process affinity
(`taskset -c <sim cpus>`). Under a job cpuset the pins inside it succeed (the threads leave the sim's cpus: union
"allowed 56-119" for a sim on 64-79) and the rest fail ("pthread_setaffinity_np failed ... error code: 22").
ort_one_thread() gives every session 1 intra-op + 1 inter-op thread and no spinning (no pool, so no pinning), and
main() re-applies the start affinity to every thread after init and logs both sets.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import threading
import time
from collections import deque
from pathlib import Path

AFFINITY0 = os.sched_getaffinity(0)  # before any heavy import (ORT, torch, EGL): the cpus this sim was started on
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import rclpy  # noqa: E402
import torch  # noqa: E402
from builtin_interfaces.msg import Time  # noqa: E402
from geometry_msgs.msg import TransformStamped, Twist  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rcl_interfaces.msg import SetParametersResult  # noqa: E402
from rclpy.qos import DurabilityPolicy, QoSProfile  # noqa: E402
from rclpy.qos import qos_profile_sensor_data as SENSOR_QOS  # noqa: E402
from rosgraph_msgs.msg import Clock  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image, LaserScan  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

from agile.sim2mujoco.leapp import (  # noqa: E402
    LeappPolicyController,
    create_leapp_command_provider,
    load_leapp_description,
    resolve_leapp_bundle,
    synthesize_sim_config,
)
from agile.sim2mujoco.simulation import MuJocoSimulation  # noqa: E402
from humanoid.contacts import ContactCounter  # noqa: E402
from humanoid.qpos_log import QposLog  # noqa: E402
from humanoid.geometry import intrinsics_from_fovy, wxyz_to_xyzw  # noqa: E402
from humanoid.room import (CAM_H, CAM_NAME, CAM_W, PITCH_CAM_NAME, CLINIC_SPAWN, SCAN_HZ, SCAN_MAX_M, SCAN_MIN_M, SCAN_N,  # noqa: E402
                           body_poses, build_clinic_scene, build_scene, load_room, place_bodies, ray_fan, yaw_quat)

EXT = Path(os.environ.get("G1_EXT", str(Path(os.environ.get("BUILD_DIR", Path.home() / "vlm_skill_agent_build")) / "ext")))
SCENE_DIR = Path(os.environ.get("G1_SCENE_DIR", str(Path(os.environ.get("BUILD_DIR", Path.home() / "vlm_skill_agent_build")) / "g1_scene")))
LEAPP_REL = "agile/agile/data/policy/velocity_g1/leapp/Velocity-G1-v0/Velocity-G1-v0.yaml"
G1_REL = "unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
CMD_TIMEOUT_S = 0.5  # sim seconds without /cmd_vel -> zero command
SETTLE_S = 2.0  # task 012 section 1: sim seconds at zero command after a layout set, before /g1/objects
CONTACT_HZ, CONTACT_LAST = 10.0, 5  # /g1/contacts rate (sim) and how many of the latest contacts it carries


def ort_one_thread() -> None:
    """Every later ort.InferenceSession (AGILE policy.py, leapp export_backend.py look it up on the module at call time)
    gets intra_op_num_threads = inter_op_num_threads = 1 and no spinning, on the caller's options or on new ones."""
    base = ort.InferenceSession
    if getattr(base, "one_thread", False):
        return

    class OneThreadSession(base):  # type: ignore[misc, valid-type]
        one_thread = True

        def __init__(self, path_or_bytes: object, sess_options: ort.SessionOptions | None = None, *a: object,
                     **kw: object) -> None:
            so = ort.SessionOptions() if sess_options is None else sess_options
            so.intra_op_num_threads = so.inter_op_num_threads = 1
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            so.add_session_config_entry("session.inter_op.allow_spinning", "0")
            super().__init__(path_or_bytes, so, *a, **kw)

    ort.InferenceSession = OneThreadSession


def cpus(s: set[int]) -> str:
    """{2, 3, 4, 7} -> '2-4,7'."""
    out, xs = [], sorted(s)
    for i, c in enumerate(xs):
        if i == 0 or c != xs[i - 1] + 1:
            out.append([c, c])
        else:
            out[-1][1] = c
    return ",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in out)


def repin(want: set[int]) -> dict[str, object]:
    """Re-apply `want` to every thread of this process; the union of the threads' affinities before, and the count of
    threads that differed from `want`."""
    before, moved = set(), 0
    for t in Path("/proc/self/task").iterdir():
        try:
            got = os.sched_getaffinity(int(t.name))
            before |= got
            if got != want:
                os.sched_setaffinity(int(t.name), want)
                moved += 1
        except (ProcessLookupError, FileNotFoundError):  # production path: a thread may exit while we walk the list
            pass
    return {"start": cpus(want), "after_init_union": cpus(before), "threads_repinned": moved}


def stamp(t_ns: int) -> Time:
    return Time(sec=t_ns // 1_000_000_000, nanosec=t_ns % 1_000_000_000)


def odom_msg(qpos: np.ndarray, qvel: np.ndarray, t_ns: int) -> Odometry:
    m = Odometry()
    m.header.stamp = stamp(t_ns)
    m.header.frame_id = "world"
    m.child_frame_id = "pelvis"
    p, o = m.pose.pose.position, m.pose.pose.orientation
    p.x, p.y, p.z = (float(v) for v in qpos[:3])
    o.x, o.y, o.z, o.w = (float(v) for v in wxyz_to_xyzw(tuple(qpos[3:7])))
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, qpos[3:7])
    v_body = rot.reshape(3, 3).T @ qvel[:3]  # free-joint linear qvel is world frame; Odometry twist is child frame
    m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z = (float(v) for v in v_body)
    m.twist.twist.angular.x, m.twist.twist.angular.y, m.twist.twist.angular.z = (float(v) for v in qvel[3:6])
    return m


def tf_msg(odom: Odometry) -> TFMessage:
    t = TransformStamped()
    t.header = odom.header
    t.child_frame_id = odom.child_frame_id
    p = odom.pose.pose.position
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = p.x, p.y, p.z
    t.transform.rotation = odom.pose.pose.orientation
    return TFMessage(transforms=[t])


def camera_info(fovy_deg: float, t_ns: int) -> CameraInfo:
    fx, fy, cx, cy = intrinsics_from_fovy(fovy_deg, CAM_W, CAM_H)
    ci = CameraInfo()
    ci.header.stamp = stamp(t_ns)
    ci.header.frame_id = f"{CAM_NAME}_optical_frame"
    ci.width, ci.height = CAM_W, CAM_H
    ci.distortion_model = "plumb_bob"
    ci.d = [0.0] * 5
    ci.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    ci.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return ci


def image_msg(arr: np.ndarray, encoding: str, t_ns: int) -> Image:
    im = Image()
    im.header.stamp = stamp(t_ns)
    im.header.frame_id = f"{CAM_NAME}_optical_frame"
    im.height, im.width = arr.shape[:2]
    im.encoding = encoding
    im.step = arr.strides[0]
    im.data = arr.tobytes()
    return im


def scan_msg(ranges: np.ndarray, t_ns: int) -> LaserScan:
    s = LaserScan()
    s.header.stamp = stamp(t_ns)
    s.header.frame_id = "pelvis_yaw"
    s.angle_min, s.angle_increment = -math.pi, 2 * math.pi / SCAN_N
    s.angle_max = s.angle_min + (SCAN_N - 1) * s.angle_increment
    s.scan_time = 1.0 / SCAN_HZ
    s.range_min, s.range_max = SCAN_MIN_M, SCAN_MAX_M
    s.ranges = ranges.tolist()
    return s


def parse_layout(text: str, mocap: list[str]) -> dict:
    """The `layout` parameter checked: int seq, optional robot [x, y, yaw], bodies {mocap gz name: [x, y, z, yaw]}."""
    req = json.loads(text)
    assert isinstance(req["seq"], int), "seq must be an int"
    assert "robot" not in req or len(req["robot"]) == 3, "robot must be [x, y, yaw]"
    bad = sorted(set(req["bodies"]) - set(mocap))
    assert not bad, f"not mocap bodies: {bad}"
    assert all(len(v) == 4 for v in req["bodies"].values()), "bodies must be [x, y, z, yaw]"
    return req


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", type=Path, default=Path(__file__).parent / "rooms/spike_room.yaml")
    ap.add_argument("--world", default=None, help="clinic world variant (closed, cabinet_open, blocked_<room>)")
    ap.add_argument("--clinic", type=Path, default=None, help="legacy: layout.yaml = --world closed")
    ap.add_argument("--ext", type=Path, default=EXT, help="pinned clones (env G1_EXT)")
    ap.add_argument("--scene-dir", type=Path, default=SCENE_DIR, help="scene XML + scene.json (env G1_SCENE_DIR)")
    ap.add_argument("--rate", type=float, default=1.0, help="target real-time factor; 0 = as fast as possible")
    ap.add_argument("--cam-hz", type=float, default=10.0)
    ap.add_argument("--truth-log", type=Path, default=None, help="jsonl of raw free-joint qpos per odom stamp")
    ap.add_argument("--qpos-log", type=Path, default=None, help="jsonl of the robot's full qpos (free joint + hinges)")
    ap.add_argument("--qpos-hz", type=float, default=30.0)
    ap.add_argument("--stats", type=Path, default=None, help="JSON written on exit (RTF, min base z, timings)")
    ap.add_argument("--pitch-control", action="store_true", help="add the pitched control camera + its segmentation topic")
    ap.add_argument("--pitch-hz", type=float, default=2.0)
    ap.add_argument("--torch-threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.torch_threads)
    ort_one_thread()
    device = torch.device("cpu")

    g1_xml = args.ext / G1_REL
    if args.clinic:
        assert args.clinic.name == "layout.yaml" and args.world in (None, "closed"), "--clinic is --world closed"
        args.world = "closed"
    if args.world:
        xml, sp = build_clinic_scene(args.world, args.scene_dir, g1_xml, args.pitch_control), CLINIC_SPAWN
    else:
        assert not args.pitch_control, "--pitch-control needs --world"
        room = load_room(args.room)
        xml, sp = build_scene(room, g1_xml, args.scene_dir), room["spawn"]
    scene_meta = json.loads((args.scene_dir / "scene.json").read_text())
    mocap = scene_meta.get("mocap_bodies", [])
    leapp_yaml = resolve_leapp_bundle(args.ext / LEAPP_REL)
    desc = load_leapp_description(leapp_yaml)
    config = synthesize_sim_config(desc, xml)
    sim = MuJocoSimulation(config, device, enable_viewer=False, mjcf_path=xml)
    provider = create_leapp_command_provider(desc, device, config=config)
    policy = LeappPolicyController(leapp_yaml, config, sim.joint_names, device, command_provider=provider)
    manager = provider.manager
    yaw0 = math.radians(sp["yaw_deg"])
    sim._default_qpos[0:2] = [sp["x"], sp["y"]]
    sim._default_qpos[3:7] = [math.cos(yaw0 / 2), 0.0, 0.0, math.sin(yaw0 / 2)]
    sim.reset()
    policy.reset()
    m, d = sim.mj_model, sim.mj_data
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
    pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    fovy = float(m.cam_fovy[cam_id])
    rgb_r = mujoco.Renderer(m, CAM_H, CAM_W)
    # Segmentation context without multisampling: MSAA averages the id colours at geom edges, which gives wrong ids
    # there and, once ids pass 255 (the clinic), ids that exist nowhere (crash in Renderer.render). RGB keeps MSAA.
    samples, m.vis.quality.offsamples = m.vis.quality.offsamples, 0
    seg_r = mujoco.Renderer(m, CAM_H, CAM_W)
    m.vis.quality.offsamples = samples
    seg_r.enable_segmentation_rendering()
    pitch_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, PITCH_CAM_NAME) if args.pitch_control else -1
    counter = ContactCounter(m, scene_meta["robot_body_ids"])
    last_contacts: deque[dict] = deque(maxlen=CONTACT_LAST)
    clear_min: list[float | None] = [None]
    geom_body = np.append(m.geom_bodyid, 0).astype(np.uint16)  # index -1 (background) -> world

    rclpy.init()
    node = rclpy.create_node("g1_sim")
    pub_clock = node.create_publisher(Clock, "/clock", SENSOR_QOS)
    pub_odom = node.create_publisher(Odometry, "/g1/odom", SENSOR_QOS)
    pub_tf = node.create_publisher(TFMessage, "/tf", SENSOR_QOS)
    pub_rgb = node.create_publisher(Image, f"/{CAM_NAME}/image_raw", 2)
    pub_seg = node.create_publisher(Image, f"/{CAM_NAME}/segmentation", 2)
    pub_pseg = node.create_publisher(Image, f"/{PITCH_CAM_NAME}/segmentation", 2) if args.pitch_control else None
    pub_info = node.create_publisher(CameraInfo, f"/{CAM_NAME}/camera_info", SENSOR_QOS)
    pub_scan = node.create_publisher(LaserScan, "/g1/scan", SENSOR_QOS)
    pub_con = node.create_publisher(String, "/g1/contacts", SENSOR_QOS)
    pub_obj = node.create_publisher(String, "/g1/objects",
                                    QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    sim_t = [0.0]
    last_cmd_t = [-1.0]
    settle_until = [-1.0]  # sim time until which /cmd_vel is ignored after a layout set; < 0: none pending
    pending: deque[dict] = deque()  # layout requests queued by the parameter callback, applied on the main thread
    settling: list[dict] = []

    def on_cmd(msg: Twist) -> None:
        if settle_until[0] >= 0:
            return
        manager.set_command(msg.linear.x, msg.linear.y, msg.angular.z)
        last_cmd_t[0] = sim_t[0]

    def on_params(params: list) -> SetParametersResult:
        for p in params:
            if p.name == "layout" and p.value:
                try:
                    pending.append(parse_layout(p.value, mocap))
                except (AssertionError, KeyError, TypeError, ValueError) as e:  # a bad request is refused, not fatal
                    return SetParametersResult(successful=False, reason=f"layout refused: {e!r}")
        return SetParametersResult(successful=True)

    node.declare_parameter("layout", "")
    node.add_on_set_parameters_callback(on_params)
    node.create_subscription(Twist, "/cmd_vel", on_cmd, 10)
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()

    aff = repin(AFFINITY0)
    node.get_logger().info(f"cpu affinity: start {aff['start']}, after init (union over threads) "
                           f"{aff['after_init_union']}, {aff['threads_repinned']} threads re-pinned to {aff['start']}")
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    truth = args.truth_log.open("w") if args.truth_log else None
    qlog = None
    if args.qpos_log:  # the robot's joints (free joint first), asserted to be the leading qpos entries of the scene
        jn = [j for j in range(m.njnt) if int(m.jnt_bodyid[j]) in set(scene_meta["robot_body_ids"])]
        nq_r = sum(7 if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1 for j in jn)
        assert [int(m.jnt_qposadr[j]) for j in jn][0] == 0 and m.jnt_type[jn[0]] == mujoco.mjtJoint.mjJNT_FREE \
            and all(m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE for j in jn[1:]) and nq_r == 7 + len(jn) - 1, \
            "robot joints are not a free joint + hinges at qpos 0"
        qlog = QposLog(args.qpos_log, ["root"] + [m.joint(j).name for j in jn[1:]], args.qpos_hz)
        qlog_nq = nq_r
        node.get_logger().info(f"qpos log: nq {nq_r} of scene nq {m.nq}")
    cam_every = max(1, round(1.0 / (args.cam_hz * sim.dt)))
    pitch_every = max(1, round(1.0 / (args.pitch_hz * sim.dt)))
    scan_every = max(1, round(1.0 / (SCAN_HZ * sim.dt)))
    con_every = max(1, round(1.0 / (CONTACT_HZ * sim.dt)))
    min_z, step, scans, layouts = float("inf"), 0, 0, 0
    t_pol = t_phys = t_cam = t_scan = t_con = t_clear = 0.0
    wall0 = time.perf_counter()
    cpu0 = time.process_time()
    node.get_logger().info(f"G1 up: scene {xml}, physics {1 / sim.physics_dt:.0f} Hz, policy {1 / sim.dt:.0f} Hz, "
                           f"camera every {cam_every} steps, fovy {fovy:.3f}")
    while not stop.is_set():
        if pending:
            req = pending.popleft()
            if "robot" in req:
                x, y, yaw = req["robot"]
                d.qpos[:] = sim._default_qpos
                d.qpos[0:2] = [x, y]
                d.qpos[3:7] = yaw_quat(yaw)
                d.qvel[:] = 0.0
                d.ctrl[:] = 0.0
                policy.reset()
            place_bodies(m, d, req["bodies"])
            mujoco.mj_forward(m, d)
            manager.set_command(0.0, 0.0, 0.0)
            last_cmd_t[0] = -1.0
            settle_until[0] = d.time + SETTLE_S
            settling[:] = [req]
            layouts += 1
            node.get_logger().info(f"layout seq {req['seq']}: robot {req.get('robot')}, {len(req['bodies'])} bodies")
        if settling and d.time >= settle_until[0]:
            mujoco.mj_kinematics(m, d)
            pub_obj.publish(String(data=json.dumps({"seq": settling[0]["seq"], "t_sim": d.time,
                                                    "bodies": body_poses(m, d, mocap), "robot_z": float(d.qpos[2])})))
            settling.clear()
            settle_until[0] = -1.0
        if last_cmd_t[0] >= 0 and sim_t[0] - last_cmd_t[0] > CMD_TIMEOUT_S:
            manager.set_command(0.0, 0.0, 0.0)
            last_cmd_t[0] = -1.0
        a = time.perf_counter()
        joint_cmd = policy.process(sim.get_state())
        b = time.perf_counter()
        for _ in range(sim.decimation):
            sim.step(joint_cmd)
            min_z = min(min_z, float(d.qpos[2]))
            k = time.perf_counter()
            for rb, ob, _mo, dist in counter.scan(d):
                last_contacts.append({"robot_body": m.body(rb).name, "other_body": m.body(ob).name, "t_sim": d.time,
                                      "dist": dist})
            t_con += time.perf_counter() - k
        k = time.perf_counter()
        cl = counter.clearance(d, SCAN_MAX_M)
        if cl is not None and (clear_min[0] is None or cl < clear_min[0]):
            clear_min[0] = cl
        c = time.perf_counter()
        t_clear += c - k
        t_pol += b - a
        t_phys += k - b
        step += 1
        sim_t[0] = d.time
        t_ns = int(round(d.time * 1e9))
        pub_clock.publish(Clock(clock=stamp(t_ns)))
        od = odom_msg(d.qpos[:7].copy(), d.qvel[:6].copy(), t_ns)
        pub_odom.publish(od)
        pub_tf.publish(tf_msg(od))
        if truth is not None:
            # H4's second access path: the pelvis body pose from forward kinematics. mj_step leaves xpos / xquat one
            # physics step behind qpos, so recompute them for the published state (derived fields only, no dynamics).
            mujoco.mj_kinematics(m, d)
            truth.write(json.dumps({"t_ns": t_ns, "qpos": d.qpos[:7].tolist(), "body_xpos": d.xpos[pelvis_id].tolist(),
                                    "body_xquat": d.xquat[pelvis_id].tolist()}) + "\n")
            if step % 50 == 0:
                truth.flush()
        if qlog is not None:
            qlog.write(d.time, d.qpos[:qlog_nq])
        if step % cam_every == 0:
            rgb_r.update_scene(d, camera=cam_id)
            rgb = rgb_r.render()
            seg_r.update_scene(d, camera=cam_id)
            seg = seg_r.render()
            body = geom_body[np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)]
            pub_rgb.publish(image_msg(np.ascontiguousarray(rgb), "rgb8", t_ns))
            pub_seg.publish(image_msg(np.ascontiguousarray(body), "16UC1", t_ns))
            pub_info.publish(camera_info(fovy, t_ns))
            t_cam += time.perf_counter() - c
        if pub_pseg is not None and step % pitch_every == 0:
            seg_r.update_scene(d, camera=pitch_id)
            pseg = seg_r.render()
            pbody = geom_body[np.where(pseg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), pseg[..., 0], -1)]
            pub_pseg.publish(image_msg(np.ascontiguousarray(pbody), "16UC1", t_ns))
        if step % scan_every == 0:
            e = time.perf_counter()
            w, qx, qy, qz = d.qpos[3:7]  # frame pelvis_yaw: the pelvis heading from the free-joint quaternion
            yaw = math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            pub_scan.publish(scan_msg(ray_fan(m, d, float(d.qpos[0]), float(d.qpos[1]), yaw), t_ns))
            t_scan += time.perf_counter() - e
            scans += 1
        if step % con_every == 0:
            pub_con.publish(String(data=json.dumps({"t_sim": d.time, "n_mocap_total": counter.n_mocap,
                                                    "n_static_total": counter.n_static, "clearance_m": clear_min[0],
                                                    "last": list(last_contacts)})))
            clear_min[0] = None
        if step % 500 == 0:
            wall = time.perf_counter() - wall0
            node.get_logger().info(f"sim {d.time:.1f} s, RTF {d.time / wall:.2f}, base z {d.qpos[2]:.3f}, "
                                   f"min z {min_z:.3f}, cmd {manager.get_navigation_command().tolist()}")
        if args.rate > 0:
            lag = wall0 + d.time / args.rate - time.perf_counter()
            if lag > 0:
                time.sleep(lag)
    wall = time.perf_counter() - wall0
    stats = {"sim_s": d.time, "wall_s": wall, "rtf": d.time / wall, "rate_target": args.rate,
             "min_base_z": min_z, "control_steps": step, "process_cpu_s": time.process_time() - cpu0,
             "ms_per_step": {"policy": 1e3 * t_pol / step, "physics_10_substeps": 1e3 * t_phys / step,
                             "camera_render_per_frame": 1e3 * t_cam / max(1, step // cam_every),
                             "scan": 1e3 * t_scan / max(1, scans), "contacts_10_substeps": 1e3 * t_con / step,
                             "clearance": 1e3 * t_clear / step},
             "contacts_mocap_total": counter.n_mocap, "contacts_static_total": counter.n_static,
             "cam_every_steps": cam_every, "scan_every_steps": scan_every, "layouts": layouts,
             "fovy_deg": fovy, "torch_threads": args.torch_threads, "world": args.world, "affinity": aff}
    if args.stats:
        args.stats.write_text(json.dumps(stats, indent=1))
    if truth is not None:
        truth.close()
    if qlog is not None:
        qlog.close()
    rgb_r.close()
    seg_r.close()
    rclpy.try_shutdown()
    spin.join(timeout=2.0)


if __name__ == "__main__":
    main()
