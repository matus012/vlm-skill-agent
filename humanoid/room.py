"""Scene loader: a world + the G1 MJCF (with its head camera) -> one scene MJCF + scene.json.

Two worlds: `build_scene()` for a room YAML (walls as segments, objects as primitives; the Phase 9 spike room) and
`build_clinic_scene()` for a clinic world variant (closed / cabinet_open / blocked_<room>) via
sim/worlds/clinic/generate_mjcf.py, with the spec objects (realize.SPEC_OBJECTS) as mocap bodies (task 012 section 1).
Pure MuJoCo helpers for g1_node (no ROS): `place_bodies` / `body_poses` (the layout transform and its read-back) and
`ray_fan` (the /g1/scan range fan, mj_ray only).

Body order in the compiled model: world (0), then the robot tree (pelvis ...), then one static body per wall /
object. `scene.json` next to the XML records the robot's body-id range (self-view checks) and the truth label of every
body id (`label_of_body`; clinic only, all 0 for a room YAML).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from humanoid.geometry import fovy_for_hfov

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim/worlds/clinic"))
import realize  # noqa: E402
from generate_mjcf import generate_mjcf  # noqa: E402

CAM_NAME = "head_rgb"
PITCH_CAM_NAME = "head_rgb_pitch"  # H-P2 G7 negative control (a recorded deviation addendum): same mount, pitched down
PITCH_DEG = 47.6  # the D435 angle in Unitree's URDF d435_link
CAM_W, CAM_H = 640, 480
CAM_HFOV = math.pi / 2.0  # project decision 2026-11-03: 90 deg HFOV, 640x480
# Camera on torso_link at the G1's head camera spot (Unitree URDF d435_link xyz 0.0576 0.0175 0.4299), moved
# forward of the face shell and centred, looking level along torso +x (decision recorded in README).
CAM_POS = (0.10, 0.0, 0.43)
CLINIC_LAYOUT = Path(__file__).resolve().parents[1] / "sim/worlds/clinic/layout.yaml"
CLINIC_PREFIX = "clinic/"
CLINIC_SPAWN = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}  # layout.yaml: "robot spawns at 0,0" (hall); Stage 1 yaw 0
# Clinic geoms go to group 2 (visible to the renderers' default mjvOption, like groups 0/1), the robot's stay in 0/1,
# so the range fan sees only the clinic (walls, furniture, mocap objects, blocker) and never the robot's own body.
RAY_GROUP = 2
# /g1/scan (task 012, a recorded deviation): planar fan from the pelvis xy at 0.40 m above the floor, 360 deg at 1 deg,
# 0.1-4.0 m, 10 Hz sim.
SCAN_Z, SCAN_N, SCAN_MIN_M, SCAN_MAX_M, SCAN_HZ = 0.40, 360, 0.1, 4.0, 10.0


def world_layout(world: str) -> tuple[Path, str | None]:
    """(layout yaml, blocked room) of a clinic world variant: closed, cabinet_open, blocked_<room>."""
    if world == "cabinet_open":
        return CLINIC_LAYOUT.with_name("layout_cabinet_open.yaml"), None
    if world.startswith("blocked_"):
        return CLINIC_LAYOUT, world.removeprefix("blocked_")
    assert world == "closed", f"unknown world {world!r}"
    return CLINIC_LAYOUT, None


def mocap_names(world: str) -> list[str]:
    """gz model names of realize.SPEC_OBJECTS in the world's frozen truth.json, in SPEC_OBJECTS order."""
    frozen = {o["id"]: o for o in json.loads(realize.S.world_truth_path(world).read_text())["objects"]}
    return [frozen[oid]["model_name"] for oid in realize.SPEC_OBJECTS]


def yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def place_bodies(m: mujoco.MjModel, d: mujoco.MjData, bodies: dict[str, list[float]]) -> None:
    """Mocap pose of each clinic body {gz model name: [x, y, z, yaw]} (no mj_forward here)."""
    for name, (x, y, z, yaw) in bodies.items():
        k = m.body_mocapid[m.body(CLINIC_PREFIX + name).id]
        assert k >= 0, f"{name} is not a mocap body"
        d.mocap_pos[k] = [x, y, z]
        d.mocap_quat[k] = yaw_quat(yaw)


def body_poses(m: mujoco.MjModel, d: mujoco.MjData, names: list[str]) -> dict[str, list[float]]:
    """{gz model name: [x, y, z, yaw]} from xpos / xquat (yaw of a z-axis rotation)."""
    out = {}
    for name in names:
        b = m.body(CLINIC_PREFIX + name).id
        w, qx, qy, qz = d.xquat[b]
        out[name] = [*map(float, d.xpos[b]), math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))]
    return out


def scan_dirs(yaw: float, n: int = SCAN_N) -> np.ndarray:
    """Unit ray directions (world frame) of the fan: angle i = -pi + i * 2 pi / n in the pelvis yaw-only frame."""
    a = yaw - math.pi + np.arange(n) * (2 * math.pi / n)
    return np.stack([np.cos(a), np.sin(a), np.zeros(n)], axis=1)


def ray_fan(m: mujoco.MjModel, d: mujoco.MjData, x: float, y: float, yaw: float, n: int = SCAN_N,
            z: float = SCAN_Z, max_m: float = SCAN_MAX_M) -> np.ndarray:
    """Ranges (float32, inf = no hit within max_m) of n horizontal rays from (x, y, z), geoms of RAY_GROUP only.
    A sensor: geometry through mj_multiRay, nothing else (no truth, no object poses)."""
    group = np.zeros(6, np.uint8)
    group[RAY_GROUP] = 1
    geom, dist = np.zeros(n, np.int32), np.zeros(n)
    mujoco.mj_multiRay(m, d, np.array([x, y, z]), scan_dirs(yaw, n).ravel(), group, 1, -1, geom, dist, None, n, max_m)
    dist[(geom < 0) | (dist > max_m)] = np.inf
    return dist.astype(np.float32)


def load_room(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def _level_cam_quat(pitch_down_deg: float = 0.0) -> np.ndarray:
    # MuJoCo cameras look along their -z with +y up; columns = camera x, y, z axes in the body frame.
    rot = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    a = math.radians(pitch_down_deg)  # rotation about the body +y tilts the forward axis x toward -z (down)
    rot = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]]) @ rot
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, rot.flatten())
    return q


def _g1_spec(g1_xml: Path, name: str, pitch_control: bool = False) -> mujoco.MjSpec:
    """The G1 MJCF with the head camera and offscreen buffer, ready for a world to be added."""
    spec = mujoco.MjSpec.from_file(str(g1_xml))
    spec.meshdir = str((g1_xml.parent / "meshes").resolve())
    spec.modelname = f"g1_{name}"
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]
    spec.visual.global_.offwidth = max(CAM_W, 1280)
    spec.visual.global_.offheight = max(CAM_H, 720)
    spec.body("torso_link").add_camera(name=CAM_NAME, pos=list(CAM_POS), quat=list(_level_cam_quat()),
                                       fovy=fovy_for_hfov(CAM_HFOV, CAM_W, CAM_H))
    if pitch_control:
        spec.body("torso_link").add_camera(name=PITCH_CAM_NAME, pos=list(CAM_POS), quat=list(_level_cam_quat(PITCH_DEG)),
                                           fovy=fovy_for_hfov(CAM_HFOV, CAM_W, CAM_H))
    return spec


def _write(spec: mujoco.MjSpec, out_dir: Path, name: str, spawn: dict, body_labels: dict[str, int],
           **extra: Any) -> Path:
    """Compile (fail fast before writing), write `<out_dir>/<name>.xml` + `scene.json`; return the XML path.
    scene.json `label_of_body[i]` = truth label of body id i (0 = robot / world / unlabelled)."""
    model = spec.compile()
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / f"{name}.xml"
    xml_path.write_text(spec.to_xml())
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    robot_ids = [b for b in range(model.nbody) if model.body_rootid[b] == pelvis]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)]
    meta = {"xml": str(xml_path), "robot_body_ids": robot_ids, "body_names": names, "spawn": spawn,
            "label_of_body": [body_labels.get(n, 0) for n in names], **extra}
    (out_dir / "scene.json").write_text(json.dumps(meta, indent=1))
    return xml_path


def build_clinic_scene(world: str, out_dir: Path, g1_xml: Path, pitch_control: bool = False) -> Path:
    """G1 + the clinic MJCF of a world variant (sim/worlds/clinic/generate_mjcf.py), spawned in the hall at
    CLINIC_SPAWN, spec objects as mocap bodies. Clinic bodies carry the prefix `clinic/` and geom group RAY_GROUP;
    scene.json maps every body id to its truth label and records world + mocap_bodies (gz model names)."""
    layout, blocked = world_layout(world)
    mocap = mocap_names(world)
    xml, labels = generate_mjcf(layout, blocked=blocked, mocap=frozenset(mocap))
    clinic = mujoco.MjSpec.from_string(xml)
    for g in clinic.geoms:
        g.group = RAY_GROUP
    spec = _g1_spec(g1_xml, "clinic", pitch_control)
    spec.attach(clinic, frame=spec.worldbody.add_frame(), prefix=CLINIC_PREFIX)
    body_labels = {CLINIC_PREFIX + n: v for n, v in json.loads(labels)["body_labels"].items()}
    return _write(spec, out_dir, "clinic", CLINIC_SPAWN, body_labels, world=world, mocap_bodies=mocap)


def build_scene(room: dict[str, Any], g1_xml: Path, out_dir: Path) -> Path:
    """Write `<out_dir>/<room name>.xml` (+ `scene.json`) and return the XML path."""
    spec = _g1_spec(g1_xml, room["name"])

    h, t = float(room["wall_height"]), float(room["wall_thickness"])
    wb = spec.worldbody
    fx, fy = room["floor"]["size"]
    floor = wb.add_body(name="floor")
    floor.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[fx / 2 + 1, fy / 2 + 1, 0.05],
                   rgba=room["floor"]["rgba"])
    if room.get("ceiling"):
        ceil = wb.add_body(name="ceiling", pos=[0, 0, h + t / 2])
        ceil.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[fx / 2 + t, fy / 2 + t, t / 2],
                      rgba=room["ceiling_rgba"], contype=0, conaffinity=0)
    for i, (x1, y1, x2, y2) in enumerate(room["walls"]):
        length = math.hypot(x2 - x1, y2 - y1)
        yaw = math.atan2(y2 - y1, x2 - x1)
        body = wb.add_body(name=f"wall_{i}", pos=[(x1 + x2) / 2, (y1 + y2) / 2, h / 2],
                           quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[length / 2 + t / 2, t / 2, h / 2],
                      rgba=room["wall_rgba"])
    for obj in room["objects"]:
        yaw = math.radians(obj.get("yaw_deg", 0.0))
        body = wb.add_body(name=obj["name"], pos=obj["pos"], quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        gtype = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
                 "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}[obj["shape"]]
        body.add_geom(type=gtype, size=obj["size"], rgba=obj["rgba"])
    for i, lp in enumerate(room.get("lights", [])):
        wb.add_light(name=f"light_{i}", pos=lp, dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5], castshadow=False)

    return _write(spec, out_dir, room["name"], room["spawn"], {})
