#!/usr/bin/env python3
"""Real robot bodies for the demo recorder's sim view (task 013). Pure parts first (tests/test_robot_bodies.py).

    python probes/robot_bodies.py build-tb4        # one-time; needs ROS Jazzy installed, trimesh + pycollada (G1 venv)

TurtleBot 4: URDF (xacro of our turtlebot4_description overlay + irobot_create_description) -> every visual link at zero
joint angles composed into the base_link frame -> MJCF fragment + converted STL meshes in BODIES/tb4/ (licensed assets:
a cache outside git, never committed). Wheels / caster are static (rigid body). Unitree G1: the articulated MJCF of
unitree_mujoco, attached to the clinic scene with collisions off; only mj_kinematics / mj_forward ever runs on it.
"""
from __future__ import annotations

import os
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BODIES = Path(os.environ.get("BUILD_DIR", Path.home() / "vlm_skill_agent_build")) / "bodies"
TB4_DIR = BODIES / "tb4"
BUILD_CMD = "$BUILD_DIR/.venv-g1/bin/python probes/robot_bodies.py build-tb4"
G1_XML = Path(os.environ.get("BUILD_DIR", Path.home() / "vlm_skill_agent_build")) / "ext/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
XACRO = ROOT / "ros_ws/src/turtlebot4_description/urdf/standard/turtlebot4.urdf.xacro"
PKG_DIRS = {"turtlebot4_description": ROOT / "ros_ws/src/turtlebot4_description",
            "irobot_create_description": Path("/opt/ros/jazzy/share/irobot_create_description")}
GREY = (0.5, 0.5, 0.5, 1.0)


# ---------------------------------------------------------------- pure: URDF transforms
def rpy_xyz_to_mat(rpy: tuple[float, float, float], xyz: tuple[float, float, float]) -> np.ndarray:
    """URDF origin -> 4x4: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = rz @ ry @ rx
    T[:3, 3] = xyz
    return T


def mat_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """3x3 rotation -> (w, x, y, z)."""
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        return (w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w))
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(0.0, R[i, i] - R[j, j] - R[k, k] + 1)) * 2
    q = [0.0] * 4
    q[1 + i], q[0] = s / 4, (R[k, j] - R[j, k]) / s
    q[1 + j], q[1 + k] = (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
    return tuple(q)  # type: ignore[return-value]


def _origin(el: ET.Element | None) -> np.ndarray:
    if el is None:
        return np.eye(4)
    return rpy_xyz_to_mat(tuple(map(float, el.get("rpy", "0 0 0").split())),  # type: ignore[arg-type]
                          tuple(map(float, el.get("xyz", "0 0 0").split())))  # type: ignore[arg-type]


@dataclass
class Visual:
    link: str
    T: np.ndarray                 # visual frame in the root link frame, joints at zero
    kind: str                     # mesh | cylinder | sphere | box
    mesh: str | None = None       # package://... uri
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    size: tuple[float, ...] = ()  # cylinder (radius, length) | sphere (radius,) | box (x, y, z)
    rgba: tuple[float, float, float, float] | None = None


def urdf_visuals(urdf: str) -> list[Visual]:
    """Every <visual> of the URDF with its static pose in the root link frame (all joints at zero)."""
    root = ET.fromstring(urdf)
    parent_of: dict[str, tuple[str, np.ndarray]] = {}
    for j in root.findall("joint"):
        parent_of[j.find("child").get("link")] = (j.find("parent").get("link"), _origin(j.find("origin")))  # type: ignore[union-attr]
    cache: dict[str, np.ndarray] = {}

    def pose(link: str) -> np.ndarray:
        if link not in cache:
            cache[link] = np.eye(4) if link not in parent_of else pose(parent_of[link][0]) @ parent_of[link][1]
        return cache[link]

    out: list[Visual] = []
    for ln in root.findall("link"):
        for v in ln.findall("visual"):
            g = v.find("geometry")[0]  # type: ignore[index]
            col = v.find("material/color")
            rgba = tuple(map(float, col.get("rgba").split())) if col is not None else None  # type: ignore[union-attr]
            T = pose(ln.get("name")) @ _origin(v.find("origin"))  # type: ignore[arg-type]
            if g.tag == "mesh":
                out.append(Visual(ln.get("name"), T, "mesh", g.get("filename"),  # type: ignore[arg-type]
                                  tuple(map(float, g.get("scale", "1 1 1").split())), (), rgba))  # type: ignore[arg-type]
            else:
                size = {"cylinder": ("radius", "length"), "sphere": ("radius",), "box": ()}[g.tag]
                sz = tuple(float(g.get(k)) for k in size) if size else tuple(map(float, g.get("size").split()))  # type: ignore[arg-type]
                out.append(Visual(ln.get("name"), T, g.tag, None, (1, 1, 1), sz, rgba))
    return out


def resolve_package(uri: str) -> Path:
    pkg, rel = uri.removeprefix("package://").split("/", 1)
    return PKG_DIRS[pkg] / rel


# ---------------------------------------------------------------- pure: time + qpos
def wall_to_tsim(hist_wall: np.ndarray, hist_t: np.ndarray, wall: np.ndarray | float) -> np.ndarray:
    """gt_history (wall_t, t_sim) -> sim time of arbitrary wall stamps (clamped outside the run)."""
    return np.interp(wall, hist_wall, hist_t)


def read_qpos(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """qpos.jsonl -> (header, t_sim [n], qpos [n, nq]). A cut last line is skipped; t must be non-decreasing."""
    lines = path.read_text().splitlines()
    header, rows = json.loads(lines[0]), []
    for i, l in enumerate(lines[1:]):
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            assert i == len(lines) - 2, f"{path}: unparsable line {i + 2} is not the last"
    t = np.array([r["t_sim"] for r in rows])
    q = np.array([r["qpos"] for r in rows], dtype=float)
    assert q.shape[1] == header["nq"], f"{path}: rows have {q.shape[1]} values, header says nq={header['nq']}"
    assert np.all(np.diff(t) >= 0), f"{path}: t_sim goes backwards"
    return header, t, q


def interp_qpos(t: np.ndarray, q: np.ndarray, ts: float) -> np.ndarray:
    """Linear in sim time (clamped); the free joint's quaternion (cols 3:7) is renormalised."""
    out = np.array([np.interp(ts, t, q[:, i]) for i in range(q.shape[1])])
    qq = out[3:7]
    out[3:7] = qq / np.linalg.norm(qq)
    return out


def choose_stance(keyframe_qpos: np.ndarray | None, qpos0: np.ndarray, bent: dict[int, float]) -> tuple[np.ndarray, str]:
    """Default pose when a run has no joint log: the model's first keyframe if it has one, else qpos0 with the
    bent-knee offsets `bent` (qpos index -> angle). Returns (qpos, source)."""
    if keyframe_qpos is not None:
        return np.array(keyframe_qpos, dtype=float), "keyframe"
    q = np.array(qpos0, dtype=float)
    for i, v in bent.items():
        q[i] = v
    return q, "default stance"


def history_pose_at(h: np.ndarray, ts: float) -> tuple[float, float, float, float]:
    """h columns t_sim, x, y, z, yaw -> (x, y, z, yaw) at sim time ts (yaw unwrapped before interpolating)."""
    yaw = np.unwrap(h[:, 4])
    return tuple(float(np.interp(ts, h[:, 0], c)) for c in (h[:, 1], h[:, 2], h[:, 3], yaw))  # type: ignore[return-value]


def read_history(path: Path) -> np.ndarray:
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    return np.array([[r["t_sim"], r["x"], r["y"], r["z"], r["yaw"]] for r in rows])


# ---------------------------------------------------------------- TurtleBot 4 build (needs trimesh)
def _xacro_urdf() -> str:
    cmd = ("source /opt/ros/jazzy/setup.bash && "
           f"xacro {XACRO} gazebo:=ignition namespace:=turtlebot4")
    return subprocess.run(["bash", "-c", cmd], check=True, capture_output=True, text=True).stdout


def build_tb4() -> None:
    import trimesh
    urdf = _xacro_urdf()
    TB4_DIR.mkdir(parents=True, exist_ok=True)
    asset, body = ET.Element("asset"), ET.Element("body", name="robot_body", mocap="true")
    n, cache = 0, {}
    for v in urdf_visuals(urdf):
        if v.kind != "mesh":
            q = mat_to_quat(v.T[:3, :3])
            sz = (v.size[0], v.size[1] / 2) if v.kind == "cylinder" else v.size if v.kind == "sphere" else tuple(s / 2 for s in v.size)
            ET.SubElement(body, "geom", type=v.kind, size=" ".join(f"{s:.5f}" for s in sz),
                          pos=" ".join(f"{s:.5f}" for s in v.T[:3, 3]), quat=" ".join(f"{s:.6f}" for s in q),
                          rgba=" ".join(map(str, v.rgba or GREY)), contype="0", conaffinity="0")
            continue
        path = resolve_package(v.mesh)  # type: ignore[arg-type]
        if path not in cache:
            cache[path] = trimesh.load(path, force="scene").dump()  # transforms of the DAE scene graph applied
        T = v.T @ np.diag([*v.scale, 1.0])
        parts: dict[tuple, list] = {}
        for m in cache[path]:
            mat = getattr(m.visual, "material", None)
            c = tuple(np.round(np.asarray(mat.main_color, dtype=float) / 255, 3)) if mat is not None else (v.rgba or GREY)
            parts.setdefault(c, []).append(m)
        for rgba, ms in parts.items():
            m = trimesh.util.concatenate(ms)
            m.apply_transform(T)
            name = f"tb4_{n}_{v.link}"
            n += 1
            m.export(TB4_DIR / f"{name}.stl")
            ET.SubElement(asset, "mesh", name=name, file=f"{name}.stl")
            ET.SubElement(body, "geom", type="mesh", mesh=name, rgba=" ".join(map(str, rgba)),
                          contype="0", conaffinity="0")
    mj = ET.Element("mujoco", model="tb4_visual")
    mj.append(asset)
    ET.SubElement(mj, "worldbody").append(body)
    ET.indent(mj)
    (TB4_DIR / "tb4.xml").write_text(ET.tostring(mj, encoding="unicode"))
    (TB4_DIR / "SOURCE.txt").write_text(
        "TurtleBot 4 visual meshes (turtlebot4_description overlay, irobot_create_description), joints at zero.\n"
        "Licensed assets: cache only, never commit.\n")
    print(f"wrote {TB4_DIR}/tb4.xml and {n} meshes")


def load_tb4() -> tuple[ET.Element, dict[str, bytes]]:
    """(fragment root, {mesh file name: bytes}); fails loudly when the cache is missing."""
    if not (TB4_DIR / "tb4.xml").exists():
        raise SystemExit(f"TurtleBot 4 body cache missing: {TB4_DIR}/tb4.xml\nbuild it once: {BUILD_CMD}")
    frag = ET.fromstring((TB4_DIR / "tb4.xml").read_text())
    return frag, {m.get("file"): (TB4_DIR / m.get("file")).read_bytes() for m in frag.iter("mesh")}


if __name__ == "__main__":
    if sys.argv[1:] == ["build-tb4"]:
        build_tb4()
    else:
        raise SystemExit(__doc__)
