"""Clinic MJCF writer (sim/worlds/clinic/generate_mjcf.py) + the clinic scene: deterministic, every truth object at
its pose with its truth label, geometry and colours equal to the SDF world, segmentation -> truth labels."""
from __future__ import annotations

import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")  # as g1_node; before mujoco picks its GL backend

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from humanoid.geometry import wrap  # noqa: E402
from humanoid.room import CAM_NAME, CLINIC_LAYOUT, CLINIC_PREFIX, CLINIC_SPAWN, build_clinic_scene  # noqa: E402

CLINIC = CLINIC_LAYOUT.parent
sys.path.insert(0, str(CLINIC))
import generate  # noqa: E402
from generate_mjcf import generate_mjcf  # noqa: E402

G1_XML = Path(os.environ.get("BUILD_DIR", Path.home() / "vlm_skill_agent_build")) / "ext/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
TRUTH = json.loads((CLINIC / "generated/truth.json").read_text())
STRUCTURE = {"structure_floor": 241, "structure_walls": 240, "structure_ceiling": 242, "structure_trim": 243}


@pytest.fixture(scope="module")
def clinic() -> tuple[mujoco.MjModel, mujoco.MjData, dict]:
    xml, labels = generate_mjcf(CLINIC_LAYOUT)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    return m, d, json.loads(labels)


def test_deterministic_and_same_layout() -> None:
    a, b = generate_mjcf(CLINIC_LAYOUT), generate_mjcf(CLINIC_LAYOUT)
    assert a == b
    assert json.loads(a[1])["layout_sha256"] == TRUTH["layout_sha256"]


def test_every_truth_object_at_its_pose_with_its_label(clinic: tuple) -> None:
    m, d, labels = clinic
    assert labels["body_labels"] == {**{o["model_name"]: o["label"] for o in TRUTH["objects"]}, **STRUCTURE}
    for o in TRUTH["objects"]:
        x, y, z, _, _, yaw = o["pose"]
        b = m.body(o["model_name"])
        assert d.xpos[b.id] == pytest.approx([x, y, z], abs=1e-6), o["id"]
        q = d.xquat[b.id]
        assert abs(wrap(math.atan2(2 * q[0] * q[3], 1 - 2 * q[3] ** 2) - yaw)) < 1e-6, o["id"]
    assert m.nbody == 1 + len(TRUTH["objects"]) + len(STRUCTURE)


def test_geometry_and_colour_equal_the_sdf(clinic: tuple) -> None:
    """Each SDF visual's world centre (model pose then visual pose, computed here independently) and diffuse colour
    is a MuJoCo geom of the same body, and the object's own colour from layout.yaml is on it."""
    m, d, _ = clinic
    sdf, _ = generate.generate(CLINIC_LAYOUT)
    for model in ET.fromstring(sdf).find("world").findall("model"):
        name = model.get("name")
        mx, my, mz, _, _, yaw = (float(v) for v in model.findtext("pose").split())
        visuals = model.find("link").findall("visual")
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(name).id]
        assert len(geoms) == len(visuals), name
        for v, g in zip(visuals, geoms):
            vx, vy, vz = (float(s) for s in v.findtext("pose").split()[:3])
            c, s = math.cos(yaw), math.sin(yaw)
            want = [mx + c * vx - s * vy, my + s * vx + c * vy, mz + vz]
            assert d.geom_xpos[g] == pytest.approx(want, abs=1e-4), (name, v.get("name"))
            rgb = [float(x) for x in v.find("material").findtext("diffuse").split()[:3]]
            assert m.geom_rgba[g][:3] == pytest.approx(rgb, abs=1e-6), (name, v.get("name"))
    for o in TRUTH["objects"]:
        rgbs = [m.geom_rgba[g][:3] for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(o["id"]).id]
        assert any(np.allclose(c, generate.COLOURS[o["colour"]], atol=1e-6) for c in rgbs), o["id"]


@pytest.mark.skipif(not G1_XML.exists(), reason="run humanoid/setup.sh first")
def test_clinic_scene_labels_and_segmentation(tmp_path: Path) -> None:
    """G1 + clinic: robot ids first, scene.json label_of_body right, and the head camera at the storage place facing
    the toolbox (yaw 180 deg) sees it under its truth label after body id -> label."""
    xml = build_clinic_scene("closed", tmp_path, G1_XML)
    meta = json.loads((tmp_path / "scene.json").read_text())
    m = mujoco.MjModel.from_xml_path(str(xml))
    robot = meta["robot_body_ids"]
    assert robot == list(range(1, len(robot) + 1)) and m.body(1).name == "pelvis"
    assert meta["spawn"] == CLINIC_SPAWN
    want = {CLINIC_PREFIX + o["model_name"]: o["label"] for o in TRUTH["objects"]}
    want.update({CLINIC_PREFIX + k: v for k, v in STRUCTURE.items()})
    got = {n: lab for n, lab in zip(meta["body_names"], meta["label_of_body"]) if lab}
    assert got == want
    assert all(meta["label_of_body"][i] == 0 for i in [0, *robot])

    d = mujoco.MjData(m)
    d.qpos[:7] = [2.5, -3.9, 0.8, 0.0, 0.0, 0.0, 1.0]  # storage place, yaw 180 deg -> toolbox at x 0.4 ahead
    mujoco.mj_forward(m, d)
    m.vis.quality.offsamples = 0  # as g1_node's segmentation renderer (MSAA blends id colours at edges)
    r = mujoco.Renderer(m, 480, 640)
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=CAM_NAME)
    seg = r.render()
    d.qpos[:7] = [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]  # spawn view down the hall: crashed render() under MSAA (ids > 255)
    mujoco.mj_forward(m, d)
    r.update_scene(d, camera=CAM_NAME)
    r.render()
    r.close()
    body = np.append(m.geom_bodyid, 0)[np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)]
    label = np.asarray(meta["label_of_body"], np.uint8)[body]
    toolbox = next(o["label"] for o in TRUTH["objects"] if o["id"] == "toolbox_red_1")
    assert (label == toolbox).sum() > 1000
    assert (label == 241).sum() > 1000 and (label == 240).sum() > 1000
    assert set(np.unique(label)) <= {0, *meta["label_of_body"]}
