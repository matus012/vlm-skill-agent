"""Room loader: the generated scene has the 90-deg head camera, every wall/object as its own body, robot ids first."""
from __future__ import annotations

import os
import json
from pathlib import Path

import mujoco
import pytest

from humanoid.room import CAM_NAME, build_scene, load_room

G1_XML = Path(os.environ.get("BUILD_DIR", Path.home() / "vlm_skill_agent_build")) / "ext/unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
ROOM = Path(__file__).resolve().parents[1] / "rooms/spike_room.yaml"


@pytest.mark.skipif(not G1_XML.exists(), reason="run humanoid/setup.sh first")
def test_spike_room_scene(tmp_path: Path) -> None:
    room = load_room(ROOM)
    xml = build_scene(room, G1_XML, tmp_path)
    m = mujoco.MjModel.from_xml_path(str(xml))
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
    assert m.cam_fovy[cam] == pytest.approx(73.7398, abs=1e-3)
    assert m.body(m.cam_bodyid[cam]).name == "torso_link"
    names = {m.body(i).name for i in range(m.nbody)}
    assert {o["name"] for o in room["objects"]} <= names
    assert {f"wall_{i}" for i in range(len(room["walls"]))} <= names
    meta = json.loads((tmp_path / "scene.json").read_text())
    robot = meta["robot_body_ids"]
    assert robot == list(range(1, len(robot) + 1)) and m.body(1).name == "pelvis"
    assert all(m.body_rootid[i] != 1 for i in range(len(robot) + 1, m.nbody))
