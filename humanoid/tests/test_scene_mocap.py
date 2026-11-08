"""Task 012 section 1 (P1): one G1 scene per world, spec objects as mocap bodies, placed per run.

For every synthetic spec of tasks/phase8_dev.yaml + tasks/phase8_regress.yaml (both worlds), run_task.layout_plan's
teleport targets applied by room.place_bodies give body xpos / yaw == truth_for_spec poses (1e-6), found through the
scene.json label of each truth object (so a label swap fails); parked ids sit at realize.park_pose and are not in the
truth; every truth label is on exactly one body. Also: generate_mjcf's default output is byte-identical to before task
012, the place() constants equal teleport()'s, the read-back check, and the /g1/scan ray fan.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402

from humanoid import scene_g1  # noqa: E402
from humanoid.room import (CLINIC_LAYOUT, CLINIC_PREFIX, CLINIC_SPAWN, RAY_GROUP, body_poses,  # noqa: E402
                           build_clinic_scene, place_bodies, ray_fan)

ROOT = Path(__file__).resolve().parents[2]
CLINIC = CLINIC_LAYOUT.parent
sys.path.insert(0, str(CLINIC))
from generate_mjcf import generate_mjcf  # noqa: E402

from agent import truth as agent_truth  # noqa: E402

G1_XML = Path(os.environ.get("G1_EXT", str(Path(os.environ.get("BUILD_DIR", Path.home() / "vlm_skill_agent_build")) / "ext"))) / \
    "unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
LISTS = ("tasks/phase8_dev.yaml", "tasks/phase8_regress.yaml")  # synthetic only (never the pass lists)
# sha256 of generate_mjcf(layout) (xml, labels) at HEAD 758ce21, recorded before the task 012 edit
GOLDEN = {"layout.yaml": ("94f7d6ed8bd21b55e8aa04bde61d4622480ff13ca575494349d78984698a603a",
                          "8a4a283e60171dcfcd1b2aad3a53b149d36149d691896b5197849f934c277da3"),
          "layout_cabinet_open.yaml": ("fab109e2a3b1c35f3bbae5b795aa655b233e09e2b38a01c9522e6ac363f87492",
                                       "9878072a59a7df18f6dcc6eb83778d57c495a90c0046b0c4367dff026cb24610"),
          "layout_nomovables.yaml": ("ed28982220d2ce196eaffe07024959ca9877f7435c95446f5bafe26e4310cf87",
                                     "debfedf9c1b73102d40aa5b30a86c8b25e048cec55fbd4ec2199c360fd339738")}
needs_g1 = pytest.mark.skipif(not G1_XML.exists(), reason="run humanoid/setup.sh first")


def _run_task_ast() -> ast.Module:
    return ast.parse((ROOT / "adapters/run_task.py").read_text())


def layout_plan_fn() -> Any:
    """adapters/run_task.layout_plan, its own source: run_task itself does not import in the G1 venv (openai)."""
    fn = next(n for n in _run_task_ast().body if isinstance(n, ast.FunctionDef) and n.name == "layout_plan")
    ns: dict[str, Any] = {"Any": Any, "realizer": agent_truth.realizer, "seeded": agent_truth.seeded}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "adapters/run_task.py", "exec"), ns)
    return ns["layout_plan"]


def run_task_const(name: str) -> Any:
    for n in _run_task_ast().body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple):
            names = [t.id for t in n.targets[0].elts]
            if name in names:
                return ast.literal_eval(n.value)[names.index(name)]
    raise KeyError(name)


def specs() -> list[tuple[str, dict]]:
    """(world, spec) of every distinct synthetic spec in LISTS."""
    out: dict[str, tuple[str, dict]] = {}
    for path in LISTS:
        for r in yaml.safe_load((ROOT / path).read_text()):
            out.setdefault(json.dumps(r["layout_spec"], sort_keys=True), (r["world"], r["layout_spec"]))
    return list(out.values())


@pytest.fixture(scope="module")
def scenes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, tuple]:
    out = {}
    for world in ("closed", "cabinet_open"):
        d_ = tmp_path_factory.mktemp(world)
        m = mujoco.MjModel.from_xml_path(str(build_clinic_scene(world, d_, G1_XML)))
        out[world] = (m, mujoco.MjData(m), json.loads((d_ / "scene.json").read_text()))
    return out


def check_scene(m: mujoco.MjModel, d: mujoco.MjData, label_of_body: list[int], truth: dict, parked: list[str],
                frozen: dict[str, dict]) -> None:
    """Truth object -> its label -> the one body with that label -> that body's pose == the truth pose."""
    rz = agent_truth.realizer()
    for o in truth["objects"]:
        ids = [b for b, lab in enumerate(label_of_body) if lab == o["label"]]
        assert len(ids) == 1, (o["id"], ids)
        name = m.body(ids[0]).name
        assert name == CLINIC_PREFIX + o["model_name"], (o["id"], name)
        x, y, z, yaw = body_poses(m, d, [name.removeprefix(CLINIC_PREFIX)])[name.removeprefix(CLINIC_PREFIX)]
        p = o["pose"]
        assert [x, y, z] == pytest.approx([p[0], p[1], p[2]], abs=1e-6), o["id"]
        assert abs(math.remainder(yaw - p[5], 2 * math.pi)) < 1e-6, o["id"]
    ids = {o["id"] for o in truth["objects"]}
    for oid in parked:
        assert oid not in ids, oid
        gz = frozen[oid]["model_name"]
        px, py, pyaw = rz.park_pose(oid)
        x, y, _, yaw = body_poses(m, d, [gz])[gz]
        assert [x, y] == pytest.approx([px, py], abs=1e-6) and abs(math.remainder(yaw - pyaw, 2 * math.pi)) < 1e-6


def place_and_check(scene: tuple, world: str, spec: dict, label_of_body: list[int] | None = None) -> None:
    m, d, meta = scene
    frozen_truth = json.loads(agent_truth.world_truth_path(world).read_text())
    want, truth, keys, _ = layout_plan_fn()(world, frozen_truth, None, spec)
    assert sorted(want) == sorted(meta["mocap_bodies"])  # every spec object hosted by a mocap body
    place_bodies(m, d, {k: list(v) for k, v in want.items()})
    mujoco.mj_forward(m, d)
    check_scene(m, d, meta["label_of_body"] if label_of_body is None else label_of_body, truth, keys["parked"],
                {o["id"]: o for o in frozen_truth["objects"]})


@needs_g1
def test_every_synthetic_spec_places_as_truth_for_spec(scenes: dict) -> None:
    got = specs()
    assert {w for w, _ in got} == {"closed", "cabinet_open"} and len(got) >= 12
    for world, spec in got:
        place_and_check(scenes[world], world, spec)


@needs_g1
def test_label_swap_mutant_fails(scenes: dict) -> None:
    """Mutant: toolbox and chair_orange_1 swap labels in scene.json -> the check must fail."""
    world, spec = next((w, s) for w, s in specs() if "chair_orange_1" in s["objects"])
    m, _, meta = scenes[world]
    lab = list(meta["label_of_body"])
    a, b = (m.body(CLINIC_PREFIX + n).id for n in ("toolbox_red_1", "chair_orange_1"))
    lab[a], lab[b] = lab[b], lab[a]
    with pytest.raises(AssertionError):
        place_and_check(scenes[world], world, spec, lab)


@needs_g1
def test_scene_json_and_mocap(scenes: dict) -> None:
    rz = agent_truth.realizer()
    for world, (m, _, meta) in scenes.items():
        assert meta["world"] == world and m.nmocap == len(rz.SPEC_OBJECTS) == len(meta["mocap_bodies"])
        mocap = {m.body(b).name for b in range(m.nbody) if m.body_mocapid[b] >= 0}
        assert mocap == {CLINIC_PREFIX + n for n in meta["mocap_bodies"]}
        assert set(m.geom_group[m.geom_bodyid == 0]) <= {RAY_GROUP}  # robot ids stay out of the ray group
        robot = set(meta["robot_body_ids"])
        assert all((m.geom_group[g] == RAY_GROUP) != (m.geom_bodyid[g] in robot) for g in range(m.ngeom))


@needs_g1
def test_blocked_world_scene(tmp_path: Path) -> None:
    m = mujoco.MjModel.from_xml_path(str(build_clinic_scene("blocked_storage", tmp_path, G1_XML)))
    meta = json.loads((tmp_path / "scene.json").read_text())
    blocker = json.loads(agent_truth.world_truth_path("blocked_storage").read_text())["blocker"]
    assert meta["world"] == "blocked_storage" and m.nmocap == 9
    assert meta["label_of_body"][m.body(CLINIC_PREFIX + blocker["id"]).id] == blocker["label"]


def test_generate_mjcf_default_byte_identical() -> None:
    for name, (xml_sha, lab_sha) in GOLDEN.items():
        xml, labels = generate_mjcf(CLINIC / name)
        assert hashlib.sha256(xml.encode()).hexdigest() == xml_sha, name
        assert hashlib.sha256(labels.encode()).hexdigest() == lab_sha, name
    xml, _ = generate_mjcf(CLINIC_LAYOUT, mocap=frozenset({"toolbox_red_1"}))
    assert xml.count('mocap="true"') == 1 and '<body name="toolbox_red_1" mocap="true"' in xml


def test_place_constants_equal_teleport() -> None:
    assert (scene_g1.LAND_M, scene_g1.LAND_DEG) == (run_task_const("LAND_M"), run_task_const("LAND_DEG"))
    assert (scene_g1.READBACKS, scene_g1.READBACK_WAIT_S) == (run_task_const("READBACKS"),
                                                              run_task_const("READBACK_WAIT_S"))
    assert (scene_g1.SET_RETRIES, scene_g1.SET_WAIT_S) == (run_task_const("SET_POSE_RETRIES"),
                                                           run_task_const("SET_POSE_WAIT_S"))
    assert scene_g1.HALL == (CLINIC_SPAWN["x"], CLINIC_SPAWN["y"], math.radians(CLINIC_SPAWN["yaw_deg"]))
    node = ast.parse((ROOT / "humanoid/g1_node.py").read_text())
    settle = next(n.value.value for n in node.body if isinstance(n, ast.Assign)
                  and getattr(n.targets[0], "id", "") == "SETTLE_S")
    assert scene_g1.SETTLE_S == settle


def test_readback_check() -> None:
    want = {"a": (1.0, 2.0, 0.0, 0.5)}
    ok = {"bodies": {"a": [1.0, 2.0 + 0.019, 0.0, 0.5 + math.radians(0.9)]}, "robot_z": 0.75}
    assert scene_g1.check(want, ok) == []
    assert scene_g1.check(want, {**ok, "bodies": {"a": [1.0, 2.021, 0.0, 0.5]}})
    assert scene_g1.check(want, {**ok, "bodies": {"a": [1.0, 2.0, 0.0, 0.5 + math.radians(1.1)]}})
    assert scene_g1.check(want, {**ok, "bodies": {}})
    assert scene_g1.check(want, {**ok, "robot_z": 0.49})


def test_ray_fan_wall_box_and_self_exclusion() -> None:
    """Wall face at x = 2.0, box face at y = -1.25, a robot-like sphere (group 0) around the origin: the fan sees the
    wall (angle 0) and the box (angle -90 deg) within 1 cm, not the sphere; nothing within 4 m behind (angle 180)."""
    xml = f"""<mujoco><worldbody>
      <body name="wall" pos="2.05 0 1"><geom type="box" size="0.05 3 1" group="{RAY_GROUP}"/></body>
      <body name="box" pos="0 -1.5 0.4" mocap="true"><geom type="box" size="0.3 0.25 0.4" group="{RAY_GROUP}"/></body>
      <body name="robot" pos="0 0 0.4"><freejoint/><geom type="sphere" size="0.3"/></body>
    </worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = ray_fan(m, d, 0.0, 0.0, 0.0)
    assert r.shape == (360,) and r.dtype == np.float32
    assert abs(r[180] - 2.0) < 0.01 and abs(r[90] - 1.25) < 0.01 and np.isinf(r[0])
    r = ray_fan(m, d, 0.0, 0.0, math.pi / 2)  # robot turned left: the wall is now at -90 deg in the pelvis_yaw frame
    assert abs(r[90] - 2.0) < 0.01 and abs(r[0] - 1.25) < 0.01
    d.mocap_pos[0] = [0.0, -3.0, 0.4]  # a mocap move is seen by the sensor (geometry only)
    mujoco.mj_forward(m, d)
    assert abs(ray_fan(m, d, 0.0, 0.0, 0.0)[90] - 2.75) < 0.01
