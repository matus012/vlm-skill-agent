"""One robot factory for the runners (task 012 design section 2): env ROBOT (default turtlebot4) -> make(ROBOT).

    rb = make(robot_name())
    try: run_one(rb.ad, rb.gt_pose, ..., extra_meta={**meta, **rb.scene_meta}, **rb.run_kw)
    finally: rb.close()

turtlebot4: TB4Adapter + SimTruth().gt_pose, run_kw {} (run_one's defaults: teleport, GtTrace, TB4 hand help).
g1: G1Adapter(scene_json=$G1_SCENES/$GZ_PARTITION/scene.json, map_path=WORLD_DIRS[SIM_WORLD]/map_knowledge.json) (the
adapter reads no truth module, so the paths come from here, via run_task's world lookup), G1SimTruth().gt_pose, run_kw {robot g1, hand help at
the CommandManager clamp |(0.5, 0.5)| = 0.71 m/s, place_objects + restore_world G1Scene().place (a frozen-world run
puts the movables back and resets the robot: a layout run earlier on the same node moved them), trace_factory G1Trace}, scene_meta
{"scene": scene_meta(scene.json)} (run.json `scene`). Every import is lazy: the TB4 path never imports humanoid/ (or ROS
before make() is called)."""
import functools
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.robot import Adapter

DEFAULT = "turtlebot4"
G1_V_MAX_MPS = 0.71  # design section 2: |(0.5, 0.5)| m/s, humanoid/geometry.py:72-73 clamp box


@dataclass(frozen=True)
class Robot:
    ad: Adapter
    gt_pose: Callable[[], tuple[float, float, float]]
    run_kw: dict[str, Any]
    scene_meta: dict[str, Any]
    close: Callable[[], None]


def scene_meta(scene_json: Path) -> dict[str, Any]:
    """run.json `scene` of a G1 sim (design section 2) from humanoid/room.py's scene.json: world, mocap bodies, the
    sha256 of the scene XML (next to scene.json) and of label_of_body as canonical JSON."""
    sj = json.loads(scene_json.read_text())
    labels = json.dumps(sj["label_of_body"], separators=(",", ":")).encode()
    return {"world": sj["world"], "xml_sha256": hashlib.sha256(scene_json.with_name(Path(sj["xml"]).name).read_bytes())
            .hexdigest(), "labels_sha256": hashlib.sha256(labels).hexdigest(), "mocap_bodies": sj["mocap_bodies"]}


def robot_name() -> str:
    return os.environ.get("ROBOT", DEFAULT)


def make(name: str) -> Robot:
    if name == "turtlebot4":
        from adapters.turtlebot4 import SimTruth, TB4Adapter  # ROS: only when driving the sim
        ad, st = TB4Adapter(), SimTruth()
        return Robot(ad, st.gt_pose, {}, {}, ad.close)
    if name == "g1":
        from adapters.hand_help import HandHelp
        from adapters.run_task import WORLD_DIRS, sim_world
        from humanoid.adapter_g1 import G1Adapter
        from humanoid.scene_g1 import G1Scene
        from humanoid.sim_truth_g1 import G1SimTruth, G1Trace
        scene_json = Path(os.environ["G1_SCENES"]) / os.environ["GZ_PARTITION"] / "scene.json"
        ad = G1Adapter(scene_json=scene_json, map_path=WORLD_DIRS[sim_world()] / "map_knowledge.json")
        st, scene = G1SimTruth(), G1Scene()

        def close() -> None:  # idempotent; the adapter last: it shuts down the default context G1Scene's node lives on
            try:
                st.close()
                scene.close()
            finally:
                ad.close()
        return Robot(ad, st.gt_pose, {"robot": "g1", "hand_help": functools.partial(HandHelp, v_max_mps=G1_V_MAX_MPS),
                                      "place_objects": scene.place, "restore_world": scene.place,
                                      "trace_factory": G1Trace},
                     {"scene": scene_meta(scene_json)}, close)
    raise SystemExit(f"ROBOT={name!r} is not a robot (turtlebot4, g1)")
