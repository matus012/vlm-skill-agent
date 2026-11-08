#!/usr/bin/env python3
"""G1 vet list (task 012 design section 5, no bar; design notes a recorded deviation): AGGREGATE counts only, over the
unique layouts of the test list. Read by script, never shown; nothing per entry is printed or written.

    /opt/venv-g1/bin/python humanoid/probes/g1_vet_list.py render --scene closed=DIR --scene cabinet_open=DIR --out R.json
    /opt/venv/bin/python    humanoid/probes/g1_vet_list.py count --render R.json --out V.json

render (MuJoCo, no ROS): each layout in its world's scene (DIR/scene.json, the scene a g1_node built): spec objects
  realized (realize.realize; parked ones at park_pose) as the mocap bodies clinic/<model_name>, z from the world's frozen
  truth (as run_task.layout_plan), the robot's free joint on the place point at --base-z (the standing pelvis z a sim's
  ready.json measured; default the model's qpos0 z), joints at qpos0, so the head camera's own geometry (recorded:
  base_z_m, cam_z_m), head camera at place yaw + 0/90/180/270 at all 5 places; per photo the label
  histogram (scene.json label_of_body; robot / world -> 0). R.json holds layout INDEXES + histograms + the scenes'
  mocap body names only.
count: (a) layouts failing G2 / G3 / G4 / G7 (phase1_checks rules and numbers, on the histograms) at the G1 camera;
  (b) layouts with a placed spec object's footprint (seeded.footprint) within LEG_CLEAR_M of a straight geometry.route()
  leg between any two of the 5 places (the G1 has no costmap); (c) placed SPEC_OBJECTS with no mocap body
  (hosting: such an entry is dropped from both columns, a recorded deviation; expected 0). Prints and writes counts only.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "probes"), str(ROOT / "sim/worlds/clinic")]
import realize as rz  # noqa: E402
import seeded as S  # noqa: E402

import yaml  # noqa: E402

LIST = ROOT / "tasks/phase8_pass2.yaml"
LEG_CLEAR_M = 0.35  # design section 5: an object footprint + 0.35 m on a route() leg
PREFIX = "clinic/"  # humanoid/room.py CLINIC_PREFIX
HEADINGS = (0, 90, 180, 270)


def layouts(list_path: Path) -> list[tuple[str, dict]]:
    """Unique (world, spec) in list order (one per spec sha)."""
    seen, out = set(), []
    for e in yaml.safe_load(list_path.read_text()):
        sha = rz.spec_sha256(e["layout_spec"])
        if sha not in seen:
            seen.add(sha)
            out.append((e["world"], e["layout_spec"]))
    return out


def want_of(world: str, spec: dict) -> dict[str, tuple[float, float, float, float]]:
    """{model name: (x, y, z, yaw)} of every spec object, placed or parked (run_task.layout_plan's targets)."""
    frozen = {o["id"]: o for o in json.loads(S.world_truth_path(world).read_text())["objects"]}
    r = rz.realize(spec)
    poses = {**r["poses"], **{oid: rz.park_pose(oid) for oid in r["parked"]}}
    return {frozen[o]["model_name"]: (x, y, frozen[o]["pose"][2], yaw)
            for o, (x, y, yaw) in ((o, poses[o]) for o in rz.SPEC_OBJECTS)}


def render(scenes: dict[str, Path], list_path: Path, out: Path, base_z: float | None) -> None:
    import mujoco
    import numpy as np
    res: dict = {"scenes": {}, "layouts": []}
    ctx = {}
    for world, d in scenes.items():
        sj = json.loads((d / "scene.json").read_text())
        m = mujoco.MjModel.from_xml_path(sj["xml"])
        m.vis.quality.offsamples = 0  # no MSAA: exact ids (as g1_node.py)
        r = mujoco.Renderer(m, 480, 640)
        r.enable_segmentation_rendering()
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
        mocap = {n: int(m.body_mocapid[b]) for b, n in enumerate(names) if m.body_mocapid[b] >= 0}
        cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_rgb")
        ctx[world] = (m, mujoco.MjData(m), r, cam, mocap, np.asarray(sj["label_of_body"], np.uint8),
                      np.append(m.geom_bodyid, 0).astype(np.int64))
        res["scenes"][world] = {"mocap_bodies": sorted(mocap), "xml": sj["xml"]}
    for world, spec in layouts(list_path):
        m, d, r, cam, mocap, lut, gb = ctx[world]
        for name, (x, y, z, yaw) in want_of(world, spec).items():
            if PREFIX + name in mocap:
                d.mocap_pos[mocap[PREFIX + name]] = (x, y, z)
                d.mocap_quat[mocap[PREFIX + name]] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
        counts = {}
        for place, p in S.PLACES.items():
            counts[place] = []
            for h in HEADINGS:
                a = p["yaw"] + math.radians(h)
                d.qpos[:] = m.qpos0
                d.qpos[:7] = (p["x"], p["y"], base_z or m.qpos0[2], math.cos(a / 2), 0, 0, math.sin(a / 2))
                mujoco.mj_forward(m, d)
                r.update_scene(d, camera=cam)
                seg = r.render()
                body = gb[np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)]
                counts[place].append(np.bincount(lut[body].ravel(), minlength=256).tolist())
        res["layouts"].append({"world": world, "counts": counts})
        res["cam_z_m"], res["base_z_m"] = float(d.cam_xpos[cam][2]), float(d.qpos[2])
    out.write_text(json.dumps(res))
    print(f"rendered {len(res['layouts'])} layouts x 5 places x 4 photos; head camera z {res['cam_z_m']:.3f} m")


def view_fails(truth: dict, counts: dict[str, list[list[int]]]) -> set[str]:
    """Gates (g2, g3, g4, g7) failing on these photos: phase1_checks.check_place's rules on label histograms."""
    import phase1_checks as pc
    fails = set()
    for place, cs in counts.items():
        if any(1000 * c[0] > pc.G7_MAX_UNLABELLED_PERMILLE * sum(c) for c in cs):
            fails.add("g7")
        if place == "hall":
            continue
        for o in truth["objects"]:
            px = [c[o["label"]] for c in cs]
            if o["room"] == place:
                if max(px) < pc.G2_MIN_PX:
                    fails.add("g2")
                if o["countable"] and not (sum(px) > 0 and 100 * max(px) >= pc.G3_ONE_PHOTO_PCT * sum(px)):
                    fails.add("g3")
            elif o["countable"] and max(px) >= pc.G4_MAX_LEAK_PX:
                fails.add("g4")
    return fails


def legs(world: str) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    from agent.truth import WORLD_DIRS
    from agent.world import load_doors, load_rooms
    from humanoid.geometry import route
    mk = WORLD_DIRS[world] / "map_knowledge.json"
    rooms, doors = load_rooms(mk), load_doors(mk)
    out = []
    for a, b in itertools.permutations(S.PLACES.values(), 2):
        pts = [(a["x"], a["y"]), *route(a["x"], a["y"], b["x"], b["y"], rooms, doors)]
        out += list(zip(pts, pts[1:]))
    return out


def leg_hit(spec: dict, world_legs: list) -> bool:
    r = rz.realize(spec)
    state = spec["objects"].get(rz.CABINET, {}).get("state")
    foots = [S.footprint(S.CLASS_OF[o], state if o == rz.CABINET else None, *pose) for o, pose in r["poses"].items()]
    return any(S.gap([a, b], f) < LEG_CLEAR_M for a, b in world_legs for f in foots)


def unhosted(world: str, spec: dict, mocap_bodies: set[str]) -> int:
    frozen = {o["id"]: o for o in json.loads(S.world_truth_path(world).read_text())["objects"]}
    return sum(PREFIX + frozen[o]["model_name"] not in mocap_bodies for o in rz.realize(spec)["poses"])


def count(render_path: Path, list_path: Path, out: Path) -> None:
    rend = json.loads(render_path.read_text())
    lays = layouts(list_path)
    assert len(lays) == len(rend["layouts"]), "render and list disagree on the layouts"
    gates: Counter = Counter()
    view = hits = host_lay = host_obj = 0
    world_legs = {}
    for (world, spec), r in zip(lays, rend["layouts"]):
        assert r["world"] == world
        f = view_fails(rz.truth_for_spec(spec), r["counts"])
        gates.update(f)
        view += bool(f)
        hits += leg_hit(spec, world_legs.setdefault(world, legs(world)))
        n = unhosted(world, spec, set(rend["scenes"][world]["mocap_bodies"]))
        host_lay, host_obj = host_lay + bool(n), host_obj + n
    res = {"list": list_path.name, "layouts": len(lays), "cam_z_m": rend.get("cam_z_m"), "base_z_m": rend.get("base_z_m"),
           "a_view_fail_layouts": view, "a_fail_layouts_per_gate": {g: gates[g] for g in ("g2", "g3", "g4", "g7")},
           "b_leg_clearance_layouts": hits, "b_clear_m": LEG_CLEAR_M,
           "c_unhosted_layouts": host_lay, "c_unhosted_objects": host_obj}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["render", "count"])
    ap.add_argument("--scene", action="append", default=[], help="render: world=scene dir")
    ap.add_argument("--base-z", type=float, default=None, help="render: standing pelvis z (ready.json base_z)")
    ap.add_argument("--render", type=Path)
    ap.add_argument("--list", type=Path, default=LIST)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "render":
        render({k: Path(v) for k, v in (s.split("=", 1) for s in a.scene)}, a.list, a.out, a.base_z)
    else:
        count(a.render, a.list, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
