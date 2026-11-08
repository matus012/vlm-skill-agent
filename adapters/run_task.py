#!/usr/bin/env python3
"""One task, one planner, one run dir (agent/INTERFACES.md sections 0 and 8).

    source sim/env.sh
    python3 adapters/run_task.py --task find-toolbox --out RUN_DIR [--script tasks/scripts/find-toolbox.yaml]
        [--planner scripted|vlm [--seed N]] [--start-place P] [--extra-place name:x,y,yaw ...]

With the clinic sim + AMCL + Nav2 running (sim/launch_clinic.sh localization:=true nav2:=true); the operator holds
the gz lock (sim/INTERFACES.md section 6), this script never takes it.
1. Setup (not a step): go_to(start place x, y, yaw), default the hall; logs run.json setup {start_place, reached,
   gt_err_m} (+ nav status, sim / wall time, GT pose). reached = Nav2 reached AND GT within 0.30 m of the start place
   (002 finding 14).
2. agent.runtime.run with the SimTruth hook (gt_pose_after, place_err_m after every step), the prompt-only Task (no
   answers) and the room names of map_knowledge.json. adapters/hand_help.py watches GT jumps + /initialpose during the
   run; run.json hand_help / hand_help_why are set from it before scoring (N7).
   VLM planner: malformed_limit = runtime.MALFORMED_LIMIT (two malformed outputs in a row end the run); scripted: none.
   During the run adapters/gt_trace.py samples GT at 2 Hz into RUN_DIR/gt_trace.jsonl (demo recorder only; never
   affects the run); run.json gt_trace {samples, errors, last_error}. run.json gz_reads {reads, attempts, deaths}: every
   `gz topic` CLI read of the run and each signal death (probes/gz_cli.py, a recorded deviation: <= 2 retries per read).
3. agent.scorer.write_score -> RUN_DIR/score.json.
world_tag (run.json) = the newest phase1-world-frozen* tag whose clinic.sdf + truth.json match the files on disk (N5).
World variant (hpc/PERUN_SIM.md section 5): env SIM_WORLD (default closed; agent/truth.py WORLD_DIRS) picks the
truth.json (task check, truth hook, layout sha, scoring) and map_knowledge.json of the run; run.json records `world`.
cabinet_open's world_tag: the newest phase4-world-cabinet-open* tag matching its generated files.
--extra-place: test-only places (yaw in rad, map frame), added to this run's navigate_to enum, never to places.yaml.
Seeded layout (--layout-seed / run_one(layout_seed=S), INTERFACES section 10): before the setup move every movable of
seeded.placements(S) is teleported (one `gz service -s /world/clinic/set_pose` per model, z from the world's frozen
truth.json) and read back from one /world/clinic/pose/info message (probes/gz_cli.py); each must land within 2 cm /
1 deg, else the run fails with outcome crash, crash_origin layout_teleport (gz_read if the read-back died), scorer label
sim, no setup move and no VLM step. Every seeded run teleports (never assumes the sim's previous state). The run's truth
(task check, truth hook, layout sha, scoring) is seeded.truth_for(S, world); run.json records layout_seed.
Layout spec (--layout-spec FILE / run_one(layout_spec=spec), Phase 8, INTERFACES section 12): the same teleport and
read-back rule for every spec object (realize.SPEC_OBJECTS: toolbox, cabinet, beds, chairs): placed ones to
realize(spec)["poses"] (beds / cabinet at their frozen poses: a previous run may have parked them), parked ones to
realize.park_pose; the sim's SIM_WORLD must be realize(spec)["world"] (else refused before any gz call). The run's truth
is realize.truth_for_spec(spec); run.json records layout_spec (canonical), layout_spec_sha256 and parked.
World restore (run_one(restore_world=place), robots.make sets it for the G1 only, task 012): a run with no layout seed /
spec puts every realize.SPEC_OBJECTS model back at its pose in the world's frozen truth.json through `place` (the G1:
G1Scene.place, which also resets the robot to the hall), same rule, failure record and run.json `teleport` as a layout;
the run's truth stays the world's truth.json (no layout keys). None (the TB4): the frozen world is assumed, as before.
`--planner vlm` is VlmPlanner (VLM_BASE_URL / VLM_API_KEY / VLM_MODEL) with sampling seed --seed.
"""
import argparse
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "probes"))
from agent import runtime, scorer  # noqa: E402
from agent.planners import Planner, ScriptedPlanner, VlmConfig, VlmPlanner  # noqa: E402
from agent.robot import Adapter  # noqa: E402
from agent.context import system_prompt  # noqa: E402
from agent.runlog import RunLog, git_state, utc_now  # noqa: E402
from agent.skills import NAV_TIMEOUT_S, PLACE_BOUND_M, SkillContext, build_tools, load_places  # noqa: E402
from agent.tasks import SCRIPTS_DIR, Task, load_script, task_by_id as prompt_task  # noqa: E402
from agent.truth import (BLOCKED_ROOMS, DEFAULT_WORLD, WORLD_DIRS, derive, load_truth, realizer, seeded,  # noqa: E402
                         task_by_id, world_truth_path)
from agent.world import load_map  # noqa: E402
from adapters.gt_trace import GtTrace  # noqa: E402
from adapters.hand_help import HandHelp, InitialPoseWatch  # noqa: E402
import gz_cli  # noqa: E402

PLACES_YAML = scorer.PLACES_YAML
WORLD_TAG_GLOB = "refs/tags/phase1-world-frozen*"
WORLD_FILES = ("sim/worlds/clinic/generated/clinic.sdf", "sim/worlds/clinic/generated/truth.json")
# world -> world_tag() kwargs; closed = the defaults (phase1-world-frozen*)
WORLD_TAG_KW: dict[str, dict[str, Any]] = {"closed": {}, "cabinet_open": {
    "files": ("sim/worlds/clinic/generated_cabinet_open/clinic.sdf", "sim/worlds/clinic/generated_cabinet_open/truth.json"),
    "glob": "refs/tags/phase4-world-cabinet-open*"}, **{f"blocked_{r}": {  # Phase 6 (INTERFACES section 11)
        "files": tuple(f"sim/worlds/clinic/generated_blocked_{r}/{f}" for f in ("clinic.sdf", "truth.json")),
        "glob": "refs/tags/phase6-world-blocked*"} for r in BLOCKED_ROOMS}}
START_PLACE = "hall"
Pose = tuple[float, float, float]
# seeded layouts (INTERFACES section 10): the gz service form of sim/launch_clinic.sh's dock move
SET_POSE = ["gz", "service", "-s", "/world/clinic/set_pose", "--reqtype", "gz.msgs.Pose", "--reptype",
            "gz.msgs.Boolean", "--timeout", "5000", "--req"]
POSE_INFO = "/world/clinic/pose/info"
LAND_M, LAND_DEG = 0.02, 1.0  # INTERFACES section 10: a teleported movable lands within 2 cm / 1 deg


def set_pose_req(name: str, x: float, y: float, z: float, yaw: float) -> str:
    return (f'name: "{name}", position: {{x: {x}, y: {y}, z: {z}}}, '
            f'orientation: {{x: 0, y: 0, z: {math.sin(yaw / 2)}, w: {math.cos(yaw / 2)}}}')


def model_poses() -> dict[str, tuple[float, float, float, float]]:
    """{name: (x, y, z, yaw)} of every entity in one pose/info message (first entry per name, as probes/photos.py)."""
    msg = json.loads(gz_cli.echo_once(POSE_INFO, 20).splitlines()[0])
    out: dict[str, tuple[float, float, float, float]] = {}
    for p in msg["pose"]:
        pos, q = p.get("position", {}), p.get("orientation", {})
        qx, qy, qz, qw = (*(q.get(k, 0.0) for k in "xyz"), q.get("w", 1.0))
        out.setdefault(p.get("name", ""), (*(pos.get(k, 0.0) for k in "xyz"),
                                           math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))))
    return out


READBACKS, READBACK_WAIT_S = 3, 1.0  # pose/info read-backs after the set_pose calls (bounded; a miss on all 3 fails the run)
SET_POSE_RETRIES, SET_POSE_WAIT_S = 2, 1.0  # a recorded deviation.2: a set_pose that times out (empty reply) is sent again, at most twice, each logged


def teleport(want: dict[str, tuple[float, float, float, float]], log: dict[str, Any] | None = None) -> str | None:
    """gz set_pose of each model to want[name] = (x, y, z, yaw), in order, then one pose/info read-back; None when every
    model landed within LAND_M / LAND_DEG, else why. A dead read-back raises (gz_cli).
    Deviation 20.2: a set_pose with rc 0 and an empty reply (gz: "Service call timed out") is sent again, at most
    SET_POSE_RETRIES times (set_pose is absolute, so a resend is safe); a reply `data: false` or rc != 0 is not retried.
    `log` (run.json `teleport`) gets the number of set_pose sends and one line per retry."""
    log = {} if log is None else log
    log.update(models=len(want), sends=0, retries=[])
    for name, (x, y, z, yaw) in want.items():
        for k in range(SET_POSE_RETRIES + 1):
            p = subprocess.run([*SET_POSE, set_pose_req(name, x, y, z, yaw)], capture_output=True, text=True)
            log["sends"] += 1
            if p.returncode == 0 and "data: true" in p.stdout:
                break
            why = f"rc {p.returncode}, reply {p.stdout.strip()!r}, stderr {p.stderr.strip()[-200:]!r}"
            if p.returncode != 0 or p.stdout.strip() or k == SET_POSE_RETRIES:
                return f"set_pose {name}: {why}" + (f" (send {k + 1} of {SET_POSE_RETRIES + 1})" if k else "")
            log["retries"].append(f"set_pose {name}: send {k + 1} {why}; sent again")
            time.sleep(SET_POSE_WAIT_S)
    for k in range(READBACKS):  # a pose/info message from before the moves can still be in flight: read again
        got, bad = model_poses(), []
        for name, w in want.items():
            if name not in got:
                bad.append(f"{name} not in {POSE_INFO}")
                continue
            g = got[name]
            dm, da = math.dist(g[:3], w[:3]), abs(math.degrees(math.remainder(g[3] - w[3], 2 * math.pi)))
            if dm > LAND_M or da > LAND_DEG:
                bad.append(f"{name} at {[round(v, 4) for v in g]}, want {[round(v, 4) for v in w]}: "
                           f"{dm:.3f} m, {da:.2f} deg")
        if not bad:
            return None
        if k + 1 < READBACKS:
            time.sleep(READBACK_WAIT_S)
    return f"after {READBACKS} read-backs: " + "; ".join(bad)


def sim_world() -> str:
    """The world variant of this sim (env SIM_WORLD, default closed); refuses an unknown name."""
    w = os.environ.get("SIM_WORLD", DEFAULT_WORLD)
    if w not in WORLD_DIRS:
        raise SystemExit(f"SIM_WORLD={w!r} is not a world variant {sorted(WORLD_DIRS)}")
    return w


def world_tag(repo: Path = ROOT, files: tuple[str, ...] = WORLD_FILES, glob: str = WORLD_TAG_GLOB) -> str:
    """The newest `glob` tag whose committed `files` hash equal to the files on disk (N5)."""
    def git(*a: str) -> bytes:
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, check=True).stdout
    disk = [hashlib.sha256((repo / f).read_bytes()).hexdigest() for f in files]
    tags = git("for-each-ref", "--sort=-creatordate", "--format=%(refname:short)", glob).decode().split()
    for tag in tags:
        if [hashlib.sha256(git("show", f"{tag}:{f}")).hexdigest() for f in files] == disk:
            return tag
    raise RuntimeError(f"no tag {glob} matches {files} on disk (tags: {tags})")


def layout_plan(world: str, frozen_truth: dict[str, Any], layout_seed: int | None,
                layout_spec: dict[str, Any] | None) -> tuple[dict[str, tuple[float, float, float, float]], dict[str, Any],
                                                             dict[str, Any], str]:
    """(teleport targets {model name: (x, y, z, yaw)}, the run's truth, run.json keys, label) of a seeded or a spec
    layout; z and the gz model name from the world's frozen truth. A spec realizing in another world is refused."""
    frozen = {o["id"]: o for o in frozen_truth["objects"]}

    def want(poses: dict[str, tuple[float, float, float]]) -> dict[str, tuple[float, float, float, float]]:
        return {frozen[oid]["model_name"]: (x, y, frozen[oid]["pose"][2], yaw) for oid, (x, y, yaw) in poses.items()}
    if layout_spec is None:
        return (want(seeded().placements(layout_seed)), seeded().truth_for(layout_seed, world),
                {"layout_seed": layout_seed}, f"layout seed {layout_seed}")
    assert layout_seed is None, "a run has a layout seed or a layout spec, not both"
    rz = realizer()
    r = rz.realize(layout_spec)
    if r["world"] != world:
        raise ValueError(f"layout spec realizes in world {r['world']!r}; this sim is SIM_WORLD {world!r}: refused")
    poses = {**r["poses"], **{oid: rz.park_pose(oid) for oid in r["parked"]}}
    sha = rz.spec_sha256(layout_spec)
    return (want({oid: poses[oid] for oid in rz.SPEC_OBJECTS}), rz.truth_for_spec(layout_spec),
            {"layout_spec": rz.canonical(layout_spec), "layout_spec_sha256": sha, "parked": r["parked"]},
            f"layout spec {sha[:12]}")


def frozen_plan(frozen_truth: dict[str, Any]) -> dict[str, tuple[float, float, float, float]]:
    """{gz model name: (x, y, z, yaw)} of every realize.SPEC_OBJECTS object at its pose in the world's frozen truth."""
    frozen = {o["id"]: o for o in frozen_truth["objects"]}
    return {frozen[oid]["model_name"]: (*frozen[oid]["pose"][:3], frozen[oid]["pose"][5])
            for oid in realizer().SPEC_OBJECTS}


def run_one(ad: Adapter, gt_pose: Callable[[], Pose], task: Task, planner: Planner, run_dir: Path, *,
            robot: str = "turtlebot4", extra_places: dict[str, Pose] | None = None,
            places_path: Path = PLACES_YAML, truth_path: Path | None = None, map_path: Path | None = None,
            tag: str | None = None, watch_initialpose: bool = False,
            hand_help: Callable[[], HandHelp] = HandHelp, start_place: str = START_PLACE,
            malformed_limit: int | None = None, extra_meta: dict[str, Any] | None = None,
            layout_seed: int | None = None, layout_spec: dict[str, Any] | None = None,
            place_objects: Callable[..., str | None] | None = None,
            restore_world: Callable[..., str | None] | None = None,
            trace_factory: Callable[[Callable[[], Pose], Path], Any] = GtTrace) -> dict[str, Any]:
    """Setup move to `start_place` (default the hall), the run, the score. `gt_pose`: evaluator ground truth (SimTruth.gt_pose).
    `task` is passed to the runtime prompt-only; its truth is derived again by the scorer. `tag`: world tag (None:
    world_tag()). `watch_initialpose`: run the /initialpose watch (needs ROS; off in unit tests). `hand_help`: makes
    the detector (default: TB4 limits, adapters/hand_help.py). `extra_meta`: more run.json keys (phase3_run: the
    invocation_id, 005 B1); None adds nothing. The world is sim_world() (env SIM_WORLD); `truth_path` / `map_path`
    None: that world's truth.json / map_knowledge.json (tests pass a fake world's). `layout_seed`: the seeded layout
    (module doc); None = the frozen world, as before 008. `layout_spec`: a Phase 8 layout spec (module doc).
    `place_objects(want, log) -> why | None`: the layout placer (None: this module's teleport, looked up at call time);
    `restore_world(want, log) -> why | None`: run without a layout: the frozen world's placer (module doc; None: none);
    `trace_factory(gt_pose, path)`: the GT trace (default GtTrace); a `history_samples` in its stop() summary goes into
    run.json gt_trace (task 012: robots.make passes the G1 forms)."""
    world = sim_world()
    truth_path = truth_path or world_truth_path(world)
    map_path = map_path or WORLD_DIRS[world] / "map_knowledge.json"
    run_dir = Path(run_dir)
    assert not run_dir.exists(), f"{run_dir} exists; one run per dir"
    places = {**load_places(places_path), **(extra_places or {})}
    truth, known = load_truth(truth_path), load_map(map_path)
    assert known["layout_sha256"] == truth["layout_sha256"], "map_knowledge.json and truth.json from different layouts"
    task = Task(task.id, task.kind, task.prompt, task.target)  # prompt only: no answers reach the runtime
    gz0 = gz_cli.mark()  # this run's gz CLI reads (teleport read-back, setup, truth hook, GT trace) from here
    layout_keys: dict[str, Any] = {}
    tp_keys: dict[str, Any] = {}  # run.json `teleport` of a layout or a world restore
    laid = layout_seed is not None or layout_spec is not None
    place = (place_objects or teleport) if laid else restore_world
    if place is not None:
        if laid:
            want, truth, layout_keys, what = layout_plan(world, truth, layout_seed, layout_spec)
            derive(task.kind, task.target, truth)  # the task must be valid on this layout (TaskError otherwise)
        else:
            want, what = frozen_plan(truth), f"frozen world {world} restore"
        head = {"robot": robot, "world": world, "layout_sha256": truth["layout_sha256"], **layout_keys,
                "world_tag": tag or world_tag(**WORLD_TAG_KW[world]), **git_state(ROOT), "start_place": start_place,
                **(extra_meta or {})}
        wall0, origin = time.perf_counter(), "layout_teleport"
        tp: dict[str, Any] = {}
        head["teleport"] = tp_keys["teleport"] = tp  # a recorded deviation.2: sends + retries, per run (run.json)
        try:
            why = place(want, tp)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:  # the read-back failed
            why = f"read-back {type(e).__name__}: {e}"
            origin = "gz_read" if getattr(e, "gz_read_failed", False) else origin
        if why is not None:
            return teleport_failed(run_dir, task, planner, places, list(known["rooms"]), head, malformed_limit,
                                   f"{what} teleport did not land: {why}", origin,
                                   time.perf_counter() - wall0, gz0, places_path, None if laid else truth_path)
    hx, hy, hyaw = places[start_place]
    go = ad.go_to(hx, hy, hyaw, NAV_TIMEOUT_S)
    gt = gt_pose()
    err = math.dist(gt[:2], (hx, hy))  # reached = Nav2 reached and GT within the place bound (design review N2)
    setup = {"start_place": start_place, "reached": go.status == "reached" and err <= PLACE_BOUND_M, "gt_err_m": err,
             "nav_status": go.nav_status, "sim_s": go.sim_s, "wall_s": go.wall_s, "gt_pose": list(gt)}

    hh, n_step = hand_help(), [0]
    hh.gt(0, ad.sim_time(), gt[:2], drove=False)

    def truth_hook(place: str | None) -> dict[str, Any]:
        g = gt_pose()
        n_step[0] += 1
        hh.gt(n_step[0], ad.sim_time(), g[:2], drove=place is not None)
        return {"gt_pose_after": list(g),
                "place_err_m": math.dist(g[:2], places[place][:2]) if place is not None else None}

    meta = {"robot": robot, "world": world, "layout_sha256": truth["layout_sha256"],
            **layout_keys, **tp_keys,
            "world_tag": tag or world_tag(**WORLD_TAG_KW[world]),
            **git_state(ROOT), "start_place": start_place, "setup": setup, **(extra_meta or {})}
    run_dir.mkdir(parents=True)
    watch = InitialPoseWatch(run_dir / "initialpose.log") if watch_initialpose else None
    trace = trace_factory(gt_pose, run_dir / "gt_trace.jsonl").start()  # demo only (T10); never raises
    try:
        res = runtime.run(task, SkillContext(ad, places, task.kind, list(known["rooms"])), planner, run_dir, meta,
                          truth=truth_hook, malformed_limit=malformed_limit)
    finally:
        tr = trace.stop()
        if watch is not None:
            hh.initialpose(watch.stop())
    gt_trace = {k: tr.get(k) for k in ("samples", "errors", "last_error")}
    if "history_samples" in tr:  # G1Trace only (task 012); a TB4 run.json is unchanged
        gt_trace["history_samples"] = tr["history_samples"]
    run_json = run_dir / "run.json"
    run_json.write_text(json.dumps({**json.loads(run_json.read_text()), **hh.verdict(), "gt_trace": gt_trace,
                                    "gz_reads": gz_cli.since(gz0)}, indent=1))
    # a seeded / spec run is scored by the scorer's own rule (run.json world + layout_seed / layout_spec)
    score = scorer.write_score(run_dir, truth_path=None if laid else truth_path, places_path=places_path)
    return {"run_dir": str(run_dir), "outcome": res.outcome, "steps": res.steps, "setup": setup, "score": score,
            "git_commit": meta["git_commit"], "git_dirty": meta["git_dirty"], "hand_help": hh.verdict()}


def teleport_failed(run_dir: Path, task: Task, planner: Planner, places: dict[str, Pose], rooms: list[str],
                    meta: dict[str, Any], malformed_limit: int | None, why: str, origin: str, wall_s: float,
                    gz0: tuple[int, int, int], places_path: Path, truth_path: Path | None = None) -> dict[str, Any]:
    """The record of a seeded run whose movables did not land: run.json as the runtime writes it (the prompt and tools
    that would have been sent), outcome crash with `origin`, no setup move (setup.reached null), no step; scored."""
    sp = system_prompt(list(places))
    setup = {"start_place": meta["start_place"], "reached": None, "gt_err_m": None, "skipped": "layout teleport failed"}
    now = utc_now()
    log = RunLog(run_dir, {"task": task.run_json(), "planner": planner.name, "planner_info": planner.info,
                           "system_prompt": sp, "system_prompt_sha256": hashlib.sha256(sp.encode()).hexdigest(),
                           "tools": build_tools(list(places), task.kind, rooms), "malformed_limit": malformed_limit,
                           "hand_help": False, "hand_help_why": [], **{**meta, "setup": setup},
                           "start_utc": now, "end_utc": now, "outcome": "crash", "steps": 0, "report": None,
                           "sim_s": 0.0, "wall_s": wall_s, "crash": why, "crash_origin": origin,
                           "gz_reads": gz_cli.since(gz0)})
    log.finish({})
    score = scorer.write_score(run_dir, truth_path=truth_path, places_path=places_path)
    return {"run_dir": str(run_dir), "outcome": "crash", "steps": 0, "setup": setup, "score": score,
            "git_commit": meta["git_commit"], "git_dirty": meta["git_dirty"],
            "hand_help": {"hand_help": False, "hand_help_why": []}}


def parse_place(s: str) -> tuple[str, Pose]:
    name, xyz = s.split(":")
    x, y, yaw = (float(v) for v in xyz.split(","))
    return name, (x, y, yaw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="task id: tasks/dev.yaml, then tasks/phase5_tasks.yaml")
    ap.add_argument("--out", type=Path, required=True, help="run dir (must not exist)")
    ap.add_argument("--planner", choices=["scripted", "vlm"], default="scripted")
    ap.add_argument("--script", type=Path, help="scripted: default tasks/scripts/<task>.yaml")
    ap.add_argument("--extra-place", action="append", default=[], help="name:x,y,yaw (rad), test-only")
    ap.add_argument("--start-place", default=START_PLACE, help="places.yaml place of the setup move")
    ap.add_argument("--seed", type=int, default=None, help="vlm: sampling seed")
    ap.add_argument("--layout-seed", type=int, default=None, help="seeded layout (INTERFACES section 10)")
    ap.add_argument("--layout-spec", type=Path, default=None, help="YAML / JSON file of a Phase 8 layout spec (section 12)")
    a = ap.parse_args()
    spec = None if a.layout_spec is None else yaml.safe_load(a.layout_spec.read_text())
    # the task must be valid in this world (a seeded / spec layout: checked by run_one on that layout)
    task = (task_by_id(a.task, truth_path=world_truth_path(sim_world())) if a.layout_seed is None and spec is None
            else prompt_task(a.task))
    if a.planner == "vlm":
        planner: Planner = VlmPlanner(dataclasses.replace(VlmConfig.from_env(), seed=a.seed))
    else:
        path = a.script or SCRIPTS_DIR / f"{a.task}.yaml"
        tid, steps = load_script(path)
        assert tid == a.task, f"script {path} is for task {tid}, not {a.task}"
        planner = ScriptedPlanner(steps, path.name.removesuffix(".yaml"))
    from adapters import robots
    rb = robots.make(robots.robot_name())  # env ROBOT, default turtlebot4
    try:
        out = run_one(rb.ad, rb.gt_pose, task, planner, a.out, extra_places=dict(map(parse_place, a.extra_place)),
                      watch_initialpose=True, start_place=a.start_place, layout_seed=a.layout_seed,
                      layout_spec=spec,
                      malformed_limit=runtime.MALFORMED_LIMIT if a.planner == "vlm" else None,
                      extra_meta=rb.scene_meta or None, **rb.run_kw)
    finally:
        rb.close()
    print(json.dumps(out, indent=1))
    return 0 if out["score"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
