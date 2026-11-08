#!/usr/bin/env python3
"""Task 009 preconditions P1 + P2 (INTERFACES section 11), live, in ONE sim whose world is a blocked variant (env
SIM_WORLD=blocked_<room>, sim_job.sbatch w<i>=) on the map clinic_static:

    source sim/env.sh   # with the sim's ROS_DOMAIN_ID / GZ_PARTITION / SIM_WORLD
    python3 probes/phase6_pre.py {p1,p2,all} --out DIR [--rehearsal]

P1 (fails fast and cleanly): 10 runtime runs (adapters/run_task.run_one + ScriptedPlanner, as Phase 2 G3), try k from
the reachable place starts[k % n] (hall first, then the reachable rooms in places.yaml order): navigate_to(<blocked
room>) -> 'failed: blocked' with skill_detail.nav_sim_s <= 120 s and GT after it outside the blocked room's rectangle,
then navigate_to(hall) -> 'reached', then a report so the run ends (not scored here). Run dirs DIR/p1/<k>-<start>/.
P2 (the rest still works): 10 NavigateToPose tries under the Phase 1 G1 rule (phase1_gates / map_world_check machinery),
targets cycling [r1, r2, r3, hall] over the reachable rooms, each from where the previous ended (first: hall, uncounted).
Writes DIR/p1.json, DIR/p2.json. Git: as phase2_gates (no --rehearsal: refuse a dirty tree, stop once the tree moves;
P1 / P2 fail on a dirty or moved tree).
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "probes")]
import map_world_check as mwc  # noqa: E402
import phase1_gates as pg1  # noqa: E402
import phase2_gates as pg2  # noqa: E402
from adapters.run_task import run_one, sim_world  # noqa: E402
from agent.planners import ScriptedPlanner  # noqa: E402
from agent.runlog import git_state  # noqa: E402
from agent.scorer import read_run  # noqa: E402
from agent.skills import NAV_TIMEOUT_S  # noqa: E402
from agent.truth import load_tasks, load_truth, world_truth_path  # noqa: E402

TRIES = 10              # P1 / P2: 10/10 per blocked layout (task 009 Preconditions)
P1_ID, P1_TASK = "p6-p1", pg2.G3_TASK  # the task only feeds the prompt; the run's score is not part of P1
GZ_WORLD = "clinic"     # gz world name of every clinic variant


def blocked_room(truth: dict[str, Any], world: str) -> str:
    if "blocker" not in truth:
        raise SystemExit(f"refusing world {world!r}: its truth.json has no 'blocker' (not a blocked variant)")
    return truth["blocker"]["room"]


def room_rect(truth: dict[str, Any], room: str) -> tuple[float, float, float, float]:
    """(xmin, ymin, xmax, ymax) of the room's polygon."""
    xs, ys = zip(*truth["rooms"][room]["polygon"])
    return min(xs), min(ys), max(xs), max(ys)


def inside(p: list[float], rect: tuple[float, float, float, float]) -> bool:
    return rect[0] <= p[0] <= rect[2] and rect[1] <= p[1] <= rect[3]


def reachable(places: dict[str, Any], room: str) -> list[str]:
    """Places minus the blocked room, hall first, then the rooms in places.yaml order."""
    return ["hall"] + [p for p in places if p not in ("hall", room)]


def p1_starts(reach: list[str]) -> list[str]:
    return [reach[k % len(reach)] for k in range(TRIES)]


def p2_targets(reach: list[str]) -> list[str]:
    cyc = reach[1:] + ["hall"]
    return [cyc[k % len(cyc)] for k in range(TRIES)]


def stats(xs: list[float | None]) -> dict[str, float | None]:
    v = [x for x in xs if x is not None]
    return {"min": min(v), "median": statistics.median(v), "max": max(v)} if v else \
        {"min": None, "median": None, "max": None}


# ------------------------------------------------------------------------------------------------------------ P1
def p1_script(room: str) -> list[dict[str, Any]]:
    return [{"name": "navigate_to", "args": {"place": room}}, {"name": "navigate_to", "args": {"place": "hall"}},
            {"name": "report", "args": {"answer": room, "photo": 1, "note": "P1 run; the answer is not scored here"}}]


def p1_numbers(run: dict[str, Any], steps: list[dict[str, Any]], rect: tuple[float, float, float, float]) -> dict:
    b, h = steps[0], steps[1]
    sd = b.get("skill_detail") or {}
    gt = b["truth"]["gt_pose_after"]
    return {"setup_reached": run["setup"]["reached"], "setup_gt_err_m": run["setup"]["gt_err_m"],
            "blocked_result": b["result"], "nav_status": sd.get("nav_status"), "nav_sim_s": sd.get("nav_sim_s"),
            "skill_sim_s": b["skill_sim_s"], "gt_after_blocked": gt, "gt_in_room": inside(gt, rect),
            "hall_result": h["result"], "hall_gt_err_m": h["truth"]["place_err_m"]}


def p1_problems(run_dir: Path, room: str, rect: tuple[float, float, float, float]) -> list[str]:
    run, steps = read_run(run_dir)
    p = ["run crashed"] if run["outcome"] == "crash" else []
    calls = [(s["call"]["name"], (s["call"]["args"] or {}).get("place")) for s in steps[:2]]
    if calls != [("navigate_to", room), ("navigate_to", "hall")]:
        return p + [f"steps 1-2 are {calls}, want navigate_to [{room}, hall]"]
    n = p1_numbers(run, steps, rect)
    if not n["setup_reached"]:
        p.append(f"setup did not reach {run['setup']['start_place']} (GT {n['setup_gt_err_m']:.3f} m): invalid start")
    if n["blocked_result"] != "failed: blocked":
        p.append(f"blocked step: result {n['blocked_result']!r}, want 'failed: blocked'")
    if n["nav_sim_s"] is None or not n["nav_sim_s"] <= NAV_TIMEOUT_S:
        p.append(f"blocked step: nav_sim_s {n['nav_sim_s']}, want <= {NAV_TIMEOUT_S} s")
    if n["gt_in_room"]:
        p.append(f"blocked step: GT {n['gt_after_blocked'][:2]} inside {room} {rect} (it got in)")
    if n["hall_result"] != "reached":
        p.append(f"hall step: result {n['hall_result']!r}, want 'reached' (robot stuck)")
    return p


def p1_verdict(dirs: list[Path], world: str, room: str, rect: tuple[float, float, float, float]) -> dict[str, Any]:
    tries = []
    for d in dirs:
        run, steps = read_run(d)
        p = p1_problems(d, room, rect)
        tries.append({"run_dir": str(d), "start": run["start_place"], "problems": p, "pass": not p,
                      "numbers": p1_numbers(run, steps, rect) if len(steps) >= 2 else None})
    gp = pg2.git_problems([pg2.git_of(d) for d in dirs])
    nums = [t["numbers"] or {} for t in tries]
    return {"criterion": f"{TRIES} runtime runs from cycling reachable starts (hall first; setup reached = valid start): "
                         f"navigate_to({room}) -> 'failed: blocked', skill_detail.nav_sim_s <= {NAV_TIMEOUT_S} s sim, "
                         f"GT after it not inside {room}'s rectangle; then navigate_to(hall) -> 'reached'; "
                         f"clean tree, one commit; P1 = {TRIES}/{TRIES}",
            "world": world, "blocked_room": room, "room_rect": rect, "passed": sum(t["pass"] for t in tries),
            "total": len(tries), "time_to_fail": {"nav_sim_s": stats([n.get("nav_sim_s") for n in nums]),
                                                  "skill_sim_s": stats([n.get("skill_sim_s") for n in nums]),
                                                  "nav_status": [n.get("nav_status") for n in nums]},
            "git_problems": gp, "tries": tries,
            "pass": len(tries) == TRIES and all(t["pass"] for t in tries) and not gp}


def run_p1(ad, gt_pose, world: str, room: str, rect: tuple, places: dict, out: Path,  # noqa: ANN001
           guard=lambda: None, **kw: Any) -> dict[str, Any]:  # noqa: ANN001
    """kw: more run_one kwargs (tests: the fake world's paths)."""
    task = {t.id: t for t in load_tasks()}[P1_TASK]
    dirs = []
    for k, start in enumerate(p1_starts(reachable(places, room))):
        d = out / "p1" / f"{k:02d}-{start}"
        guard()
        r = run_one(ad, gt_pose, task, ScriptedPlanner(p1_script(room), P1_ID), d, start_place=start, **kw)
        dirs.append(d)
        print(f"P1 {k + 1:2d} from {start:9s} {r['outcome']:12s} setup {r['setup']['gt_err_m']:.3f} m  "
              f"{'ok' if not p1_problems(d, room, rect) else 'FAIL'}", flush=True)
    return p1_verdict(dirs, world, room, rect)


# ------------------------------------------------------------------------------------------------------------ P2
def p2_record(cur: str, target: str, r: dict, gt0: tuple, gt1: tuple, places: dict, succeeded: int) -> dict[str, Any]:
    """One counted try under the Phase 1 G1 rule."""
    goal_w = pg1.to_world(places[target])
    err, start_err = math.dist(gt1[:2], goal_w[:2]), math.dist(gt0[:2], pg1.to_world(places[cur])[:2])
    valid = start_err <= pg1.G1_MAX_ERR_M
    return {"from": cur, "to": target, **r, "gt_start": gt0, "gt_end": gt1, "goal_world": goal_w, "err_m": err,
            "start_err_m": start_err, "valid_start": valid,
            "pass": bool(valid and r["status"] == succeeded and err <= pg1.G1_MAX_ERR_M
                         and r["sim_s"] <= pg1.G1_MAX_SIM_S)}


def run_p2(robot, places: dict, world: str, room: str, guard=lambda: None) -> dict[str, Any]:  # noqa: ANN001
    reach = reachable(places, room)
    git0 = git_state(ROOT)
    move = {"to": "hall", **robot.navigate(places["hall"])}  # uncounted
    cur, tries = "hall", []
    for target in p2_targets(reach):
        guard()
        gt0, _ = mwc.gt_pose(GZ_WORLD)
        r = robot.navigate(places[target], sim_limit_s=pg1.G1_MAX_SIM_S)
        robot.spin_for_sim(0.5)
        gt1, _ = mwc.gt_pose(GZ_WORLD)
        tries.append(p2_record(cur, target, r, gt0, gt1, places, robot.SUCCEEDED))
        t = tries[-1]
        print(f"P2 {len(tries):2d} {cur:9s} -> {target:9s} {r['status_name']:10s} {r['sim_s']:6.1f} s sim  "
              f"start {t['start_err_m']:.3f} m  err {t['err_m']:.3f} m  {'ok' if t['pass'] else 'FAIL'}", flush=True)
        cur = target
    gp = pg2.git_problems([git0, git_state(ROOT)])
    n = sum(t["pass"] for t in tries)
    return {"criterion": f"{TRIES} NavigateToPose tries over the reachable places (targets {p2_targets(reach)}), each "
                         f"from where the previous ended (first from the hall, reached by an uncounted move); pass = "
                         f"SUCCEEDED, GT within {pg1.G1_MAX_ERR_M} m of the place point (map->world "
                         f"{pg1.T_WORLD_FROM_MAP}), <= {pg1.G1_MAX_SIM_S} s sim, valid start (GT within "
                         f"{pg1.G1_MAX_ERR_M} m of the from place); clean tree, one commit; P2 = {TRIES}/{TRIES}",
            "world": world, "blocked_room": room, "reachable": reach, "passed": n, "total": len(tries),
            "worst_err_m": max(t["err_m"] for t in tries), "worst_sim_s": max(t["sim_s"] for t in tries),
            "invalid_start_tries": sum(not t["valid_start"] for t in tries), "git_problems": gp,
            "tries": tries, "uncounted_moves": [move],
            "pass": len(tries) == TRIES and n == TRIES and not gp}


# ------------------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["p1", "p2", "all"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rehearsal", action="store_true")
    a = ap.parse_args()
    world = sim_world()
    truth = load_truth(world_truth_path(world))
    room = blocked_room(truth, world)
    rect, places = room_rect(truth, room), mwc.load_places(pg1.PLACES_YAML)
    start = git_state(ROOT)
    if not a.rehearsal and start["git_dirty"]:
        sys.exit("refusing a counted run on a dirty tree (commit first, or pass --rehearsal)")

    def guard() -> None:
        if not a.rehearsal and (why := pg2.tree_moved(start, git_state(ROOT))):
            sys.exit(f"stopping the invocation before the next run: {why}")
    print(f"world {world}, blocked {room} {rect}, reachable {reachable(places, room)}", flush=True)
    results = []
    if a.gate in ("p1", "all"):
        from adapters.turtlebot4 import SimTruth, TB4Adapter  # ROS: only live
        ad, st = TB4Adapter(), SimTruth()
        try:
            v = run_p1(ad, st.gt_pose, world, room, rect, places, a.out, guard, watch_initialpose=True)
        finally:
            ad.close()  # rclpy.shutdown: P2's node does its own rclpy.init
        results.append(pg2.write(a.out, "p1", v, a.rehearsal))
    if a.gate in ("p2", "all"):
        results.append(pg2.write(a.out, "p2", run_p2(mwc.make_robot(), places, world, room, guard), a.rehearsal))
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
