#!/usr/bin/env python3
"""Task 003 gate runner: one command per gate, each writes `<out>/<gate>.json` (verdict + numbers); run dirs under
`<out>/g1/`, `<out>/g2/`, `<out>/g3/`; `<out>/measure.json` (no bar) after any sim gate.

    source sim/env.sh   # with the sim's ROS_DOMAIN_ID / GZ_PARTITION
    python3 probes/phase2_gates.py {g1,g2,g3,g4,g5,all} --out DIR [--rehearsal] [--only <script-id,...>]

g1/g2/g3 need the clinic sim with AMCL + Nav2 (sim/launch_clinic.sh localization:=true nav2:=true). The operator holds
the gz lock (sim/INTERFACES.md section 6); this runner never takes it. g4/g5 need no sim.
`--only`: run only these script ids (G1 correct scripts, each once; G2 controls `<task>.neg-<name>`; `g3-wall`).
A subset never passes G1 / G2. `--rehearsal` tags every JSON as a rehearsal (not the counted run). Without it the runner refuses a dirty tree, stops
before any sim run once the tree is dirty or on another commit than at start (exit non-zero), and G1 / G2 / G3 fail if any run is dirty or the runs disagree on git_commit (design review B4).

Gate text: not included in this repo.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.run_task import run_one  # noqa: E402
from agent.context import MAX_STEPS  # noqa: E402
from agent.planners import ScriptedPlanner  # noqa: E402
from agent.runlog import check_complete, git_state  # noqa: E402
from agent.scorer import read_run  # noqa: E402
from agent.skills import NAV_TIMEOUT_S, PLACE_BOUND_M  # noqa: E402
from agent.tasks import KINDS, SCRIPTS_DIR, load_script  # noqa: E402
from agent.truth import TRUTH_JSON, load_tasks, load_truth  # noqa: E402
from probes.replay import replay  # noqa: E402

G1_PER_KIND = 3                         # task 003 G1: 3 of 3 runs per type
G1_TOTAL = G1_PER_KIND * len(KINDS)     # 12 / 12
# G2: control name (tasks/scripts/<task>.neg-<name>.yaml) -> the right reason it must fail for.
G2_FAILS = {  # scorer fails on exactly these checks, label vlm
    "wrong-answer": {"answer"},
    "no-proof": {"proof"},
    "13-steps": {"answer", "proof", "steps"},  # named check: steps; no report -> answer / proof cannot hold either
}
G2_ERRORS = {  # error texts sent back to the caller, in order; then the run reaches its correct report (passes)
    "unknown-call": ("unknown skill", "invalid navigate_to call at place"),
    "bad-format": ("invalid report call at answer",),
    "raw-invalid": ("are not valid JSON", "call exactly one tool"),  # extra control, not a G2 row
}
G2_REQUIRED = {"wrong-answer", "no-proof", "13-steps", "unknown-call", "bad-format"}  # the gate's 5 rows
G3_ID, G3_PLACE, G3_TASK = "g3-wall", "g3_wall", "find-toolbox"
DOOR_MARGIN_M = 0.3  # G3: the wall point is more than 0.3 m from any doorway (lead's task text)
G3_MIN_MOVE_M = 1.0  # G3: the hall step must move the robot (GT) at least this far (design review B1)
HANG_GUARD_S = 1.0   # G3: nav_sim_s may exceed the 120 s limit by one hang-guard spin (design review B2)


def git_problems(states: list[dict[str, Any]]) -> list[str]:
    """Counted runs are code-pinned (design review B4): no run on a dirty tree, all runs on one commit."""
    p = [f"{len([s for s in states if s['git_dirty']])} run(s) on a dirty tree"] if any(s["git_dirty"] for s in states) else []
    commits = sorted({s["git_commit"] for s in states})
    return p + ([f"runs disagree on git_commit: {commits}"] if len(commits) > 1 else [])


def tree_moved(start: dict[str, Any], now: dict[str, Any]) -> str | None:
    """Why the tree is no longer the invocation's start tree (dirty, or another commit); None if unchanged. The runners
    check it before every run and stop the invocation on it (2026-11-04: a tree dirtied mid-invocation)."""
    if now["git_dirty"]:
        return f"tree dirty on commit {now['git_commit']}"
    return None if now["git_commit"] == start["git_commit"] else (
        f"tree on commit {now['git_commit']}, not the start commit {start['git_commit']}")


def git_of(run_dir: Path) -> dict[str, Any]:
    run = read_run(run_dir)[0]
    return {"git_commit": run["git_commit"], "git_dirty": run["git_dirty"]}


def write(out: Path, name: str, verdict: dict, rehearsal: bool) -> dict:
    verdict = {"gate": name, "rehearsal": rehearsal, "written_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
               **verdict}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(verdict, indent=2, default=float))
    print(f"{name}: {'PASS' if verdict.get('pass') else 'FAIL'}", flush=True)
    return verdict


def steps_of(run_dir: Path) -> list[dict[str, Any]]:
    return read_run(run_dir)[1]


# ------------------------------------------------------------------------------------------------------------ G1
def g1_plan(tasks: list) -> list[tuple[str, str]]:  # noqa: ANN001
    """[(kind, script id)]: per kind G1_PER_KIND runs cycling through that kind's tasks in dev.yaml order."""
    by_kind = {k: [t.id for t in tasks if t.kind == k] for k in KINDS}
    return [(k, by_kind[k][i % len(by_kind[k])]) for k in KINDS for i in range(G1_PER_KIND)]


def g1_verdict(runs: list[dict[str, Any]], full: bool) -> dict[str, Any]:
    """runs: run_one outputs + kind. A run counts if the setup reached the hall and the scorer passed."""
    ok = [bool(r["setup"]["reached"] and r["score"]["pass"]) for r in runs]
    gp = git_problems(runs)
    per_kind = {k: f"{sum(o for r, o in zip(runs, ok) if r['kind'] == k)}/{sum(r['kind'] == k for r in runs)}"
                for k in KINDS}
    return {"criterion": f"per task type {G1_PER_KIND} runs of correct scripts (cycling through that type's dev tasks), "
                         f"each from the hall (setup reached); pass = scorer pass on {G1_TOTAL}/{G1_TOTAL}",
            "passed": sum(ok), "total": len(runs), "per_kind": per_kind,
            "verdict_scope": "full plan" if full else "--only subset: never the G1 verdict, pass false",
            "hall_setup_gt_err_m": [r["setup"]["gt_err_m"] for r in runs], "git_problems": gp,
            "runs": [{k: r[k] for k in ("kind", "script", "run_dir", "outcome", "steps", "setup", "score")} for r in runs],
            "pass": full and len(runs) == G1_TOTAL and sum(ok) == G1_TOTAL and not gp}


# ------------------------------------------------------------------------------------------------------------ G2
def control_name(script_id: str) -> str:
    return script_id.split(".neg-", 1)[1]


def g2_problems(name: str, run_dir: Path) -> list[str]:
    """Why control `name`'s run did NOT fail for its right reason (empty = it did)."""
    run, steps = read_run(run_dir)
    score = json.loads((Path(run_dir) / "score.json").read_text())
    failed = {c for c, v in score["checks"].items() if not v}
    p: list[str] = []
    if name in G2_FAILS:
        if score["pass"] or failed != G2_FAILS[name]:
            p.append(f"failed checks {sorted(failed)} != {sorted(G2_FAILS[name])} (pass {score['pass']})")
        if score["label"] != "vlm":
            p.append(f"label {score['label']} != vlm")
        if name == "13-steps" and (run["outcome"] != "fail_step_limit" or len(steps) != MAX_STEPS):
            p.append(f"outcome {run['outcome']} after {len(steps)} steps, want fail_step_limit at {MAX_STEPS}")
        return p
    if name not in G2_ERRORS:
        return [f"control {name!r} has no G2 mapping"]
    if run["outcome"] != "reported" or not score["pass"]:
        p.append(f"did not continue to its correct report: outcome {run['outcome']}, pass {score['pass']}")
    if any("traceback" in s for s in steps):
        p.append("a step crashed")
    errs = [s for s in steps if s["check"] != "ok"]
    want = G2_ERRORS[name]
    if len(errs) != len(want) or not all(w in s["check"] and s["check"].startswith("error:") for w, s in zip(want, errs)):
        p.append(f"error steps {[s['check'] for s in errs]} != expected {list(want)}")
    for s in errs:
        nxt = [t for t in steps if t["step"] == s["step"] + 1]
        if s["result"] is not None or s["photos_out"]:
            p.append(f"step {s['step']}: an invalid call reached the adapter")
        if not nxt or nxt[0]["packet"]["last_result"] != s["check"] \
                or not nxt[0]["packet"]["history"][-1].endswith(s["check"]):
            p.append(f"step {s['step']}: error text not returned to the caller as the next last result / history")
    return p


def g2_verdict(controls: dict[str, Path], full: bool) -> dict[str, Any]:
    """controls: script id -> run dir."""
    per = {sid: g2_problems(control_name(sid), d) for sid, d in controls.items()}
    missing = sorted(G2_REQUIRED - {control_name(s) for s in controls})
    gp = git_problems([git_of(d) for d in controls.values()])
    return {"criterion": "each negative control fails for its named reason: " + json.dumps(
                {**{k: sorted(v) for k, v in G2_FAILS.items()},
                 **{k: f"errors back {list(v)}, run continues to its correct report" for k, v in G2_ERRORS.items()}}),
            "controls": {sid: {"run_dir": str(controls[sid]), "ok": not p, "problems": p} for sid, p in per.items()},
            "missing_rows": missing, "git_problems": gp,
            "verdict_scope": "all controls" if full else "--only subset: never the G2 verdict, pass false",
            "pass": full and not missing and not gp and all(not p for p in per.values())}


# ------------------------------------------------------------------------------------------------------------ G3
def _seg_dist(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def doorway_segments(truth: dict[str, Any]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Every doorway as a segment on its (horizontal: y = const) wall line."""
    out = []
    for r in truth["rooms"].values():
        if "doorway" in r:
            (cx, cy), w = r["doorway"]["center"], r["doorway"]["width"]
            out.append(((cx - w / 2, cy), (cx + w / 2, cy)))
    return out


def g3_wall_point(truth: dict[str, Any]) -> tuple[float, float]:
    """Middle of the longest solid stretch of the hall/storage wall (storage's doorway wall, y = const)."""
    r = truth["rooms"]["storage"]
    (cx, cy), w = r["doorway"]["center"], r["doorway"]["width"]
    xs = [x for x, _ in r["polygon"]]
    a, b = max([(min(xs), cx - w / 2), (cx + w / 2, max(xs))], key=lambda s: s[1] - s[0])
    p = ((a + b) / 2, cy)
    assert min(_seg_dist(p, *d) for d in doorway_segments(truth)) > DOOR_MARGIN_M, p
    return p


G3_PLACES = ("storage", G3_PLACE, "hall")  # away from the hall first, so the hall step is a real move (B1)


def g3_script() -> list[dict[str, Any]]:
    """To storage, into the wall, back to the hall, then a report so the run ends (its score is not part of G3)."""
    return [{"name": "navigate_to", "args": {"place": pl}} for pl in G3_PLACES] + [
        {"name": "report", "args": {"answer": "storage", "photo": 1, "note": "G3 run; the answer is not scored here"}}]


def g3_numbers(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """The blocked step (2) and the hall step (3): result, nav status / time, GT distance to its place, GT move."""
    b, h = steps[1], steps[2]
    g0, g1 = b["truth"]["gt_pose_after"], h["truth"]["gt_pose_after"]
    return {"blocked_result": b["result"], "blocked_nav_status": (b.get("skill_detail") or {}).get("nav_status"),
            "blocked_nav_sim_s": (b.get("skill_detail") or {}).get("nav_sim_s"), "blocked_skill_sim_s": b["skill_sim_s"],
            "blocked_gt_err_m": b["truth"]["place_err_m"], "hall_result": h["result"],
            "hall_gt_err_m": h["truth"]["place_err_m"], "hall_gt_move_m": math.dist(g0[:2], g1[:2])}


def g3_problems(run_dir: Path) -> list[str]:
    run, steps = read_run(run_dir)
    p = ["run crashed"] if run["outcome"] == "crash" else []
    calls = [(s["call"]["name"], (s["call"]["args"] or {}).get("place")) for s in steps[:3]]
    if calls != [("navigate_to", pl) for pl in G3_PLACES]:
        return p + [f"steps 1-3 are {calls}, want navigate_to {list(G3_PLACES)}"]
    n = g3_numbers(steps)
    if n["blocked_result"] != "failed: blocked":
        p.append(f"blocked step: result {n['blocked_result']!r}, want 'failed: blocked'")
    if not n["blocked_gt_err_m"] > PLACE_BOUND_M:
        p.append(f"blocked step: GT {n['blocked_gt_err_m']:.3f} m from the wall point, want > {PLACE_BOUND_M} m")
    if n["blocked_nav_sim_s"] is None or not n["blocked_nav_sim_s"] <= NAV_TIMEOUT_S + HANG_GUARD_S:
        p.append(f"blocked step: nav_sim_s {n['blocked_nav_sim_s']}, want <= {NAV_TIMEOUT_S + HANG_GUARD_S} s")
    if n["hall_result"] != "reached":
        p.append(f"hall step: result {n['hall_result']!r}, want 'reached' (robot stuck)")
    if not n["hall_gt_move_m"] >= G3_MIN_MOVE_M:
        p.append(f"hall step: GT moved {n['hall_gt_move_m']:.3f} m, want >= {G3_MIN_MOVE_M} m (a real move)")
    if not n["hall_gt_err_m"] <= PLACE_BOUND_M:
        p.append(f"hall step: GT {n['hall_gt_err_m']:.3f} m from the hall place, want <= {PLACE_BOUND_M} m")
    return p


def g3_verdict(run_dir: Path, wall_point: tuple[float, float]) -> dict[str, Any]:
    p = g3_problems(run_dir) + git_problems([git_of(run_dir)])
    steps = steps_of(run_dir)
    return {"criterion": f"navigate_to(storage), then navigate_to(test-only place inside a wall, > {DOOR_MARGIN_M} m "
                         f"from any doorway) -> 'failed: blocked', GT > {PLACE_BOUND_M} m from it, Nav2 time "
                         f"skill_detail.nav_sim_s <= {NAV_TIMEOUT_S} s + {HANG_GUARD_S} s (one hang-guard spin; the "
                         f"photo after it not counted); then navigate_to(hall) -> 'reached', GT moved >= "
                         f"{G3_MIN_MOVE_M} m in that step and ends <= {PLACE_BOUND_M} m from the hall place; "
                         "clean tree, one commit",
            "wall_point": wall_point, "run_dir": str(run_dir), "problems": p,
            "numbers": g3_numbers(steps) if len(steps) >= 3 else None,
            "steps": [{k: s.get(k) for k in ("step", "call", "check", "result", "skill_sim_s", "skill_wall_s",
                                             "skill_detail", "truth")} for s in steps[:3]],
            "pass": not p}


# ------------------------------------------------------------------------------------------------------------ G4
def run_dirs(out: Path) -> list[Path]:
    return sorted(p.parent for p in out.glob("g[123]/*/run.json"))


def png_refs(steps: list[dict[str, Any]]) -> set[str]:
    return {ph[k] for s in steps for ph in (s["packet"] or {}).get("photos", []) + s["photos_out"]
            for k in ("rgb", "seg") if ph.get(k)}


def g4_verdict(dirs: list[Path], expected: int | None, **score_kw: Any) -> dict[str, Any]:
    """expected: G1_TOTAL + controls + 1 (G3) on a full run; None for an --only subset (never passes)."""
    per = {}
    for d in dirs:
        refs = png_refs(steps_of(d))
        missing = sorted(r for r in refs if not (d / r).is_file())
        same, diff = replay(d, **score_kw)
        per[str(d)] = {"incomplete": check_complete(d), "pngs": len(refs), "missing_pngs": missing,
                       "replay_same": same, "replay_diff": diff}
    return {"criterion": "every G1-G3 run dir: agent.runlog.check_complete empty (it also checks every referenced PNG "
                         "is on disk; missing_pngs repeats that list for reading), probes/replay.py re-scores to the "
                         "saved verdict",
            "n_runs": len(per), "expected_runs": expected, "runs": per,
            "pass": len(per) == expected and all(not r["incomplete"] and r["replay_same"] for r in per.values())}


# ------------------------------------------------------------------------------------------------------------ G5
def gate_g5(out: Path, rehearsal: bool) -> dict:
    junit = out / "g5_junit.xml"
    out.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"],
                       cwd=ROOT, capture_output=True, text=True, timeout=1800)
    cases = ET.parse(junit).getroot().iter("testcase")
    res = defaultdict(list)
    for c in cases:
        bad = [e.tag for e in c if e.tag in ("failure", "error", "skipped")]
        res[bad[0] if bad else "passed"].append(f"{c.get('classname')}::{c.get('name')}")
    no_ros = [n for n in res["passed"] if "test_agent_no_ros" in n]
    no_ros_bad = [n for k in ("failure", "error", "skipped") for n in res[k] if "test_agent_no_ros" in n]
    return write(out, "g5", {
        "criterion": "the full pytest suite passes, incl. tests/test_agent_no_ros.py (no ROS import in agent/)",
        "returncode": p.returncode, "summary": (p.stdout.strip().splitlines() or [""])[-1],
        "counts": {k: len(v) for k, v in res.items()}, "not_passed": {k: v for k, v in res.items() if k != "passed"},
        "no_ros_tests_passed": no_ros, "junit": str(junit),
        "pass": p.returncode == 0 and bool(no_ros) and not no_ros_bad}, rehearsal)


# ------------------------------------------------------------------------------------------------------- Measure
def _stats(xs: list[float]) -> dict[str, float | int]:
    return {"n": len(xs), "mean": statistics.fmean(xs), "max": max(xs)} if xs else {"n": 0}


def measure(dirs: list[Path], out: Path) -> dict[str, Any]:
    runs, skill_sim, skill_wall = [], defaultdict(list), defaultdict(list)
    for d in dirs:
        run, steps = read_run(d)
        su = run["setup"]
        runs.append({"run": str(d.relative_to(out)), "outcome": run["outcome"], "steps": run["steps"],
                     "sim_s": run["sim_s"], "wall_s": run["wall_s"], "rtf": run["sim_s"] / run["wall_s"],
                     "setup_gt_err_m": su["gt_err_m"], "setup_sim_s": su.get("sim_s"), "setup_wall_s": su.get("wall_s")})
        for s in steps:
            if s["check"] == "ok":
                skill_sim[s["call"]["name"]].append(s["skill_sim_s"])
                skill_wall[s["call"]["name"]].append(s["skill_wall_s"])
    m = {"note": "no bar (task 003 Measure): sizes Phase 3+ runs", "runs": runs,
         "per_run_sim_s": _stats([r["sim_s"] for r in runs]), "per_run_wall_s": _stats([r["wall_s"] for r in runs]),
         "rtf_overall": (sum(r["sim_s"] for r in runs) / sum(r["wall_s"] for r in runs)) if runs else None,
         "hall_setup_gt_err_m": _stats([r["setup_gt_err_m"] for r in runs]),
         "per_skill": {k: {"sim_s": _stats(skill_sim[k]), "wall_s": _stats(skill_wall[k]),
                           "rtf": sum(skill_sim[k]) / sum(skill_wall[k]) if sum(skill_wall[k]) else None}
                       for k in sorted(skill_sim)}}
    (out / "measure.json").write_text(json.dumps(m, indent=2))
    return m


# ------------------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["g1", "g2", "g3", "g4", "g5", "all"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of script ids (see module doc)")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    start = git_state(ROOT)
    if not a.rehearsal and start["git_dirty"]:
        sys.exit("refusing a counted run on a dirty tree (commit first, or pass --rehearsal)")

    def guard() -> None:  # before every sim run: the tree must still be the start tree (non-rehearsal)
        if not a.rehearsal and (why := tree_moved(start, git_state(ROOT))):
            sys.exit(f"stopping the invocation before the next run: {why}")
    want = {g: a.gate in (g, "all") for g in ("g1", "g2", "g3", "g4", "g5")}
    tasks = {t.id: t for t in load_tasks()}
    plan = g1_plan(list(tasks.values()))
    if only is not None:
        plan = list(dict.fromkeys(p for p in plan if p[1] in only))  # each listed script once
    controls = sorted(p.name.removesuffix(".yaml") for p in SCRIPTS_DIR.glob("*.neg-*.yaml"))
    expected = None if only is not None else G1_TOTAL + len(controls) + 1
    if only is not None:
        controls = [c for c in controls if c in only]
    want["g1"] &= bool(plan)
    want["g2"] &= bool(controls)
    want["g3"] &= only is None or G3_ID in only
    results = []
    if want["g1"] or want["g2"] or want["g3"]:
        from adapters.robots import make, robot_name  # ROS: only for the sim gates; env ROBOT (task 012 H-P4)
        rb = make(robot_name())
        ad, gt_pose, kw = rb.ad, rb.gt_pose, {"extra_meta": rb.scene_meta or None, **rb.run_kw}
        try:
            if want["g1"]:
                runs = []
                for i, (kind, sid) in enumerate(plan):
                    tid, steps = load_script(SCRIPTS_DIR / f"{sid}.yaml")
                    guard()
                    r = run_one(ad, gt_pose, tasks[tid], ScriptedPlanner(steps, sid), a.out / "g1" / f"{i:02d}-{sid}",
                                watch_initialpose=True, **kw)
                    runs.append({"kind": kind, "script": sid, **r})
                    print(f"G1 {i + 1:2d} {sid:24s} {r['outcome']:15s} pass {r['score']['pass']}  hall setup "
                          f"{r['setup']['gt_err_m']:.3f} m", flush=True)
                results.append(write(a.out, "g1", g1_verdict(runs, only is None), a.rehearsal))
            if want["g2"]:
                done = {}
                for sid in controls:
                    tid, steps = load_script(SCRIPTS_DIR / f"{sid}.yaml")
                    done[sid] = a.out / "g2" / sid
                    guard()
                    r = run_one(ad, gt_pose, tasks[tid], ScriptedPlanner(steps, sid), done[sid],
                                watch_initialpose=True, **kw)
                    print(f"G2 {sid:40s} {r['outcome']:15s} checks {r['score']['checks']}", flush=True)
                results.append(write(a.out, "g2", g2_verdict(done, only is None), a.rehearsal))
            if want["g3"]:
                wp = g3_wall_point(load_truth(TRUTH_JSON))
                d = a.out / "g3" / G3_ID
                guard()
                run_one(ad, gt_pose, tasks[G3_TASK], ScriptedPlanner(g3_script(), G3_ID), d,
                        extra_places={G3_PLACE: (wp[0], wp[1], 0.0)}, watch_initialpose=True, **kw)
                results.append(write(a.out, "g3", g3_verdict(d, wp), a.rehearsal))
        finally:
            rb.close()
        measure(run_dirs(a.out), a.out)
    if want["g4"]:
        results.append(write(a.out, "g4", g4_verdict(run_dirs(a.out), expected), a.rehearsal))
    if want["g5"]:
        results.append(gate_g5(a.out, a.rehearsal))
    return 0 if results and all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
