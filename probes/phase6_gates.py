#!/usr/bin/env python3
"""Task 009 gate runner (Phase 6, blocked doorways: a block makes one room unreachable). Reads the merged run dirs
`<out>/<run_id>/` of one cross-job counted invocation of tasks/phase6_counted.yaml (hpc/PERUN_SIM.md section 5); needs
no sim and no VLM. The list mixes a blocked set (entries whose `world` starts with `blocked_`) and a guard set
(`closed` / `cabinet_open`), run as one invocation. Each gate writes `<out>/<gate>.json`; `measure` writes
`<out>/measure.json` (no bar). Every gate first calls probes/phase3_gates.require_merged(out).

    source sim/env.sh   # G3's no-ROS test needs ROS on the path, as phase5's G2
    $BUILD_DIR/.venv/bin/python probes/phase6_gates.py {g1,g2,g3,measure,all} --out DIR \
        [--list tasks/phase6_counted.yaml] [--seeds tasks/phase5_seeds.json] [--rehearsal] [--job-gpu-h H]

G1  blocked: >= 7 of the 10 blocked runs pass all four scorer checks against truth_for(layout_seed, world) (per run
    phase4_gates.g1_run: entry match incl. world + layout seed, setup reached, saved score = re-score) AND 0 of them
    crashed (run.json outcome "crash"). Common (whole list, phase5_gates.common_problems): one commit, clean tree, one
    system prompt, one merged invocation of the committed list. Also recorded, not part of the pass (task 009 Design):
    block_hits = blocked runs with >= 1 step navigate_to(<its blocked room>) whose result is "failed: blocked",
    block_mattered = block_hits >= 6. Split by type and by start type (hall / answer room / other room).
G2  guard: >= 7 of the 10 guard runs pass all four checks, same per-run and common rules; split by type.
G3  phase5's G2 (= 007's G3, phase4_gates.g3_verdict) over all 20 runs: full JSONL, one failure label + one line why per
    failed run, no decimal / coordinate / robot, sensor or map word in any packet, tests/test_agent_no_ros.py +
    tests/test_truth_readers.py pass (run with sim/env.sh sourced).
kill_crash  every gate JSON carries the list of crashed runs anywhere in the list (task 009 Kill: any crash -> a
    diagnosed fix).
measure  phase5_gates.measure per set (blocked / guard); steps per run per set; every "failed: blocked" step with the
    run's blocked room, its skill_detail.nav_sim_s and skill_sim_s (time-to-fail) and the next step's call (how the VLM
    reacted); per blocked run the number of navigate_to(<blocked room>) calls; sim/wall per run and per node, < 0.6
    flagged.
The blocked room of a run is its truth's: agent.truth.run_truth(world, layout_seed)["blocker"]["room"] (INTERFACES
section 11). Gate text: not included in this repo.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.scorer import read_run  # noqa: E402
from agent.truth import run_truth  # noqa: E402
from probes import phase3_gates as p3  # noqa: E402
from probes import phase4_gates as p4  # noqa: E402
from probes import phase5_gates as p5  # noqa: E402

COUNTED = ROOT / "tasks" / "phase6_counted.yaml"
SEEDS_JSON = p5.SEEDS_JSON  # the ray-caster vetting record (Count photo split)
NEED, PER_SET = 7, 10  # task 009 G1 / G2: >= 7 of 10 per set
BLOCK_MATTERED = 6  # task 009 Design: >= 6 of 10 blocked runs hit "failed: blocked" (recorded, no gate)
BLOCKED = "failed: blocked"
SETS = ("blocked", "guard")
START_TYPES = ("hall", "answer_room", "other_room")


def set_of(entry: dict[str, Any]) -> str:
    return "blocked" if p4.world_of(entry).startswith("blocked_") else "guard"


def load_list(path: Path, seeds: Path = SEEDS_JSON) -> list[dict[str, Any]]:
    """phase5_gates.load_list + per entry its set and blocked room (from the truth). A blocked entry must have a
    blocker, a guard entry none; the answer room and the start are never the blocked room (INTERFACES section 11)."""
    entries = p5.load_list(path, seeds)
    for e in entries:
        ex = e["expected"]
        ex["set"] = set_of(e)
        ex["blocked_room"] = (run_truth(p4.world_of(e), e.get("layout_seed")).get("blocker") or {}).get("room")
        assert (ex["blocked_room"] is not None) == (ex["set"] == "blocked"), f"{e['run_id']}: blocker {ex}"
        assert ex["blocked_room"] not in (ex["answer_room"], e["start_place"]), f"{e['run_id']}: blocked room {ex}"
    return entries


def is_counted(path: Path) -> bool:
    return Path(path).resolve() == COUNTED.resolve()


def nav_place(step: dict[str, Any]) -> str | None:
    return (step["call"]["args"] or {}).get("place") if step["call"]["name"] == "navigate_to" else None


def kill_crash(out: Path, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every run of the list whose run.json outcome is "crash" (task 009 Kill)."""
    dirs = p3.present(out, entries)
    runs = {e["run_id"]: (e, read_run(dirs[e["run_id"]])[0]) for e in entries if e["run_id"] in dirs}
    return [{"run_id": k, "set": e["expected"]["set"], "crash_origin": r.get("crash_origin")}
            for k, (e, r) in runs.items() if r["outcome"] == "crash"]


# ---------------------------------------------------------------------------------------------------------- G1 / G2
def set_rows(out: Path, entries: list[dict[str, Any]], which: str) -> list[dict[str, Any]]:
    dirs, rows = p3.present(out, entries), []
    for e in entries:
        if e["expected"]["set"] != which:
            continue
        d, room = dirs.get(e["run_id"]), e["expected"]["blocked_room"]
        r = p4.g1_run(e, d)
        run, steps = read_run(d) if d is not None else ({}, [])
        r |= {"kind": e["expected"]["kind"], "start_type": p4.start_type(e, e["expected"]["answer_room"]),
              "world": p4.world_of(e), "blocked_room": room, "crash": run.get("outcome") == "crash",
              "block_hit": room is not None and any(nav_place(s) == room and s["result"] == BLOCKED for s in steps)}
        rows.append(r)
    return rows


def _split(rows: list[dict[str, Any]], key: str, values: tuple[str, ...] | None = None) -> dict[str, Any]:
    return {v: {"runs": sum(r[key] == v for r in rows), "passed": sum(r["pass"] for r in rows if r[key] == v)}
            for v in (values or sorted({r[key] for r in rows}))}


def set_verdict(out: Path, entries: list[dict[str, Any]], counted: bool, list_path: Path,
                which: str) -> dict[str, Any]:
    p3.require_merged(out)
    common = p5.common_problems(out, entries, counted, list_path)
    rows = set_rows(out, entries, which)
    n, crashes = sum(r["pass"] for r in rows), [r["run_id"] for r in rows if r["crash"]]
    ok = counted and len(rows) == PER_SET and n >= NEED and not common
    return {"common_problems": common, "passed": n, "total": len(rows), "crashes": crashes,
            "by_type": _split(rows, "kind"), "runs": rows, "kill_crash": kill_crash(out, entries),
            "verdict_scope": "counted list" if counted else "not the counted list (or --rehearsal): pass false",
            "pass": ok and not (which == "blocked" and crashes)}


def g1_verdict(out: Path, entries: list[dict[str, Any]], counted: bool,
               list_path: Path | None = None) -> dict[str, Any]:
    v = set_verdict(out, entries, counted, list_path or COUNTED, "blocked")
    hits = sum(r["block_hit"] for r in v["runs"])
    return {"criterion": f"blocked set: >= {NEED} of its {PER_SET} counted runs (tasks/phase6_counted.yaml) pass all "
                         "four scorer checks against truth_for(layout_seed, world) and 0 of them crashed; each run "
                         "matches its entry (task, start, seed, world, layout_seed), reached its start, saved score = "
                         "re-score; one commit, clean tree, one system prompt, one merged invocation of the list",
            **v, "by_start": _split(v["runs"], "start_type", START_TYPES),
            "block_hits": hits, "block_mattered": hits >= BLOCK_MATTERED,
            "block_rule": f"recorded, not in pass: >= {BLOCK_MATTERED} blocked runs with a navigate_to(<blocked room>) "
                          f"step whose result is {BLOCKED!r}"}


def g2_verdict(out: Path, entries: list[dict[str, Any]], counted: bool,
               list_path: Path | None = None) -> dict[str, Any]:
    return {"criterion": f"guard set (closed / cabinet_open): >= {NEED} of its {PER_SET} counted runs pass all four "
                         "scorer checks, per-run and common rules as G1",
            **set_verdict(out, entries, counted, list_path or COUNTED, "guard")}


# ------------------------------------------------------------------------------------------------------------ G3
def g3_verdict(out: Path, entries: list[dict[str, Any]], counted: bool, no_ros: dict[str, Any]) -> dict[str, Any]:
    p3.require_merged(out)
    v = p4.g3_verdict(out, entries, counted, no_ros)
    return {**v, "criterion": "phase5 G2 on all 20 counted runs: full JSONL, one failure label + one line why per "
                              "failed run; no decimal / coordinate pair / robot, sensor or map word in anything sent "
                              "to the VLM (echoes of its own output listed, not failed); tests/test_agent_no_ros.py "
                              "and tests/test_truth_readers.py pass",
            "kill_crash": kill_crash(out, entries),
            "pass": (counted and len(entries) == PER_SET * len(SETS) and not v["missing_runs"]
                     and no_ros["returncode"] == 0 and not v["log_problems"] and not v["failing_hits"])}


# ------------------------------------------------------------------------------------------------------- Measure
def blocked_steps(run_id: str, room: str | None, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every "failed: blocked" step: where it tried, the run's blocked room, time-to-fail, the next call (reaction)."""
    out = []
    for i, s in enumerate(steps):
        if nav_place(s) is None or s["result"] != BLOCKED:
            continue
        nxt = steps[i + 1]["call"] if i + 1 < len(steps) else None
        out.append({"run_id": run_id, "step": s["step"], "place": nav_place(s), "blocked_room": room,
                    "at_blocked_room": nav_place(s) == room,
                    "nav_sim_s": (s.get("skill_detail") or {}).get("nav_sim_s"), "skill_sim_s": s.get("skill_sim_s"),
                    "next_call": None if nxt is None else {"name": nxt["name"], "args": nxt["args"]}})
    return out


def measure(out: Path, entries: list[dict[str, Any]], job_gpu_h: float | None = None) -> dict[str, Any]:
    """No bar (task 009 Measure)."""
    p3.require_merged(out)
    dirs = p3.present(out, entries)
    sets = {w: p5.measure(out, [e for e in entries if e["expected"]["set"] == w]) for w in SETS}
    for m in sets.values():
        m["note"] = "no bar (task 009 Measure); phase5_gates.measure of this set"
    fails, repeats, rows = [], {}, []
    for e in entries:
        if e["run_id"] not in dirs:
            continue
        run, steps = read_run(dirs[e["run_id"]])
        room = e["expected"]["blocked_room"]
        fails += blocked_steps(e["run_id"], room, steps)
        if room is not None:
            repeats[e["run_id"]] = sum(nav_place(s) == room for s in steps)
        ratio = run["sim_s"] / run["wall_s"] if run["wall_s"] else None
        rows.append({"run_id": e["run_id"], "set": e["expected"]["set"], "steps": run["steps"], "node": run.get("node"),
                     "sim_s": run["sim_s"], "wall_s": run["wall_s"], "sim_wall": ratio,
                     "flag_below_0.6": ratio is not None and ratio < p4.SIM_WALL_FLAG})
    nodes = sorted({str(r["node"]) for r in rows})
    m = {"note": "no bar (task 009 Measure)", "kill_crash": kill_crash(out, entries), "sets": sets,
         "steps_per_run": {w: p3._stats([r["steps"] for r in rows if r["set"] == w]) for w in SETS},
         "blocked_steps": fails,
         "time_to_fail_nav_sim_s": p3._stats([f["nav_sim_s"] for f in fails if f["at_blocked_room"]
                                              and isinstance(f["nav_sim_s"], (int, float))]),
         "reaction_next_call": dict(Counter(f["next_call"]["name"] if f["next_call"] else None for f in fails
                                            if f["at_blocked_room"])),
         "navigate_to_blocked_room_per_run": repeats,
         "sim_wall_per_node": {n: p3._stats([r["sim_wall"] for r in rows if str(r["node"]) == n
                                             and r["sim_wall"] is not None]) for n in nodes},
         "flagged_below_0.6": [r["run_id"] for r in rows if r["flag_below_0.6"]],
         "h200_h_per_run_job": None if job_gpu_h is None or not rows else job_gpu_h / len(rows),
         "runs": rows}
    (out / "measure.json").write_text(json.dumps(m, indent=2))
    return m


# ------------------------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["g1", "g2", "g3", "measure", "all"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--list", type=Path, default=COUNTED)
    ap.add_argument("--seeds", type=Path, default=SEEDS_JSON, help="vetting record for the Count photo split")
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--job-gpu-h", type=float, default=None, help="measure: the jobs' total GPU-h (hpc/JOBS.md)")
    a = ap.parse_args(argv)
    p3.require_merged(a.out)
    entries = load_list(a.list, a.seeds)
    counted = is_counted(a.list) and not a.rehearsal
    results = []
    if a.gate in ("g1", "all"):
        results.append(p3.write(a.out, "g1", g1_verdict(a.out, entries, counted, a.list), a.rehearsal))
    if a.gate in ("g2", "all"):
        results.append(p3.write(a.out, "g2", g2_verdict(a.out, entries, counted, a.list), a.rehearsal))
    if a.gate in ("g3", "all"):
        results.append(p3.write(a.out, "g3", g3_verdict(a.out, entries, counted, p3.no_ros_test()), a.rehearsal))
    if a.gate in ("measure", "all"):
        measure(a.out, entries, a.job_gpu_h)
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
