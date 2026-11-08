#!/usr/bin/env python3
"""Task 007 gate runner (Phase 4, all four task types). Reads the merged run dirs `<out>/<run_id>/` of one cross-job
counted invocation (hpc/PERUN_SIM.md section 5); needs no sim and no VLM. Each gate writes `<out>/<gate>.json`;
`measure` writes `<out>/measure.json` (no bar). Every gate first calls probes/phase3_gates.require_merged(out): an
invocation that is not merged ok is refused before any run dir is read.

    $BUILD_DIR/.venv/bin/python probes/phase4_gates.py {g1,g2,g3,measure,all} --out DIR \
        [--list tasks/phase4_counted.yaml] [--rehearsal] [--job-gpu-h H]

G1  per type: check_state, count, conditional each >= 7 of its 10 counted runs pass all four scorer checks (005 G1 per
    run: phase3_gates.g1_run, i.e. entry match, setup reached, saved score = re-score; plus the run's world = the
    entry's world). The scorer uses the truth of each run's world. All counted runs: one commit, clean tree, one
    system prompt sha256, one invocation of the committed list (invocation.json list sha256, every run's
    invocation_id). Another --list or --rehearsal never passes.
G2  Find guard: >= 7 of the 10 counted Find runs, same per-run and common rules as G1.
G3  005's G2 + G3 over all 40 counted runs (phase3_gates.run_problems, g3_run, no_ros_test): full JSONL per run, one
    failure label + one line why per failed run, no pose-like number / robot, sensor or map word in any packet.
measure  per type: steps, VLM latency per step, wall and H200-h per run, failure labels, results split by start type
    (hall / answer room / other room); per run sim_s / wall_s with its node (run.json `node`), per-node stats, runs
    below 0.6 flagged (no verdict change).
Answer room (the near-trivial start, at most 2 of 10 per type): check_state = the cabinet's room, count = the counted
room, conditional = the room the true answer names, find = the target's room.
Gate text: not included in this repo.
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
from agent.scorer import read_run, score_run  # noqa: E402
from agent.truth import DEFAULT_WORLD, WORLD_DIRS, derive, realizer, run_truth  # noqa: E402
from agent.tasks import task_by_id  # noqa: E402
from probes import phase3_gates as p3  # noqa: E402
from probes.phase2_gates import git_problems  # noqa: E402
from probes.phase3_run import load_runs, sha256_file  # noqa: E402

COUNTED = ROOT / "tasks" / "phase4_counted.yaml"
NEED, PER_TYPE = 7, 10  # task 007 G1 / G2: >= 7 of 10 counted runs per type
TYPES, GUARD = ("check_state", "count", "conditional"), "find"
SIM_WALL_FLAG = 0.6  # task 007 "Where it runs": runs below 0.6 sim_s / wall_s are flagged, no verdict change


def world_of(entry: dict[str, Any]) -> str:
    return entry.get("world", DEFAULT_WORLD)


def expected(entry: dict[str, Any]) -> dict[str, Any]:
    """kind, true answer and answer room of a list entry, from its world's truth (with `layout_seed`: that seeded
    layout's; with `layout_spec`: that Phase 8 spec's; agent/truth.run_truth), never from the list."""
    truth = run_truth(world_of(entry), entry.get("layout_seed"), entry.get("layout_spec"))
    t = task_by_id(entry["task_id"])
    answer = derive(t.kind, t.target, truth)[0]
    if t.kind == "check_state":
        room = derive("find", {"object": t.target["object"]}, truth)[0]
    else:
        room = t.target["room"] if t.kind == "count" else answer
    return {"kind": t.kind, "answer": answer, "answer_room": room}


def start_type(entry: dict[str, Any], answer_room: str) -> str:
    s = entry["start_place"]
    return "hall" if s == "hall" else "answer_room" if s == answer_room else "other_room"


def load_list(path: Path) -> list[dict[str, Any]]:
    """phase3_run.load_runs + each entry's world is a known variant and its task is valid there (expected() raises)."""
    entries = load_runs(path)
    for e in entries:
        assert world_of(e) in WORLD_DIRS, f"{e}: world not in {sorted(WORLD_DIRS)}"
        e["expected"] = expected(e)
    return entries


def is_counted(path: Path) -> bool:
    return Path(path).resolve() == COUNTED.resolve()


# ---------------------------------------------------------------------------------------------------------- G1 / G2
def g1_run(entry: dict[str, Any], d: Path | None) -> dict[str, Any]:
    """005's per-run G1 (scored against the run's world truth) + the run's world and layout seed are the entry's."""
    r = p3.g1_run(entry, d)
    if d is not None and (got := read_run(d)[0].get("world", DEFAULT_WORLD)) != world_of(entry):
        r["problems"].append(f"run world {got!r}, list says {world_of(entry)!r}")
        r["pass"] = False
    if d is not None and (seed := read_run(d)[0].get("layout_seed")) != entry.get("layout_seed"):
        r["problems"].append(f"run layout_seed {seed!r}, list says {entry.get('layout_seed')!r}")
        r["pass"] = False
    want = None if entry.get("layout_spec") is None else realizer().spec_sha256(entry["layout_spec"])
    if d is not None and (got := read_run(d)[0].get("layout_spec_sha256")) != want:  # Phase 8 (INTERFACES section 12)
        r["problems"].append(f"run layout_spec_sha256 {got!r}, list's spec has {want!r}")
        r["pass"] = False
    ex = entry["expected"]
    return {**r, "kind": ex["kind"], "world": world_of(entry), "truth_answer": ex["answer"],
            "start_type": start_type(entry, ex["answer_room"])}


def invocation_problems(out: Path, runs: dict[str, dict[str, Any]], list_path: Path = COUNTED) -> list[str]:
    """One invocation of the committed list; the ledgers and the split are require_merged's (PERUN_SIM.md s5)."""
    if not (out / "invocation.json").is_file():
        return ["invocation.json missing"]
    inv = json.loads((out / "invocation.json").read_text())
    p = []
    if inv.get("list_sha256") != sha256_file(list_path):
        p.append(f"invocation list_sha256 {inv.get('list_sha256')} != {list_path.name} {sha256_file(list_path)}")
    other = sorted(k for k, r in runs.items() if r.get("invocation_id") != inv.get("invocation_id"))
    if other:
        p.append(f"runs not of invocation {inv.get('invocation_id')}: {other}")
    return p


def common_problems(out: Path, entries: list[dict[str, Any]], counted: bool) -> list[str]:
    runs = {k: read_run(d)[0] for k, d in p3.present(out, entries).items()}
    return (git_problems([{k: r[k] for k in ("git_commit", "git_dirty")} for r in runs.values()])
            + p3.prompt_problems(list(runs.values())) + (invocation_problems(out, runs) if counted else []))


def type_verdicts(out: Path, entries: list[dict[str, Any]], counted: bool, kinds: tuple[str, ...]) -> dict[str, Any]:
    p3.require_merged(out)
    dirs, common = p3.present(out, entries), common_problems(out, entries, counted)
    per: dict[str, Any] = {}
    for kind in kinds:
        rows = [g1_run(e, dirs.get(e["run_id"])) for e in entries if e["expected"]["kind"] == kind]
        n = sum(r["pass"] for r in rows)
        per[kind] = {"passed": n, "total": len(rows), "runs": rows,
                     "pass": counted and len(rows) == PER_TYPE and n >= NEED and not common}
    return {"common_problems": common, "types": per,
            "verdict_scope": "counted list" if counted else "not the counted list (or --rehearsal): pass false",
            "pass": all(v["pass"] for v in per.values())}


def g1_verdict(out: Path, entries: list[dict[str, Any]], counted: bool) -> dict[str, Any]:
    return {"criterion": f"per type {', '.join(TYPES)}: >= {NEED} of its {PER_TYPE} counted runs (tasks/"
                         "phase4_counted.yaml) pass all four scorer checks against the run's world truth; each run "
                         "matches its entry (task, start, seed, world), reached its start, saved score = re-score; one "
                         "commit, clean tree, one system prompt, one merged invocation of the committed list",
            **type_verdicts(out, entries, counted, TYPES)}


def g2_verdict(out: Path, entries: list[dict[str, Any]], counted: bool) -> dict[str, Any]:
    return {"criterion": f"Find guard: >= {NEED} of the {PER_TYPE} counted Find runs pass, rules as G1",
            **type_verdicts(out, entries, counted, (GUARD,))}


# ------------------------------------------------------------------------------------------------------------ G3
def g3_verdict(out: Path, entries: list[dict[str, Any]], counted: bool, no_ros: dict[str, Any]) -> dict[str, Any]:
    p3.require_merged(out)
    dirs = p3.present(out, entries)
    log = {k: p3.run_problems(d) for k, d in dirs.items()}
    hits = {k: p3.g3_run(d) for k, d in dirs.items()}
    bad = {k: [h for h in v if not h["echo"]] for k, v in hits.items()}
    missing = sorted({e["run_id"] for e in entries} - set(dirs))
    return {"criterion": "005 G2 + G3 on every counted run: full JSONL, one failure label + one line why per failed "
                         "run; no decimal / coordinate pair / robot, sensor or map word in anything sent to the VLM "
                         "(echoes of its own output listed, not failed); tests/test_agent_no_ros.py passes",
            "no_ros_test": no_ros, "missing_runs": missing,
            "log_problems": {k: v for k, v in log.items() if v}, "hits": {k: v for k, v in hits.items() if v},
            "failing_hits": {k: v for k, v in bad.items() if v},
            "verdict_scope": "counted list" if counted else "not the counted list (or --rehearsal): pass false",
            "pass": (counted and len(entries) == PER_TYPE * (len(TYPES) + 1) and not missing
                     and no_ros["returncode"] == 0 and not any(log.values()) and not any(bad.values()))}


# ------------------------------------------------------------------------------------------------------- Measure
def measure(out: Path, entries: list[dict[str, Any]], job_gpu_h: float | None = None) -> dict[str, Any]:
    """No bar (task 007 Measure). H200-h per run = planner wall x 1 GPU (lower bound, as 005); with `job_gpu_h` also the
    jobs' GPU-h divided by the runs present."""
    dirs = p3.present(out, entries)
    rows = []
    for e in entries:
        if e["run_id"] not in dirs:
            continue
        d = dirs[e["run_id"]]
        run, steps = read_run(d)
        sc = score_run(d)
        pw = [s["planner_wall_s"] for s in steps if isinstance(s.get("planner_wall_s"), (int, float))]
        ratio = run["sim_s"] / run["wall_s"] if run["wall_s"] else None
        rows.append({"run_id": e["run_id"], "kind": e["expected"]["kind"], "world": world_of(e),
                     "start_type": start_type(e, e["expected"]["answer_room"]), "pass": sc["pass"],
                     "label": sc["label"], "steps": run["steps"], "planner_wall_s": pw, "wall_s": run["wall_s"],
                     "sim_s": run["sim_s"], "h200_h": sum(pw) / 3600, "node": run.get("node"),
                     "job": run.get("job"), "shard": run.get("shard"), "sim_wall": ratio,
                     "flag_below_0.6": ratio is not None and ratio < SIM_WALL_FLAG})
    per_type = {}
    for kind in (*TYPES, GUARD):
        rs = [r for r in rows if r["kind"] == kind]
        per_type[kind] = {
            "runs": len(rs), "passed": sum(r["pass"] for r in rs),
            "steps_per_run": p3._stats([r["steps"] for r in rs]),
            "vlm_latency_per_step_s": p3._stats([x for r in rs for x in r["planner_wall_s"]]),
            "wall_per_run_s": p3._stats([r["wall_s"] for r in rs]),
            "h200_h_per_run": p3._stats([r["h200_h"] for r in rs]),
            "failure_labels": dict(Counter(r["label"] for r in rs if not r["pass"])),
            "by_start": {st: {"runs": sum(r["start_type"] == st for r in rs),
                              "passed": sum(r["pass"] for r in rs if r["start_type"] == st)}
                         for st in ("hall", "answer_room", "other_room")}}
    nodes = sorted({str(r["node"]) for r in rows})
    m = {"note": "no bar (task 007 Measure)", "per_type": per_type,
         "sim_wall_per_node": {n: p3._stats([r["sim_wall"] for r in rows if str(r["node"]) == n
                                             and r["sim_wall"] is not None]) for n in nodes},
         "flagged_below_0.6": [r["run_id"] for r in rows if r["flag_below_0.6"]],
         "h200_h_per_run_note": "lower bound: planner wall time only; the job holds the GPU for its whole wall",
         "h200_h_per_run_job": None if job_gpu_h is None or not rows else job_gpu_h / len(rows),
         "runs": [{k: v for k, v in r.items() if k != "planner_wall_s"} for r in rows]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "measure.json").write_text(json.dumps(m, indent=2))
    return m


# ------------------------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["g1", "g2", "g3", "measure", "all"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--list", type=Path, default=COUNTED)
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--job-gpu-h", type=float, default=None, help="measure: the jobs' total GPU-h (hpc/JOBS.md)")
    a = ap.parse_args(argv)
    entries = load_list(a.list)
    counted = is_counted(a.list) and not a.rehearsal
    results = []
    if a.gate in ("g1", "all"):
        results.append(p3.write(a.out, "g1", g1_verdict(a.out, entries, counted), a.rehearsal))
    if a.gate in ("g2", "all"):
        results.append(p3.write(a.out, "g2", g2_verdict(a.out, entries, counted), a.rehearsal))
    if a.gate in ("g3", "all"):
        results.append(p3.write(a.out, "g3", g3_verdict(a.out, entries, counted, p3.no_ros_test()), a.rehearsal))
    if a.gate in ("measure", "all"):
        measure(a.out, entries, a.job_gpu_h)
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
