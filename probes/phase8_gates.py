#!/usr/bin/env python3
"""Task 011 gate runner (Phase 8, Stage 1 eval on the sealed test entries, INTERFACES section 12). Reads the merged run
dirs `<out>/<run_id>/` of one cross-job counted invocation of tasks/phase8_pass<N>.yaml; needs no sim and no VLM. Each
gate writes `<out>/<gate>.json`; `measure` writes measure.json, `gallery` gallery.json + gallery.md (no bars). Every
gate first calls probes/phase3_gates.require_merged(out).

    source sim/env.sh   # G2's no-ROS test needs ROS on the path, as phase5 / phase6
    $BUILD_DIR/.venv/bin/python probes/phase8_gates.py {g1,g2,measure,gallery,all} --out DIR \
        --list tasks/phase8_pass1.yaml [--freeze-tag skill-interface-frozen-v1] [--rehearsal] [--job-gpu-h H]

G1  per type find, check_state, count, conditional: >= 18 of its 25 counted runs pass all four scorer checks against
    truth_for_spec(layout_spec) (per run phase4_gates.g1_run: entry match, setup reached, saved score = re-score, run
    world = entry world, run.json layout_spec_sha256 = the entry spec's). Common (phase5_gates.common_problems): one
    commit, clean tree, one system prompt, one merged invocation of the list. Per type the rate's 95 % Wilson interval.
G2  phase5's G2 (phase4_gates.g3_verdict: full JSONL, one label + one line why per failed run, packet scan,
    tests/test_agent_no_ros.py + tests/test_truth_readers.py) over all 100 runs, plus tests/test_interface_freeze.py
    passes, the runs' one commit is recorded, and tests/interface_freeze.json at that commit equals the one at the
    freeze tag (the interface the runs used = the frozen one). Deviation 20.3: a teleport_failed record (outcome crash,
    setup.skipped, setup.reached / gt_err_m null by design) is complete, as phase3_gates has had since 008; its crash
    still fails G1. g2.json keeps the as-written verdict too (`pass_as_written`, `log_problems_as_written`).
measure  phase4_gates.measure (per type: steps, VLM latency, wall, H200-h, failure labels, start split; sim/wall per
    node, < 0.6 flagged) + per type sim/wall, per node steps and VLM latency + per sim (job / shard, in run order) the
    set_pose sends, retries and teleport crashes (a recorded deviation.2).
gallery  per failed G1 run: task id, prompt, kind, truth answer vs answer, label + why, the photos the VLM saw (run-dir
    paths, per step), each step's call + result + the last ~400 chars of its reasoning.
Gate text: not included in this repo.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import NormalDist
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.scorer import read_run  # noqa: E402
from agent.tasks import task_by_id  # noqa: E402
from probes import phase3_gates as p3  # noqa: E402
from probes import phase4_gates as p4  # noqa: E402
from probes import phase5_gates as p5  # noqa: E402
from probes.phase3_run import is_counted  # noqa: E402

LIST = ROOT / "tasks" / "phase8_pass1.yaml"
TYPES = ("find", "check_state", "count", "conditional")
NEED, PER_TYPE = 18, 25  # task 011 G1: >= 70 % of 25 (>= 18/25) per type
Z95 = NormalDist().inv_cdf(0.975)  # task 011: 95 % Wilson interval
FREEZE_TAG = "skill-interface-frozen-v1"
FREEZE_TEST, FREEZE_JSON = "tests/test_interface_freeze.py", "tests/interface_freeze.json"
EXCERPT = 400  # task: reasoning excerpt = the last ~400 chars
SKIPPED_SETUP = "layout teleport failed"  # adapters/run_task.teleport_failed: setup.skipped
SKIPPED_NULLS = ("check_complete: run.setup.reached: null", "check_complete: run.setup.gt_err_m: null")


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    if n == 0:
        return None
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load_list(path: Path) -> list[dict[str, Any]]:
    """phase4_gates.load_list (phase3_run.load_runs + expected from truth_for_spec via agent.truth.run_truth); every
    entry carries a layout_spec."""
    entries = p4.load_list(path)
    assert all("layout_spec" in e for e in entries), f"{path}: an entry without layout_spec"
    return entries


# ------------------------------------------------------------------------------------------------------------ G1
def g1_verdict(out: Path, entries: list[dict[str, Any]], counted: bool, list_path: Path) -> dict[str, Any]:
    p3.require_merged(out)
    dirs, common = p3.present(out, entries), p5.common_problems(out, entries, counted, list_path)
    per: dict[str, Any] = {}
    for kind in TYPES:
        rows = [p4.g1_run(e, dirs.get(e["run_id"])) for e in entries if e["expected"]["kind"] == kind]
        n = sum(r["pass"] for r in rows)
        per[kind] = {"passed": n, "total": len(rows), "rate": n / len(rows) if rows else None,
                     "wilson95": wilson(n, len(rows)), "runs": rows,
                     "pass": counted and len(rows) == PER_TYPE and n >= NEED and not common}
    return {"criterion": f"per type {', '.join(TYPES)}: >= {NEED} of its {PER_TYPE} counted runs "
                         f"({Path(list_path).name}) pass all four scorer checks against truth_for_spec(layout_spec); "
                         "each run matches its entry "
                         "(task, start, seed, world, layout spec sha256), reached its start, saved score = "
                         "re-score; one commit, clean tree, one system prompt, one merged invocation of the list",
            "common_problems": common, "types": per,
            "verdict_scope": "counted list" if counted else "not the counted list (or --rehearsal): pass false",
            "pass": all(v["pass"] for v in per.values())}


# ------------------------------------------------------------------------------------------------------------ G2
def git_show(rev: str, path: str) -> str | None:
    p = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


def freeze_test() -> dict[str, Any]:
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", FREEZE_TEST],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    return {"returncode": p.returncode, "file": FREEZE_TEST, "summary": (p.stdout.strip().splitlines() or [""])[-1]}


def freeze_check(out: Path, entries: list[dict[str, Any]], tag: str, test: dict[str, Any]) -> dict[str, Any]:
    commits = sorted({str(read_run(d)[0].get("git_commit")) for d in p3.present(out, entries).values()})
    run_json = git_show(commits[0], FREEZE_JSON) if len(commits) == 1 else None
    tag_json = git_show(tag, FREEZE_JSON)
    return {"freeze_test": test, "run_commits": commits, "run_commit": commits[0] if len(commits) == 1 else None,
            "freeze_tag": tag, "freeze_json_equal_tag": run_json is not None and run_json == tag_json,
            "pass": test["returncode"] == 0 and len(commits) == 1 and run_json is not None and run_json == tag_json}


def skipped_setup(out: Path, entries: list[dict[str, Any]],
                  log_problems: dict[str, list[str]]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Deviation 20.3: (kept, accepted). The setup nulls of a teleport_failed record are accepted (complete by design,
    phase3_gates' 008 exception); every other problem is kept."""
    dirs, kept, accepted = p3.present(out, entries), {}, {}
    for rid, probs in log_problems.items():
        run = read_run(dirs[rid])[0] if rid in dirs else {}
        ok = run.get("outcome") == "crash" and (run.get("setup") or {}).get("skipped") == SKIPPED_SETUP
        rest = [x for x in probs if not (ok and x in SKIPPED_NULLS)]
        if len(rest) < len(probs):
            accepted[rid] = [x for x in probs if x not in rest]
        if rest:
            kept[rid] = rest
    return kept, accepted


def g2_verdict(out: Path, entries: list[dict[str, Any]], counted: bool, no_ros: dict[str, Any],
               freeze: dict[str, Any]) -> dict[str, Any]:
    p3.require_merged(out)
    v = p4.g3_verdict(out, entries, counted, no_ros)
    kept, accepted = skipped_setup(out, entries, v["log_problems"])
    rest = (counted and len(entries) == PER_TYPE * len(TYPES) and not v["missing_runs"]
            and no_ros["returncode"] == 0 and not v["failing_hits"] and freeze["pass"])
    return {**v, "criterion": f"phase5 G2 on all {PER_TYPE * len(TYPES)} counted runs: full JSONL, one failure label + "
                              "one line why per failed run; no decimal / coordinate pair / robot, sensor or map word "
                              "in anything sent to the VLM (echoes listed, not failed); tests/test_agent_no_ros.py and "
                              f"tests/test_truth_readers.py pass; {FREEZE_TEST} passes; the runs' one commit has "
                              f"{FREEZE_JSON} equal to the freeze tag's; a teleport_failed record is complete "
                              "(a recorded deviation.3)",
            "freeze": freeze, "log_problems": kept, "log_problems_as_written": v["log_problems"],
            "log_problems_accepted_skipped_setup": accepted,
            "pass_as_written": rest and not v["log_problems"], "pass": rest and not kept}


# ------------------------------------------------------------------------------------------------------- Measure
def measure(out: Path, entries: list[dict[str, Any]], job_gpu_h: float | None = None) -> dict[str, Any]:
    """No bar (task 011 Report)."""
    p3.require_merged(out)
    m = p4.measure(out, entries, job_gpu_h)
    m["note"] = "no bar (task 011 Report: failure labels, steps / latency / sim-wall per type)"
    m["sim_wall_per_type"] = {k: p3._stats([r["sim_wall"] for r in m["runs"] if r["kind"] == k
                                            and r["sim_wall"] is not None]) for k in TYPES}
    lat: dict[str, list[float]] = {}
    for d in p3.present(out, entries).values():
        run, steps = read_run(d)
        lat.setdefault(str(run.get("node")), []).extend(
            s["planner_wall_s"] for s in steps if isinstance(s.get("planner_wall_s"), (int, float)))
    m["per_node"] = {n: {"steps_per_run": p3._stats([r["steps"] for r in m["runs"] if str(r["node"]) == n]),
                         "vlm_latency_per_step_s": p3._stats(lat[n])} for n in sorted(lat)}
    m["teleports_per_sim"] = teleports_per_sim(out, entries)
    (out / "measure.json").write_text(json.dumps(m, indent=2))
    return m


def teleports_per_sim(out: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Deviation 20.2: per sim (slurm job / shard), runs in start order with set_pose sends, retries, teleport crash
    (runs from before 20.2 have no run.json `teleport`: sends / retries null)."""
    per: dict[str, list[tuple[str, str, Any, Any, bool]]] = {}
    for rid, d in p3.present(out, entries).items():
        run = read_run(d)[0]
        tp = run.get("teleport")
        per.setdefault(f"{run.get('slurm_job_id')}/{run.get('shard')}", []).append(
            (str(run.get("start_utc")), rid, None if tp is None else tp.get("sends"),
             None if tp is None else len(tp.get("retries") or []), run.get("crash_origin") == "layout_teleport"))
    return {k: {"runs": [{"order": i + 1, "run_id": r, "sends": s, "retries": n, "crash": c}
                         for i, (_, r, s, n, c) in enumerate(sorted(v))],
                "retries": sum(x[3] or 0 for x in v), "crashes": sum(x[4] for x in v),
                "crash_orders": [i + 1 for i, x in enumerate(sorted(v)) if x[4]]} for k, v in sorted(per.items())}


# ------------------------------------------------------------------------------------------------------- Gallery
def excerpt(s: dict[str, Any]) -> str:
    po = s.get("planner_out") or {}
    return str(po.get("reasoning") or po.get("content") or "")[-EXCERPT:]


def gallery(out: Path, entries: list[dict[str, Any]], g1: dict[str, Any]) -> list[dict[str, Any]]:
    """Every failed G1 run (from the g1 verdict), written as gallery.json + gallery.md."""
    by_id = {e["run_id"]: e for e in entries}
    rows = []
    for kind in TYPES:
        for r in g1["types"][kind]["runs"]:
            if r["pass"]:
                continue
            e, d = by_id[r["run_id"]], out / r["run_id"]
            t = task_by_id(e["task_id"])
            row = {"run_id": r["run_id"], "task_id": e["task_id"], "prompt": t.prompt, "kind": kind,
                   "truth_answer": e["expected"]["answer"], "answer": None, "label": r.get("label"),
                   "why": r.get("label_why") or "; ".join(r["problems"]), "photos": [], "steps": []}
            if (d / "run.json").is_file():
                run, steps = read_run(d)
                row["answer"] = (run.get("report") or {}).get("answer")
                row["photos"] = [{"step": s["step"], "n": p["n"], "rgb": str(d / p["rgb"])}
                                 for s in steps for p in ((s.get("packet") or {}).get("photos") or [])]
                row["steps"] = [{"step": s["step"], "call": {"name": s["call"]["name"], "args": s["call"]["args"]},
                                 "result": s.get("result"), "reasoning_excerpt": excerpt(s)} for s in steps]
            rows.append(row)
    (out / "gallery.json").write_text(json.dumps(rows, indent=2))
    md = [f"# Phase 8 failure gallery ({len(rows)} failed runs)", ""]
    for g in rows:
        md += [f"## {g['run_id']} — {g['kind']} — {g['task_id']}", "", f"- prompt: {g['prompt']}",
               f"- truth {g['truth_answer']!r}, answer {g['answer']!r}", f"- label {g['label']}: {g['why']}",
               "- photos the VLM saw: " + (", ".join(f"step {p['step']} #{p['n']} `{p['rgb']}`" for p in g["photos"])
                                           or "none"), ""]
        for s in g["steps"]:
            md += [f"{s['step']}. `{s['call']['name']}({json.dumps(s['call']['args'])})` -> {s['result']}",
                   f"   > {' '.join(s['reasoning_excerpt'].split())}"]
        md.append("")
    (out / "gallery.md").write_text("\n".join(md) + "\n")
    return rows


# ------------------------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["g1", "g2", "measure", "gallery", "all"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--list", type=Path, default=LIST)
    ap.add_argument("--freeze-tag", default=FREEZE_TAG, help="g2: the pass's freeze tag (pass 2 / 3: the fix tag)")
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--job-gpu-h", type=float, default=None, help="measure: the jobs' total GPU-h (hpc/JOBS.md)")
    a = ap.parse_args(argv)
    p3.require_merged(a.out)
    entries = load_list(a.list)
    counted = is_counted(a.list) and not a.rehearsal
    results = []
    if a.gate in ("g1", "gallery", "all"):
        g1 = g1_verdict(a.out, entries, counted, a.list)
        if a.gate != "gallery":
            results.append(p3.write(a.out, "g1", g1, a.rehearsal))
    if a.gate in ("g2", "all"):
        fz = freeze_check(a.out, entries, a.freeze_tag, freeze_test())
        results.append(p3.write(a.out, "g2", g2_verdict(a.out, entries, counted, p3.no_ros_test(), fz), a.rehearsal))
    if a.gate in ("measure", "all"):
        measure(a.out, entries, a.job_gpu_h)
    if a.gate in ("gallery", "all"):
        print(f"gallery: {len(gallery(a.out, entries, g1))} failed runs", flush=True)
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
