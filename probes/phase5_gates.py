#!/usr/bin/env python3
"""Task 008 gate runner (Phase 5, seeded layouts: the toolbox and the chairs move between runs). Reads the merged run
dirs `<out>/<run_id>/` of one cross-job counted invocation (hpc/PERUN_SIM.md section 5); needs no sim and no VLM.
Each gate writes `<out>/<gate>.json`; `measure` writes `<out>/measure.json` (no bar). Every gate first calls
probes/phase3_gates.require_merged(out): an invocation that is not merged ok is refused before anything is read.

    $BUILD_DIR/.venv/bin/python probes/phase5_gates.py {g1,g2,g3,measure,all} --out DIR \
        [--list tasks/phase5_counted.yaml] [--spot SPOT] [--rehearsal] [--job-gpu-h H]

G1  per type: find, check_state, count, conditional each >= 7 of its 10 counted runs pass all four scorer checks
    against truth_for(run layout_seed, world) (phase4_gates.g1_run: 005's per-run G1 + the run's world and layout seed
    are the entry's). Common: one commit, clean tree, one system prompt, one invocation of the committed list.
G2  007's G3 over all 40 counted runs (phase4_gates.g3_verdict): full JSONL per run, one failure label + one line why
    per failed run, no pose-like number / robot, sensor or map word in any packet, phase3_gates.no_ros_test
    (tests/test_agent_no_ros.py + tests/test_truth_readers.py) passes.
G3  layout fitness (design ruling, a recorded deviation): every counted layout seed is listed ok in tasks/phase5_seeds.json
    (ray-caster vetting); the live spot-check summary (--spot: a JSON file, or a dir holding spot.json; written by
    probes/spot_check.py on PERUN, {"seeds": {seed: {"agree": bool, "diffs": [...], ...}}, ...}) has >= 5 ok-listed
    seeds that agree; and no seed that disagreed is in the counted or the dev list ("a seed that fails is never
    used"). agree_all is not required; g3.json lists each disagreeing seed's record verbatim.
measure  phase4_gates.measure (per type: steps, VLM latency, wall, H200-h, failure labels, start-type split; per-run
    sim/wall per node, < 0.6 flagged) + Count split one photo / several photos (phase5_seeds.json spans_2_photos of the
    counted room) + the memory signal: of the runs whose target is the toolbox or the chairs and whose answer room is
    not where Phase 3/4's fixed layout had it (toolbox -> storage, chairs -> reception) and whose prompt does not name
    that room, the share whose first navigate_to goes there.
Answer room (at most 2 of 10 starts per type): check_state = the cabinet's room, count = the counted room, conditional
= the room the true answer names, find = the target's room (per layout seed).
Gate text: not included in this repo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.scorer import read_run  # noqa: E402
from agent.tasks import task_by_id  # noqa: E402
from probes import phase3_gates as p3  # noqa: E402
from probes import phase4_gates as p4  # noqa: E402
from probes.phase2_gates import git_problems  # noqa: E402
from probes.phase3_run import load_runs  # noqa: E402

COUNTED = ROOT / "tasks" / "phase5_counted.yaml"
DEV = ROOT / "tasks" / "phase5_dev.yaml"
SEEDS_JSON = ROOT / "tasks" / "phase5_seeds.json"
TYPES = ("find", "check_state", "count", "conditional")
NEED, PER_TYPE = p4.NEED, p4.PER_TYPE  # task 008 G1: >= 7 of 10 per type (as 007)
SPOT_MIN = 5  # task 008 Design: a live sim spot check on >= 5 seeds
MEMORY_ROOM = {"toolbox": "storage", "chair": "reception"}  # where Phase 3/4's fixed layout had the target


def ok_seeds(path: Path = SEEDS_JSON) -> dict[int, dict[str, Any]]:
    """seed -> its tasks/phase5_seeds.json "ok" record (only records with ok true)."""
    return {e["seed"]: e for e in json.loads(Path(path).read_text())["ok"] if e.get("ok") is True}


def target_class(target: dict[str, Any]) -> str:
    return target["class"] if "class" in target else target["object"].split("_")[0]


def load_list(path: Path, seeds: Path = SEEDS_JSON) -> list[dict[str, Any]]:
    """phase4_gates.load_list (load_runs + expected from truth_for(layout_seed, world)) + per entry: the Count photo
    split from phase5_seeds.json and the memory room (None if the target never sat in a fixed room)."""
    entries, ok = p4.load_list(path), ok_seeds(seeds)
    for e in entries:
        t = task_by_id(e["task_id"])
        ex = e["expected"]
        named = {t.target[k] for k in ("room", "if_not_in", "else_room") if k in t.target}
        mem = None if t.kind == "check_state" else MEMORY_ROOM.get(target_class(t.target))
        ex["memory_room"] = mem
        ex["memory_eligible"] = mem is not None and mem != ex["answer_room"] and mem not in named
        ex["photos"] = None
        if t.kind == "count" and e["layout_seed"] in ok:
            ex["photos"] = "several" if ok[e["layout_seed"]]["spans_2_photos"].get(ex["answer_room"]) else "one"
    return entries


def is_counted(path: Path) -> bool:
    return Path(path).resolve() == COUNTED.resolve()


# ------------------------------------------------------------------------------------------------------------ G1
def common_problems(out: Path, entries: list[dict[str, Any]], counted: bool, list_path: Path) -> list[str]:
    runs = {k: read_run(d)[0] for k, d in p3.present(out, entries).items()}
    return (git_problems([{k: r[k] for k in ("git_commit", "git_dirty")} for r in runs.values()])
            + p3.prompt_problems(list(runs.values()))
            + (p4.invocation_problems(out, runs, list_path) if counted else []))


def g1_verdict(out: Path, entries: list[dict[str, Any]], counted: bool,
               list_path: Path | None = None) -> dict[str, Any]:
    p3.require_merged(out)
    dirs, common = p3.present(out, entries), common_problems(out, entries, counted, list_path or COUNTED)
    per: dict[str, Any] = {}
    for kind in TYPES:
        rows = [p4.g1_run(e, dirs.get(e["run_id"])) for e in entries if e["expected"]["kind"] == kind]
        n = sum(r["pass"] for r in rows)
        per[kind] = {"passed": n, "total": len(rows), "runs": rows,
                     "pass": counted and len(rows) == PER_TYPE and n >= NEED and not common}
    return {"criterion": f"per type {', '.join(TYPES)}: >= {NEED} of its {PER_TYPE} counted runs (tasks/"
                         "phase5_counted.yaml) pass all four scorer checks against truth_for(layout_seed, world); each "
                         "run matches its entry (task, start, seed, world, layout_seed), reached its start, saved "
                         "score = re-score; one commit, clean tree, one system prompt, one merged invocation of "
                         "the list",
            "common_problems": common, "types": per,
            "verdict_scope": "counted list" if counted else "not the counted list (or --rehearsal): pass false",
            "pass": all(v["pass"] for v in per.values())}


# ------------------------------------------------------------------------------------------------------------ G2
def g2_verdict(out: Path, entries: list[dict[str, Any]], counted: bool, no_ros: dict[str, Any]) -> dict[str, Any]:
    p3.require_merged(out)
    return {**p4.g3_verdict(out, entries, counted, no_ros),
            "criterion": "007 G3 on every counted run: full JSONL, one failure label + one line why per failed run; no "
                         "decimal / coordinate pair / robot, sensor or map word in anything sent to the VLM (echoes of "
                         "its own output listed, not failed); tests/test_agent_no_ros.py and "
                         "tests/test_truth_readers.py pass"}


# ------------------------------------------------------------------------------------------------------------ G3
def read_spot(spot: Path | None) -> dict[str, Any] | None:
    if spot is None:
        return None
    f = Path(spot) / "spot.json" if Path(spot).is_dir() else Path(spot)
    return json.loads(f.read_text()) if f.is_file() else None


def g3_verdict(out: Path, entries: list[dict[str, Any]], counted: bool, spot: Path | None,
               seeds: Path = SEEDS_JSON, dev: Path | None = None) -> dict[str, Any]:
    p3.require_merged(out)
    ok = ok_seeds(seeds)
    used = sorted({e["layout_seed"] for e in entries})
    dev_used = sorted({e["layout_seed"] for e in load_runs(dev or DEV)})
    not_ok = [s for s in used if s not in ok]
    summ, problems = read_spot(spot), []
    spot_seeds: dict[int, Any] = {}
    if summ is None:
        problems.append(f"no spot-check summary at {spot}")
    else:
        spot_seeds = {int(k): v for k, v in (summ.get("seeds") or {}).items()}
    agree = sorted(s for s, v in spot_seeds.items() if isinstance(v, dict) and v.get("agree") is True and s in ok)
    disagree = {s: v for s, v in sorted(spot_seeds.items()) if not (isinstance(v, dict) and v.get("agree") is True)}
    if summ is not None and len(agree) < SPOT_MIN:
        problems.append(f"{len(agree)} spot seeds agree (ok-listed), need >= {SPOT_MIN}")
    problems += [f"spot seed {s} disagreed and is in the counted list" for s in disagree if s in used]
    problems += [f"spot seed {s} disagreed and is in the dev list" for s in disagree if s in dev_used]
    return {"criterion": f"every counted layout seed is ok in {Path(seeds).name} (ray-caster G2-G4); >= {SPOT_MIN} "
                         "ok-listed spot seeds agree live; no disagreeing spot seed in the counted or dev list",
            "layout_seeds": used, "not_ok": not_ok, "spot": str(spot), "spot_seeds": sorted(spot_seeds),
            "spot_agree": agree, "spot_disagree": disagree,
            "agree_all": None if summ is None else summ.get("agree_all"),
            "spot_seeds_in_counted": sorted(set(spot_seeds) & set(used)), "spot_problems": problems,
            "verdict_scope": "counted list" if counted else "not the counted list (or --rehearsal): pass false",
            "pass": counted and len(entries) == PER_TYPE * len(TYPES) and not not_ok and not problems}


# ------------------------------------------------------------------------------------------------------- Measure
def first_nav(d: Path) -> str | None:
    """The place of the run's first navigate_to call (the VLM's first move), None if it never called one."""
    for s in read_run(d)[1]:
        if s["call"]["name"] == "navigate_to":
            return (s["call"]["args"] or {}).get("place")
    return None


def _split(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return {v: {"runs": sum(r[key] == v for r in rows), "passed": sum(r["pass"] for r in rows if r[key] == v)}
            for v in sorted({r[key] for r in rows})}


def measure(out: Path, entries: list[dict[str, Any]], job_gpu_h: float | None = None) -> dict[str, Any]:
    """No bar (task 008 Measure)."""
    p3.require_merged(out)
    m = p4.measure(out, entries, job_gpu_h)
    m["note"] = "no bar (task 008 Measure)"
    passed = {r["run_id"]: r["pass"] for r in m["runs"]}
    dirs = p3.present(out, entries)
    rows = []
    for e in entries:
        if e["run_id"] not in dirs:
            continue
        ex = e["expected"]
        nav = first_nav(dirs[e["run_id"]])
        rows.append({"run_id": e["run_id"], "kind": ex["kind"], "pass": passed[e["run_id"]], "photos": ex["photos"],
                     "answer_room": ex["answer_room"], "memory_room": ex["memory_room"],
                     "memory_eligible": ex["memory_eligible"], "first_navigate_to": nav,
                     "went_to_memory_room": ex["memory_eligible"] and nav == ex["memory_room"]})
    m["count_by_photos"] = _split([r for r in rows if r["kind"] == "count"], "photos")
    elig = [r for r in rows if r["memory_eligible"]]
    m["memory_signal"] = {
        "rule": "runs whose target (toolbox / chairs) is not in its Phase 3/4 room (toolbox storage, chairs reception) "
                "and whose prompt does not name that room; share whose first navigate_to goes there",
        "eligible": len(elig), "went": sum(r["went_to_memory_room"] for r in elig),
        "share": sum(r["went_to_memory_room"] for r in elig) / len(elig) if elig else None,
        "per_type": {k: {"eligible": sum(r["kind"] == k for r in elig),
                         "went": sum(r["went_to_memory_room"] for r in elig if r["kind"] == k)} for k in TYPES},
        "runs": rows}
    (out / "measure.json").write_text(json.dumps(m, indent=2))
    return m


# ------------------------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["g1", "g2", "g3", "measure", "all"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--list", type=Path, default=COUNTED)
    ap.add_argument("--spot", type=Path, default=None,
                    help="g3: spot-check summary JSON, or a dir with spot.json (hpc/plans/p5_spot.sh)")
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--job-gpu-h", type=float, default=None, help="measure: the jobs' total GPU-h (hpc/JOBS.md)")
    a = ap.parse_args(argv)
    p3.require_merged(a.out)
    entries = load_list(a.list)
    counted = is_counted(a.list) and not a.rehearsal
    results = []
    if a.gate in ("g1", "all"):
        results.append(p3.write(a.out, "g1", g1_verdict(a.out, entries, counted, a.list), a.rehearsal))
    if a.gate in ("g2", "all"):
        results.append(p3.write(a.out, "g2", g2_verdict(a.out, entries, counted, p3.no_ros_test()), a.rehearsal))
    if a.gate in ("g3", "all"):
        results.append(p3.write(a.out, "g3", g3_verdict(a.out, entries, counted, a.spot), a.rehearsal))
    if a.gate in ("measure", "all"):
        measure(a.out, entries, a.job_gpu_h)
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
