#!/usr/bin/env python3
"""Task 005 gate runner (Phase 3, VLM Find). Reads run dirs `<out>/<run_id>/` written by probes/phase3_run.py; needs
no sim and no VLM. Each gate writes `<out>/<gate>.json`; `measure` writes `<out>/measure.json` (no bar).

    $BUILD_DIR/.venv/bin/python probes/phase3_gates.py {g1,g2,g3,measure,all} --out DIR \
        [--list tasks/phase3_counted.yaml] [--rehearsal] [--ledger PATH] [--job-gpu-h H]

G1  >= 7 of the 10 counted runs pass all four scorer checks. Exactly the counted list: a missing run fails; a run counts
    only if its run.json matches its list entry (task, start place, seed), its setup reached the start place (a miss:
    row label nav, N6), the saved score.json says pass and a fresh re-score agrees. Runs must be on one commit, clean
    tree (design review B4), one system prompt sha256. One invocation (B1): OUT/invocation.json with the committed list
    sha256, every run.json carrying its invocation_id, and the ledger holding exactly this one counted invocation
    (start, then only resume / abort, then one end). Another --list or --rehearsal never passes G1.
G2  every counted run's steps.jsonl carries the packet actually sent (user text parts, system prompt sha256, photos by
    path + file sha256 + sent-JPEG sha256), the raw VLM output (content, reasoning, finish_reason, usage, error; the
    reasoning key must exist, empty is listed, N4), the tool call, check result, skill result and timings; run.json
    carries the prompt, tools, model, sampling, seed; agent.runlog.check_complete is empty; each failed run has exactly
    one label + a one-line why.
G3  no pose-like number or robot / sensor / map word in anything the VLM was sent (system prompt, tool schemas, every
    step's text parts and history); a hit that is an echo of the VLM's own earlier output is listed, not failed.
    Plus tests/test_agent_no_ros.py passes.
Cross-job (hpc/PERUN_SIM.md section 5): an OUT whose invocation.json has a "split" passes no gate (SystemExit) until
`phase3_run.py --merge OUT` wrote merged.json with ok true (require_merged); its G1 ledger check reads the invocation's
own lines (split start, merge end / abort), the job lines (with "job") were checked by the merge. Measure adds sim_s /
wall_s per run and per node and flags runs below RTF_FLAG (no verdict).
Gate text: not included in this repo.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.runlog import check_complete  # noqa: E402
from agent.scorer import no_call_steps, read_run, score_run, toolcall_in_reasoning  # noqa: E402
from probes.phase2_gates import git_problems  # noqa: E402
from probes.phase3_run import COUNTED, COUNTED_SHA256, default_ledger, is_counted, load_runs, sha256_file  # noqa: E402

G1_NEED, G1_RUNS = 7, 10  # task 005 G1: >= 7 of 10 counted runs
RTF_FLAG = 0.6  # hpc/PERUN_SIM.md section 5: counted runs with sim_s / wall_s below this are flagged (no verdict)
LABELS = {"vlm", "skill", "nav", "sim", "scorer"}
TIMINGS = ("sim_t0", "sim_t1", "wall_t0", "wall_t1", "planner_wall_s", "skill_sim_s", "skill_wall_s")
PLANNER_INFO = ("model", "thinking", "sampling", "max_tokens", "seed")
# G3: decimals, coordinate pairs / triples, and words that name the body, its sensors or the map.
POSE_RES = [re.compile(r"-?\d+\.\d+"), re.compile(r"[(\[]\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?")]
WORD_RE = re.compile(r"turtlebot|create3|oakd|\bg1\b|unitree|lidar|\bmap\b|coordinat|odom|\byaw\b|humanoid", re.I)


def write(out: Path, name: str, verdict: dict, rehearsal: bool) -> dict:
    verdict = {"gate": name, "rehearsal": rehearsal,
               "written_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), **verdict}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(verdict, indent=2, default=float))
    print(f"{name}: {'PASS' if verdict.get('pass') else 'FAIL'}", flush=True)
    return verdict


def require_merged(out: Path) -> None:
    """SystemExit with the reason if OUT is a cross-job invocation (invocation.json has a "split") without a merged.json
    of this invocation saying ok true. An OUT without a split (single-job invocation) passes unchanged."""
    inv_p, m_p = out / "invocation.json", out / "merged.json"
    inv = json.loads(inv_p.read_text()) if inv_p.is_file() else {}
    if "split" not in inv:
        return
    if not m_p.is_file():
        sys.exit(f"refusing {out}: split over jobs but not merged (run probes/phase3_run.py --merge {out})")
    m = json.loads(m_p.read_text())
    if m.get("invocation_id") != inv["invocation_id"] or m.get("ok") is not True:
        sys.exit(f"refusing {out}: merged.json of {m.get('invocation_id')} ok {m.get('ok')}: {m.get('problems')}")


def present(out: Path, runs: list[dict[str, Any]]) -> dict[str, Path]:
    """run_id -> run dir, for the listed runs whose run dir has a run.json."""
    return {r["run_id"]: out / r["run_id"] for r in runs if (out / r["run_id"] / "run.json").is_file()}


# ------------------------------------------------------------------------------------------------------------ G1
def entry_problems(entry: dict[str, Any], run: dict[str, Any]) -> list[str]:
    """The run is the listed one: same task, start place and sampling seed."""
    got = (run["task"]["id"], run["start_place"], (run.get("planner_info") or {}).get("seed"))
    want = (entry["task_id"], entry["start_place"], entry["seed"])
    return [] if got == want else [f"run is (task, start, seed) {got}, list says {want}"]


def g1_run(entry: dict[str, Any], d: Path | None, **score_kw: Any) -> dict[str, Any]:
    if d is None:
        return {"run_id": entry["run_id"], "pass": False, "problems": ["run dir missing"]}
    run = read_run(d)[0]
    saved = json.loads((d / "score.json").read_text()) if (d / "score.json").is_file() else None
    fresh = score_run(d, **score_kw)
    p, label, why = entry_problems(entry, run), fresh["label"], fresh["label_why"]
    if run["setup"].get("skipped"):  # 008: no setup move (layout teleport failed); the scorer's crash label stands
        p.append(f"setup skipped: {run['setup']['skipped']}")
    elif not run["setup"]["reached"]:  # N6: a G1 fail with exactly one label, also when the scorer passed it
        label, why = "nav", f"setup did not reach {run['start_place']} (GT {run['setup']['gt_err_m']:.2f} m)"
        p.append(why)
    if saved is None:
        p.append("score.json missing")
    elif saved["pass"] != fresh["pass"] or saved["checks"] != fresh["checks"]:
        p.append(f"score.json {saved['checks']} != re-score {fresh['checks']}")
    return {"run_id": entry["run_id"], "run_dir": str(d), "task_id": run["task"]["id"], "start": run["start_place"],
            "outcome": run["outcome"], "steps": run["steps"], "checks": fresh["checks"], "label": label,
            "label_why": why, "problems": p, "pass": bool(fresh["pass"]) and not p}


def prompt_problems(runs: list[dict[str, Any]]) -> list[str]:
    shas = sorted({str(r.get("system_prompt_sha256")) for r in runs})
    return [f"runs disagree on system_prompt_sha256: {shas}"] if len(shas) > 1 else []


def invocation_problems(out: Path, runs: dict[str, dict[str, Any]], ledger: Path) -> list[str]:
    """B1: the counted runs are one invocation of the committed list, and the ledger knows no other."""
    if not (out / "invocation.json").is_file():
        return ["invocation.json missing"]
    inv = json.loads((out / "invocation.json").read_text())
    iid, p = inv.get("invocation_id"), []
    if inv.get("list_sha256") != COUNTED_SHA256:
        p.append(f"invocation list_sha256 {inv.get('list_sha256')} != committed counted list {COUNTED_SHA256}")
    other = sorted(k for k, r in runs.items() if r.get("invocation_id") != iid)
    if other:
        p.append(f"runs not of invocation {iid}: {other}")
    lines = ledger.read_text().splitlines() if ledger.is_file() else []
    events = [e for e in map(json.loads, filter(str.strip, lines)) if e["list_sha256"] == COUNTED_SHA256]
    ids = sorted({e["invocation_id"] for e in events})
    if ids != [iid]:
        p.append(f"ledger {ledger} holds counted invocations {ids}, want exactly [{iid}]")
    kinds = [e["event"] for e in events if e["invocation_id"] == iid and "job" not in e]  # job lines: --merge
    if kinds[:1] != ["start"] or kinds[-1:] != ["end"] or set(kinds[1:-1]) - {"resume", "abort"}:
        p.append(f"ledger events of {iid}: {kinds}, want start, then only resume / abort, then one end")
    return p


def g1_verdict(out: Path, entries: list[dict[str, Any]], counted: bool, ledger: Path | None = None,
               **score_kw: Any) -> dict[str, Any]:
    require_merged(out)
    dirs = present(out, entries)
    per = [g1_run(e, dirs.get(e["run_id"]), **score_kw) for e in entries]
    runs = {k: read_run(d)[0] for k, d in dirs.items()}
    gp = (git_problems([{k: r[k] for k in ("git_commit", "git_dirty")} for r in runs.values()])
          + prompt_problems(list(runs.values())))
    ip = invocation_problems(out, runs, ledger or default_ledger()) if counted else []
    n = sum(r["pass"] for r in per)
    return {"criterion": f">= {G1_NEED} of the {G1_RUNS} counted runs (tasks/phase3_counted.yaml) pass all four scorer "
                         "checks; a missing run fails; each run matches its entry (task, start, seed), reached its "
                         "start, saved score = re-score; one commit, clean tree, one system prompt; one invocation "
                         "of the committed list, alone in the ledger",
            "passed": n, "total": len(entries), "git_problems": gp, "invocation_problems": ip,
            "setup_misses": [r["run_id"] for r in per if any(x.startswith("setup did not reach") for x in r["problems"])],
            "system_prompt_sha256": sorted({str(r.get("system_prompt_sha256")) for r in runs.values()}),
            "verdict_scope": "counted list" if counted else "not the counted list (or --rehearsal): pass false",
            "runs": per, "pass": counted and len(entries) == G1_RUNS and n >= G1_NEED and not gp and not ip}


# ------------------------------------------------------------------------------------------------------------ G2
def step_problems(s: dict[str, Any], d: Path, run: dict[str, Any]) -> list[str]:
    at, p = f"step {s.get('step')}", []
    crashed = "traceback" in s
    pk = s.get("packet") or {}
    if pk.get("system_sha256") != run.get("system_prompt_sha256"):
        p.append(f"{at}: packet.system_sha256 {pk.get('system_sha256')} != run.json system_prompt_sha256")
    texts = pk.get("texts")
    if not (isinstance(texts, list) and texts and all(isinstance(t, str) for t in texts)):
        p.append(f"{at}: packet.texts missing (the text parts sent)")
    photos = pk.get("photos") or []
    if not photos:
        p.append(f"{at}: packet has no photos")
    for ph in photos:
        f = d / str(ph.get("rgb"))
        if not f.is_file() or ph.get("rgb_sha256") != sha256_file(f):
            p.append(f"{at}: photo {ph.get('n')} rgb {ph.get('rgb')} missing or rgb_sha256 does not match the file")
        if not re.fullmatch(r"[0-9a-f]{64}", str(ph.get("sent_sha256"))):
            p.append(f"{at}: photo {ph.get('n')} sent_sha256 missing")
    po = s.get("planner_out")
    if not isinstance(po, dict) or set(po) < {"content", "reasoning", "finish_reason", "usage", "error"}:
        p.append(f"{at}: planner_out lacks content / reasoning / finish_reason / usage / error")
    elif po["error"] is None:
        if po["finish_reason"] is None or po["usage"] is None:
            p.append(f"{at}: planner_out finish_reason / usage null without a planner error")
    if not isinstance((s.get("call") or {}).get("raw"), list):
        p.append(f"{at}: call.raw (the tool calls as received) missing")
    if not isinstance(s.get("check"), str):
        p.append(f"{at}: check missing")
    if "result" not in s or (s.get("check") == "ok" and s["result"] is None and not crashed):
        p.append(f"{at}: skill result missing")
    missing = [k for k in TIMINGS if not isinstance(s.get(k), (int, float))]
    if missing and not crashed:
        p.append(f"{at}: timings missing {missing}")
    return p


def run_problems(d: Path) -> list[str]:
    run, steps = read_run(d)
    p = [f"check_complete: {e}" for e in check_complete(d)]
    sp = run.get("system_prompt")
    if not isinstance(sp, str) or hashlib.sha256(sp.encode()).hexdigest() != run.get("system_prompt_sha256"):
        p.append("run.json system_prompt missing or its sha256 != system_prompt_sha256")
    if not isinstance(run.get("tools"), list) or not run["tools"]:
        p.append("run.json tools (the schemas sent) missing")
    info = run.get("planner_info") or {}
    if any(info.get(k) is None for k in PLANNER_INFO):
        p.append(f"run.json planner_info lacks {[k for k in PLANNER_INFO if info.get(k) is None]}")
    for s in steps:
        p += step_problems(s, d, run)
    sc = json.loads((d / "score.json").read_text()) if (d / "score.json").is_file() else None
    if sc is None:
        return p + ["score.json missing"]
    why = sc.get("label_why")
    if sc["pass"] and (sc.get("label") is not None or why is not None):
        p.append("passing run carries a failure label")
    if not sc["pass"] and not (sc.get("label") in LABELS and isinstance(why, str) and why.strip() and "\n" not in why):
        p.append(f"failed run needs exactly one label in {sorted(LABELS)} + one line why; has "
                 f"{sc.get('label')!r} / {why!r}")
    return p


def g2_verdict(out: Path, entries: list[dict[str, Any]], counted: bool) -> dict[str, Any]:
    require_merged(out)
    dirs = present(out, entries)
    per = {e["run_id"]: (run_problems(dirs[e["run_id"]]) if e["run_id"] in dirs else ["run dir missing"])
           for e in entries}
    empty = {k: [s["step"] for s in read_run(d)[1] if isinstance(s.get("planner_out"), dict)
                 and s["planner_out"].get("error") is None and not s["planner_out"].get("reasoning")]
             for k, d in dirs.items()}
    return {"criterion": "every counted run: full JSONL (packet sent: text parts + system prompt sha256 + photos by "
                         "path + sha256; raw VLM output incl. the reasoning key (empty listed), finish_reason, usage; "
                         "tool call; check; skill result; timings), run.json prompt + tools + model / sampling / seed, "
                         "check_complete empty; each failed run exactly one label + one line why",
            "runs": {k: {"ok": not v, "problems": v} for k, v in per.items()},
            "empty_reasoning_steps": {k: v for k, v in empty.items() if v},
            "verdict_scope": "counted list" if counted else "not the counted list (or --rehearsal): pass false",
            "pass": counted and len(entries) == G1_RUNS and all(not v for v in per.values())}


# ------------------------------------------------------------------------------------------------------------ G3
def scan(text: str, echoes: str) -> list[dict[str, Any]]:
    """Pose-like numbers and banned words in `text`; `echo` = the hit also occurs in the VLM's own earlier output."""
    hits = [m.group(0) for r in [*POSE_RES, WORD_RE] for m in r.finditer(text)]
    return [{"hit": h, "echo": h.lower() in echoes} for h in hits]


def vlm_said(s: dict[str, Any]) -> str:
    """Everything the VLM itself produced in one step (tool calls as received + content), lower case."""
    po = s.get("planner_out") or {}
    return (json.dumps((s.get("call") or {}).get("raw")) + " " + str(po.get("content") or "")).lower()


def g3_run(d: Path) -> list[dict[str, Any]]:
    """Every hit in what this run sent the VLM, with where it was."""
    run, steps = read_run(d)
    found = [{"where": "system_prompt", **h} for h in scan(str(run.get("system_prompt")), "")]
    found += [{"where": "tools", **h} for h in scan(json.dumps(run.get("tools")), "")]
    echoes = ""
    for s in steps:
        pk = s.get("packet") or {}
        for i, t in enumerate([*(pk.get("texts") or []), *(pk.get("history") or []), str(pk.get("last_result"))]):
            found += [{"where": f"step {s['step']} text {i}", **h} for h in scan(t, echoes)]
        echoes += vlm_said(s)
    return found


NO_ANSWER_TESTS = ("tests/test_agent_no_ros.py", "tests/test_truth_readers.py")  # the second: design review 007 audit


def no_ros_test() -> dict[str, Any]:
    """G3's rule tests: the agent side imports no ROS, and no answer reaches the agent or the robot adapters."""
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *NO_ANSWER_TESTS],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    return {"returncode": p.returncode, "files": list(NO_ANSWER_TESTS),
            "summary": (p.stdout.strip().splitlines() or [""])[-1]}


def g3_verdict(out: Path, entries: list[dict[str, Any]], counted: bool, no_ros: dict[str, Any]) -> dict[str, Any]:
    require_merged(out)
    dirs = present(out, entries)
    per = {k: g3_run(d) for k, d in dirs.items()}
    bad = {k: [h for h in v if not h["echo"]] for k, v in per.items()}
    missing = sorted({e["run_id"] for e in entries} - set(dirs))
    return {"criterion": "no decimal / coordinate pair / robot, sensor or map word in the system prompt, tool schemas "
                         "or any step's text parts, history, last result of a counted run (echoes of the VLM's own "
                         "output listed, not failed); tests/test_agent_no_ros.py and tests/test_truth_readers.py pass",
            "patterns": [r.pattern for r in [*POSE_RES, WORD_RE]], "no_ros_test": no_ros, "missing_runs": missing,
            "hits": {k: v for k, v in per.items() if v}, "failing_hits": {k: v for k, v in bad.items() if v},
            "verdict_scope": "counted list" if counted else "not the counted list (or --rehearsal): pass false",
            "pass": (counted and len(entries) == G1_RUNS and not missing and no_ros["returncode"] == 0
                     and not any(bad.values()))}


# ------------------------------------------------------------------------------------------------------- Measure
def _stats(xs: list[float]) -> dict[str, float | int]:
    return ({"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs), "max": max(xs)}
            if xs else {"n": 0})


def measure(out: Path, entries: list[dict[str, Any]], job_gpu_h: float | None = None) -> dict[str, Any]:
    """No bar (task 005 Measure): steps per run, VLM latency per step, wall per run, H200-h per run (planner wall x 1
    GPU: the time the job's GPU served this run's calls; the idle job time between calls is not in it, so it is a lower
    bound, N5), and with `job_gpu_h` (the job's total GPU-h from hpc/JOBS.md) that divided by the runs of this block.
    B2: per block, the `length` finishes and the tool calls left inside reasoning (dev exit: zero of both), and the
    largest completion_tokens. Per run and per node (run.json `node`) sim_s / wall_s; runs below RTF_FLAG listed."""
    require_merged(out)
    rows, lat, length, inside, tokens = [], [], [], [], []
    for k, d in present(out, entries).items():
        run, steps = read_run(d)
        pw = [s["planner_wall_s"] for s in steps if isinstance(s.get("planner_wall_s"), (int, float))]
        lat += pw
        rows.append({"run_id": k, "outcome": run["outcome"], "steps": run["steps"], "wall_s": run["wall_s"],
                     "sim_s": run["sim_s"], "vlm_s": sum(pw), "h200_h": sum(pw) / 3600, "node": run.get("node"),
                     "sim_per_wall": run["sim_s"] / run["wall_s"]})
        pos = [s for s in steps if isinstance(s.get("planner_out"), dict)]
        length += [f"{k} step {s['step']}" for s in pos if s["planner_out"].get("finish_reason") == "length"]
        inside += [f"{k} step {s['step']}" for s in no_call_steps(steps) if toolcall_in_reasoning(s)]
        tokens += [(s["planner_out"].get("usage") or {}).get("completion_tokens") or 0 for s in pos]
    m = {"note": "no bar (task 005 Measure); sets the Phase 4-8 budgets", "runs": rows,
         "steps_per_run": _stats([r["steps"] for r in rows]), "vlm_latency_per_step_s": _stats(lat),
         "wall_per_run_s": _stats([r["wall_s"] for r in rows]), "h200_h_per_run": _stats([r["h200_h"] for r in rows]),
         "h200_h_per_run_note": "lower bound: planner wall time only; the job holds the GPU for its whole wall",
         "h200_h_per_run_job": None if job_gpu_h is None or not rows else job_gpu_h / len(rows),
         "sim_per_wall_by_node": {str(n): _stats([r["sim_per_wall"] for r in rows if r["node"] == n])
                                  for n in sorted({r["node"] for r in rows}, key=str)},
         f"sim_per_wall_below_{RTF_FLAG}": [r["run_id"] for r in rows if r["sim_per_wall"] < RTF_FLAG],
         "vlm_output": {"length_finishes": len(length), "toolcall_in_reasoning": len(inside),
                        "max_completion_tokens": max(tokens, default=None), "length_steps": length,
                        "toolcall_in_reasoning_steps": inside}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "measure.json").write_text(json.dumps(m, indent=2))
    return m


# ------------------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["g1", "g2", "g3", "measure", "all"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--list", type=Path, default=COUNTED)
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--ledger", type=Path, default=None, help="counted ledger (default: phase3_run.default_ledger())")
    ap.add_argument("--job-gpu-h", type=float, default=None, help="measure: the vLLM job's GPU-h (hpc/JOBS.md)")
    a = ap.parse_args()
    entries = load_runs(a.list)
    counted = is_counted(a.list) and not a.rehearsal
    results = []
    if a.gate in ("g1", "all"):
        results.append(write(a.out, "g1", g1_verdict(a.out, entries, counted, a.ledger), a.rehearsal))
    if a.gate in ("g2", "all"):
        results.append(write(a.out, "g2", g2_verdict(a.out, entries, counted), a.rehearsal))
    if a.gate in ("g3", "all"):
        results.append(write(a.out, "g3", g3_verdict(a.out, entries, counted, no_ros_test()), a.rehearsal))
    if a.gate in ("measure", "all"):
        measure(a.out, entries, a.job_gpu_h)
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
