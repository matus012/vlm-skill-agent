"""Scorer (INTERFACES section 7): score a run directory (section 5) from disk. No ROS imports.

    python -m agent.scorer <run_dir>     # writes <run_dir>/score.json, prints it

Pass only if all four hold: (1) answer == truth answer, (2) the reported photo's cropped seg has >= 1000 px of one
proof label, (3) outcome "reported" within 12 steps, (4) no crash and no hand help.
Truth: the run's world variant (run.json `world`, absent = closed; agent/truth.py WORLD_DIRS), with run.json
`layout_seed` that seeded layout's (agent/truth.run_truth, INTERFACES section 10); the run's layout sha256 must equal
that truth's.
"""
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from agent.truth import DEFAULT_WORLD, ROOT, derive, load_truth, run_truth

MAX_STEPS = 12
PROOF_MIN_PX = 1000
# run.json crash_origin (INTERFACES section 4) -> failure label; absent (runs before 005) -> "sim", the old rule
CRASH_LABEL = {"adapter": "sim", "skill": "skill", "runtime": "skill", "agent": "skill", "planner": "vlm",
               "truth_hook": "scorer", "gz_read": "sim",  # gz_read: a gz CLI read dead after its retries (dev. 13)
               "layout_teleport": "sim"}  # 008: a seeded layout's movables did not land (adapters/run_task.py)
PLACES_YAML = ROOT / "sim" / "worlds" / "clinic" / "places.yaml"  # the reachable places (a test-only place is not)
# B2: a tool call the parsers did not extract (left in the reasoning / content): hermes, qwen3_coder or bare JSON form
TOOLCALL_RE = re.compile(r'<tool_call>|<function=|"name"\s*:\s*"(?:navigate_to|look_around|report)"')


def read_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    steps = [json.loads(line) for line in (run_dir / "steps.jsonl").read_text().splitlines() if line.strip()]
    return run, steps


def _report_step(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok = [s for s in steps if s["call"]["name"] == "report" and s["check"] == "ok"]
    if len(ok) > 1:
        raise ValueError(f"{len(ok)} accepted report steps; a run ends on the first")
    return ok[0] if ok else None


def proof_seg(steps: list[dict[str, Any]], report: dict[str, Any] | None) -> str | None:
    """Run-dir path of the reported photo's seg PNG (as logged in the report step's packet), None if there is none."""
    if report is None:
        return None
    st = _report_step(steps)
    if st is None:
        raise ValueError("run.json has a report but steps.jsonl has no accepted report call")
    hit = [p for p in st["packet"]["photos"] if p["n"] == report["photo"]]
    return hit[0]["seg"] if hit else None


def proof_pixels(run_dir: Path, steps: list[dict[str, Any]], report: dict[str, Any] | None,
                 labels: tuple[int, ...]) -> tuple[int, int | None]:
    """(max px of a single proof label in the reported photo's seg, that label). 0 if no report or no such photo."""
    path = proof_seg(steps, report)
    if path is None:
        return 0, None
    seg = np.asarray(Image.open(Path(run_dir) / path))
    px = {lab: int(np.count_nonzero(seg == lab)) for lab in labels}
    best = max(px, key=lambda lab: px[lab])
    return px[best], best


def _blocked_reachable(steps: list[dict[str, Any]], places_path: Path, truth: dict[str, Any]) -> str | None:
    """A failed navigate_to on a reachable place: places.yaml minus the blocked room's place (INTERFACES section 11)."""
    reachable = set(yaml.safe_load(Path(places_path).read_text())["places"]) - {truth.get("blocker", {}).get("room")}
    for s in steps:
        if (s["call"]["name"] == "navigate_to" and s["check"] == "ok" and str(s["result"]).startswith("failed")
                and s["call"]["args"]["place"] in reachable):
            return f"step {s['step']}: navigate_to({s['call']['args']['place']}) -> {s['result']} on a reachable place"
    return None


def toolcall_in_reasoning(s: dict[str, Any]) -> bool:
    po = s.get("planner_out") or {}
    return TOOLCALL_RE.search(f"{po.get('reasoning') or ''} {po.get('content') or ''}") is not None


def no_call_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """B2: steps whose planner call came back (error null) with no tool call in it."""
    return [s for s in steps if isinstance(s.get("planner_out"), dict) and s["planner_out"].get("error") is None
            and s["call"]["raw"] == []]


def no_call_why(steps: list[dict[str, Any]]) -> str:
    return "; no-call steps: " + ", ".join(
        f"step {s['step']} finish_reason {s['planner_out'].get('finish_reason')}, completion_tokens "
        f"{(s['planner_out'].get('usage') or {}).get('completion_tokens')}, tool call inside reasoning "
        f"{'yes' if toolcall_in_reasoning(s) else 'no'}" for s in steps)


def endpoint_error(run: dict[str, Any], steps: list[dict[str, Any]]) -> str | None:
    """B3: a VLM run's first step whose HTTP call failed (VlmPlanner sets planner_out.error only then)."""
    if not str(run.get("planner")).startswith("vlm:"):
        return None
    for s in steps:
        err = (s.get("planner_out") or {}).get("error")
        if err is not None:
            return f"VLM endpoint error at step {s['step']}: {' '.join(str(err).split())[:80]}"
    return None


def score_run(run_dir: Path, truth_path: Path | None = None, places_path: Path = PLACES_YAML) -> dict[str, Any]:
    """truth_path None: the truth of the run's world (run.json `world`, default closed: runs before 007) and layout
    seed (run.json `layout_seed`, absent = the frozen world: runs before 008) or Phase 8 layout spec (run.json
    `layout_spec`, INTERFACES section 12)."""
    run_dir = Path(run_dir)
    run, steps = read_run(run_dir)
    truth = (load_truth(truth_path) if truth_path is not None
             else run_truth(run.get("world", DEFAULT_WORLD), run.get("layout_seed"), run.get("layout_spec")))
    if run["layout_sha256"] != truth["layout_sha256"]:
        raise ValueError(f"run layout {run['layout_sha256']} != truth layout {truth['layout_sha256']}")
    task = run["task"]
    truth_answer, labels = derive(task["kind"], task["target"], truth)
    report = run["report"]
    proof_px, proof_label = proof_pixels(run_dir, steps, report, labels)
    n_steps = len(steps)
    checks = {
        "answer": report is not None and report["answer"] == truth_answer,
        "proof": proof_px >= PROOF_MIN_PX,
        "steps": run["outcome"] == "reported" and n_steps <= MAX_STEPS,
        "clean": run["outcome"] != "crash" and not run.get("hand_help", False),
    }
    passed = all(checks.values())
    label = why = None
    if not passed:
        blocked = _blocked_reachable(steps, places_path, truth)
        setup, endpoint = run.get("setup") or {}, endpoint_error(run, steps)
        if setup.get("reached") is False:  # N6: the run did not start where its list entry says
            label, why = "nav", f"setup did not reach {setup['start_place']} (GT {setup['gt_err_m']:.2f} m)"
        elif run["outcome"] == "crash":
            origin = run.get("crash_origin")
            label = CRASH_LABEL[origin] if origin else "sim"
            why = f"run crashed at step {n_steps} (outcome crash, origin {origin or 'not recorded'})"
        elif endpoint:  # B3: the instrument failed, not the VLM
            label, why = "sim", endpoint
        elif blocked:
            label, why = "nav", blocked
        else:
            label = "vlm"
            if run["outcome"] == "fail_malformed":
                bad = steps[-run["malformed_limit"]:]
                why = (f"{len(bad)} malformed planner outputs in a row (steps {', '.join(str(x['step']) for x in bad)}), "
                       f"last: {' '.join(str(bad[-1]['check']).split())[:200]}")
            elif not checks["steps"]:
                why = f"no report: outcome {run['outcome']} after {n_steps} steps (limit {MAX_STEPS})"
            elif not checks["clean"]:
                why = "hand help recorded in run.json"
            elif not checks["answer"]:
                why = f"answer {report['answer']!r} != truth {truth_answer!r}"
            else:
                why = (f"proof photo {report['photo']} has {proof_px} px of proof labels {list(labels)}, "
                       f"need {PROOF_MIN_PX}")
        if no_call_steps(steps):  # B2: a truncation or a call swallowed by the reasoning parser shows in the why
            why += no_call_why(no_call_steps(steps))
    return {"pass": passed, "checks": checks, "truth_answer": truth_answer, "proof_px": proof_px,
            "proof_label": proof_label, "label": label, "label_why": why}


def file_hashes(run_dir: Path) -> dict[str, str]:
    """sha256 of the files the verdict rests on: run.json, steps.jsonl and the reported photo's seg PNG."""
    run, steps = read_run(run_dir)
    seg = proof_seg(steps, run["report"])
    return {f: hashlib.sha256((Path(run_dir) / f).read_bytes()).hexdigest()
            for f in ("run.json", "steps.jsonl", *([seg] if seg else []))}


def write_score(run_dir: Path, **kw: Any) -> dict[str, Any]:
    sc = {**score_run(run_dir, **kw), "sha256": file_hashes(run_dir)}
    (Path(run_dir) / "score.json").write_text(json.dumps(sc, indent=1) + "\n")
    return sc


if __name__ == "__main__":
    print(json.dumps(write_score(Path(sys.argv[1])), indent=1))
