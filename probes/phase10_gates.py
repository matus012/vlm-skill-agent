#!/usr/bin/env python3
"""Task 012 gate runner (Phase 10: the frozen agent on the G1; design contract section 6). Reads the merged run dirs
`<D>/<run_id>/` of one cross-job counted invocation of tasks/phase8_pass2.yaml on the G1 and, for `side`, the TB4 pass-2
run dir T. No sim, no VLM. phase8_gates.py stays untouched (TB4 verdicts stay reproducible); this file reuses it.

    uv run python probes/phase10_gates.py {g1,g2,g3,side,measure,diag,all} --out D --tb4-out T \
        [--list tasks/phase8_pass2.yaml] [--freeze-tag skill-interface-frozen-v2] [--job-gpu-h H] [--drops FILE]

Every gate first refuses D unless it is merged ok, its invocation robot is g1 and every run.json says robot "g1".
G1    phase8_gates.g1_verdict as is (>= 18 of 25 per type, Wilson).
G2    `git diff <tag> HEAD -- <SWAP_PATHSPEC>` saved verbatim to g2_swap_diff.txt, PASS iff 0 lines; phase8 freeze_check;
      the list's sha256 == its COUNTED_LISTS pin.
G3    packet scan over all G1 packets (phase4_gates.g3_verdict) + tests/test_agent_no_ros.py + tests/test_truth_readers.py.
side  per type TB4 k/25 vs G1 k/25, 2x2 agreement, per run_id both columns and the G1 label. No prompts. `--drops FILE`
      (one run_id per line) removes hosting drops from both columns.
measure / diag: no bar; diag = failure aggregates only (label x crash_origin x nav status x step skill).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.scorer import read_run  # noqa: E402
from probes import phase3_gates as p3  # noqa: E402
from probes import phase4_gates as p4  # noqa: E402
from probes import phase8_gates as p8  # noqa: E402
from probes.phase3_run import COUNTED_LISTS, is_counted, sha256_file  # noqa: E402

LIST = ROOT / "tasks" / "phase8_pass2.yaml"
FREEZE_TAG = "skill-interface-frozen-v2"
ROBOT = "g1"
SWAP_PATHSPEC = [
    "agent/", "sim/worlds/clinic/places.yaml", "sim/worlds/clinic/layout.yaml",
    "sim/worlds/clinic/layout_cabinet_open.yaml", "sim/worlds/clinic/generate.py", "sim/worlds/clinic/seeded.py",
    "sim/worlds/clinic/realize.py", "sim/worlds/clinic/generated/*", "sim/worlds/clinic/generated_cabinet_open/*",
    "tests/interface_freeze.json", "tests/test_interface_freeze.py", "hpc/sim_lib.sh", "hpc/serve_vllm.sbatch",
    "hpc/VLLM_VERSION.md"]
FALL_RE = re.compile(r"RobotFell|robot fell|\bfell\b", re.I)  # drive.RobotFell traceback in run.json `crash`
NON_DRIVE = ("look_around", "report", "turn_to", "wait")


def require_robot(out: Path, entries: list[dict[str, Any]], robot: str = ROBOT) -> None:
    """Refuse a run dir that is not a merged G1 invocation (SystemExit with the reason)."""
    p3.require_merged(out)
    inv_p = out / "invocation.json"
    inv = json.loads(inv_p.read_text()).get("robot", "turtlebot4") if inv_p.is_file() else None
    bad = {rid: (read_run(d)[0].get("robot")) for rid, d in p3.present(out, entries).items()
           if read_run(d)[0].get("robot") != robot}
    if inv != robot or bad:
        sys.exit(f"refusing {out}: invocation robot {inv!r}, runs with robot != {robot!r}: {dict(sorted(bad.items()))}")


# ------------------------------------------------------------------------------------------------------------ G2
def swap_diff(tag: str, cwd: Path = ROOT, pathspec: list[str] = SWAP_PATHSPEC) -> str:
    p = subprocess.run(["git", "diff", tag, "HEAD", "--", *pathspec], cwd=cwd, capture_output=True, text=True)
    assert p.returncode == 0, f"git diff failed: {p.stderr}"
    return p.stdout


def list_pin(list_path: Path, pins: dict[Path, str] = COUNTED_LISTS) -> dict[str, Any]:
    pin = next((v for k, v in pins.items() if k.name == Path(list_path).name), None)
    sha = sha256_file(list_path)
    return {"list": Path(list_path).name, "sha256": sha, "pin": pin, "pass": pin is not None and sha == pin}


def g2_verdict(out: Path, entries: list[dict[str, Any]], tag: str, diff: str, freeze: dict[str, Any],
               pin: dict[str, Any]) -> dict[str, Any]:
    require_robot(out, entries)
    lines = diff.splitlines()
    (out / "g2_swap_diff.txt").write_text(diff)
    return {"criterion": f"`git diff {tag} HEAD -- <swap pathspec>` is 0 lines (saved verbatim to g2_swap_diff.txt); "
                         f"{p8.FREEZE_TEST} passes and the runs' one commit has {p8.FREEZE_JSON} equal to the tag's; "
                         "the list's sha256 equals its COUNTED_LISTS pin",
            "diff_lines": len(lines), "pathspec": SWAP_PATHSPEC, "freeze": freeze, "list_pin": pin,
            "pass": not lines and freeze["pass"] and pin["pass"]}


# ------------------------------------------------------------------------------------------------------------ G3
def g3_verdict(out: Path, entries: list[dict[str, Any]], counted: bool, no_ros: dict[str, Any]) -> dict[str, Any]:
    require_robot(out, entries)
    v = p4.g3_verdict(out, entries, counted, no_ros)
    return {**v, "criterion": "no decimal / coordinate pair / robot, sensor or map word (incl. g1, unitree, humanoid) "
                              "in anything sent to the VLM over all G1 packets (echoes listed, not failed); "
                              + " and ".join(p3.NO_ANSWER_TESTS) + " pass",
            "pass": (counted and len(entries) == p8.PER_TYPE * len(p8.TYPES) and not v["missing_runs"]
                     and no_ros["returncode"] == 0 and not v["failing_hits"])}


# ------------------------------------------------------------------------------------------------------------ side
def is_fall(run: dict[str, Any]) -> bool:
    """A G1 fall: outcome crash, origin adapter (label sim), crash text names RobotFell / a fall."""
    return run.get("crash_origin") == "adapter" and bool(FALL_RE.search(str(run.get("crash"))))


def is_stall(d: Path | None, label: Any) -> bool:
    """navigate_to `failed: blocked` (stall or timeout, label nav)."""
    return d is not None and label == "nav" and any(
        s.get("result") == "failed: blocked" for s in read_run(d)[1])


def repeat_look_arounds(steps: list[dict[str, Any]]) -> int:
    """look_around steps with an earlier look_around at the same place: no navigate_to `reached` in between (design review, 012:
    the drift stress case -- a repeated set shares its anchor)."""
    n, seen = 0, False
    for s in steps:
        name = (s.get("call") or {}).get("name")
        if name == "navigate_to" and s.get("result") == "reached":
            seen = False
        elif name == "look_around":
            n, seen = n + seen, True
    return n


def repeats(out: Path, kept: list[dict[str, Any]]) -> dict[str, Any]:
    per = {rid: read_run(d) for rid, d in p3.present(out, kept).items()}
    hit = {rid: repeat_look_arounds(steps) for rid, (_, steps) in per.items()}
    return {"runs": len(per), "runs_with_repeat": sum(1 for v in hit.values() if v), "repeats": sum(hit.values()),
            "hand_help_runs": sum(1 for run, _ in per.values() if run.get("hand_help")),
            "hand_help_runs_with_repeat": sum(1 for rid, (run, _) in per.items() if run.get("hand_help") and hit[rid])}


def side(d_out: Path, t_out: Path, entries: list[dict[str, Any]], drops: set[str] = frozenset()) -> dict[str, Any]:
    """Join T (TB4) and D (G1) by run_id; drops leave both columns. Aggregates + ids + labels only, no prompts."""
    require_robot(d_out, entries)
    p3.require_merged(t_out)
    kept = [e for e in entries if e["run_id"] not in drops]
    td, dd = p3.present(t_out, kept), p3.present(d_out, kept)
    rows = []
    for e in kept:
        rid = e["run_id"]
        tb, g = p4.g1_run(e, td.get(rid)), p4.g1_run(e, dd.get(rid))
        rows.append({"run_id": rid, "kind": e["expected"]["kind"], "tb4_pass": bool(tb["pass"]),
                     "g1_pass": bool(g["pass"]), "g1_label": g.get("label") if not g["pass"] else None,
                     "g1_fall": g["pass"] is False and rid in dd and is_fall(read_run(dd[rid])[0]),
                     "g1_stall": not g["pass"] and is_stall(dd.get(rid), g.get("label"))})
    types = {}
    for k in p8.TYPES:
        rs = [r for r in rows if r["kind"] == k]
        c = Counter((r["tb4_pass"], r["g1_pass"]) for r in rs)
        n = len(rs)
        types[k] = {"n": n, "tb4": sum(r["tb4_pass"] for r in rs), "g1": sum(r["g1_pass"] for r in rs),
                    "both": c[(True, True)], "tb4_only": c[(True, False)], "g1_only": c[(False, True)],
                    "neither": c[(False, False)],
                    "g1_falls": sum(r["g1_fall"] for r in rs), "g1_stalls": sum(r["g1_stall"] for r in rs),
                    "g1_other_sim": sum(r["g1_label"] == "sim" and not r["g1_fall"] for r in rs),
                    "agreement_pct": 100 * (c[(True, True)] + c[(False, False)]) / n if n else None}
    return {"note": "no bar", "dropped": sorted(drops & {e["run_id"] for e in entries}), "types": types,
            "repeat_look_arounds": {"tb4": repeats(t_out, kept), "g1": repeats(d_out, kept)}, "runs": rows}


# ------------------------------------------------------------------------------------------------------ measure
def _num(x: Any) -> float:
    return len(x) if isinstance(x, (list, tuple)) else x if isinstance(x, (int, float)) else 0


def measure(out: Path, entries: list[dict[str, Any]], job_gpu_h: float | None = None) -> dict[str, Any]:
    """phase8 measure + G1 extras; every key tolerated missing. Sources: run.json falls / robot_resets / recentres,
    node {rtf, restarts}; step skill_detail recentres / drift_m (non-drive steps only)."""
    require_robot(out, entries)
    m = p8.measure(out, entries, job_gpu_h)
    falls = resets = recentres = 0
    drift = 0.0
    nodes: dict[str, dict[str, Any]] = {}
    for d in p3.present(out, entries).values():
        run, steps = read_run(d)
        falls += int(_num(run.get("falls")) or "RobotFell" in str(run.get("crash")))
        resets += int(_num(run.get("robot_resets")) or run.get("teleport") is not None)
        for s in steps:
            det = s.get("skill_detail") or {}
            recentres += int(_num(det.get("recentres")))
            if (s.get("call") or {}).get("name") in NON_DRIVE and isinstance(det.get("drift_m"), (int, float)):
                drift = max(drift, det["drift_m"])
        nd = run.get("node")
        if isinstance(nd, dict):
            nodes[f"{run.get('slurm_job_id')}/{run.get('shard')}"] = nd
    m["g1"] = {"falls": falls, "robot_resets": resets, "recentres": recentres, "max_nondrive_step_drift_m": drift,
               "node_restarts": sum(int(_num(n.get("restarts"))) for n in nodes.values()),
               "rtf_per_node": {k: n.get("rtf") for k, n in sorted(nodes.items())}}
    (out / "measure.json").write_text(json.dumps(m, indent=2))
    return m


# --------------------------------------------------------------------------------------------------------- diag
def diag(out: Path, entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregates of failed runs only: label x crash_origin x nav status x skill of the last step."""
    require_robot(out, entries)
    c: Counter = Counter()
    for d in p3.present(out, entries).values():
        sc = p4.score_run(d)
        if sc["pass"]:
            continue
        run, steps = read_run(d)
        last = steps[-1] if steps else {}
        c[(str(sc.get("label")), str(run.get("crash_origin")),
           str((last.get("skill_detail") or {}).get("nav_status")), str((last.get("call") or {}).get("name")))] += 1
    rows = [dict(zip(("label", "crash_origin", "nav_status", "step_skill"), k), n=n) for k, n in c.most_common()]
    res = {"note": "failure aggregates only; no run ids, prompts or photos", "failed": sum(c.values()), "groups": rows}
    (out / "diag.json").write_text(json.dumps(res, indent=2))
    return res


# ------------------------------------------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=["g1", "g2", "g3", "side", "measure", "diag", "all"])
    ap.add_argument("--out", type=Path, required=True, help="D: the merged G1 run dir")
    ap.add_argument("--tb4-out", type=Path, required=True, help="T: the TB4 pass-2 run dir (side)")
    ap.add_argument("--list", type=Path, default=LIST)
    ap.add_argument("--freeze-tag", default=FREEZE_TAG)
    ap.add_argument("--job-gpu-h", type=float, default=None)
    ap.add_argument("--drops", type=Path, default=None, help="side: file of hosting-dropped run_ids, one per line")
    ap.add_argument("--rehearsal", action="store_true")
    a = ap.parse_args(argv)
    entries = p8.load_list(a.list)
    require_robot(a.out, entries)
    counted = is_counted(a.list) and not a.rehearsal
    g, results = a.gate, []
    if g in ("g1", "all"):
        v = p8.g1_verdict(a.out, entries, counted, a.list)
        results.append(p3.write(a.out, "g1", {**v, "robot": ROBOT}, a.rehearsal))
    if g in ("g2", "all"):
        fz = p8.freeze_check(a.out, entries, a.freeze_tag, p8.freeze_test())
        results.append(p3.write(a.out, "g2", g2_verdict(a.out, entries, a.freeze_tag, swap_diff(a.freeze_tag), fz,
                                                        list_pin(a.list)), a.rehearsal))
    if g in ("g3", "all"):
        results.append(p3.write(a.out, "g3", g3_verdict(a.out, entries, counted, p3.no_ros_test()), a.rehearsal))
    if g in ("side", "all"):
        drops = set(a.drops.read_text().split()) if a.drops else set()
        (a.out / "side.json").write_text(json.dumps(side(a.out, a.tb4_out, entries, drops), indent=2))
    if g in ("measure", "all"):
        measure(a.out, entries, a.job_gpu_h)
    if g in ("diag", "all"):
        diag(a.out, entries)
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
