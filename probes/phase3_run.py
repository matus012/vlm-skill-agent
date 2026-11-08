#!/usr/bin/env python3
"""Task 005: run a run list (tasks/phase3_counted.yaml / tasks/phase3_dev.yaml) with the VLM planner, one sim session.

    source sim/env.sh; source $BUILD_DIR/vlm_link.env
    export VLM_BASE_URL=$BASE_URL VLM_API_KEY=$API_KEY VLM_MODEL=<served model name>
    python3 probes/phase3_run.py --list tasks/phase3_counted.yaml --out DIR [--min-job-left-min N]
    python3 probes/phase3_run.py --list tasks/phase3_counted.yaml --resume DIR      # after the runner died / aborted
    python3 probes/phase3_run.py --list tasks/phase3_dev.yaml --out DIR [--only d01,d02] [--rehearsal]
    python3 probes/phase3_run.py --list L --out DIR --shards 4 --shard-env 0=40:j7s0 ... --shard-env 3=43:j7s3

Each entry {run_id, task_id, start_place, seed} (+ optional world, layout_seed or layout_spec) ->
adapters.run_task.run_one(..., start_place, layout_seed, layout_spec) with a VlmPlanner whose sampling seed is the entry's seed; run dir DIR/<run_id>. One line
per run. Needs the clinic sim with AMCL + Nav2; the operator holds the gz lock (sim/INTERFACES.md section 6).
Without --rehearsal it refuses a dirty tree (design review B4; probes/phase3_gates.py also fails runs that are dirty or
on mixed commits). Gates: probes/phase3_gates.py.
Counted-run rules (agent/INTERFACES.md section 8, 005 design review B1 / B3): the counted list (by path or sha256) refuses
--rehearsal, --only and an existing --out; DIR/invocation.json (uuid, list sha256, commit, segments) is written first
and its uuid goes into every run.json; a counted invocation appends start / resume / end / abort lines to the ledger
outside the repo; --resume runs only entries with no run dir (never re-runs one); before each entry the endpoint
must answer GET /models (else abort, "endpoint down") and, without --rehearsal, the tree must still be clean
on the invocation's commit (else abort, "tree dirty"); --min-job-left-min N needs N min left on the vLLM job
(in a SLURM job: squeue of $SLURM_JOB_ID run locally; else ssh $LOGIN_NODE with the JOBID of vlm_link.env).
Sharded (hpc/PERUN_SIM.md section 3): --shards N --shard-env <i>=<ROS_DOMAIN_ID>:<GZ_PARTITION> (one per sim). This
process stays the coordinator: plan(), the refusals, invocation.json and the ledger lines are exactly as serial. It
assigns the segment's entries (all of them; on --resume those with no run dir) in list order to sim k % N, writes that
assignment into the segment before any run, then starts one worker subprocess per sim (this script, --shard-worker,
that sim's ROS_DOMAIN_ID / GZ_PARTITION) into the same OUT. Workers report "run" / "done" / "end" lines on stdout;
only the coordinator writes invocation.json and the ledger. A worker checks the tree and the endpoint before each of
its entries and refuses an entry whose run dir exists. A worker that stops early (abort or death) leaves its remaining
entries without a run dir (--resume runs them); the others finish theirs; the segment then ends in abort.
--shard-env I=DOMAIN:PARTITION:WORLD also sets that worker's SIM_WORLD; a worker refuses (abort) an entry whose `world`
(list key, default "closed") is not its SIM_WORLD (default "closed").
Cross-job (hpc/PERUN_SIM.md section 5): one counted list over K SLURM jobs x N sims, one invocation.
    python3 probes/phase3_run.py --list L --out D --split K:N [--shard-world J:I=WORLD ...]      # login node, no run
    python3 probes/phase3_run.py --list L --job-part D --job J --shards N --shard-env I=DOM:PART[:WORLD] ... [--resume]
    python3 probes/phase3_run.py --list L --merge D                                               # after every job
--split fixes {run_id: [job, shard]} in D/invocation.json before any run (list order; each entry to the next shard
of its world, job-major order j0s0, j0s1, j1s0 ..., cycling per world; a world with no shard: refused), with the
refusals of a counted --out; ledger line "start" to <ledger dir>/<invocation_id>.plan.jsonl. --job-part is job J's
coordinator: its entries on their split shards through the sharded worker protocol; its segments go to
D/invocation.j<J>.json (it never writes D/invocation.json), its ledger lines (with "job") to
<invocation_id>.j<J>.jsonl; --resume runs J's entries with no run dir. --merge checks every split entry has exactly one
run dir whose run.json carries D's invocation_id and the split's job / shard, appends the plan and job ledger lines in
time order to the counted ledger plus one "end" (ok) or "abort" line, and writes D/merged.json;
probes/phase3_gates.py refuses D until merged.json says ok. Every run.json gets
run_id, slurm_job_id, node, job, shard, world.
"""
import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.robots import DEFAULT as ROBOT_DEFAULT, make, robot_name  # noqa: E402
from agent.runlog import git_state  # noqa: E402
from agent.skills import load_places  # noqa: E402
from agent.tasks import all_tasks, task_by_id  # noqa: E402
from probes.phase2_gates import tree_moved  # noqa: E402

COUNTED = ROOT / "tasks" / "phase3_counted.yaml"
COUNTED_SHA256 = "c757e075c7d7b02fc8c102b8469f5d5e27dfe2380c587f00a0d2a167c7bd1c7b"  # blob 081c3d8e (548ad65)
PLACES_YAML = ROOT / "sim" / "worlds" / "clinic" / "places.yaml"
VLM_LINK_ENV = Path(os.environ.get("BUILD_DIR", Path.home() / "vlm_skill_agent_build")) / "vlm_link.env"
KEYS = {"run_id", "task_id", "start_place", "seed"}
# layout_seed: 008 seeded layout (INTERFACES section 10); layout_spec: Phase 8 layout spec (section 12); to run_one
OPTIONAL = {"world", "layout_seed", "layout_spec"}
PROTO = "@@shard "  # worker -> coordinator event lines on the worker's stdout
WORKER = [sys.executable, str(Path(__file__).resolve())]  # the shard worker command (tests swap in a fake runner)
RunEntry = Callable[[dict[str, Any], Path, dict[str, Any]], Any]  # (entry, run dir, extra run.json meta)


def load_runs(path: Path = COUNTED, places_path: Path = PLACES_YAML) -> list[dict[str, Any]]:
    """The run list, validated: exact keys (+ optional `world`, a non-empty str, `layout_seed`, an int, or
    `layout_spec`, a mapping), unique run ids and seeds, known task ids (agent/tasks.all_tasks) and places, int seeds.
    The list is all-seeded, all-spec or neither: an unseeded run after a seeded one on the same sim would start from
    the seeded placement, not the frozen world."""
    runs = yaml.safe_load(Path(path).read_text())
    tasks, places = {t.id for t in all_tasks()}, set(load_places(places_path))
    for r in runs:
        assert KEYS <= set(r) <= KEYS | OPTIONAL, f"{r}: keys {sorted(r)} != {sorted(KEYS)} (+ {sorted(OPTIONAL)})"
        assert isinstance(r.get("world", "closed"), str) and r.get("world", "closed"), r
        assert isinstance(r.get("layout_seed", 0), int) and not isinstance(r.get("layout_seed"), bool), r
        assert isinstance(r.get("layout_spec", {}), dict) and not {"layout_seed", "layout_spec"} <= set(r), r
        assert r["task_id"] in tasks and r["start_place"] in places, r
        assert isinstance(r["seed"], int) and not isinstance(r["seed"], bool), r
    for k in ("run_id", "seed"):
        assert len({r[k] for r in runs}) == len(runs), f"{path}: duplicate {k}"
    assert len({"layout_seed" in r for r in runs}) <= 1, f"{path}: layout_seed on some entries only"
    assert len({"layout_spec" in r for r in runs}) <= 1, f"{path}: layout_spec on some entries only"
    return runs


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# Every counted list (each runs once, under the counted-run rules), by path and by committed sha256 (a copy elsewhere is
# still counted): 005's Phase 3 list and 007's Phase 4 list (pinned when committed, before any 007 VLM run).
COUNTED_LISTS = {COUNTED: COUNTED_SHA256,
                 ROOT / "tasks" / "phase4_counted.yaml": "958a7d71aefe9c098791caedf33a126f83afa2d56ae9fac5d1e25cd1203e5889",
                 ROOT / "tasks" / "phase5_counted.yaml": "3b5416cdd5f9add874517587a7fe311606b3f0008524e63cf5899d47f11d6fca",
                 ROOT / "tasks" / "phase6_counted.yaml": "33d4f5889ee57029c79cf1eb15a5d1699bb0f3b70a1fc8ba9a84b50b3bb10275",
                 ROOT / "tasks" / "phase8_pass1.yaml": "2d7c5eee2d37c0ae132d3bb81413b0505a1f379433a6c203fdd5f4b75ca4a025",
    ROOT / "tasks" / "phase8_pass2.yaml": "b14ca498ca3ee21441235c1f54197c94bf88bd0d3585a297c54c42be15b7ba33"}


def is_counted(path: Path) -> bool:
    return (Path(path).resolve() in {c.resolve() for c in COUNTED_LISTS}
            or sha256_file(path) in COUNTED_LISTS.values())


def default_ledger() -> Path:
    """<ws>/ledger/phase3_counted_ledger.jsonl; ws = $WS, else <repo>/runs."""
    ws = Path(os.environ["WS"]) if os.environ.get("WS") else ROOT / "runs"
    return ws / "ledger" / "phase3_counted_ledger.jsonl"


def utc() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def endpoint_down(base_url: str, api_key: str, timeout_s: float = 10.0) -> str | None:
    """None if GET {base_url}/models answers 200 with the key, else why (B3)."""
    req = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return None if r.status == 200 else f"HTTP {r.status}"
    except OSError as e:  # URLError, HTTPError, timeouts: the endpoint is down for us
        return f"{type(e).__name__}: {e}"[:200]


def slurm_minutes(left: str) -> float:
    """squeue %L ([D-]HH:MM:SS, MM:SS) -> minutes. Anything else (INVALID, empty) raises."""
    days, _, hms = left.strip().rpartition("-")
    h, m, s = ([0, 0] + [int(x) for x in hms.split(":")])[-3:]
    return int(days or 0) * 1440 + h * 60 + m + s / 60


def job_left_min(env_path: Path = VLM_LINK_ENV) -> float:
    """Remaining wall of the vLLM job: inside a SLURM job (PERUN, vLLM in the same job) squeue of $SLURM_JOB_ID run
    locally; else the JOBID in vlm_link.env, from squeue over ssh."""
    if job := os.environ.get("SLURM_JOB_ID"):
        cmd = ["squeue", "-h", "-j", job, "-o", "%L"]
    else:
        job = dict(x.split("=", 1) for x in env_path.read_text().splitlines() if "=" in x)["JOBID"]
        cmd = ["ssh", "-o", "BatchMode=yes", "$LOGIN_NODE", f"squeue -h -j {job} -o %L"]
    return slurm_minutes(subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True).stdout)


def inv_name(job: int | None) -> str:
    """invocation.json, or job j's segment file invocation.j<j>.json (cross-job)."""
    return "invocation.json" if job is None else f"invocation.j{job}.json"


def write_inv(out: Path, inv: dict[str, Any], name: str = "invocation.json") -> None:
    """Atomic (tmp + rename): shard workers read it while the coordinator rewrites it."""
    tmp = out / f".{name}.tmp"
    tmp.write_text(json.dumps(inv, indent=1) + "\n")
    os.replace(tmp, out / name)


def refuse_robot(flag: str, out: Path, inv: dict[str, Any]) -> None:
    """SystemExit when `inv` (no "robot" key = turtlebot4) was planned for another robot than env ROBOT."""
    if (got := inv.get("robot", ROBOT_DEFAULT)) != robot_name():
        sys.exit(f"refusing {flag}: {out} is robot {got!r}, this is ROBOT={robot_name()!r}")


def world_of(e: dict[str, Any]) -> str:
    return e.get("world", "closed")


def run_meta(e: dict[str, Any], job: int | None, shard: int | None, base: dict[str, Any]) -> dict[str, Any]:
    """The extra run.json keys of entry e (base: the invocation_id)."""
    return {**base, "run_id": e["run_id"], "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "node": socket.gethostname(), "job": job, "shard": shard, "world": world_of(e)}


def ledger_line(ledger: Path, inv: dict[str, Any], event: str, **kw: Any) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as f:
        f.write(json.dumps({"utc": utc(), "event": event, "invocation_id": inv["invocation_id"], "out": inv["out"],
                            "list_sha256": inv["list_sha256"], "git_commit": inv["git_commit"], **kw}) + "\n")


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=Path, default=COUNTED)
    ap.add_argument("--out", type=Path, help="new invocation into OUT")
    ap.add_argument("--resume", type=Path, nargs="?", const=True,
                    help="OUT of an interrupted invocation: run only entries with no run dir; bare with --job-part")
    ap.add_argument("--shard-worker", type=Path, help="internal: run OUT's last segment's entries assigned to --shard")
    ap.add_argument("--shard", type=int, default=None, help="internal: the worker's sim index")
    ap.add_argument("--split", metavar="K:N", help="with --out: fix the split over K jobs x N sims; no run")
    ap.add_argument("--shard-world", action="append", default=[], metavar="J:I=WORLD",
                    help="--split: sim I of job J runs WORLD (default closed)")
    ap.add_argument("--job-part", type=Path, help="split OUT: run job --job's entries (needs --shards)")
    ap.add_argument("--job", type=int, default=None, help="the job index of --job-part (internal: of --shard-worker)")
    ap.add_argument("--merge", type=Path, help="split OUT, after every job: check, merge the ledgers, merged.json")
    ap.add_argument("--shards", type=int, default=0, help="N parallel sims (0: serial, one sim, as before)")
    ap.add_argument("--shard-env", action="append", default=[], metavar="I=DOMAIN:PARTITION",
                    help="sim I's ROS_DOMAIN_ID and GZ_PARTITION, once per sim 0..N-1")
    ap.add_argument("--only", default=None, help="comma list of run ids (not for the counted list)")
    ap.add_argument("--rehearsal", action="store_true", help="dev / rehearsal: allow a dirty tree")
    ap.add_argument("--ledger", type=Path, default=None, help="counted ledger (default: default_ledger())")
    ap.add_argument("--min-job-left-min", type=float, default=0.0, help="refuse to start with less left on the job")
    a = ap.parse_args(argv)
    a.argv = sys.argv[1:] if argv is None else argv
    a.job_resume = a.resume is True
    if a.job_resume:
        a.resume = None
    if sum(x is not None for x in (a.out, a.resume, a.shard_worker, a.job_part, a.merge)) != 1:
        ap.error("exactly one of --out, --resume OUT, --shard-worker, --job-part, --merge")
    a.shard_envs = shard_envs(a.shards, a.shard_env) if a.shards else []
    if a.shard_env and not a.shards:
        ap.error("--shard-env needs --shards")
    if (a.shard_worker is None) != (a.shard is None):
        ap.error("--shard-worker and --shard go together")
    if a.job_resume and a.job_part is None:
        ap.error("a bare --resume needs --job-part (else --resume OUT)")
    if a.job is not None and a.job_part is None and a.shard_worker is None:
        ap.error("--job goes with --job-part (or --shard-worker)")
    if a.job_part is not None and not (a.shards and a.job is not None and not a.only):
        ap.error("--job-part needs --job and --shards, and takes no --only (the split fixed the entries)")
    if a.split is not None and (a.out is None or a.shards):
        ap.error("--split needs --out and runs nothing (no --shards)")
    if a.shard_world and a.split is None:
        ap.error("--shard-world needs --split")
    a.split_shape = split_shape(a.split) if a.split is not None else None
    a.shard_worlds = shard_worlds(a.split_shape, a.shard_world) if a.split is not None else {}
    return a


def split_shape(spec: str) -> tuple[int, int]:
    k, _, n = spec.partition(":")
    assert k.isdigit() and n.isdigit() and int(k) > 0 and int(n) > 0, f"bad --split {spec!r} (want K:N)"
    return int(k), int(n)


def shard_worlds(shape: tuple[int, int], specs: list[str]) -> dict[tuple[int, int], str]:
    """--shard-world J:I=WORLD ... -> {(j, i): world} for every sim of the split (unnamed: closed)."""
    k, n = shape
    w = {(j, i): "closed" for j in range(k) for i in range(n)}
    seen = set()
    for spec in specs:
        ji, _, world = spec.partition("=")
        j, _, i = ji.partition(":")
        assert j.isdigit() and i.isdigit() and world and (int(j), int(i)) in w and ji not in seen, \
            f"bad --shard-world {spec!r} (want J:I=WORLD, J < {k}, I < {n}, once each)"
        seen.add(ji)
        w[int(j), int(i)] = world
    return w


def shard_envs(n: int, specs: list[str]) -> list[dict[str, str]]:
    """--shard-env I=DOMAIN:PARTITION[:WORLD] x N -> [{ROS_DOMAIN_ID, GZ_PARTITION[, SIM_WORLD]}] for sims 0..N-1,
    domains and partitions all distinct."""
    envs: dict[int, dict[str, str]] = {}
    for spec in specs:
        i, _, rest = spec.partition("=")
        dom, _, rest = rest.partition(":")
        part, _, world = rest.partition(":")
        assert i.isdigit() and dom.isdigit() and part and int(i) not in envs, f"bad --shard-env {spec!r}"
        envs[int(i)] = {"ROS_DOMAIN_ID": dom, "GZ_PARTITION": part, **({"SIM_WORLD": world} if world else {})}
    assert sorted(envs) == list(range(n)), f"--shards {n} needs --shard-env for sims 0..{n - 1}, got {sorted(envs)}"
    for k in ("ROS_DOMAIN_ID", "GZ_PARTITION"):
        assert len({e[k] for e in envs.values()}) == n, f"--shard-env: two sims share a {k}"
    return [envs[i] for i in range(n)]


def plan(a: argparse.Namespace) -> tuple[list[dict[str, Any]], Path, dict[str, Any], str]:
    """(entries, out, invocation, "start" | "resume"), or SystemExit naming the refusal."""
    counted, git = is_counted(a.list), git_state(ROOT)
    if counted and (a.rehearsal or a.only):
        sys.exit("refusing --rehearsal / --only on the counted list (it runs once, whole; task 005 B1)")
    if not a.rehearsal and git["git_dirty"]:
        sys.exit("refusing counted runs on a dirty tree (commit first, or pass --rehearsal)")
    runs = load_runs(a.list)
    if a.only:
        only = set(a.only.split(","))
        assert only <= {r["run_id"] for r in runs}, f"--only {sorted(only)} not all in {a.list}"
        runs = [r for r in runs if r["run_id"] in only]
    if a.resume is not None:
        inv = json.loads((a.resume / "invocation.json").read_text())
        if "split" in inv:
            sys.exit(f"refusing --resume: {a.resume} is split over jobs (use --job-part {a.resume} --job J --resume)")
        if (inv["list_sha256"], inv["git_commit"]) != (sha256_file(a.list), git["git_commit"]):
            sys.exit(f"refusing --resume: {a.resume} is list {inv['list_sha256']} on {inv['git_commit']}, "
                     f"this is {sha256_file(a.list)} on {git['git_commit']}")
        if inv["segments"] and inv["segments"][-1]["end"] == "end":
            sys.exit(f"refusing --resume: invocation {inv['invocation_id']} already ended")
        refuse_robot("--resume", a.resume, inv)
        return runs, a.resume, inv, "resume"
    if counted and a.out.exists():
        sys.exit(f"refusing an existing --out {a.out} for the counted list (use --resume {a.out} after a crash)")
    inv = {"invocation_id": str(uuid.uuid4()), "argv": a.argv, "list": str(a.list), "list_sha256": sha256_file(a.list),
           "git_commit": git["git_commit"], "out": str(a.out), "counted": counted, "segments": []}
    if robot_name() != ROBOT_DEFAULT:  # task 012: a TB4 invocation.json stays byte-identical
        inv["robot"] = robot_name()
    return runs, a.out, inv, "start"


def run_list(entries: list[dict[str, Any]], out: Path, inv: dict[str, Any], kind: str, run_entry: RunEntry,
             check_endpoint: Callable[[], str | None], ledger: Path | None,
             check_tree: Callable[[], str | None] = lambda: None) -> list[Any]:
    """One segment of an invocation: each entry (a resume skips every entry whose run dir exists), tree and endpoint
    checked before each; invocation.json rewritten around each run; ledger lines (counted only) at start and at the end."""
    out.mkdir(parents=True, exist_ok=True)
    seg = {"kind": kind, "start_utc": utc(), "end_utc": None, "entries_run": [], "end": None}
    inv["segments"].append(seg)
    write_inv(out, inv)
    if ledger is not None:
        ledger_line(ledger, inv, kind)
    results = []
    try:
        for e in entries:
            if kind == "resume" and (out / e["run_id"]).exists():
                continue  # finished or partial: scored as it stands, never re-run
            moved = check_tree()
            if moved is not None:
                seg["end"] = f"abort: {moved}"
                break
            down = check_endpoint()
            if down is not None:
                seg["end"] = f"abort: endpoint down ({down})"
                break
            seg["entries_run"].append(e["run_id"])
            write_inv(out, inv)
            meta = run_meta(e, None, None, {"invocation_id": inv["invocation_id"]})
            results.append(run_entry(e, out / e["run_id"], meta))
        else:
            seg["end"] = "end"
    except BaseException as ex:  # the runner itself died: say so in the ledger, then die
        seg["end"] = f"abort: {type(ex).__name__}: {ex}"[:300]
        raise
    finally:
        seg["end_utc"] = utc()
        write_inv(out, inv)
        if ledger is not None:
            ledger_line(ledger, inv, "end" if seg["end"] == "end" else "abort", reason=seg["end"],
                        entries_run=seg["entries_run"])
    return results


def run_sharded(entries: list[dict[str, Any]], out: Path, inv: dict[str, Any], kind: str,
                envs: list[dict[str, str]], spawn: Callable[[int, dict[str, str]], subprocess.Popen],
                ledger: Path | None, job: int | None = None, shard_of: dict[str, int] | None = None) -> list[bool]:
    """One segment of an invocation over len(envs) sims in parallel (module doc). Returns the scorer pass of each run.
    Cross-job part: `job` (segments in invocation.j<job>.json, ledger lines carry "job") and `shard_of` (the split's
    sim of each entry, instead of list order k % N)."""
    out.mkdir(parents=True, exist_ok=True)
    todo = [e for e in entries if not (kind == "resume" and (out / e["run_id"]).exists())]
    n, name, tag = len(envs), inv_name(job), {} if job is None else {"job": job}
    seg: dict[str, Any] = {"kind": kind, "start_utc": utc(), "end_utc": None, "entries_run": [], "end": None,
                           "shards": {str(i): env for i, env in enumerate(envs)},
                           "assignment": {e["run_id"]: k % n if shard_of is None else shard_of[e["run_id"]]
                                          for k, e in enumerate(todo)}, "shard_end": {}}
    inv["segments"].append(seg)
    write_inv(out, inv, name)  # the assignment is on disk before any worker starts
    if ledger is not None:
        ledger_line(ledger, inv, kind, **tag)
    events: queue.Queue = queue.Queue()
    procs: dict[int, subprocess.Popen] = {}
    passed: list[bool] = []

    def pump(i: int, p: subprocess.Popen) -> None:
        for line in p.stdout:
            events.put((i, line))
        events.put((i, None))
    try:
        for i in range(n):
            if i in seg["assignment"].values():
                procs[i] = spawn(i, envs[i])
                threading.Thread(target=pump, args=(i, procs[i]), daemon=True).start()
            else:
                seg["shard_end"][str(i)] = "end"  # nothing assigned: no worker
        open_ = len(procs)
        while open_:
            i, line = events.get()
            if line is None:
                open_ -= 1
            elif not line.startswith(PROTO):
                print(f"s{i} | {line}", end="", flush=True)
            else:
                ev = json.loads(line[len(PROTO):])
                if ev["ev"] == "run":
                    assert seg["assignment"].get(ev["run_id"]) == i and ev["run_id"] not in seg["entries_run"], ev
                    seg["entries_run"].append(ev["run_id"])
                    write_inv(out, inv, name)
                elif ev["ev"] == "done":
                    passed.append(ev["passed"])
                else:
                    seg["shard_end"][str(i)] = ev["end"]
        for i, p in procs.items():
            rc = p.wait()
            said = seg["shard_end"].get(str(i))
            if rc != 0 or said is None:
                seg["shard_end"][str(i)] = said if said not in (None, "end") else f"abort: worker died (rc {rc})"
        missing = sorted(set(seg["assignment"]) - {d.name for d in out.iterdir() if d.is_dir()})
        bad = [f"s{i} {r}" for i, r in sorted(seg["shard_end"].items(), key=lambda x: int(x[0])) if r != "end"]
        if missing and not bad:
            bad = [f"entries without a run dir after every worker ended: {missing}"]
        seg["end"] = "end" if not bad else f"abort: {'; '.join(bad)}"[:300]
    except BaseException as ex:  # the coordinator itself died: stop the workers, say so in the ledger, then die
        for p in procs.values():
            p.kill()
        seg["end"] = f"abort: {type(ex).__name__}: {ex}"[:300]
        raise
    finally:
        seg["end_utc"] = utc()
        write_inv(out, inv, name)
        if ledger is not None:
            ledger_line(ledger, inv, "end" if seg["end"] == "end" else "abort", reason=seg["end"],
                        entries_run=seg["entries_run"], **tag)
    return passed


def spawner(a: argparse.Namespace, out: Path) -> Callable[[int, dict[str, str]], subprocess.Popen]:
    """Start sim i's worker: WORKER --shard-worker OUT --shard i [--job j], in this env with sim i's domain and
    partition (and world)."""
    def spawn(i: int, env: dict[str, str]) -> subprocess.Popen:
        cmd = [*WORKER, "--list", str(a.list), "--shard-worker", str(out), "--shard", str(i)]
        cmd += [] if a.job is None else ["--job", str(a.job)]
        return subprocess.Popen(cmd + ["--rehearsal"] * a.rehearsal, env={**os.environ, **env},
                                stdout=subprocess.PIPE, text=True, bufsize=1)
    return spawn


def shard_worker(a: argparse.Namespace, runner: Callable[[], tuple[RunEntry, Callable[[], str | None],
                 Callable[[], None]]]) -> int:
    """Run the entries of OUT's last segment (of job a.job's file, cross-job) assigned to sim a.shard, in list order;
    events on stdout. An entry whose world is not this sim's SIM_WORLD (default closed) aborts the worker before it."""
    out = a.shard_worker
    inv = json.loads((out / inv_name(a.job)).read_text())
    sim_world = os.environ.get("SIM_WORLD", "closed")
    assign = inv["segments"][-1]["assignment"]
    assert inv["list_sha256"] == sha256_file(a.list), f"{a.list} is not the list of {out}"
    mine = [e for e in load_runs(a.list) if assign.get(e["run_id"]) == a.shard]
    start = {"git_commit": inv["git_commit"]}

    def emit(**kw: Any) -> None:
        print(PROTO + json.dumps(kw), flush=True)
    end = "end"
    run_entry, check_endpoint, close = runner()
    try:
        for e in mine:
            if (out / e["run_id"]).exists():
                end = f"abort: run dir {e['run_id']} exists (never run twice)"
                break
            if world_of(e) != sim_world:
                end = f"abort: entry {e['run_id']} needs world {world_of(e)}, this sim is SIM_WORLD {sim_world}"
                break
            moved = None if a.rehearsal else tree_moved(start, git_state(ROOT))
            if moved is not None:
                end = f"abort: {moved}"
                break
            down = check_endpoint()
            if down is not None:
                end = f"abort: endpoint down ({down})"
                break
            emit(ev="run", run_id=e["run_id"])
            meta = run_meta(e, a.job, a.shard, {"invocation_id": inv["invocation_id"]})
            o = run_entry(e, out / e["run_id"], meta)
            emit(ev="done", run_id=e["run_id"], passed=bool(o["score"]["pass"]))
    except BaseException as ex:
        end = f"abort: {type(ex).__name__}: {ex}"[:300]
        raise
    finally:
        emit(ev="end", end=end)
        close()
    return 0 if end == "end" else 1


def need_job_left(a: argparse.Namespace, job_left: Callable[[], float]) -> None:
    if a.min_job_left_min > 0 and (left := job_left()) < a.min_job_left_min:
        sys.exit(f"refusing to start: vLLM job has {left:.1f} min left, need {a.min_job_left_min:g}")


def split_runs(entries: list[dict[str, Any]], worlds: dict[tuple[int, int], str]) -> dict[str, list[int]]:
    """{run_id: [job, shard]}: list order, each entry to the next sim of its world in job-major order (each world
    cycles through its own sims); a world no sim runs -> SystemExit."""
    by_world: dict[str, list[tuple[int, int]]] = {}
    for js in sorted(worlds):
        by_world.setdefault(worlds[js], []).append(js)
    used = dict.fromkeys(by_world, 0)
    split = {}
    for e in entries:
        w = world_of(e)
        if w not in by_world:
            sys.exit(f"refusing --split: entry {e['run_id']} needs world {w!r}, no sim runs it ({sorted(by_world)})")
        split[e["run_id"]] = list(by_world[w][used[w] % len(by_world[w])])
        used[w] += 1
    return split


def split_invocation(a: argparse.Namespace, entries: list[dict[str, Any]], out: Path, inv: dict[str, Any],
                     ledger: Path | None) -> int:
    """--split: D/invocation.json with the split (+ plan ledger "start"), before any run; runs nothing."""
    inv |= {"split": split_runs(entries, a.shard_worlds), "split_shape": list(a.split_shape),
            "shard_worlds": {f"{j}:{i}": w for (j, i), w in sorted(a.shard_worlds.items())},
            "rehearsal": a.rehearsal}
    out.mkdir(parents=True, exist_ok=True)
    write_inv(out, inv)
    if ledger is not None:
        ledger_line(ledger.parent / f"{inv['invocation_id']}.plan.jsonl", inv, "start", split=inv["split"],
                    shard_worlds=inv["shard_worlds"])
    print(f"split {inv['invocation_id']}: {len(inv['split'])} entries over {a.split} (jobs x sims) -> {out}; "
          f"then per job: --job-part {out} --job J --shards N --shard-env ...")
    return 0


def job_part(a: argparse.Namespace, job_left: Callable[[], float]) -> int:
    """--job-part D --job j: job j's coordinator (module doc). Refuses another list / commit, a dirty tree (unless the
    split was a rehearsal), a D merged ok, a start of a started job, a resume of an ended one, sims whose world is not the
    split's; then run_sharded on job j's entries with the split's sims."""
    out, j = a.job_part, a.job
    top, git = json.loads((out / "invocation.json").read_text()), git_state(ROOT)
    if "split" not in top:
        sys.exit(f"refusing --job-part: {out} has no split (made by --out D --split K:N)")
    refuse_robot("--job-part", out, top)
    if (out / "merged.json").is_file() and json.loads((out / "merged.json").read_text())["ok"]:
        sys.exit(f"refusing --job-part: {out} is already merged ok")
    if (top["list_sha256"], top["git_commit"]) != (sha256_file(a.list), git["git_commit"]):
        sys.exit(f"refusing --job-part: {out} is list {top['list_sha256']} on {top['git_commit']}, "
                 f"this is {sha256_file(a.list)} on {git['git_commit']}")
    a.rehearsal = top["rehearsal"]
    if not a.rehearsal and git["git_dirty"]:
        sys.exit("refusing counted runs on a dirty tree (commit first)")
    k, n = top["split_shape"]
    if not (0 <= j < k and a.shards == n):
        sys.exit(f"refusing --job-part: split is {k} jobs x {n} sims, got --job {j} --shards {a.shards}")
    worlds = {i: env.get("SIM_WORLD", os.environ.get("SIM_WORLD", "closed")) for i, env in enumerate(a.shard_envs)}
    want = {i: top["shard_worlds"][f"{j}:{i}"] for i in range(n)}
    if worlds != want:
        sys.exit(f"refusing --job-part: sims' worlds {worlds} != the split's {want} for job {j}")
    path = out / inv_name(j)
    if a.job_resume:
        if not path.is_file():
            sys.exit(f"refusing --resume: job {j} never started ({path} missing)")
        inv, kind = json.loads(path.read_text()), "resume"
        if [seg["end"] for seg in inv["segments"]][-1:] == ["end"]:
            sys.exit(f"refusing --resume: job {j} of {top['invocation_id']} already ended")
    else:
        if path.exists():
            sys.exit(f"refusing --job-part: job {j} already started ({path}); use --resume")
        inv, kind = {**{x: top[x] for x in ("invocation_id", "list", "list_sha256", "git_commit", "out", "counted",
                                             "rehearsal")}, "job": j, "argv": a.argv, "segments": []}, "start"
    need_job_left(a, job_left)
    mine = {r: i for r, (jj, i) in top["split"].items() if jj == j}
    entries = [e for e in load_runs(a.list) if e["run_id"] in mine]
    ledger = (a.ledger or default_ledger()).parent / f"{top['invocation_id']}.j{j}.jsonl" if top["counted"] else None
    passed = run_sharded(entries, out, inv, kind, a.shard_envs, spawner(a, out), ledger, job=j, shard_of=mine)
    end = inv["segments"][-1]["end"]
    print(f"{sum(passed)}/{len(passed)} scorer pass ({a.list.name}, job {j}, {kind}: {end}); "
          f"after every job: probes/phase3_run.py --merge {out}")
    return 0 if end == "end" else 1


def merge(out: Path, ledger: Path) -> int:
    """--merge D (module doc): per split entry exactly one run dir (by run.json run_id, else its dir name) with D's
    invocation_id and the split's job / shard, run by no other job; no run dir outside the split; each job with
    entries ended its last segment "end". Counted: appends the plan and job ledger lines not yet appended (counts kept
    in merged.json) in time order to `ledger`, then one "end" (ok) / "abort" line. Writes D/merged.json."""
    inv = json.loads((out / "invocation.json").read_text())
    if "split" not in inv:
        sys.exit(f"refusing --merge: {out} has no split (a single-job invocation needs no merge)")
    refuse_robot("--merge", out, inv)
    mpath = out / "merged.json"
    prev = json.loads(mpath.read_text()) if mpath.is_file() else {}
    if prev.get("ok"):
        sys.exit(f"refusing --merge: {out} is already merged ok")
    iid, split = inv["invocation_id"], inv["split"]
    jobs = {j: json.loads((out / inv_name(j)).read_text()) for j in range(inv["split_shape"][0])
            if (out / inv_name(j)).is_file()}
    dirs: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    bare = []  # run dirs without run.json
    for d in sorted(x for x in out.iterdir() if x.is_dir()):
        if (d / "run.json").is_file():
            r = json.loads((d / "run.json").read_text())
            dirs.setdefault(r.get("run_id", d.name), []).append((d.name, r))
        else:
            bare.append(d.name)
    ran: dict[str, set[int]] = {}
    for j, ji in jobs.items():
        for seg in ji["segments"]:
            for rid in seg["entries_run"]:
                ran.setdefault(rid, set()).add(j)
    entries, problems = {}, []
    for rid, (j, i) in split.items():
        got = dirs.get(rid, [])
        r = got[0][1] if len(got) == 1 else {}
        if not got:
            st = "missing" + (" (run dir without run.json)" if rid in bare else "")
        elif len(got) > 1:
            st = f"duplicated: run dirs {[x for x, _ in got]}"
        elif r.get("invocation_id") != iid:
            st = f"wrong invocation_id {r.get('invocation_id')}"
        elif (r.get("job"), r.get("shard"), got[0][0]) != (j, i, rid):
            st = f"misplaced: run dir {got[0][0]} ran on job {r.get('job')} sim {r.get('shard')}, split [{j}, {i}]"
        elif ran.get(rid, {j}) != {j}:
            st = f"duplicated: run by jobs {sorted(ran[rid])}, split job {j}"
        else:
            st = "ok"
        entries[rid] = {"status": st, "job": j, "shard": i}
        problems += [] if st == "ok" else [f"{rid}: {st}"]
    problems += [f"run dir {x} not in the split" for x in sorted(set(dirs) - set(split))]
    problems += [f"run dir {x} (no run.json) not in the split" for x in bare if x not in split]
    ends = {str(j): [seg["end"] for seg in jobs[j]["segments"]] if j in jobs else None
            for j in sorted({j for j, _ in split.values()})}
    problems += [f"job {j}: segment ends {e}, want the last 'end'" for j, e in ends.items() if not e or e[-1] != "end"]
    ok, counts = not problems, dict(prev.get("ledger_lines", {}))
    if inv["counted"]:
        new = []
        for key in ["plan", *(f"j{j}" for j in sorted(jobs))]:
            f = ledger.parent / f"{iid}.{key}.jsonl"
            lines = f.read_text().splitlines() if f.is_file() else []
            new += lines[counts.get(key, 0):]
            counts[key] = len(lines)
        new.sort(key=lambda x: json.loads(x)["utc"])  # stable: plan, then job order within one second
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a") as f:
            f.writelines(x + "\n" for x in new)
        ledger_line(ledger, inv, "end" if ok else "abort", reason="merged" if ok else "; ".join(problems)[:300],
                    merged=str(mpath))
    m = {"invocation_id": iid, "ok": ok, "merged_utc": utc(), "entries": entries, "job_segment_ends": ends,
         "problems": problems, "ledger_lines": counts}
    mpath.write_text(json.dumps(m, indent=1) + "\n")
    print(f"merge {iid}: {'ok' if ok else 'NOT ok'}" + "".join(f"\n  {x}" for x in problems))
    return 0 if ok else 1


def entry_runner(ad: Any, gt_pose: Callable[[], tuple[float, float, float]],
                 planner: Callable[[int], Any], scene_meta: dict[str, Any] | None = None, **run_kw: Any) -> RunEntry:
    """run_entry: one list entry -> adapters.run_task.run_one (start place, sampling seed, layout seed) on `ad`;
    `scene_meta` (more run.json keys) and `run_kw` (run_one kwargs): adapters/robots.make's, both empty for the TB4."""
    from adapters.run_task import run_one
    from agent.runtime import MALFORMED_LIMIT

    def run_entry(r: dict[str, Any], d: Path, meta: dict[str, Any]) -> dict[str, Any]:
        o = run_one(ad, gt_pose, task_by_id(r["task_id"]), planner(r["seed"]), d, start_place=r["start_place"],
                    watch_initialpose=True, malformed_limit=MALFORMED_LIMIT,
                    extra_meta={**meta, **scene_meta} if scene_meta else meta,
                    layout_seed=r.get("layout_seed"), layout_spec=r.get("layout_spec"), **run_kw)
        print(f"{r['run_id']} {r['task_id']:18s} from {r['start_place']:9s} seed {r['seed']:10d}  "
              f"setup {'ok' if o['setup']['reached'] else 'NOT REACHED'}  {o['outcome']:16s} steps {o['steps']:2d}  "
              f"pass {o['score']['pass']}  {o['score']['label'] or ''} {o['score']['label_why'] or ''}", flush=True)
        return o
    return run_entry


def sim_runner() -> tuple[RunEntry, Callable[[], str | None], Callable[[], None]]:
    """(run_entry, check_endpoint, close) on the clinic sim with the VLM from VLM_BASE_URL / VLM_API_KEY / VLM_MODEL;
    the robot is adapters/robots.make(env ROBOT, default turtlebot4)."""
    from agent.planners import VlmConfig, VlmPlanner
    cfg = VlmConfig.from_env()
    rb = make(robot_name())
    run_entry = entry_runner(rb.ad, rb.gt_pose, lambda seed: VlmPlanner(dataclasses.replace(cfg, seed=seed)),
                             rb.scene_meta, **rb.run_kw)
    return run_entry, lambda: endpoint_down(cfg.base_url, cfg.api_key), rb.close


def main(argv: list[str] | None = None, runner: Callable[[], tuple[RunEntry, Callable[[], str | None],
         Callable[[], None]]] = sim_runner, job_left: Callable[[], float] = job_left_min) -> int:
    a = parse(argv)
    if a.shard_worker is not None:
        return shard_worker(a, runner)
    if a.merge is not None:
        return merge(a.merge, a.ledger or default_ledger())
    if a.job_part is not None:
        return job_part(a, job_left)
    entries, out, inv, kind = plan(a)
    if a.split is not None:
        return split_invocation(a, entries, out, inv, (a.ledger or default_ledger()) if inv["counted"] else None)
    need_job_left(a, job_left)
    ledger = (a.ledger or default_ledger()) if inv["counted"] else None
    if a.shards:
        passed = run_sharded(entries, out, inv, kind, a.shard_envs, spawner(a, out), ledger)
        print(f"{sum(passed)}/{len(passed)} scorer pass ({a.list.name}, {kind}, {a.shards} shards: "
              f"{inv['segments'][-1]['end']}); verdicts: probes/phase3_gates.py all --out {out}")
        return 0 if inv["segments"][-1]["end"] == "end" else 1
    start = {"git_commit": inv["git_commit"]}

    def check_tree() -> str | None:  # non-rehearsal: the tree must stay the invocation's commit, clean
        return None if a.rehearsal else tree_moved(start, git_state(ROOT))
    run_entry, check_endpoint, close = runner()
    try:
        res = run_list(entries, out, inv, kind, run_entry, check_endpoint, ledger, check_tree)
    finally:
        close()
    n_pass = sum(bool(o["score"]["pass"]) for o in res)
    print(f"{n_pass}/{len(res)} scorer pass ({a.list.name}, {kind}: {inv['segments'][-1]['end']}); "
          f"verdicts: probes/phase3_gates.py all --out {out}")
    return 0 if inv["segments"][-1]["end"] == "end" else 1


if __name__ == "__main__":
    raise SystemExit(main())
