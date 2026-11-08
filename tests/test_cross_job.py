"""probes/phase3_run.py cross-job counted lists (task 007 Step 0, hpc/PERUN_SIM.md section 5): --split fixes
{run_id: [job, sim]} before any run, --job-part runs one job's entries on their split sims through the sharded worker
protocol, --merge checks and merges, probes/phase3_gates.require_merged refuses an unmerged / bad D. No sim, no VLM:
2 "jobs" x 2 sims in one process tree, the workers are real subprocesses of a fake worker script (phase3_run.main
with a fake runner); the gate test runs adapters.run_task.run_one on the FakeAdapter through
tests/test_phase3_gates.make_run."""
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from probes import phase3_gates as g  # noqa: E402
from probes import phase3_run as pr  # noqa: E402
from test_phase3_shards import OK_NO_ROS, verdicts, world  # noqa: E402,F401  (world: module fixture)

IDS = [f"c{i:02d}" for i in range(1, 11)]
# job-major sims j0s0, j0s1, j1s0, j1s1, cycled in list order (all closed)
SPLIT = {r: [k % 4 // 2, k % 2] for k, r in enumerate(IDS)}
ENVS = {0: ["0=40:j0s0", "1=41:j0s1"], 1: ["0=50:j1s0", "1=51:j1s1"]}

# The worker: phase3_run.main(--shard-worker ... --job j) with git_state, the host name and the runner faked. FAKE
# (json): mode "plain" (run.json = meta) or "gates" (make_run on the fake world, c01-c07 correct, c08-c10 wrong);
# down_at: endpoint down before that entry. FAKE_HOST: the node name.
WORKER = r'''
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1)); sys.path.insert(0, sys.argv.pop(1))
from probes import phase3_run as pr
C = json.loads(os.environ["FAKE"])
pr.git_state = lambda repo: {"git_commit": "c0ffee", "git_dirty": False}
pr.socket.gethostname = lambda: os.environ["FAKE_HOST"]

def runner():
    a = pr.parse(sys.argv[1:])
    seg = json.loads((a.shard_worker / pr.inv_name(a.job)).read_text())["segments"][-1]

    def run_entry(e, d, meta):
        assert seg["assignment"][e["run_id"]] == a.shard  # on disk before the run
        if C["mode"] == "gates":
            from test_phase3_gates import correct, make_run, wrong
            w = {k: Path(v) for k, v in C["world"].items()}
            make_run(w, d, e, (correct if int(e["run_id"][1:]) <= 7 else wrong)(e["task_id"]), meta)
        else:
            d.mkdir(parents=True)
            (d / "run.json").write_text(json.dumps(meta))
        return {"score": {"pass": True}}

    def check():
        nxt = [e for e in pr.load_runs(a.list) if seg["assignment"].get(e["run_id"]) == a.shard
               and not (a.shard_worker / e["run_id"]).exists()]
        return "URLError: refused" if nxt and nxt[0]["run_id"] == C.get("down_at") else None
    return run_entry, check, lambda: None

sys.exit(pr.main(sys.argv[1:], runner=runner))
'''


@pytest.fixture(autouse=True)
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr, "git_state", lambda repo: {"git_commit": "c0ffee", "git_dirty": False})
    w = tmp_path / "fake_worker.py"
    w.write_text(WORKER)
    monkeypatch.setattr(pr, "WORKER", [sys.executable, str(w), str(ROOT), str(ROOT / "tests")])
    monkeypatch.delenv("SIM_WORLD", raising=False)
    cfg(monkeypatch, tmp_path)


def cfg(mp: pytest.MonkeyPatch, tmp: Path, **kw: Any) -> None:
    mp.setenv("FAKE", json.dumps({"mode": "plain", "tmp": str(tmp), **kw}))


def main(tmp: Path, *args: str) -> int:
    return pr.main([*args, "--ledger", str(tmp / "ledger.jsonl")],
                   runner=lambda: pytest.fail("the coordinator never drives a sim"),
                   job_left=lambda: pytest.fail("job_left called"))


def job(mp: pytest.MonkeyPatch, tmp: Path, out: Path, j: int, *args: str, envs: list[str] | None = None) -> int:
    """Job j's coordinator as inside SLURM job 10<j> on node n<j>."""
    mp.setenv("SLURM_JOB_ID", f"10{j}")
    mp.setenv("FAKE_HOST", f"n{j}")
    envs = ENVS[j] if envs is None else envs
    ev = [x for e in envs for x in ("--shard-env", e)]
    return main(tmp, "--job-part", str(out), "--job", str(j), "--shards", str(len(envs)), *ev, *args)


def jl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text().splitlines()] if p.is_file() else []


def inv_of(out: Path, name: str = "invocation.json") -> dict[str, Any]:
    return json.loads((out / name).read_text())


def runs_present(out: Path) -> list[str]:
    return sorted(d.name for d in out.iterdir() if d.is_dir())


def split(tmp: Path, *args: str) -> tuple[Path, str]:
    out = tmp / "out"
    assert main(tmp, "--out", str(out), "--split", "2:2", *args) == 0
    return out, inv_of(out)["invocation_id"]


def both_jobs(mp: pytest.MonkeyPatch, tmp: Path, *args: str) -> tuple[Path, str]:
    out, iid = split(tmp, *args)
    assert job(mp, tmp, out, 0) == 0 and job(mp, tmp, out, 1) == 0
    return out, iid


# ------------------------------------------------------------------------------------------------------ split
def test_split_is_fixed_and_recorded_before_any_run(tmp_path: Path) -> None:
    out, iid = split(tmp_path)
    inv = inv_of(out)
    assert inv["split"] == SPLIT and inv["split_shape"] == [2, 2] and inv["segments"] == [] and inv["counted"]
    assert inv["shard_worlds"] == {"0:0": "closed", "0:1": "closed", "1:0": "closed", "1:1": "closed"}
    assert runs_present(out) == [] and sorted(p.name for p in out.iterdir()) == ["invocation.json"]
    plan = jl(tmp_path / f"{iid}.plan.jsonl")
    assert [x["event"] for x in plan] == ["start"] and plan[0]["split"] == SPLIT and "job" not in plan[0]
    assert not (tmp_path / "ledger.jsonl").exists()  # the counted ledger is written by --merge only


@pytest.mark.parametrize("args", [["--rehearsal"], ["--only", "c01"]])
def test_split_keeps_the_counted_refusals(tmp_path: Path, args: list[str]) -> None:
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="--rehearsal / --only"):
        main(tmp_path, "--out", str(out), "--split", "2:2", *args)
    assert not out.exists()
    out.mkdir()
    with pytest.raises(SystemExit, match="existing --out"):
        main(tmp_path, "--out", str(out), "--split", "2:2")
    assert not any(out.iterdir()) and not list(tmp_path.glob("*.jsonl"))


def test_split_refuses_a_dirty_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr, "git_state", lambda repo: {"git_commit": "c0ffee", "git_dirty": True})
    with pytest.raises(SystemExit, match="dirty tree"):
        main(tmp_path, "--out", str(tmp_path / "out"), "--split", "2:2")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("args", [["--split", "2"], ["--split", "0:2"], ["--split", "2:2", "--shard-world", "2:0=x"],
                                  ["--split", "2:2", "--shard-world", "0:0=x", "--shard-world", "0:0=y"]])
def test_split_spec_errors(tmp_path: Path, args: list[str]) -> None:
    with pytest.raises(AssertionError):
        main(tmp_path, "--out", str(tmp_path / "out"), *args)
    assert not (tmp_path / "out").exists()


# ------------------------------------------------------------------------------------------------- job parts
def test_each_job_runs_only_its_entries_then_merge_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out, iid = split(tmp_path)
    top = (out / "invocation.json").read_bytes()
    with pytest.raises(SystemExit, match="not merged"):
        g.require_merged(out)
    assert job(monkeypatch, tmp_path, out, 0) == 0
    assert runs_present(out) == sorted(r for r, (j, _) in SPLIT.items() if j == 0)
    assert (out / "invocation.json").read_bytes() == top  # a job part never writes D/invocation.json
    j0 = inv_of(out, "invocation.j0.json")
    assert j0["invocation_id"] == iid and j0["job"] == 0 and len(j0["segments"]) == 1
    assert j0["segments"][0]["assignment"] == {r: i for r, (j, i) in SPLIT.items() if j == 0}
    assert job(monkeypatch, tmp_path, out, 1) == 0
    assert runs_present(out) == IDS and (out / "invocation.json").read_bytes() == top
    for r, (j, i) in SPLIT.items():
        assert json.loads((out / r / "run.json").read_text()) == {
            "invocation_id": iid, "run_id": r, "slurm_job_id": f"10{j}", "node": f"n{j}", "job": j, "shard": i,
            "world": "closed"}
    for j in (0, 1):  # one writer per ledger file: job j's lines only, all tagged with j
        lines = jl(tmp_path / f"{iid}.j{j}.jsonl")
        assert [x["event"] for x in lines] == ["start", "end"] and {x["job"] for x in lines} == {j}
    assert not (tmp_path / "ledger.jsonl").exists()
    assert main(tmp_path, "--merge", str(out)) == 0
    m = inv_of(out, "merged.json")
    assert m["ok"] and m["problems"] == [] and m["invocation_id"] == iid
    assert {r: x["status"] for r, x in m["entries"].items()} == dict.fromkeys(IDS, "ok")
    assert m["job_segment_ends"] == {"0": ["end"], "1": ["end"]}
    led = jl(tmp_path / "ledger.jsonl")
    assert [(x["event"], x.get("job")) for x in led] == [("start", None), ("start", 0), ("end", 0), ("start", 1),
                                                         ("end", 1), ("end", None)]
    assert [x["utc"] for x in led] == sorted(x["utc"] for x in led) and {x["invocation_id"] for x in led} == {iid}
    g.require_merged(out)
    with pytest.raises(SystemExit, match="already merged"):
        main(tmp_path, "--merge", str(out))
    with pytest.raises(SystemExit, match="already merged"):
        job(monkeypatch, tmp_path, out, 0, "--resume")


def test_job_part_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out, _ = split(tmp_path)
    with pytest.raises(SystemExit, match="never started"):
        job(monkeypatch, tmp_path, out, 0, "--resume")
    with pytest.raises(SystemExit, match="split is 2 jobs x 2 sims"):
        job(monkeypatch, tmp_path, out, 2, envs=ENVS[0])
    with pytest.raises(SystemExit, match="split is 2 jobs x 2 sims"):
        job(monkeypatch, tmp_path, out, 0, envs=["0=40:a", "1=41:b", "2=42:c"])
    with pytest.raises(SystemExit, match="refusing --resume: .* split over jobs"):
        main(tmp_path, "--resume", str(out), "--shards", "2", "--shard-env", "0=40:a", "--shard-env", "1=41:b")
    monkeypatch.setattr(pr, "git_state", lambda repo: {"git_commit": "beef", "git_dirty": False})
    with pytest.raises(SystemExit, match="refusing --job-part: .* on c0ffee"):
        job(monkeypatch, tmp_path, out, 0)
    monkeypatch.setattr(pr, "git_state", lambda repo: {"git_commit": "c0ffee", "git_dirty": True})
    with pytest.raises(SystemExit, match="dirty tree"):
        job(monkeypatch, tmp_path, out, 0)
    assert runs_present(out) == [] and not (out / "invocation.j0.json").exists()
    monkeypatch.setattr(pr, "git_state", lambda repo: {"git_commit": "c0ffee", "git_dirty": False})
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "invocation.json").write_text(json.dumps({"invocation_id": "x"}))
    with pytest.raises(SystemExit, match="--merge: .* no split"):
        main(tmp_path, "--merge", str(tmp_path / "plain"))


def test_resume_of_one_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out, iid = split(tmp_path)
    assert job(monkeypatch, tmp_path, out, 0) == 0
    cfg(monkeypatch, tmp_path, down_at="c08")  # job 1 sim 1: c04, c08 -> c08 left
    assert job(monkeypatch, tmp_path, out, 1) == 1
    assert runs_present(out) == [r for r in IDS if r != "c08"]
    with pytest.raises(SystemExit, match="already started"):
        job(monkeypatch, tmp_path, out, 1)
    with pytest.raises(SystemExit, match="already ended"):
        job(monkeypatch, tmp_path, out, 0, "--resume")
    assert main(tmp_path, "--merge", str(out)) == 1  # merge before the resume: c08 missing, job 1 aborted
    m = inv_of(out, "merged.json")
    assert not m["ok"] and m["entries"]["c08"]["status"] == "missing" and "job 1" in m["problems"][-1]
    with pytest.raises(SystemExit, match="ok False"):
        g.require_merged(out)
    cfg(monkeypatch, tmp_path)
    assert job(monkeypatch, tmp_path, out, 1, "--resume") == 0
    seg = inv_of(out, "invocation.j1.json")["segments"]
    assert [s["kind"] for s in seg] == ["start", "resume"] and seg[1]["assignment"] == {"c08": 1}
    assert seg[1]["entries_run"] == ["c08"] and runs_present(out) == IDS
    assert main(tmp_path, "--merge", str(out)) == 0
    g.require_merged(out)
    led = jl(tmp_path / "ledger.jsonl")  # each plan / job line once, though merged twice
    assert [x["event"] for x in led if "job" not in x] == ["start", "abort", "end"]
    assert [(x["event"], x["job"]) for x in led if "job" in x] == [("start", 0), ("end", 0), ("start", 1),
                                                                   ("abort", 1), ("resume", 1), ("end", 1)]


# ------------------------------------------------------------------------------------------------------ worlds
def world_list(tmp: Path) -> Path:
    """Dev entries d01-d06 (not the counted list), d02 and d05 in cabinet_open."""
    runs = pr.load_runs(ROOT / "tasks" / "phase3_dev.yaml")[:6]
    for r in runs:
        if r["run_id"] in ("d02", "d05"):
            r["world"] = "cabinet_open"
    p = tmp / "worlds.yaml"
    p.write_text(yaml.safe_dump(runs))
    return p


def test_world_routing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lst = world_list(tmp_path)
    with pytest.raises(SystemExit, match="needs world 'cabinet_open', no sim runs it"):
        main(tmp_path, "--list", str(lst), "--out", str(tmp_path / "out"), "--split", "2:2", "--rehearsal")
    assert not (tmp_path / "out").exists()
    out, iid = split(tmp_path, "--list", str(lst), "--rehearsal", "--shard-world", "1:1=cabinet_open")
    assert inv_of(out)["split"] == {"d01": [0, 0], "d02": [1, 1], "d03": [0, 1], "d04": [1, 0], "d05": [1, 1],
                                    "d06": [0, 0]}
    with pytest.raises(SystemExit, match="worlds .* != the split's"):  # sim 1 of job 1 not started in cabinet_open
        job(monkeypatch, tmp_path, out, 1, "--list", str(lst))
    assert job(monkeypatch, tmp_path, out, 0, "--list", str(lst)) == 0
    assert job(monkeypatch, tmp_path, out, 1, "--list", str(lst), envs=["0=50:a", "1=51:b:cabinet_open"]) == 0
    assert {r: json.loads((out / r / "run.json").read_text())["world"] for r in runs_present(out)} == {
        "d01": "closed", "d02": "cabinet_open", "d03": "closed", "d04": "closed", "d05": "cabinet_open",
        "d06": "closed"}
    assert inv_of(out, "invocation.j1.json")["segments"][0]["shards"]["1"]["SIM_WORLD"] == "cabinet_open"
    assert main(tmp_path, "--merge", str(out)) == 0 and not list(tmp_path.glob("*.jsonl"))  # not counted: no ledger


def test_worker_refuses_an_entry_of_another_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Single-job sharded run (k % 2), sims in closed (SIM_WORLD unset): d02 (sim 1) and d05 (sim 0), both
    cabinet_open, are refused and left without a run dir, which stops their sims."""
    monkeypatch.setenv("FAKE_HOST", "n0")
    out = tmp_path / "out"
    assert main(tmp_path, "--list", str(world_list(tmp_path)), "--out", str(out), "--rehearsal", "--shards", "2",
                "--shard-env", "0=40:a", "--shard-env", "1=41:b") == 1
    seg = inv_of(out)["segments"][0]
    assert seg["shard_end"]["1"] == "abort: entry d02 needs world cabinet_open, this sim is SIM_WORLD closed"
    assert seg["shard_end"]["0"] == "abort: entry d05 needs world cabinet_open, this sim is SIM_WORLD closed"
    assert runs_present(out) == ["d01", "d03"]


# ------------------------------------------------------------------------------------------------------- merge
def plant_other_invocation(out: Path) -> str:
    rj = out / "c04" / "run.json"
    rj.write_text(json.dumps({**json.loads(rj.read_text()), "invocation_id": "another-run"}))
    return "c04: wrong invocation_id another-run"


def plant_duplicate(out: Path) -> str:
    shutil.copytree(out / "c04", out / "c04_again")
    return "c04: duplicated: run dirs ['c04', 'c04_again']"


def plant_missing(out: Path) -> str:
    shutil.rmtree(out / "c04")
    return "c04: missing"


def plant_misplaced(out: Path) -> str:  # c04 is split to job 1 sim 1; its run.json says job 0 sim 1
    rj = out / "c04" / "run.json"
    rj.write_text(json.dumps({**json.loads(rj.read_text()), "job": 0}))
    return "c04: misplaced"


def plant_run_by_two_jobs(out: Path) -> str:
    p = out / "invocation.j0.json"
    inv = json.loads(p.read_text())
    inv["segments"][0]["entries_run"].append("c04")
    p.write_text(json.dumps(inv))
    return "c04: duplicated: run by jobs [0, 1]"


def plant_foreign_dir(out: Path) -> str:
    shutil.copytree(out / "c04", out / "x99")
    rj = out / "x99" / "run.json"
    rj.write_text(json.dumps({**json.loads(rj.read_text()), "run_id": "x99"}))
    return "run dir x99 not in the split"


@pytest.mark.parametrize("plant", [plant_other_invocation, plant_duplicate, plant_missing, plant_misplaced,
                                   plant_run_by_two_jobs, plant_foreign_dir])
def test_merge_refuses_missing_duplicated_misplaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    plant: Any) -> None:
    out, iid = both_jobs(monkeypatch, tmp_path)
    want = plant(out)
    assert main(tmp_path, "--merge", str(out)) == 1
    m = inv_of(out, "merged.json")
    assert not m["ok"] and [p for p in m["problems"] if p.startswith(want)], m["problems"]
    assert jl(tmp_path / "ledger.jsonl")[-1]["event"] == "abort"
    for gate in (lambda: g.require_merged(out), lambda: g.g2_verdict(out, pr.load_runs(pr.COUNTED), True),
                 lambda: g.g3_verdict(out, pr.load_runs(pr.COUNTED), True, OK_NO_ROS),
                 lambda: g.g1_verdict(out, pr.load_runs(pr.COUNTED), True, tmp_path / "ledger.jsonl"),
                 lambda: g.measure(out, pr.load_runs(pr.COUNTED))):
        with pytest.raises(SystemExit, match="merged.json .* ok False"):
            gate()


def test_require_merged_passes_a_single_job_out(tmp_path: Path) -> None:
    g.require_merged(tmp_path)  # no invocation.json at all
    (tmp_path / "invocation.json").write_text(json.dumps({"invocation_id": "x", "segments": []}))
    g.require_merged(tmp_path)


def test_require_merged_wants_this_invocations_merge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out, _ = both_jobs(monkeypatch, tmp_path)
    assert main(tmp_path, "--merge", str(out)) == 0
    m = inv_of(out, "merged.json")
    (out / "merged.json").write_text(json.dumps({**m, "invocation_id": "another"}))
    with pytest.raises(SystemExit, match="merged.json of another"):
        g.require_merged(out)


# ------------------------------------------------------------------------------------------------------- gates
def test_gates_pass_on_a_merged_cross_job_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              world: dict[str, Path]) -> None:
    cfg(monkeypatch, tmp_path, mode="gates", world={k: str(v) for k, v in world.items()})
    out, iid = both_jobs(monkeypatch, tmp_path)
    led = tmp_path / "ledger.jsonl"
    with pytest.raises(SystemExit, match="not merged"):
        verdicts(out, led, world)
    assert main(tmp_path, "--merge", str(out)) == 0
    v = verdicts(out, led, world)
    assert v["g1"]["pass"] and v["g1"]["passed"] == 7 and v["g2"]["pass"] and v["g3"]["pass"], v["g1"]
    assert v["g1"]["invocation_problems"] == []
    m = g.measure(out, pr.load_runs(pr.COUNTED))
    assert sorted(m["sim_per_wall_by_node"]) == ["n0", "n1"]
    assert m["sim_per_wall_by_node"]["n0"]["n"] == 6 and m["sim_per_wall_by_node"]["n1"]["n"] == 4
    assert {r["node"] for r in m["runs"]} == {"n0", "n1"}
    slow = [r["run_id"] for r in m["runs"] if r["sim_per_wall"] < g.RTF_FLAG]
    assert m[f"sim_per_wall_below_{g.RTF_FLAG}"] == slow
