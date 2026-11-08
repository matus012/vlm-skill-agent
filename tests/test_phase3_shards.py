"""probes/phase3_run.py sharded mode (task 006, hpc/PERUN_SIM.md section 3): one coordinator, N worker subprocesses
into one OUT, each entry once, one invocation, one ledger; and probes/phase3_gates.py G1 / G2 / G3 unchanged on a
sharded run. No sim, no VLM: the workers are real subprocesses of a fake worker script (phase3_run.main with a fake
runner); the gate tests run adapters.run_task.run_one on the FakeAdapter through tests/test_phase3_gates.make_run."""
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from probes import phase3_gates as g  # noqa: E402
from probes import phase3_run as pr  # noqa: E402
from test_phase3_gates import LAYOUT, PLACES, ROOMS, TRUTH, correct, make_run, wrong  # noqa: E402

DEV = ROOT / "tasks" / "phase3_dev.yaml"
IDS = [f"c{i:02d}" for i in range(1, 11)]
ENVS = ["0=40:j7s0", "1=41:j7s1", "2=42:j7s2"]
OK_NO_ROS = {"returncode": 0, "summary": "2 passed"}

# The worker: phase3_run.main(--shard-worker ...) with git_state and the runner faked. FAKE (json, env) says what to do:
# mode "plain" (run.json = meta, env.json = the worker's ROS_DOMAIN_ID / GZ_PARTITION) or "gates" (make_run on the fake
# world, c01-c07 correct, c08-c10 wrong); raise_at / exit_at: die inside that entry (exception / os._exit 3);
# down_at: endpoint down before that entry; dirty_after: the tree is dirty once that entry ran; rogue: announce an
# entry of another sim first.
WORKER = r'''
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1)); sys.path.insert(0, sys.argv.pop(1))
from probes import phase3_run as pr
C = json.loads(os.environ["FAKE"])
dirty = Path(C["tmp"]) / "dirty"
pr.git_state = lambda repo: {"git_commit": "c0ffee", "git_dirty": dirty.exists()}

def runner():
    a = pr.parse(sys.argv[1:])
    seg = json.loads((a.shard_worker / "invocation.json").read_text())["segments"][-1]
    me = {"ROS_DOMAIN_ID": os.environ["ROS_DOMAIN_ID"], "GZ_PARTITION": os.environ["GZ_PARTITION"]}
    assert seg["shards"][str(a.shard)] == me, (seg["shards"], a.shard, me)
    if C.get("rogue"):
        other = next(r for r, i in seg["assignment"].items() if i != a.shard)
        print(pr.PROTO + json.dumps({"ev": "run", "run_id": other}), flush=True)
    downs = iter([])

    def run_entry(e, d, meta):
        assert seg["assignment"][e["run_id"]] == a.shard  # on disk before the run
        if C["mode"] == "gates":
            from test_phase3_gates import correct, make_run, wrong
            world = {k: Path(v) for k, v in C["world"].items()}
            steps = (correct if int(e["run_id"][1:]) <= 7 else wrong)(e["task_id"])
            make_run(world, d, e, steps, meta)
        else:
            d.mkdir(parents=True)
            if e["run_id"] == C.get("raise_at"):
                raise RuntimeError("sim died")
            if e["run_id"] == C.get("exit_at"):
                os._exit(3)
            (d / "run.json").write_text(json.dumps(meta))
            (d / "env.json").write_text(json.dumps(me))
        if e["run_id"] == C.get("dirty_after"):
            dirty.touch()
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
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SIM_WORLD", raising=False)
    cfg(monkeypatch, tmp_path)


def cfg(mp: pytest.MonkeyPatch, tmp: Path, **kw: Any) -> None:
    mp.setenv("FAKE", json.dumps({"mode": "plain", "tmp": str(tmp), **kw}))


def run(tmp: Path, *args: str, envs: list[str] = ENVS, ledger: Path | None = None) -> int:
    ev = [x for e in envs for x in ("--shard-env", e)]
    return pr.main([*args, "--shards", str(len(envs)), *ev, "--ledger", str(ledger or tmp / "ledger.jsonl")],
                   runner=lambda: pytest.fail("the coordinator never drives a sim"),
                   job_left=lambda: pytest.fail("job_left called"))


def ledger(tmp: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (tmp / "ledger.jsonl").read_text().splitlines()]


def inv_of(out: Path) -> dict[str, Any]:
    return json.loads((out / "invocation.json").read_text())


def runs_present(out: Path) -> list[str]:
    return sorted(d.name for d in out.iterdir() if d.is_dir())


# ------------------------------------------------------------------------------------------------------ runner
def test_sharded_counted_run_each_entry_once_in_its_sim(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert run(tmp_path, "--out", str(out)) == 0
    inv = inv_of(out)
    seg = inv["segments"][0]
    assert seg["assignment"] == {r: k % 3 for k, r in enumerate(IDS)}  # list order, k % N
    assert seg["shard_end"] == {"0": "end", "1": "end", "2": "end"} and seg["end"] == "end"
    assert sorted(seg["entries_run"]) == IDS and len(seg["entries_run"]) == 10
    assert seg["shards"]["1"] == {"ROS_DOMAIN_ID": "41", "GZ_PARTITION": "j7s1"}
    for k, r in enumerate(IDS):
        assert json.loads((out / r / "run.json").read_text()) == {
            "invocation_id": inv["invocation_id"], "run_id": r, "slurm_job_id": None, "node": pr.socket.gethostname(),
            "job": None, "shard": k % 3, "world": "closed"}
        assert json.loads((out / r / "env.json").read_text()) == seg["shards"][str(k % 3)]
    lines = ledger(tmp_path)
    assert [x["event"] for x in lines] == ["start", "end"] and sorted(lines[1]["entries_run"]) == IDS
    assert {x["invocation_id"] for x in lines} == {inv["invocation_id"]} and inv["counted"]


@pytest.mark.parametrize("args,needle", [(["--rehearsal"], "--rehearsal / --only"),
                                         (["--only", "c01"], "--rehearsal / --only")])
def test_sharded_keeps_the_counted_refusals(tmp_path: Path, args: list[str], needle: str) -> None:
    with pytest.raises(SystemExit, match=needle):
        run(tmp_path, "--out", str(tmp_path / "out"), *args)
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit, match="existing --out"):
        run(tmp_path, "--out", str(tmp_path / "out"))
    assert not (tmp_path / "ledger.jsonl").exists() and not any((tmp_path / "out").iterdir())


@pytest.mark.parametrize("envs", [["0=40:a", "2=42:c"], ["0=40:a", "1=40:b"], ["0=40:a", "1=41:a"],
                                  ["0=40:a", "0=41:b"], ["0=x:a", "1=41:b"], ["0=40", "1=41:b"]])
def test_shard_env_must_name_each_sim_once_distinct(tmp_path: Path, envs: list[str]) -> None:
    with pytest.raises(AssertionError):
        run(tmp_path, "--out", str(tmp_path / "out"), envs=envs)
    assert not (tmp_path / "out").exists()


def test_worker_exception_leaves_its_rest_for_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    cfg(monkeypatch, tmp_path, raise_at="c05")  # sim 1 holds c02, c05, c08
    assert run(tmp_path, "--out", str(out)) == 1
    assert runs_present(out) == [r for r in IDS if r != "c08"] and not (out / "c05/run.json").exists()
    seg = inv_of(out)["segments"][0]
    assert seg["shard_end"]["1"] == "abort: RuntimeError: sim died" and seg["shard_end"]["0"] == "end"
    last = ledger(tmp_path)[-1]
    assert last["event"] == "abort" and last["reason"] == "abort: s1 abort: RuntimeError: sim died"
    assert sorted(last["entries_run"]) == [r for r in IDS if r != "c08"]
    cfg(monkeypatch, tmp_path)
    assert run(tmp_path, "--resume", str(out)) == 0
    seg2 = inv_of(out)["segments"][1]
    assert seg2["kind"] == "resume" and seg2["assignment"] == {"c08": 0} and seg2["entries_run"] == ["c08"]
    assert not (out / "c05/run.json").exists()  # the partial c05 is never re-run
    assert [x["event"] for x in ledger(tmp_path)] == ["start", "abort", "resume", "end"]
    iid = inv_of(out)["invocation_id"]
    assert {json.loads((out / r / "run.json").read_text())["invocation_id"] for r in IDS if r != "c05"} == {iid}


def test_worker_hard_death_is_an_abort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg(monkeypatch, tmp_path, exit_at="c03")  # sim 2: c03, c06, c09
    assert run(tmp_path, "--out", str(tmp_path / "out")) == 1
    assert runs_present(tmp_path / "out") == [r for r in IDS if r not in ("c06", "c09")]
    assert inv_of(tmp_path / "out")["segments"][0]["shard_end"]["2"] == "abort: worker died (rc 3)"
    assert [x["event"] for x in ledger(tmp_path)] == ["start", "abort"]


def test_endpoint_down_stops_that_sim_before_the_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg(monkeypatch, tmp_path, down_at="c04")  # sim 0: c01, c04, c07, c10
    assert run(tmp_path, "--out", str(tmp_path / "out")) == 1
    assert runs_present(tmp_path / "out") == ["c01", "c02", "c03", "c05", "c06", "c08", "c09"]
    assert ledger(tmp_path)[-1]["reason"] == "abort: s0 abort: endpoint down (URLError: refused)"


def test_tree_dirtied_mid_invocation_stops_the_sim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg(monkeypatch, tmp_path, dirty_after="c01")
    out = tmp_path / "out"
    assert run(tmp_path, "--out", str(out), envs=["0=40:a"]) == 1
    assert runs_present(out) == ["c01"]
    assert ledger(tmp_path)[-1]["reason"] == "abort: s0 abort: tree dirty on commit c0ffee"


def test_worker_refuses_an_existing_run_dir(tmp_path: Path) -> None:
    out = tmp_path / "out"
    (out / "d03").mkdir(parents=True)  # dev list, existing --out allowed: d03 must not be run a second time
    assert run(tmp_path, "--list", str(DEV), "--out", str(out), "--rehearsal") == 1
    seg = inv_of(out)["segments"][0]
    assert "d03" not in seg["entries_run"] and not (out / "d03/run.json").exists()
    assert "run dir d03 exists" in seg["end"]


def test_coordinator_rejects_a_run_of_another_sims_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg(monkeypatch, tmp_path, rogue=True)
    with pytest.raises(AssertionError):
        run(tmp_path, "--out", str(tmp_path / "out"))
    last = ledger(tmp_path)[-1]
    assert last["event"] == "abort" and last["reason"].startswith("abort: AssertionError")


def test_resume_of_a_finished_or_foreign_invocation_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    assert run(tmp_path, "--out", str(out)) == 0
    with pytest.raises(SystemExit, match="already ended"):
        run(tmp_path, "--resume", str(out))
    monkeypatch.setattr(pr, "git_state", lambda repo: {"git_commit": "beef", "git_dirty": False})
    with pytest.raises(SystemExit, match="refusing --resume"):
        run(tmp_path, "--resume", str(out))
    assert [x["event"] for x in ledger(tmp_path)] == ["start", "end"]


def test_job_left_in_a_slurm_job_asks_squeue_locally(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen = []

    def fake_run(cmd: list[str], **kw: Any) -> Any:
        seen.append(cmd)
        return pr.subprocess.CompletedProcess(cmd, 0, stdout="1:05:00\n", stderr="")
    monkeypatch.setattr(pr.subprocess, "run", fake_run)
    monkeypatch.setenv("SLURM_JOB_ID", "150001")
    assert pr.job_left_min(tmp_path / "no_such_env_file") == 65
    assert seen == [["squeue", "-h", "-j", "150001", "-o", "%L"]]


# ------------------------------------------------------------------------------- G1 / G2 / G3 on a sharded run
@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    import yaml
    root = tmp_path_factory.mktemp("w")
    w = {"truth_path": root / "truth.json", "places_path": root / "places.yaml"}
    w["truth_path"].write_text(json.dumps(TRUTH))
    w["places_path"].write_text(yaml.safe_dump({"places": {n: {"x": x, "y": y, "yaw": a}
                                                           for n, (x, y, a) in PLACES.items()}}))
    (root / "map_knowledge.json").write_text(json.dumps({"world": "fake", "layout_sha256": LAYOUT,
                                                         "rooms": {r: {} for r in ROOMS}}))
    return w


def verdicts(out: Path, led: Path, w: dict[str, Path]) -> dict[str, Any]:
    entries = pr.load_runs(pr.COUNTED)
    return {"g1": g.g1_verdict(out, entries, True, led, **w), "g2": g.g2_verdict(out, entries, True),
            "g3": g.g3_verdict(out, entries, True, OK_NO_ROS)}


def gates_cfg(mp: pytest.MonkeyPatch, tmp: Path, w: dict[str, Path], **kw: Any) -> None:
    cfg(mp, tmp, mode="gates", world={k: str(v) for k, v in w.items()}, **kw)


def serial_out(tmp: Path, w: dict[str, Path]) -> tuple[Path, Path]:
    out, led = tmp / "serial", tmp / "serial_ledger.jsonl"
    entries, _, inv, kind = pr.plan(pr.parse(["--out", str(out)]))
    pr.run_list(entries, out, inv, kind, lambda e, d, meta: make_run(w, d, e, (correct if int(e["run_id"][1:]) <= 7
                else wrong)(e["task_id"]), meta), lambda: None, led)
    return out, led


def same(a: dict[str, Any], b: dict[str, Any]) -> None:
    """The verdicts of two run sets agree: pass, counts and per-run results (run dirs aside)."""
    strip = lambda rs: [{k: v for k, v in r.items() if k != "run_dir"} for r in rs]  # noqa: E731
    assert (a["g1"]["pass"], a["g1"]["passed"], strip(a["g1"]["runs"])) == \
        (b["g1"]["pass"], b["g1"]["passed"], strip(b["g1"]["runs"]))
    assert a["g1"]["invocation_problems"] == b["g1"]["invocation_problems"] == []
    assert (a["g2"]["pass"], a["g2"]["runs"]) == (b["g2"]["pass"], b["g2"]["runs"])
    assert (a["g3"]["pass"], a["g3"]["hits"]) == (b["g3"]["pass"], b["g3"]["hits"])


def test_gates_pass_on_a_sharded_run_exactly_as_serial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       world: dict[str, Path]) -> None:
    gates_cfg(monkeypatch, tmp_path, world)
    out = tmp_path / "out"
    assert run(tmp_path, "--out", str(out)) == 0
    sharded = verdicts(out, tmp_path / "ledger.jsonl", world)
    assert sharded["g1"]["pass"] and sharded["g1"]["passed"] == 7 and sharded["g2"]["pass"] and sharded["g3"]["pass"]
    same(sharded, verdicts(*serial_out(tmp_path, world), world))


def test_gates_shard_death_fails_g1_until_resumed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  world: dict[str, Path]) -> None:
    gates_cfg(monkeypatch, tmp_path, world, down_at="c06")  # sim 2: c03, c06, c09 -> c06, c09 missing
    out, led = tmp_path / "out", tmp_path / "ledger.jsonl"
    assert run(tmp_path, "--out", str(out)) == 1
    v = verdicts(out, led, world)
    assert not v["g1"]["pass"] and not v["g2"]["pass"] and not v["g3"]["pass"]
    assert [r["run_id"] for r in v["g1"]["runs"] if r["problems"] == ["run dir missing"]] == ["c06", "c09"]
    assert "want start, then only resume / abort, then one end" in v["g1"]["invocation_problems"][0]
    gates_cfg(monkeypatch, tmp_path, world)
    assert run(tmp_path, "--resume", str(out)) == 0
    v = verdicts(out, led, world)
    assert v["g1"]["pass"] and v["g2"]["pass"] and v["g3"]["pass"], v["g1"]["invocation_problems"]
    same(v, verdicts(*serial_out(tmp_path, world), world))


def test_gates_two_sharded_invocations_fail_g1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               world: dict[str, Path]) -> None:
    gates_cfg(monkeypatch, tmp_path, world)
    led = tmp_path / "ledger.jsonl"
    assert run(tmp_path, "--out", str(tmp_path / "a")) == 0
    assert verdicts(tmp_path / "a", led, world)["g1"]["pass"]
    assert run(tmp_path, "--out", str(tmp_path / "b")) == 0  # the counted list a second time: same ledger
    for out in ("a", "b"):
        v = verdicts(tmp_path / out, led, world)["g1"]
        assert not v["pass"] and "holds counted invocations" in v["invocation_problems"][0]


def test_gates_entry_run_twice_fails_g1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                        world: dict[str, Path]) -> None:
    """An entry run twice = a second run of it (here: from a second invocation, own ledger) put in place of the
    first: its run.json carries the other invocation_id, G1 fails."""
    gates_cfg(monkeypatch, tmp_path, world)
    assert run(tmp_path, "--out", str(tmp_path / "a")) == 0
    assert run(tmp_path, "--out", str(tmp_path / "b"), ledger=tmp_path / "other.jsonl") == 0
    shutil.rmtree(tmp_path / "a" / "c04")
    shutil.copytree(tmp_path / "b" / "c04", tmp_path / "a" / "c04")
    v = verdicts(tmp_path / "a", tmp_path / "ledger.jsonl", world)["g1"]
    assert not v["pass"] and "runs not of invocation" in v["invocation_problems"][0] and "c04" in \
        v["invocation_problems"][0]
