"""probes/phase3_run.py counted-run rules (005 design review B1, B3): refusals, invocation.json, ledger, --resume,
endpoint check before each entry, vLLM job time left. No sim, no VLM: a fake run_entry writes the run dir."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from mock_openai import MockServer  # noqa: E402
from probes import phase3_run as pr  # noqa: E402

DEV = ROOT / "tasks" / "phase3_dev.yaml"
IDS = [f"c{i:02d}" for i in range(1, 11)]


@pytest.fixture(autouse=True)
def clean_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr, "git_state", lambda repo: {"git_commit": "c0ffee", "git_dirty": False})
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)


class Sim:
    """Fake runner: run_entry writes <d>/run.json with the meta; endpoint answers from `down` (one value per call)."""

    def __init__(self, down: list[str | None] | None = None, boom_at: str | None = None) -> None:
        self.ran, self.down, self.boom_at, self.closed = [], list(down or []), boom_at, False

    def run_entry(self, e: dict[str, Any], d: Path, meta: dict[str, Any]) -> dict[str, Any]:
        assert not d.exists()
        d.mkdir(parents=True)
        if e["run_id"] == self.boom_at:
            raise RuntimeError("sim died")
        (d / "run.json").write_text(json.dumps(meta))
        self.ran.append(e["run_id"])
        return {"score": {"pass": True}}

    def check(self) -> str | None:
        return self.down.pop(0) if self.down else None

    def __call__(self) -> tuple[Any, Any, Any]:
        return self.run_entry, self.check, self.close

    def close(self) -> None:
        self.closed = True


def main(tmp: Path, *args: str, sim: Sim | None = None, job_left: Any = None) -> int:
    return pr.main([*args, "--ledger", str(tmp / "ledger.jsonl")], runner=sim or Sim(),
                   job_left=job_left or (lambda: pytest.fail("job_left called")))


def events(tmp: Path) -> list[str]:
    return [json.loads(x)["event"] for x in (tmp / "ledger.jsonl").read_text().splitlines()]


# ---------------------------------------------------------------------------------------------------------- list
def test_counted_sha_is_the_committed_list(tmp_path: Path) -> None:
    assert pr.sha256_file(pr.COUNTED) == pr.COUNTED_SHA256
    copy = tmp_path / "renamed.yaml"
    shutil.copy(pr.COUNTED, copy)
    assert pr.is_counted(pr.COUNTED) and pr.is_counted(copy) and not pr.is_counted(DEV)


@pytest.mark.parametrize("args,needle", [
    (["--rehearsal"], "--rehearsal / --only"), (["--only", "c01"], "--rehearsal / --only"),
    (["--list", "COPY", "--only", "c01"], "--rehearsal / --only")])
def test_counted_refuses_rehearsal_and_only(tmp_path: Path, args: list[str], needle: str) -> None:
    shutil.copy(pr.COUNTED, tmp_path / "copy.yaml")
    args = [str(tmp_path / "copy.yaml") if a == "COPY" else a for a in args]
    with pytest.raises(SystemExit, match=needle):
        main(tmp_path, "--out", str(tmp_path / "out"), *args)
    assert not (tmp_path / "out").exists() and not (tmp_path / "ledger.jsonl").exists()


def test_counted_refuses_existing_out(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    with pytest.raises(SystemExit, match="existing --out"):
        main(tmp_path, "--out", str(tmp_path / "out"))
    assert not (tmp_path / "ledger.jsonl").exists()


def test_dirty_tree_refused_without_rehearsal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr, "git_state", lambda repo: {"git_commit": "c0ffee", "git_dirty": True})
    with pytest.raises(SystemExit, match="dirty tree"):
        main(tmp_path, "--out", str(tmp_path / "out"))


def test_dev_list_allows_only_rehearsal_existing_out_and_no_ledger(tmp_path: Path) -> None:
    (tmp_path / "out").mkdir()
    sim = Sim()
    assert main(tmp_path, "--list", str(DEV), "--out", str(tmp_path / "out"), "--only", "d01,d02", "--rehearsal",
                sim=sim) == 0
    assert sim.ran == ["d01", "d02"] and sim.closed and not (tmp_path / "ledger.jsonl").exists()
    inv = json.loads((tmp_path / "out/invocation.json").read_text())
    assert not inv["counted"] and inv["segments"][0]["entries_run"] == ["d01", "d02"]


# ------------------------------------------------------------------------------------------------- counted run
def test_counted_run_invocation_and_ledger(tmp_path: Path) -> None:
    sim = Sim()
    assert main(tmp_path, "--out", str(tmp_path / "out"), sim=sim) == 0
    inv = json.loads((tmp_path / "out/invocation.json").read_text())
    assert sim.ran == IDS and inv["counted"] and inv["list_sha256"] == pr.COUNTED_SHA256
    assert inv["git_commit"] == "c0ffee" and inv["argv"][:2] == ["--out", str(tmp_path / "out")]
    assert [s["kind"] for s in inv["segments"]] == ["start"] and inv["segments"][0]["end"] == "end"
    host = pr.socket.gethostname()
    assert all(json.loads((tmp_path / "out" / i / "run.json").read_text()) == {
        "invocation_id": inv["invocation_id"], "run_id": i, "slurm_job_id": None, "node": host, "job": None,
        "shard": None, "world": "closed"} for i in IDS)
    lines = [json.loads(x) for x in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [x["event"] for x in lines] == ["start", "end"] and lines[1]["entries_run"] == IDS
    assert {x["invocation_id"] for x in lines} == {inv["invocation_id"]}
    assert {x["list_sha256"] for x in lines} == {pr.COUNTED_SHA256}


def test_endpoint_down_aborts_before_the_entry_then_resume(tmp_path: Path) -> None:
    out = tmp_path / "out"
    sim = Sim(down=[None, None, None, "URLError: refused"])
    assert main(tmp_path, "--out", str(out), sim=sim) == 1
    assert sim.ran == IDS[:3] and not (out / "c04").exists()
    inv = json.loads((out / "invocation.json").read_text())
    assert inv["segments"][0]["end"] == "abort: endpoint down (URLError: refused)"
    last = json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[-1])
    assert last["event"] == "abort" and "endpoint down" in last["reason"]
    sim2 = Sim()
    assert main(tmp_path, "--resume", str(out), sim=sim2) == 0
    assert sim2.ran == IDS[3:] and events(tmp_path) == ["start", "abort", "resume", "end"]
    inv2 = json.loads((out / "invocation.json").read_text())
    assert inv2["invocation_id"] == inv["invocation_id"] and [s["kind"] for s in inv2["segments"]] == ["start", "resume"]
    assert {json.loads((out / i / "run.json").read_text())["invocation_id"] for i in IDS} == {inv["invocation_id"]}


def test_runner_death_is_in_the_ledger_and_partial_dir_never_rerun(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="sim died"):
        main(tmp_path, "--out", str(out), sim=Sim(boom_at="c05"))
    assert events(tmp_path) == ["start", "abort"] and (out / "c05").is_dir() and not (out / "c05/run.json").exists()
    last = json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[-1])
    assert last["reason"] == "abort: RuntimeError: sim died" and last["entries_run"] == IDS[:5]
    sim = Sim()
    assert main(tmp_path, "--resume", str(out), sim=sim) == 0
    assert sim.ran == IDS[5:] and not (out / "c05/run.json").exists()  # the partial c05 stays as it is
    assert events(tmp_path) == ["start", "abort", "resume", "end"]


def test_resume_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "out"
    main(tmp_path, "--out", str(out), sim=Sim(down=[None, "down"]))
    monkeypatch.setattr(pr, "git_state", lambda repo: {"git_commit": "beef", "git_dirty": False})
    with pytest.raises(SystemExit, match="refusing --resume"):
        main(tmp_path, "--resume", str(out))
    monkeypatch.setattr(pr, "git_state", lambda repo: {"git_commit": "c0ffee", "git_dirty": False})
    with pytest.raises(SystemExit, match="refusing --resume"):
        main(tmp_path, "--list", str(DEV), "--resume", str(out))
    assert main(tmp_path, "--resume", str(out)) == 0
    with pytest.raises(SystemExit, match="already ended"):
        main(tmp_path, "--resume", str(out))
    assert events(tmp_path) == ["start", "abort", "resume", "end"]


# ------------------------------------------------------------------------------------------ endpoint and job left
def test_endpoint_down_checks_models_with_the_key() -> None:
    srv = MockServer([], api_key="k")
    try:
        assert pr.endpoint_down(srv.base_url, "k") is None
        assert "401" in str(pr.endpoint_down(srv.base_url, "wrong"))
        url = srv.base_url
    finally:
        srv.close()
    assert pr.endpoint_down(url, "k", timeout_s=2).startswith("URLError")


@pytest.mark.parametrize("left,minutes", [("1:30:00", 90), ("45:30", 45.5), ("1-00:00:00", 1440), ("0:00", 0),
                                          ("2-01:02:03\n", 2 * 1440 + 62.05)])
def test_slurm_minutes(left: str, minutes: float) -> None:
    assert pr.slurm_minutes(left) == pytest.approx(minutes)


@pytest.mark.parametrize("left", ["INVALID", "", "UNLIMITED"])
def test_slurm_minutes_rejects(left: str) -> None:
    with pytest.raises(ValueError):
        pr.slurm_minutes(left)


def test_job_left_min_asks_squeue_for_the_link_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = tmp_path / "vlm_link.env"
    env.write_text("BASE_URL=http://x/v1\nJOBID=143210\nAPI_KEY=secret=with=eq\n")
    seen = []

    def fake_run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess:
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="1:05:00\n", stderr="")
    monkeypatch.setattr(pr.subprocess, "run", fake_run)
    assert pr.job_left_min(env) == 65
    assert seen == [["ssh", "-o", "BatchMode=yes", "$LOGIN_NODE", "squeue -h -j 143210 -o %L"]]


def test_min_job_left_refuses_before_anything(tmp_path: Path) -> None:
    sim = Sim()
    with pytest.raises(SystemExit, match="30.0 min left, need 60"):
        main(tmp_path, "--out", str(tmp_path / "out"), "--min-job-left-min", "60", sim=sim, job_left=lambda: 30.0)
    assert sim.ran == [] and not (tmp_path / "ledger.jsonl").exists() and not (tmp_path / "out").exists()
    assert main(tmp_path, "--out", str(tmp_path / "out"), "--min-job-left-min", "60", sim=sim,
                job_left=lambda: 60.0) == 0
    assert sim.ran == IDS


# --------------------------------------------------------------------- tree dirtied during an invocation (design review request)
def states(*seq: dict[str, Any]) -> Any:
    """git_state stand-in: plan()'s call and then one call per entry, in order; the last state repeats."""
    left = list(seq)
    return lambda repo: left.pop(0) if len(left) > 1 else left[0]


CLEAN, DIRTY = {"git_commit": "c0ffee", "git_dirty": False}, {"git_commit": "c0ffee", "git_dirty": True}


@pytest.mark.parametrize("later,needle", [(DIRTY, "tree dirty on commit c0ffee"),
                                          ({"git_commit": "beef", "git_dirty": False},
                                           "tree on commit beef, not the start commit c0ffee")])
def test_tree_moving_mid_invocation_stops_before_the_next_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                later: dict, needle: str) -> None:
    monkeypatch.setattr(pr, "git_state", states(CLEAN, CLEAN, CLEAN, CLEAN, later))  # plan + c01..c03 clean
    sim = Sim()
    assert main(tmp_path, "--out", str(tmp_path / "out"), sim=sim) == 1
    assert sim.ran == IDS[:3] and not (tmp_path / "out/c04").exists()
    last = json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[-1])
    assert last["event"] == "abort" and last["reason"] == f"abort: {needle}"
    assert json.loads((tmp_path / "out/invocation.json").read_text())["segments"][0]["end"] == f"abort: {needle}"


def test_rehearsal_ignores_the_tree_moving(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr, "git_state", states(CLEAN, DIRTY))
    sim = Sim()
    assert main(tmp_path, "--list", str(DEV), "--out", str(tmp_path / "out"), "--rehearsal", sim=sim) == 0
    assert len(sim.ran) == len(pr.load_runs(DEV))


def test_dev_list_without_rehearsal_also_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pr, "git_state", states(CLEAN, CLEAN, DIRTY))
    sim = Sim()
    assert main(tmp_path, "--list", str(DEV), "--out", str(tmp_path / "out"), sim=sim) == 1
    assert len(sim.ran) == 1 and not (tmp_path / "ledger.jsonl").exists()
