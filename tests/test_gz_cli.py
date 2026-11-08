"""probes/gz_cli.py (006 a recorded deviation): a `gz topic` read killed by a signal is retried at most twice, each death
recorded with its signal; a read dead 3 times raises GzReadError; a clean failure is not retried. A fake `gz` on PATH
dies by the signals listed in $FAKE_GZ_DIE (one per call, then prints the message)."""
import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))
import gz_cli  # noqa: E402

FAKE = """#!/usr/bin/env python3
import os, signal, sys
from pathlib import Path
n = Path(os.environ["FAKE_GZ_COUNT"]); k = int(n.read_text()) if n.exists() else 0; n.write_text(str(k + 1))
die = [s for s in os.environ.get("FAKE_GZ_DIE", "").split(",") if s]
if k < len(die):
    if die[k] == "EXIT3":
        sys.exit(3)
    sys.stderr.write("terminate called without an active exception\\n"); sys.stderr.flush()
    os.kill(os.getpid(), getattr(signal, die[k]))
print('{"header": {"stamp": {"sec": 7}}, "pose": []}')
"""


@pytest.fixture
def gz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    b = tmp_path / "bin"
    b.mkdir()
    (b / "gz").write_text(FAKE)
    (b / "gz").chmod(0o755)
    monkeypatch.setenv("PATH", f"{b}:{__import__('os').environ['PATH']}")
    monkeypatch.setenv("FAKE_GZ_COUNT", str(tmp_path / "count"))
    monkeypatch.setattr(gz_cli, "DEATHS", [])
    monkeypatch.setattr(gz_cli, "STATS", {"reads": 0, "attempts": 0})
    return tmp_path / "count"


def test_clean_read(gz: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = gz_cli.mark()
    assert json.loads(gz_cli.echo_once("/world/clinic/pose/info"))["header"]["stamp"]["sec"] == 7
    assert gz_cli.since(m) == {"reads": 1, "attempts": 1, "deaths": []}


def test_two_deaths_then_ok(gz: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_GZ_DIE", "SIGSEGV,SIGABRT")
    m = gz_cli.mark()
    assert json.loads(gz_cli.echo_once("/t"))["header"]["stamp"]["sec"] == 7
    s = gz_cli.since(m)
    assert s["reads"] == 1 and s["attempts"] == 3 and gz.read_text() == "3"
    assert [(d["attempt"], d["signal"]) for d in s["deaths"]] == [(1, "SIGSEGV"), (2, "SIGABRT")]
    assert "terminate called" in s["deaths"][0]["stderr"] and s["deaths"][0]["topic"] == "/t"


def test_three_deaths_raise(gz: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_GZ_DIE", "SIGSEGV,SIGABRT,SIGSEGV,SIGSEGV")
    with pytest.raises(gz_cli.GzReadError) as e:
        gz_cli.echo_once("/t")
    assert e.value.gz_read_failed and e.value.returncode == -signal.SIGSEGV
    assert isinstance(e.value, subprocess.CalledProcessError)  # callers' except CalledProcessError still hold
    assert gz.read_text() == "3" and len(gz_cli.DEATHS) == 3 and gz_cli.STATS == {"reads": 1, "attempts": 3}


def test_clean_failure_not_retried(gz: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_GZ_DIE", "EXIT3")
    with pytest.raises(subprocess.CalledProcessError) as e:
        gz_cli.echo_once("/t")
    assert not isinstance(e.value, gz_cli.GzReadError) and e.value.returncode == 3
    assert gz.read_text() == "1" and gz_cli.DEATHS == []


def test_readers_use_it() -> None:
    """Every evaluator GT reader goes through echo_once (no direct `gz topic -e` left in them)."""
    for f in ("map_world_check.py", "photos.py", "truth_check.py"):
        src = (ROOT / "probes" / f).read_text()
        assert "gz_cli.echo_once(" in src and '"topic", "-e"' not in src, f
