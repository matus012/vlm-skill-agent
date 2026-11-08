"""N5 (005 Step 0): run.json world_tag = the newest phase1-world-frozen* tag whose clinic.sdf + truth.json match disk."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.run_task import world_tag  # noqa: E402

FILES = ("w.sdf", "t.json")


@pytest.mark.skipif(subprocess.run(["git", "-C", str(ROOT), "rev-parse", "-q", "--verify", "refs/tags/phase1-world-frozen-v2"],
                                   capture_output=True).returncode != 0,
                    reason="needs the git history with tag phase1-world-frozen-v2 (absent in a one-commit snapshot)")
def test_real_repo_is_v2() -> None:
    assert world_tag() == "phase1-world-frozen-v2"


def _repo(tmp: Path) -> Path:
    def git(*a: str, date: str = "2026-01-01T00:00:00") -> None:
        env = {**os.environ, "GIT_COMMITTER_DATE": date, "GIT_AUTHOR_DATE": date, "GIT_AUTHOR_NAME": "t",
               "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        subprocess.run(["git", "-C", str(tmp), *a], check=True, capture_output=True, env=env)
    git("init", "-q")
    for i, (sdf, date) in enumerate((("a", "2026-01-01T00:00:00"), ("b", "2026-01-02T00:00:00"),
                                     ("b", "2026-01-03T00:00:00"))):
        (tmp / "w.sdf").write_text(sdf)
        (tmp / "t.json").write_text("{}")
        git("add", *FILES, date=date)
        git("commit", "-q", "--allow-empty", "-m", str(i), date=date)
        git("tag", "-a", ["phase1-world-frozen", "phase1-world-frozen-v2", "phase1-world-frozen-v3"][i], "-m", "x",
            date=date)
    git("tag", "-a", "other-newest", "-m", "x", date="2026-02-01T00:00:00")
    return tmp


def test_newest_matching_tag(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    assert world_tag(r, FILES) == "phase1-world-frozen-v3"  # v2 and v3 both match; v3 is newer
    (r / "w.sdf").write_text("a")
    assert world_tag(r, FILES) == "phase1-world-frozen"


def test_no_matching_tag_refuses(tmp_path: Path) -> None:
    r = _repo(tmp_path)
    (r / "t.json").write_text('{"changed": 1}')
    with pytest.raises(RuntimeError, match="no tag"):
        world_tag(r, FILES)
