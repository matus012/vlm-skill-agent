"""task 013: humanoid/qpos_log.py line format and probes/make_video_list.py (no ROS, no MuJoCo)."""
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanoid.qpos_log import MODEL, QposLog
from probes import make_video_list as mvl
from probes.phase3_run import is_counted, load_runs


def test_qpos_log_format_rate_and_kill_safe(tmp_path: Path) -> None:
    p = tmp_path / "q.jsonl"
    log = QposLog(p, ["root", "a_joint", "b_joint"], hz=10.0)
    head = json.loads(p.read_text().splitlines()[0])  # header is on disk before any sample (killed node)
    assert head == {"model": MODEL, "nq": 9, "qpos_names": ["root", "a_joint", "b_joint"]}  # root = 7 qpos
    for k in range(51):  # 50 Hz policy steps for 1 s -> ~10 Hz lines
        log.write(0.02 * k, [0.1 * k, 0, 0.8, 1, 0, 0, 0, 0.2, -0.2])
    with pytest.raises(AssertionError):  # a qpos that does not match the header nq
        log.write(9.0, [0.0])
    log.close()
    rows = [json.loads(ln) for ln in p.read_text().splitlines()[1:]]
    assert 9 <= len(rows) <= 12 and rows[0] == {"t_sim": 0.0, "qpos": [0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0, 0.2, -0.2]}
    assert all(set(r) == {"t_sim", "qpos"} and len(r["qpos"]) == 9 for r in rows)
    assert [r["t_sim"] for r in rows] == sorted(r["t_sim"] for r in rows)


def test_video_list_is_an_exact_copy_not_counted(tmp_path: Path) -> None:
    out = tmp_path / "v.yaml"
    out.write_text(mvl.build(["p8p2c001", "p8p2c003"]))
    src = {ln for ln in mvl.SRC.read_text().splitlines()}
    body = [ln for ln in out.read_text().splitlines() if not ln.startswith("#")]
    assert len(body) == 2 and all(ln in src for ln in body)
    assert [r["run_id"] for r in load_runs(out)] == ["p8p2c001", "p8p2c003"]
    full = {r["run_id"]: r for r in yaml.safe_load(mvl.SRC.read_text())}
    assert all(r == full[r["run_id"]] for r in load_runs(out))
    assert not is_counted(out) and not is_counted(mvl.OUT)


def test_unknown_id_refused() -> None:
    with pytest.raises(SystemExit, match="unknown run ids"):
        mvl.build(["p8p2c001", "p8p2c999"])
