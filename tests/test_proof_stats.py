"""probes/proof_stats.py (task 009 side lane): reads score.json + run.json per run, pass rate and px summary."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def mk(d: Path, rid: str, kind: str, proof: bool, px: int, reported: bool = True) -> None:
    r = d / rid
    r.mkdir(parents=True)
    (r / "run.json").write_text(json.dumps({"task": {"kind": kind}, "world": "closed",
                                            "report": {"answer": "x", "photo": 1, "note": ""} if reported else None}))
    (r / "score.json").write_text(json.dumps({"pass": proof, "checks": {"proof": proof, "answer": True},
                                              "proof_px": px, "proof_label": 9}))


def test_summary(tmp_path: Path) -> None:
    d = tmp_path / "counted"
    mk(d, "a1", "find", True, 12000)
    mk(d, "a2", "count", True, 1500)
    mk(d, "a3", "count", False, 400)
    mk(d, "a4", "find", False, 0, reported=False)  # no report: the proof check was never reached, not in the denominator
    out = tmp_path / "o.json"
    subprocess.run([sys.executable, str(ROOT / "probes/proof_stats.py"), "--out", str(out), str(d)], check=True,
                   capture_output=True)
    res = json.loads(out.read_text())
    o = res["overall"]
    assert (o["runs"], o["proof_reached"], o["proof_pass"], o["proof_pass_rate"]) == (4, 3, 2, round(2 / 3, 4))
    assert o["proof_px"]["min"] == 400 and o["proof_px"]["max"] == 12000 and o["proof_px"]["median"] == 1500
    assert o["below_2x_bar"] == 2 and o["below_5x_bar"] == 2
    assert res["by_kind"]["count"]["runs"] == 2 and res["by_kind"]["find"]["proof_pass"] == 1
    assert res["decision"]["denominator"] == 3 and res["decision"]["branch"] == "inspect-needed"  # 2/3 < 90 %
