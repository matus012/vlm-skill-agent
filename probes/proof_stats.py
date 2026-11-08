#!/usr/bin/env python3
"""Phase 7 prep (task 009 side lane, counts for no gate): proof-photo pass rate and seg pixel counts over counted runs.

    python3 probes/proof_stats.py --out FILE.json <merged counted dir> [<merged counted dir> ...]

Per run (every `<dir>/<run_id>/score.json` of each merged dir): kind, world, pass, the scorer's proof check, proof_px
(pixels of the target's labels in the reported photo's seg PNG) and proof_label, plus the answer check. Summary per dir
and overall: proof pass rate, proof_px min / p10 / median / max, runs below 2x / 5x the bar (agent/scorer.PROOF_MIN_PX),
and the same split by kind. The brief's Phase 7 rule: proof pass rate >= 90 % -> no `inspect`. By default it reads
score.json (written by the scorer at run time and re-checked by the gates): no seg PNGs, no ROS.

`--rescore` (task 010 G1, "from raw counted run dirs"): every run is scored again by agent/scorer.score_run from its
run.json + steps.jsonl + the reported photo's seg PNG (truth from run.json world + layout_seed), and that fresh result
is used; `rescore_matches_saved` says whether it equals the saved score.json. The denominator is the runs whose proof
check was reached (a report naming a photo), stated as `proof_reached`; `decision` applies the brief's rule to it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.scorer import PROOF_MIN_PX, score_run  # noqa: E402

INSPECT_RATE = 0.90  # brief, Phase 7: proof pass rate >= 90 % over the Phase 5 + 6 counted runs -> no inspect


def runs(d: Path, rescore: bool = False) -> list[dict[str, Any]]:
    out = []
    for sc in sorted(d.glob("*/score.json")):
        saved, r = json.loads(sc.read_text()), json.loads((sc.parent / "run.json").read_text())
        s = score_run(sc.parent) if rescore else saved
        row = {"run_id": sc.parent.name, "kind": r["task"]["kind"], "world": r.get("world", "closed"),
               "pass": s["pass"], "proof": s["checks"]["proof"], "answer": s["checks"]["answer"],
               "proof_px": s["proof_px"], "proof_label": s["proof_label"],
               "proof_reached": (r.get("report") or {}).get("photo") is not None}
        if rescore:
            row["rescore_matches_saved"] = all(s[k] == saved[k] for k in ("pass", "checks", "proof_px", "proof_label"))
        out.append(row)
    return out


def pct(xs: list[int], q: float) -> int:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]


def summary(rs: list[dict[str, Any]]) -> dict[str, Any]:
    px = [r["proof_px"] for r in rs if r["proof_reached"]]
    reached = [r for r in rs if r["proof_reached"]]
    if not reached:
        return {"runs": len(rs), "proof_reached": 0}
    return {"runs": len(rs), "proof_reached": len(reached), "proof_pass": sum(r["proof"] for r in reached),
            "proof_pass_rate": round(sum(r["proof"] for r in reached) / len(reached), 4),
            "proof_px": {"min": min(px), "p10": pct(px, 0.1), "median": int(statistics.median(px)), "max": max(px)},
            "below_2x_bar": sum(p < 2 * PROOF_MIN_PX for p in px), "below_5x_bar": sum(p < 5 * PROOF_MIN_PX for p in px)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rescore", action="store_true", help="score every run again from its raw files (task 010 G1)")
    a = ap.parse_args()
    per = {str(d): runs(d, a.rescore) for d in a.dirs}
    allr = [r for rs in per.values() for r in rs]
    kinds = sorted({r["kind"] for r in allr})
    res = {"note": "Phase 7 prep, no bar; brief: proof pass rate >= 90 % -> no inspect", "proof_min_px": PROOF_MIN_PX,
           "overall": summary(allr), "by_kind": {k: summary([r for r in allr if r["kind"] == k]) for k in kinds},
           "by_dir": {d: summary(rs) for d, rs in per.items()}, "rescored": a.rescore, "runs": per}
    o = res["overall"]
    res["decision"] = {"rule": f"proof pass rate over proof_reached >= {INSPECT_RATE:.0%} -> no inspect",
                       "rate": o.get("proof_pass_rate"), "denominator": o.get("proof_reached"),
                       "branch": "no-inspect" if (o.get("proof_pass_rate") or 0) >= INSPECT_RATE else "inspect-needed"}
    if a.rescore:
        res["rescore_mismatches"] = [r["run_id"] for rs in per.values() for r in rs if not r["rescore_matches_saved"]]
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("overall", "by_kind", "decision") + (("rescore_mismatches",) if a.rescore
                                                                                 else ())}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
