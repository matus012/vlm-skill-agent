#!/usr/bin/env python3
"""Re-score a saved run from disk and compare with its score.json (INTERFACES section 0, G4).
Record completeness is agent.runlog.check_complete, not repeated here.

    python probes/replay.py <run_dir>

Exit 0 iff the re-score gives the same verdict (pass, the four checks, truth answer,
proof px, label) as the saved score.json, and run.json, steps.jsonl and the reported photo's seg PNG still have the
sha256 saved in score.json (a run scored before the hashes existed fails: re-score it).
"""
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.scorer import file_hashes, score_run  # noqa: E402

VERDICT_KEYS = ("pass", "checks", "truth_answer", "proof_px", "label")


def replay(run_dir: Path, **kw: Any) -> tuple[bool, list[str]]:
    """kw: score_run's truth_path / places_path (tests)."""
    saved = json.loads((Path(run_dir) / "score.json").read_text())
    now = score_run(run_dir, **kw)
    problems = [f"{k}: saved {saved.get(k)!r} != replay {now[k]!r}" for k in VERDICT_KEYS if saved.get(k) != now[k]]
    hashes = file_hashes(run_dir)
    problems += [f"sha256 {f}: saved {saved.get('sha256', {}).get(f)} != now {h}" for f, h in hashes.items()
                 if saved.get("sha256", {}).get(f) != h]
    return not problems, problems


def main() -> int:
    run_dir = Path(sys.argv[1])
    same, problems = replay(run_dir)
    print(f"{run_dir}: {'SAME' if same else 'DIFFERENT'}")
    for p in problems:
        print(f"  {p}")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
