"""Run directory (INTERFACES section 5): run.json + steps.jsonl + lossless photos. `check_complete` is gate G4's check.

<run_dir>/run.json          metadata + summary (written at start, rewritten at the end)
<run_dir>/steps.jsonl       one line per step
<run_dir>/photos/<step>_<k>_rgb.png, <step>_<k>_seg.png   (k = photo number - 1; step 0 = the start photo)
Paths inside the JSON are relative to the run dir."""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from agent.robot import Photo

# Spec of a complete record: key -> True (null allowed) | False (must be non-null) | nested spec dict.
# A nested dict value may also be null when the key is listed as nullable via NULLABLE_DICTS.
RUN_SPEC: dict[str, Any] = {
    "task": {"id": False, "kind": False, "prompt": False, "target": False},
    "planner": False, "robot": False, "layout_sha256": False, "world_tag": False, "git_commit": False,
    "git_dirty": False, "start_place": False, "start_utc": False, "end_utc": False, "outcome": False, "steps": False,
    "report": {"answer": False, "photo": False, "note": False},
    "sim_s": False, "wall_s": False,
    "setup": {"start_place": False, "reached": False, "gt_err_m": False},
    "hand_help": False,
}
PHOTO_IN_SPEC: dict[str, Any] = {"n": False, "rgb": False, "seg": False}
PHOTO_OUT_SPEC: dict[str, Any] = {**PHOTO_IN_SPEC, "meta": False}
STEP_SPEC: dict[str, Any] = {
    "step": False,
    "packet": {"room": False, "history": False, "last_result": False, "photos": [PHOTO_IN_SPEC]},
    "call": {"raw": False, "name": True, "args": True},
    "check": False, "result": True, "photos_out": [PHOTO_OUT_SPEC],
    "sim_t0": False, "sim_t1": False, "wall_t0": False, "wall_t1": False, "skill_sim_s": False, "skill_wall_s": False,
    "truth": {"gt_pose_after": False, "place_err_m": True},
    "planner_out": {"content": True, "reasoning": True, "finish_reason": True, "usage": True, "error": True},
}
NULLABLE_DICTS = {"report"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def git_state(repo: Path) -> dict[str, Any]:
    """{git_commit, git_dirty} of `repo`, for run.json."""
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain"))}


class RunLog:
    def __init__(self, run_dir: Path, header: dict[str, Any]) -> None:
        self.dir = Path(run_dir)
        (self.dir / "photos").mkdir(parents=True, exist_ok=True)
        self.run = dict(header)
        self._write_run()
        self._steps = (self.dir / "steps.jsonl").open("w")

    def _write_run(self) -> None:
        (self.dir / "run.json").write_text(json.dumps(self.run, indent=1))

    def save_photos(self, step: int, photos: list[Photo]) -> list[dict[str, Any]]:
        """Writes each photo's frame (rgb + seg PNG, lossless) and returns the photos_out entries {n, rgb, seg, meta}."""
        out = []
        for ph in photos:
            stem = f"photos/{step}_{ph.number - 1}"
            ph.image.save(self.dir / f"{stem}_rgb.png")
            seg = None
            if ph.frame is not None and ph.frame.seg is not None:
                seg = f"{stem}_seg.png"
                Image.fromarray(ph.frame.seg).save(self.dir / seg)
            out.append({"n": ph.number, "rgb": f"{stem}_rgb.png", "seg": seg,
                        "meta": ph.frame.meta if ph.frame is not None else None})
        return out

    def step(self, rec: dict[str, Any]) -> None:
        self._steps.write(json.dumps(rec) + "\n")
        self._steps.flush()

    def finish(self, summary: dict[str, Any]) -> None:
        self._steps.close()
        self.run.update(summary)
        self._write_run()


def _check(rec: Any, spec: Any, where: str, run_dir: Path, errs: list[str]) -> None:
    if isinstance(spec, list):  # list of records
        if not isinstance(rec, list):
            errs.append(f"{where}: not a list")
            return
        for i, r in enumerate(rec):
            _check(r, spec[0], f"{where}[{i}]", run_dir, errs)
        return
    if not isinstance(rec, dict):
        errs.append(f"{where}: not an object")
        return
    for key, sub in spec.items():
        at = f"{where}.{key}"
        if key not in rec:
            errs.append(f"{at}: missing")
        elif rec[key] is None:
            if not (sub is True or key in NULLABLE_DICTS):
                errs.append(f"{at}: null")
        elif isinstance(sub, (dict, list)):
            _check(rec[key], sub, at, run_dir, errs)
        if key in ("rgb", "seg") and isinstance(rec.get(key), str) and not (run_dir / rec[key]).is_file():
            errs.append(f"{at}: file {rec[key]} missing")


def check_complete(run_dir: Path) -> list[str]:
    """Problems that make the run dir incomplete (INTERFACES section 5); empty list = complete.
    A crash step (it carries `traceback`) may have nulls where the skill never finished."""
    run_dir = Path(run_dir)
    errs: list[str] = []
    for f in ("run.json", "steps.jsonl"):
        if not (run_dir / f).is_file():
            errs.append(f"{f}: missing")
    if errs:
        return errs
    run = json.loads((run_dir / "run.json").read_text())
    _check(run, RUN_SPEC, "run", run_dir, errs)
    lines = (run_dir / "steps.jsonl").read_text().splitlines()
    for i, line in enumerate(lines):
        try:
            rec = json.loads(line)
        except ValueError:
            errs.append(f"steps[{i}]: not JSON")
            continue
        step_errs: list[str] = []
        _check(rec, STEP_SPEC, f"steps[{i}]", run_dir, step_errs)
        if "traceback" in rec:
            step_errs = [e for e in step_errs if not e.endswith(": null")]
        errs += step_errs
    if run.get("steps") != len(lines):
        errs.append(f"run.steps = {run.get('steps')} but steps.jsonl has {len(lines)} lines")
    return errs
