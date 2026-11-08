#!/usr/bin/env python3
"""Photo numbers for the correct ScriptedPlanner scripts, derived from Phase 1 photos (not typed).

    python tasks/derive_photos.py [photos_dir ...]   # writes tasks/photo_numbers.json

For each dev task: the room that holds the proof (the object's room / the counted room), its look_around photos
<photos_dir>/<room>/<k>_seg.png (cropped seg, heading k = place yaw + k*90 deg), and per k the largest single-proof-
label pixel count. Photo = k + 1 of the k with the most px (must be >= the scorer's 1000 px). Several photos_dirs
(e.g. the 002 counted set and the 003 step-0 v2 re-count) must agree on the chosen photo.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.scorer import PROOF_MIN_PX  # noqa: E402
from agent.truth import ROOT, load_tasks, load_truth  # noqa: E402

EVIDENCE = Path.home() / ".cache/vlm_skill_agent/runs/2026-11-04"
DEFAULT_DIRS = [EVIDENCE / "002-counted/photos", EVIDENCE / "003-step0/recount/photos"]
OUT = ROOT / "tasks" / "photo_numbers.json"


def proof_room(task, truth: dict) -> str:
    """The room whose photos hold the proof: the counted room, the state object's room, else the answer room."""
    if task.kind == "count":
        return task.target["room"]
    if task.kind == "check_state":
        return next(o["room"] for o in truth["objects"] if o["id"] == task.target["object"])
    return task.truth_answer


def main(dirs: list[Path]) -> int:
    truth = load_truth()
    out = {"sources": [str(d) for d in dirs], "rule": __doc__.strip().splitlines()[3:7], "tasks": {}}
    for t in load_tasks():
        room = proof_room(t, truth)
        per_dir = {}
        for d in dirs:
            if not (d / room).is_dir():
                continue
            px = []
            for k in range(4):
                seg = np.asarray(Image.open(d / room / f"{k}_seg.png"))
                px.append(max(int(np.count_nonzero(seg == lab)) for lab in t.proof_labels))
            per_dir[str(d)] = {"px_by_k": px, "photo": int(np.argmax(px)) + 1}
        photos = {v["photo"] for v in per_dir.values()}
        best = max(max(v["px_by_k"]) for v in per_dir.values())
        assert len(photos) == 1, f"{t.id}: sources disagree {per_dir}"
        assert best >= PROOF_MIN_PX, f"{t.id}: best {best} px < {PROOF_MIN_PX}"
        out["tasks"][t.id] = {"room": room, "proof_labels": list(t.proof_labels), "photo": photos.pop(),
                              "truth_answer": t.truth_answer, "by_source": per_dir}
        print(f"{t.id:22s} room {room:10s} photo {out['tasks'][t.id]['photo']}  "
              + "  ".join(f"{Path(s).parts[-2]}:{v['px_by_k']}" for s, v in per_dir.items()))
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main([Path(a) for a in sys.argv[1:]] or DEFAULT_DIRS))
