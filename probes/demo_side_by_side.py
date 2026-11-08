#!/usr/bin/env python3
"""Task 012 side-by-side demo: the first run_id (list order) that passes on both robots, each run rendered with
probes/demo_recorder.py (--gt-trace), joined with ffmpeg hstack (the shorter held on its last frame), title bar
"same prompt, same layout: TurtleBot 4 | Unitree G1". Run after the final verdict; look at the video before sending.

    uv run python probes/demo_side_by_side.py --out DIR.mp4 --tb4-out T --g1-out D [--list tasks/phase8_pass2.yaml] \
        [--recorder-python $BUILD_DIR/.venv-g1/bin/python]

The recorder needs mujoco (its sim view), which only the G1 venv has (demo_recorder module doc): --recorder-python
(default this interpreter; 145379 failed with ModuleNotFoundError under /opt/venv). g1.sif has no ffmpeg, so render
on the local machine. If the GPU's EGL hangs there (2026-11-06: both GPUs stuck in drm_modeset_lock), Mesa's software device works:
MUJOCO_EGL_DEVICE_ID=2 (the last of the 3 EGL devices mesa lists) with the recorder's default mesa vendor file.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TITLE = "same prompt, same layout: TurtleBot 4 | Unitree G1"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def pick(entries: list[dict[str, Any]], passes_tb: Any, passes_g1: Any) -> str | None:
    """First run_id in list order passing on both robots (callables run_id -> bool)."""
    return next((e["run_id"] for e in entries if passes_tb(e) and passes_g1(e)), None)


def hstack_cmd(a: Path, b: Path, dur_a: float, dur_b: float, out: Path) -> list[str]:
    """Pad the shorter input by cloning its last frame, stack, add the title bar."""
    d = max(dur_a, dur_b)
    pad = lambda i, dur: f"[{i}:v]tpad=stop_mode=clone:stop_duration={d - dur + 0.1:.3f}[v{i}]"
    fc = (f"{pad(0, dur_a)};{pad(1, dur_b)};[v0][v1]hstack=inputs=2,trim=duration={d:.3f},"
          f"pad=iw:ih+48:0:48:color=white,drawtext=fontfile={FONT}:text='{TITLE.replace(':', chr(92) + ':')}':x=(w-text_w)/2:y=10:fontsize=28:"
          "fontcolor=black[v]")
    return ["ffmpeg", "-v", "error", "-y", "-i", str(a), "-i", str(b), "-filter_complex", fc, "-map", "[v]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)]


def duration(p: Path) -> float:
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                          "csv=p=0", str(p)], text=True))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tb4-out", type=Path, required=True)
    ap.add_argument("--g1-out", type=Path, required=True)
    ap.add_argument("--list", type=Path, default=ROOT / "tasks" / "phase8_pass2.yaml")
    ap.add_argument("--recorder-python", default=sys.executable, help="python with mujoco for demo_recorder.py")
    a = ap.parse_args(argv)
    from probes import phase8_gates as p8
    from probes import phase4_gates as p4
    entries = p8.load_list(a.list)
    ok = lambda root: (lambda e: (root / e["run_id"] / "run.json").is_file() and p4.g1_run(e, root / e["run_id"])["pass"])
    rid = pick(entries, ok(a.tb4_out), ok(a.g1_out))
    assert rid, "no run_id passes on both robots"
    parts = []
    for tag, root in (("tb4", a.tb4_out), ("g1", a.g1_out)):
        o = a.out.with_name(f"{a.out.stem}_{tag}.mp4")
        subprocess.run([a.recorder_python, str(ROOT / "probes/demo_recorder.py"), str(root / rid), "--out", str(o),
                        "--gt-trace"], check=True)
        parts.append(o)
    subprocess.run(hstack_cmd(*parts, duration(parts[0]), duration(parts[1]), a.out), check=True)
    print(f"{rid} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
