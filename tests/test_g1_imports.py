"""The runner-side G1 modules import in the project venv (numpy 1.x) without mujoco or torch (012 section 9: the
numpy 1.x / 2.x split; mujoco + torch live only in the G1 venv / g1_node)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ("humanoid.adapter_g1", "humanoid.sim_truth_g1", "humanoid.drive", "humanoid.geometry")


def loaded(modules: tuple[str, ...]) -> dict:
    code = ("import importlib, json, sys\n"
            f"for m in {list(modules)!r}: importlib.import_module(m)\n"
            "import numpy\n"
            "print(json.dumps({'numpy': numpy.__version__, 'heavy': sorted(k for k in sys.modules "
            "if k.split('.')[0] in ('mujoco', 'torch'))}))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_runner_side_g1_modules_import_without_mujoco_or_torch() -> None:
    r = loaded(MODULES)
    assert r["numpy"].startswith("1."), r
    assert r["heavy"] == [], r


def test_check_sees_a_heavy_import() -> None:  # negative control: the scan does see a heavy module when it loads
    code = "import sys, types; sys.modules['mujoco'] = types.ModuleType('mujoco')"
    out = subprocess.run([sys.executable, "-c", code + "; print(sorted(k for k in sys.modules if k == 'mujoco'))"],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "['mujoco']"
