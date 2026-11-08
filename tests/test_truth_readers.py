"""No answers on the agent side (005 Step 0, N8): only the scorer side and instruments may read truth.json.

Scans every non-test .py under agent/ adapters/ humanoid/ sim/ tasks/ probes/ hpc/ for `truth.json`, `TRUTH_JSON`,
`load_truth` or an import of agent.truth; any hit outside ALLOWED fails, naming the file."""
import re
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = ("agent", "adapters", "humanoid", "sim", "tasks", "probes", "hpc")
ALLOWED = ("agent/truth.py", "agent/scorer.py", "adapters/run_task.py", "probes/*", "tasks/derive_photos.py",
           "sim/worlds/clinic/generate*.py", "sim/worlds/clinic/seeded.py",  # 008: truth per layout seed (harness)
           "sim/worlds/clinic/realize.py",  # 011: truth per Phase 8 layout spec (harness)
           "sim/worlds/clinic/blocked.py",  # 008 side lane: Phase 6 blocker variant (generator side)
           "humanoid/room.py",  # 012: G1 sim scene builder (harness, as generate*.py): mocap names from the world's frozen truth
           "humanoid/probes/*", "humanoid/tests/*")
PATTERN = re.compile(r"truth\.json|\bTRUTH_JSON\b|\bload_truth\b|\bagent\.truth\b|from\s+agent\s+import\s+[^\n]*\btruth\b")


def offenders(root: Path = ROOT) -> list[str]:
    out = []
    for d in DIRS:
        for p in sorted((root / d).rglob("*.py")):
            rel = p.relative_to(root).as_posix()
            if "__pycache__" in rel or any(fnmatch(rel, a) for a in ALLOWED):
                continue
            hits = [f"{rel}:{i}: {line.strip()}" for i, line in enumerate(p.read_text().splitlines(), 1)
                    if PATTERN.search(line)]
            out += hits
    return out


def test_only_scorer_side_reads_truth() -> None:
    bad = offenders()
    assert not bad, "truth.json readers outside the scorer side / instruments:\n" + "\n".join(bad)


def test_pattern_catches_each_form() -> None:
    for line in ('x = ROOT / "generated/truth.json"', "from agent.tasks import TRUTH_JSON", "t = load_truth()",
                 "import agent.truth", "from agent.truth import derive", "from agent import scorer, truth"):
        assert PATTERN.search(line), line
    assert not PATTERN.search("from agent.world import load_rooms  # map_knowledge.json")


AGENT_SIDE = ("agent.runtime", "agent.skills", "agent.context", "agent.planners", "agent.world", "agent.robot",
              "agent.runlog", "agent.tasks", "agent.fake_adapter", "agent.render_packet")


def loaded_answer_modules(modules: tuple[str, ...] = AGENT_SIDE) -> list[str]:
    """Import the agent-side modules in a fresh interpreter; return which of agent.truth / agent.scorer got loaded
    (design review N1: the static scan misses an indirect import, e.g. via a helper module or importlib)."""
    import subprocess
    import sys
    code = ("import importlib, sys\n"
            f"for m in {list(modules)!r}: importlib.import_module(m)\n"
            "print(','.join(sorted(k for k in sys.modules if k in ('agent.truth', 'agent.scorer')"
            " or (getattr(sys.modules[k], '__file__', '') or '').endswith(('sim/worlds/clinic/seeded.py', 'sim/worlds/clinic/realize.py')))))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    return [m for m in out.stdout.strip().split(",") if m]


def test_agent_side_imports_no_answers() -> None:
    assert loaded_answer_modules() == []


def test_dynamic_check_sees_an_import() -> None:  # negative control: the scorer side does load them
    assert loaded_answer_modules(("agent.scorer",)) == ["agent.scorer", "agent.truth"]


# The robot adapters (design review 007 audit): adapters.turtlebot4 (the VLM runs' adapter) and probes/map_world_check.make_robot
# (Phase 1 G1's) import probe modules (photos, map_world_check, gz_cli, ...). ALLOWED lets every probe read truth.json
# (instruments), so the static scan alone would not notice one the adapter imports. The adapter's probe import closure
# must be truth-clean, and importing the adapters must not load agent.truth / agent.scorer.
ADAPTERS = ("adapters.turtlebot4", "map_world_check", "humanoid.adapter_g1")


def adapter_closure(modules: tuple[str, ...] = ADAPTERS, root: Path = ROOT) -> dict[str, object]:
    """Import the adapters in a fresh interpreter (probes/ on sys.path, as the adapter does); return the answer modules
    loaded and the repo .py files (relative) of every probes/ module loaded."""
    import json
    import subprocess
    import sys
    code = ("import importlib, json, sys\n"
            f"sys.path.insert(0, {str(root / 'probes')!r})\n"
            f"for m in {list(modules)!r}: importlib.import_module(m)\n"
            f"pr = {str(root / 'probes')!r}\n"
            "files = sorted({getattr(v, '__file__', '') or '' for v in list(sys.modules.values())})\n"
            "print(json.dumps({'answers': sorted(k for k in sys.modules if k in ('agent.truth', 'agent.scorer')"
            " or (getattr(sys.modules[k], '__file__', '') or '').endswith(('sim/worlds/clinic/seeded.py', 'sim/worlds/clinic/realize.py'))),"
            " 'probes': [f[len(pr) - len('probes'):] for f in files if f.startswith(pr + '/')]}))")
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def closure_offenders(root: Path = ROOT) -> list[str]:
    c = adapter_closure(root=root)
    bad = [f"loaded by the adapters: {m}" for m in c["answers"]]
    for rel in c["probes"]:
        bad += [f"{rel}:{i}: {line.strip()}" for i, line in enumerate((root / rel).read_text().splitlines(), 1)
                if PATTERN.search(line)]
    return bad


def test_adapters_import_no_answers_and_their_probes_read_no_truth() -> None:
    c = adapter_closure()
    assert "probes/photos.py" in c["probes"] and "probes/map_world_check.py" in c["probes"], c  # the closure is real
    bad = closure_offenders()
    assert not bad, "the robot adapters reach truth:\n" + "\n".join(bad)
