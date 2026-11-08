"""Rule: agent/ imports no ROS module (INTERFACES section 0). AST scan of every agent/*.py, incl. dynamic imports by
literal name (__import__ / importlib.import_module)."""
import ast
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROS = re.compile(r"^(rclpy|geometry_msgs|nav_msgs|sensor_msgs|std_msgs|action_msgs|nav2_\w*|tf2_\w*|tf2|"
                 r"builtin_interfaces|rosgraph_msgs|launch\w*|ros_gz\w*|gz\w*|cv_bridge|ament\w*|turtlebot4\w*|"
                 r"irobot_create\w*)$")


def ros_imports(agent_dir: Path) -> list[str]:
    bad = []
    for f in sorted(agent_dir.rglob("*.py")):
        for node in ast.walk(ast.parse(f.read_text(), str(f))):
            mods: list[str] = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            elif (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant)
                  and isinstance(node.args[0].value, str)
                  and getattr(node.func, "attr", getattr(node.func, "id", None)) in ("__import__", "import_module")):
                mods = [node.args[0].value]
            bad += [f"{f.name}:{node.lineno} {m}" for m in mods if ROS.match(m.split(".")[0])]
    return bad


def test_no_ros_imports_in_agent() -> None:
    assert len(list((ROOT / "agent").glob("*.py"))) >= 8
    assert ros_imports(ROOT / "agent") == []


def test_scan_catches_a_planted_import(tmp_path: Path) -> None:
    copy = tmp_path / "agent"
    shutil.copytree(ROOT / "agent", copy)
    for i, line in enumerate(["import rclpy", "from geometry_msgs.msg import Twist", "import nav2_simple_commander",
                              "import tf2_ros", "from launch_ros.actions import Node", "import gz.transport13",
                              "importlib.import_module('rclpy')", "import os, builtin_interfaces.msg"]):
        (copy / f"planted_{i}.py").write_text(line + "\n")
    assert len(ros_imports(copy)) == 8


def test_scan_ignores_lookalikes(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("import json\nimport launcher_util_x as l\nfrom . import launch\nx = 'import rclpy'\n")
    # 'launcher_util_x' matches launch\w* by design (prefix rule); the relative import and the string do not count
    assert ros_imports(tmp_path) == ["ok.py:2 launcher_util_x"]


if __name__ == "__main__":
    print(ros_imports(Path(sys.argv[1])))
