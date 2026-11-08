"""Hand-help detector (005 Step 0, design review N7): evaluator side only, nothing in agent/ imports it.

run.json `hand_help` is true if, during the run (the harness's setup move before step 1 excepted):
- the ground-truth robot position jumps further between two consecutive samples (one per step, plus the setup end)
  than the robot can drive: > SLACK_M after a step with no drive skill (anything but navigate_to), or
  > V_MAX_MPS * sim dt + SLACK_M after a navigate_to step (a teleport / gz set_pose / manual move), or
- any message is published on /initialpose (a manual AMCL pose set).
Limits: a teleport inside a long navigate_to step that stays within V_MAX_MPS * dt is not seen; ROS service calls other
than the two above (e.g. AMCL's reinitialize) are not watched.
"""
import math
import os
import signal
import subprocess
from pathlib import Path

V_MAX_MPS = 0.26  # TB4 Nav2 velocity_smoother max_velocity[0] (stock turtlebot4_navigation nav2.yaml, freeze manifest)
SLACK_M = 0.10    # GT drift allowed across a non-driving step (design review N7: look_around / report <= 0.1 m)
INITIALPOSE_CMD = ["ros2", "topic", "echo", "/initialpose", "geometry_msgs/msg/PoseWithCovarianceStamped"]


class HandHelp:
    def __init__(self, v_max_mps: float = V_MAX_MPS, slack_m: float = SLACK_M) -> None:
        self.v_max, self.slack = v_max_mps, slack_m
        self.last: tuple[float, tuple[float, float]] | None = None
        self.why: list[str] = []

    def gt(self, step: int, sim_t: float, xy: tuple[float, float], drove: bool) -> None:
        """One GT sample after `step` (0 = end of setup); `drove`: that step ran navigate_to."""
        if self.last is not None:
            t0, xy0 = self.last
            dt, d = sim_t - t0, math.dist(xy, xy0)
            bound = self.slack + (self.v_max * max(dt, 0.0) if drove else 0.0)
            if d > bound:
                self.why.append(f"step {step}: GT moved {d:.2f} m in {dt:.1f} s sim, bound {bound:.2f} m "
                                f"({'navigate_to' if drove else 'no drive skill'})")
        self.last = (sim_t, (float(xy[0]), float(xy[1])))

    def initialpose(self, n_msgs: int) -> None:
        if n_msgs:
            self.why.append(f"{n_msgs} /initialpose message(s) during the run")

    def verdict(self) -> dict[str, object]:
        return {"hand_help": bool(self.why), "hand_help_why": list(self.why)}


def count_messages(echo_text: str) -> int:
    """Messages in `ros2 topic echo` YAML output (each one ends with a '---' line)."""
    return sum(1 for line in echo_text.splitlines() if line.strip() == "---")


class InitialPoseWatch:
    """`ros2 topic echo /initialpose` in the background, its output to `log`; stop() -> number of messages seen."""

    def __init__(self, log: Path, cmd: list[str] = INITIALPOSE_CMD) -> None:
        self.log = Path(log)
        with self.log.open("w") as f:
            self.p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.DEVNULL, start_new_session=True)

    def stop(self) -> int:
        os.killpg(self.p.pid, signal.SIGTERM)
        self.p.wait(10)
        return count_messages(self.log.read_text())
