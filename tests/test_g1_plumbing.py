"""hpc/g1_job.sbatch + hpc/g1_lib.sh plumbing (task 012 design section 3) in plain bash: no apptainer, ROS, MuJoCo or
GPU. SIM_FAKE points the libs at a fake g1_node (prints "G1 up" + "sim <t> s, RTF" lines, holds its process, writes
stats on TERM), a fake ready.py (writes ready.json) and fake params; the job runs from a throwaway clean git repo with
the job scripts, sim_lib.sh UNCHANGED, and hpc/plans/p4_part.sh UNCHANGED. Checks start args, env, ready, params, pgid
from the node, the watchdog restart, stop of every sim, and the failure paths:
    uv run --no-project --with pytest python -m pytest -q tests/test_g1_plumbing.py"""
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import pytest

from test_perun_plumbing import FASTDDS, finish, partition_pids, write

ROOT = Path(__file__).resolve().parents[1]
PROJ = "$PROJ"

PY_G1 = r"""#!/bin/bash
# fake /opt/venv-g1/bin/python: the node, or ready.py params
if [ "$1" = humanoid/g1_node.py ]; then
  echo "fake g1 node $* domain=$ROS_DOMAIN_ID partition=$GZ_PARTITION robot=$ROBOT scenes=$G1_SCENES ext=$G1_EXT pp=$PYTHONPATH"
  echo "node $(grep Cpus_allowed_list /proc/self/status)"
  all="$*"; stats=${all##*--stats }; stats=${stats%% *}
  trap 'echo "{\"rtf\": 1.0}" > "$stats"; exit 0' TERM
  echo "[INFO] [g1_sim]: G1 up"
  n=0; t=0
  stall=""; [ -n "$FAKE_STALL_ONCE" ] && [ "${GZ_PARTITION: -2}" = s0 ] && [ ! -e "$FAKE_STALL_ONCE" ] && touch "$FAKE_STALL_ONCE" && stall=1
  while true; do
    if [ -z "$stall" ] || [ $n -lt 2 ]; then t=$((t + 10)); echo "[INFO] [g1_sim]: sim $t.0 s, RTF 1.00, base z 0.780"; fi
    n=$((n + 1)); sleep 0.3 & wait $!
  done
fi
if [ "$1 $2" = "humanoid/probes/ready.py params" ]; then
  echo "cmdline: $*"
  case " $* " in *" --rate 1.0 --torch-threads 2 "*) ;; *) echo "PARAMS MISMATCH (cmdline)"; exit 1 ;; esac
  [ -n "$FAKE_PARAMS_BAD" ] && { echo "camera_info fx 300 (want 320)"; echo "PARAMS MISMATCH"; exit 1; }
  echo "PARAMS OK"; exit 0
fi
exec python3 "$@"
"""
PY_RUN = r"""#!/bin/bash
# fake /opt/venv/bin/python: ready.py wait, else the host python
if [ "$1 $2" = "humanoid/probes/ready.py wait" ]; then
  [ -n "$FAKE_NOT_READY" ] && { echo "NOT READY: odom_samples 0 < 50"; exit 1; }
  sleep 0.5; echo '{"ready": true}' > "$4"; echo "READY rtf 1.00 base_z 0.78 world $SIM_WORLD"; exit 0
fi
exec python3 "$@"
"""
PLAN = r"""sim_run 0 bash -c 'echo "env $ROS_DOMAIN_ID $GZ_PARTITION $SIM_WORLD $ROBOT $G1_SCENES $G1_EXT $PYTHONPATH $PWD"'
sim_run 1 bash -c 'echo "env $ROS_DOMAIN_ID $GZ_PARTITION $SIM_WORLD $ROBOT $G1_SCENES $G1_EXT $PYTHONPATH $PWD"'
echo "py $PY_RUN $PY_G1" > "$RUN/plan_py.txt"
sim_run 1 bash -c 'exit 7' || echo "rc=$?" > "$RUN/plan_rc.txt"
sleep "${FAKE_PLAN_SLEEP:-0}"
"""


@pytest.fixture
def job(tmp_path: Path) -> dict:
    repo, fake, run = tmp_path / "repo", tmp_path / "fake", tmp_path / "run"
    for f in ("hpc/g1_job.sbatch", "hpc/g1_lib.sh", "hpc/sim_lib.sh", "hpc/cpu_split.py", "hpc/pin_snapshot.py",
              "hpc/plans/p4_part.sh"):
        write(repo / f, (ROOT / f).read_text())
    write(repo / "hpc/plans/t.sh", PLAN)
    write(repo / "ros_ws/src/keep", "")
    for w in ("generated", "generated_cabinet_open", "generated_blocked_storage"):
        write(repo / f"sim/worlds/clinic/{w}/clinic.sdf", w)
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "t"], check=True)
    write(fake / "python_g1", PY_G1)
    write(fake / "python_run", PY_RUN)
    write(fake / "fastdds", FASTDDS)
    jid = f"8{os.getpid() % 100000:05d}"
    env = {**os.environ, "SIM_FAKE": str(fake), "SIM_JOB_ID": jid, "SIM_RUN_DIR": str(run)}
    for k in ("SLURM_SUBMIT_DIR", "CUDA_VISIBLE_DEVICES"):
        env.pop(k, None)
    return {"repo": repo, "fake": fake, "run": run, "jid": jid, "env": env}


def start(j: dict, n: int = 2, plan: str = "hpc/plans/t.sh", args: tuple[str, ...] = (), **env: str) -> subprocess.Popen:
    return subprocess.Popen(["bash", str(j["repo"] / "hpc/g1_job.sbatch"), plan, str(n), *args],
                            env={**j["env"], **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)


def test_two_g1_sims_start_ready_params_run_and_stop(job: dict) -> None:
    rc, out = finish(start(job, args=("w1=cabinet_open",)))
    run, jid, repo = job["run"], job["jid"], job["repo"]
    log = (run / "job.log").read_text()
    assert rc == 0, out
    for i, world in ((0, "closed"), (1, "cabinet_open")):
        part = f"j{jid}s{i}"
        launch = (run / f"s{i}/launch.log").read_text()
        assert (f"humanoid/g1_node.py --world {world} --scene-dir {run}/scenes/{part} --rate 1.0 --torch-threads 2 "
                f"--truth-log {run}/s{i}/truth_qpos.jsonl --stats {run}/s{i}/stats.json") in launch, launch
        assert f"robot=g1 scenes={run}/scenes ext=/opt/g1ext pp={repo}:/opt/g1ext/agile" in launch, launch
        assert f"sim {i} ready (pgid" in log and f"sim {i} params check rc=0: PARAMS OK" in log
        assert (run / f"s{i}/ready.json").exists() and (run / f"s{i}/stats.json").exists()  # stats: TERM reached the node
        cmd = (run / f"s{i}/cmd.log").read_text()
        dom = 2 + 16 * (int(jid) % 6) + 2 * i
        assert f"env {dom} {part} {world} g1 {run}/scenes /opt/g1ext {repo}:/opt/g1ext/agile {repo}" in cmd, cmd
        assert not partition_pids(part)
        pg = (run / f"s{i}/pgid.txt").read_text().strip()
        assert pg.isdigit()
    assert "sim 1 start (g1, world cabinet_open," in log and log.count("processes left: 0") == 2
    assert (run / "plan_rc.txt").read_text().strip() == "rc=7"
    assert (run / "plan_py.txt").read_text().split()[1:] == [f"{job['fake']}/python_run", f"{job['fake']}/python_g1"]
    assert log.index("sim 1 params check") < log.index("plan start") < log.index("plan end rc=0") < log.index("all stopped")
    assert "node restarts: s0=0 s1=0" in log


def test_p4_part_runs_unchanged(job: dict) -> None:
    """p4_part.sh's own lines (sim_dom / sim_part / sim_world / pin_snapshot / sim_run) work under g1_lib: the
    coordinator command gets every sim's shard env; the python path is the image's (absent on the host: rc 127)."""
    p = start(job, plan="hpc/plans/p4_part.sh",
              args=("w1=cabinet_open", "p:LIST=tasks/phase8_dev.yaml", "p:D=/x/d", "p:J=1"))
    rc = p.wait(timeout=120)  # p4_part's snapshot subshell leaves its `sleep 600` behind (harmless in a job): kill it
    os.killpg(p.pid, signal.SIGKILL)
    run, jid = job["run"], int(job["jid"])
    assert rc == 0, (run / "job.log").read_text()
    cmd = (run / "s0/cmd.log").read_text()
    d = [2 + 16 * (jid % 6) + 2 * i for i in (0, 1)]
    assert (f"probes/phase3_run.py --list tasks/phase8_dev.yaml --job-part /x/d --job 1 --shards 2 "
            f"--shard-env 0={d[0]}:j{jid}s0:closed --shard-env 1={d[1]}:j{jid}s1:cabinet_open") in cmd, cmd
    assert "pin check:" in (run / "pinning.txt").read_text()


def test_watchdog_restarts_a_stalled_node(job: dict) -> None:
    rc, out = finish(start(job, FAKE_STALL_ONCE=str(job["fake"] / "stalled"), FAKE_PLAN_SLEEP="9",
                           G1_STALL_S="2", WATCH_POLL_S="1"))
    run, log = job["run"], (job["run"] / "job.log").read_text()
    assert rc == 0, out
    assert re.search(r"g1_watch: sim 0 log stalled \d+ s \(last 'sim 20.0 s, RTF 1.00'\): restart 1", log), log
    assert "g1_watch: sim 1" not in log and "node restarts: s0=1 s1=0" in log
    assert (run / "s0/restarts.txt").read_text().count("restart") == 1
    assert "sim 20.0 s" in (run / "s0/launch.stall1.log").read_text()
    assert log.index("restart 1") < log.rindex("sim 0 ready") and (run / "s0/stats.0.json").exists()
    assert log.count("processes left: 0") == 3 and not partition_pids(f"j{job['jid']}s0")


def test_not_ready_restarted_once_then_exit_9(job: dict) -> None:
    rc, out = finish(start(job, n=1, FAKE_NOT_READY="1"))
    log = (job["run"] / "job.log").read_text()
    assert rc == 9 and log.count("sim 0 NOT READY: NOT READY: odom_samples 0 < 50") == 2 and "plan start" not in log, out
    assert (job["run"] / "s0/launch.try1.log").exists() and "processes left: 0" in log
    assert not partition_pids(f"j{job['jid']}s0")


def test_params_mismatch_stops_before_the_plan(job: dict) -> None:
    rc, out = finish(start(job, FAKE_PARAMS_BAD="1"))
    log = (job["run"] / "job.log").read_text()
    assert rc == 9 and "params check rc=1: PARAMS MISMATCH" in log and "plan start" not in log, out
    assert log.count("processes left: 0") == 2


def test_signal_mid_plan_stops_sims_and_watchdog(job: dict) -> None:
    p = start(job, FAKE_PLAN_SLEEP="60")
    log = job["run"] / "job.log"
    for _ in range(100):
        if log.exists() and "plan start" in log.read_text():
            break
        time.sleep(0.2)
    assert all(partition_pids(f"j{job['jid']}s{i}") for i in (0, 1))
    os.killpg(p.pid, signal.SIGTERM)
    rc, out = finish(p)
    assert rc == 143 and log.read_text().count("processes left: 0") == 2, out
    assert not partition_pids(f"j{job['jid']}s0") and not partition_pids(f"j{job['jid']}s1")


def test_bad_args_refused(job: dict) -> None:
    for args in (("map=clinic_static",), ("slam",), ("vlm", "vlm=3"), ("w0=nosuch",)):
        rc, out = finish(start(job, args=args))
        assert rc == 2, (args, out)
    assert "sim 0: no world dir for variant nosuch" in (job["run"] / "job.log").read_text()
    rc, out = finish(start(job, n=9))
    assert rc == 2 and "bad args" in out


def test_sim_lib_is_byte_identical_to_head() -> None:
    """design section 3: sim_lib.sh is inside the G2 pathspec; the G1 path redefines, never edits it."""
    head = subprocess.run(["git", "-C", str(ROOT), "show", "HEAD:hpc/sim_lib.sh"], capture_output=True, check=True).stdout
    assert head == (ROOT / "hpc/sim_lib.sh").read_bytes()


def test_qpos_arg_adds_the_per_sim_log_and_never_leaks(job: dict) -> None:
    """task 013: job arg `qpos` -> every node gets --qpos-log s<i>/qpos.jsonl; without it (even with G1_QPOS_LOG exported
    in the submit shell) no node does."""
    rc, out = finish(start(job, args=("qpos",)))
    assert rc == 0, out
    for i in (0, 1):
        launch = (job["run"] / f"s{i}/launch.log").read_text()
        assert f"--torch-threads 2 --qpos-log {job['run']}/s{i}/qpos.jsonl --truth-log" in launch, launch
    job["run"] = job["run"].with_name("run2")
    job["env"]["SIM_RUN_DIR"] = str(job["run"])
    rc, out = finish(start(job, G1_QPOS_LOG="1", G1_NODE_EXTRA="--pitch-control"))
    assert rc == 0, out
    assert all("--qpos-log" not in (job["run"] / f"s{i}/launch.log").read_text() for i in (0, 1))
