"""hpc/sim_job.sbatch + hpc/sim_lib.sh plumbing (task 006, hpc/PERUN_SIM.md section 2) in plain bash: no apptainer, no
ROS, no GPU. SIM_FAKE points sim_lib at a fake launch (prints the ready lines, holds a `gz sim` process) and fake ros2 /
fastdds; the job runs from a throwaway clean git repo holding the two scripts. Checks start, ready, params check,
sim_run env + rc, pgid from the gz sim process's GZ_PARTITION, stop of every sim (nothing left), and the failure
paths. Runs on the local machine and on the PERUN login node (stdlib + pytest only):
    uv run --no-project --with pytest python -m pytest -q tests/test_perun_plumbing.py"""
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hpc"))
import cpu_split as cs  # noqa: E402
PROJ = os.environ.get("PROJ", str(Path.home() / "vlm_agent_project"))  # the job scripts' default

LAUNCH = r"""echo "fake launch $* domain=$ROS_DOMAIN_ID partition=$GZ_PARTITION"
echo "launch $(grep Cpus_allowed_list /proc/self/status)"
setsid sleep 30 &  # outside the sim's pgid but under its launch, like apptainer's squashfuse_ll: the sim's, not a client
(cd "$(dirname "$0")" && exec -a gz python3 sim -s -r fake clinic.sdf) &  # ps: "gz sim -s -r fake clinic.sdf"
sleep 1
if [ -n "$FAKE_DOCK_FAIL" ] || { [ -n "$FAKE_FAIL_ONCE" ] && [ ! -e "$FAKE_FAIL_ONCE" ] && touch "$FAKE_FAIL_ONCE"; }; then
  echo "DOCK MOVE FAILED: ''"; else
  echo "[lifecycle_manager_navigation]: Managed nodes are active"
  case " $* " in *" slam:=true "*) ;; *) echo "[lifecycle_manager_localization]: Managed nodes are active" ;; esac
  echo "dock moved out of the building: data: true"; fi
wait
"""
ROS2 = r"""case "$1 $2" in
  "node list") exit 0 ;;
  "param get") case $4 in
      general_goal_checker.xy_goal_tolerance) echo "Double value is: ${FAKE_XY:-0.1}" ;;
      GridBased.tolerance) echo "Double value is: 0.0" ;;
      update_min_d) echo "Double value is: 0.05" ;;
      update_min_a) echo "Double value is: 0.1" ;; esac ;;
esac
"""
FASTDDS = 'echo "fastdds $* $(date +%s)" >> "$(dirname "$0")/fastdds.log"\n'
PLAN = r"""echo "$RUN|$PROJ|$SLURM_JOB_ID" > "$RUN/plan_vars.txt"
grep Cpus_allowed_list /proc/self/status > "$RUN/plan_cpus.txt"
sim_run 1 sleep 8 &
sleep 1; pin_snapshot "t"
wait
sim_run 0 bash -c 'echo "env $ROS_DOMAIN_ID $GZ_PARTITION $WS_LOCAL $WS $SLURM_JOB_ID $PWD"' &
sim_run 1 bash -c 'echo "env $ROS_DOMAIN_ID $GZ_PARTITION $WS_LOCAL $WS $SLURM_JOB_ID $PWD"' &
wait
sim_run 0 bash -c 'echo "gpu ${SIM_GPU:-none}"; grep Cpus_allowed_list /proc/self/status'
sim_run 1 bash -c 'echo "gpu ${SIM_GPU:-none}"; grep Cpus_allowed_list /proc/self/status'
echo "plan args ${PLAN_D:-none} ${PLAN_J:-none}" > "$RUN/plan_args.txt"
sim_run 0 bash -c 'echo "world $SIM_WORLD vlm ${VLM_BASE_URL:-none}"'
sim_run 1 bash -c 'echo "world $SIM_WORLD vlm ${VLM_BASE_URL:-none}"'
sim_run 1 bash -c 'exit 7' || echo "rc=$?" > "$RUN/plan_rc.txt"
sleep "${FAKE_PLAN_SLEEP:-0}"
"""


def write(p: Path, text: str, mode: int = 0o755) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    p.chmod(mode)
    return p


@pytest.fixture
def job(tmp_path: Path) -> dict:
    """A clean git repo with the job scripts + a plan, the fake bin dir, a run dir, a fake job id."""
    repo, fake, run = tmp_path / "repo", tmp_path / "fake", tmp_path / "run"
    for f in ("hpc/sim_job.sbatch", "hpc/sim_lib.sh", "hpc/cpu_split.py", "hpc/pin_snapshot.py"):
        write(repo / f, (ROOT / f).read_text())
    write(repo / "hpc/plans/t.sh", PLAN)
    write(repo / "ros_ws/src/keep", "")
    write(repo / "sim/worlds/clinic/generated/clinic.sdf", "closed")
    write(repo / "sim/worlds/clinic/generated_cabinet_open/clinic.sdf", "open")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "t"], check=True)
    write(fake / "launch.sh", LAUNCH)
    write(fake / "ros2", ROS2)
    write(fake / "fastdds", FASTDDS)
    write(fake / "sim", "import time\ntime.sleep(600)\n")
    jid = f"9{os.getpid() % 100000:05d}"
    env = {**os.environ, "SIM_FAKE": str(fake), "SIM_JOB_ID": jid, "SIM_RUN_DIR": str(run)}
    env.pop("SLURM_SUBMIT_DIR", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return {"repo": repo, "fake": fake, "run": run, "jid": jid, "env": env}


def start(j: dict, n: int = 2, plan: str = "hpc/plans/t.sh", args: tuple[str, ...] = (), **env: str) -> subprocess.Popen:
    return subprocess.Popen(["bash", str(j["repo"] / "hpc/sim_job.sbatch"), plan, str(n), *args],
                            env={**j["env"], **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)


def partition_pids(part: str) -> list[int]:
    """Own processes whose environ holds GZ_PARTITION=<part>."""
    hits = []
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                if f"GZ_PARTITION={part}".encode() in (p / "environ").read_bytes().split(b"\0"):
                    hits.append(int(p.name))
            except OSError:
                pass
    return hits


def finish(p: subprocess.Popen) -> tuple[int, str]:
    out, _ = p.communicate(timeout=120)
    return p.returncode, out


def test_two_sims_start_ready_run_and_stop(job: dict) -> None:
    rc, out = finish(start(job))
    run, jid = job["run"], job["jid"]
    log = (run / "job.log").read_text()
    assert rc == 0, out
    for i in (0, 1):
        assert f"sim {i} ready (pgid" in log and f"sim {i} params check rc=0: PARAMS OK" in log
        assert f"sim {i} stopped (pgid" in log and f"sim {i} stopped" in log
        cmd = (run / f"s{i}/cmd.log").read_text()
        assert f"env {2 + 16 * (int(jid) % 6) + 2 * i} j{jid}s{i} {PROJ}/build {PROJ} {jid} {job['repo']}" in cmd, cmd
        assert "localization:=true nav2:=true" in (run / f"s{i}/launch.log").read_text()
        assert not partition_pids(f"j{jid}s{i}")
    assert log.count("processes left: 0") == 2 and "plan end rc=0" in log and "all stopped" in log
    pg = [(run / f"s{i}/pgid.txt").read_text().strip() for i in (0, 1)]
    assert pg[0].isdigit() and pg[1].isdigit() and pg[0] != pg[1]
    assert (run / "plan_vars.txt").read_text().strip() == f"{run}|{PROJ}|{jid}"
    assert (run / "plan_rc.txt").read_text().strip() == "rc=7" and "rc=7" in (run / "s1/cmd.log").read_text()
    assert "shm clean" in (job["fake"] / "fastdds.log").read_text()
    assert log.index("sim 1 ready") < log.index("plan start") < log.index("plan end") < log.index("sim 0 stopped")
    # pinned (006 M5 rounds 2-3): each sim's launch on its own cpus, disjoint; sim_run commands and the plan shell on
    # the client cpus, which no sim holds
    def allowed(text: str) -> set[int]:
        return set(cs.expand(re.search(r"Cpus_allowed_list:\s*(\S+)", text).group(1)))
    sims = [allowed((run / f"s{i}/launch.log").read_text()) for i in (0, 1)]
    clients = [allowed((run / f"s{i}/cmd.log").read_text()) for i in (0, 1)]
    assert sims[0] and sims[1] and not sims[0] & sims[1], sims
    assert clients[0] == clients[1] == allowed((run / "plan_cpus.txt").read_text()), clients
    assert not clients[0] & (sims[0] | sims[1]), (clients, sims)
    assert all(f"sim {i} start" in log and "cpus " in log for i in (0, 1)) and "cpus: job " in log
    snap = (run / "pinning.txt").read_text()
    assert "pin check: PASS" in snap and " sim0 " in snap and " sim1 " in snap and " client " in snap, snap
    assert re.search(r"sleep +j\d+s1", snap) and re.search(r" client .* sleep +j\d+s1", snap), snap  # sim_run = client
    assert "pin snapshot 't': pin check: PASS" in log


def test_sims_spread_over_the_jobs_gpus(job: dict) -> None:
    """006 M5 round 2: with > 1 GPU, sim i gets GPU i mod n; with one GPU (or none) nothing is set per sim."""
    rc, out = finish(start(job, CUDA_VISIBLE_DEVICES="0,1"))
    run, log = job["run"], (job["run"] / "job.log").read_text()
    assert rc == 0 and "gpu 0" in (run / "s0/cmd.log").read_text() and "gpu 1" in (run / "s1/cmd.log").read_text(), out
    assert ", GPU 0, cpus " in log and ", GPU 1, cpus " in log


def test_one_gpu_sets_nothing_per_sim(job: dict) -> None:
    rc, out = finish(start(job, CUDA_VISIBLE_DEVICES="3"))
    run, log = job["run"], (job["run"] / "job.log").read_text()
    assert rc == 0 and all("gpu none" in (run / f"s{i}/cmd.log").read_text() for i in (0, 1)), out
    assert ", GPU" not in log


def test_cpu_split_failure_stops_before_any_sim(job: dict) -> None:
    """Job 143814: hpc/cpu_split.py crashed on the host python (3.9) -> empty cpu lists -> 2 x 180 s NOT READY.
    Now: exit 2 at once, nothing started."""
    repo = job["repo"]
    write(repo / "hpc/cpu_split.py", "raise TypeError('boom')\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "commit", "-qam", "broken split"], check=True)
    rc, out = finish(start(job))
    log = (job["run"] / "job.log").read_text()
    assert rc == 2 and "no cpus for sim 0" in log and "TypeError" in log and "sim 0 start" not in log, out


def test_pin_check_catches_a_client_on_a_sims_cpus(job: dict) -> None:
    """A process carrying sim 1's GZ_PARTITION but allowed on sim 1's cpus (a client that escaped) -> pin check FAIL."""
    repo = job["repo"]
    write(repo / "hpc/plans/bad.sh", 'taskset -c "$(sim_cpus 1)" env GZ_PARTITION="$(sim_part 1)" sleep 8 &\n'
                                     'sleep 1; pin_snapshot "bad"\nwait\n')
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "bad plan"], check=True)
    rc, out = finish(start(job, plan="hpc/plans/bad.sh"))
    snap = (job["run"] / "pinning.txt").read_text()
    assert rc == 0 and "pin check: FAIL" in snap and "client sleep allowed 6-11" in snap, out + snap
    assert "pin snapshot 'bad': pin check: FAIL" in (job["run"] / "job.log").read_text()


def test_world_variant_per_sim(job: dict) -> None:
    """PERUN_SIM.md section 5: w1=cabinet_open -> sim 1 launched with that world dir and SIM_WORLD in its env; sim 0
    stays closed (no worlddir arg)."""
    rc, out = finish(start(job, args=("w1=cabinet_open",)))
    run, log = job["run"], (job["run"] / "job.log").read_text()
    assert rc == 0, out
    assert "worlddir:=" not in (run / "s0/launch.log").read_text()
    l1 = (run / "s1/launch.log").read_text()
    assert "worlddir:=" in l1 and "generated_cabinet_open" in l1, l1
    assert "world closed vlm none" in (run / "s0/cmd.log").read_text()
    assert "world cabinet_open vlm none" in (run / "s1/cmd.log").read_text()
    assert "sim 1 start (world cabinet_open," in log and "worlds cabinet_open plan" in log, log


def test_plan_args(job: dict) -> None:
    """p:KEY=VALUE -> PLAN_KEY in the plan's env (no sbatch --export); a value with a space or $ is refused."""
    rc, out = finish(start(job, args=("p:D=/mnt/x/runs/p4/dev1", "p:J=1")))
    assert rc == 0 and (job["run"] / "plan_args.txt").read_text().strip() == "plan args /mnt/x/runs/p4/dev1 1", out
    assert "PLAN_D=/mnt/x/runs/p4/dev1 PLAN_J=1 repo" in (job["run"] / "job.log").read_text()
    for bad in ("p:D=a b", "p:D=$HOME", "p:d=x"):
        rc, out = finish(start(job, args=(bad,)))
        assert rc == 2 and "bad args" in out, bad


def test_map_arg(job: dict) -> None:
    """map=<name>: every sim launched with map:=<repo>/sim/maps/<name>.yaml; a missing map stops the job (exit 2)."""
    write(job["repo"] / "sim/maps/clinic_static.yaml", "image: x.pgm\n")
    git = ["git", "-C", str(job["repo"]), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "map"], check=True)
    rc, out = finish(start(job, args=("map=clinic_static",)))
    run = job["run"]
    assert rc == 0, out
    assert all(f"map:={job['repo']}/sim/maps/clinic_static.yaml" in (run / f"s{i}/launch.log").read_text() for i in (0, 1))
    shutil.rmtree(run)
    rc, out = finish(start(job, args=("map=nosuch",)))
    assert rc == 2 and "no map sim/maps/nosuch.yaml" in (job["run"] / "job.log").read_text(), out


def test_slam_mode(job: dict) -> None:
    """slam: launched with slam:=true (no localization), ready on one lifecycle manager, params check without AMCL."""
    rc, out = finish(start(job, args=("slam", "w0=cabinet_open")))
    run, log = job["run"], (job["run"] / "job.log").read_text()
    assert rc == 0, out
    l0 = (run / "s0/launch.log").read_text()
    assert "slam:=true nav2:=true" in l0 and "localization:=true" not in l0, l0
    assert "sim 0 ready" in log and "sim 0 params check rc=0: PARAMS OK" in log
    pc = (run / "s0/params_check.txt").read_text()
    assert "/amcl" not in pc and "/controller_server" in pc, pc
    rc, out = finish(start(job, args=()))  # without slam the same fake (one manager line only in slam) still needs 2
    assert rc == 0


def test_unknown_world_or_bad_arg_refused(job: dict) -> None:
    rc, out = finish(start(job, args=("w0=nosuch",)))
    log = (job["run"] / "job.log").read_text()
    assert rc == 2 and "sim 0: no world dir for variant nosuch" in log and "sim 0 start" not in log, out
    rc, out = finish(start(job, args=("vlm", "vlm=12")))
    assert rc == 2 and "bad args" in out


class _Models(BaseHTTPRequestHandler):
    """A fake vLLM /v1/models: 200 with the right bearer key, else 401 (or 200 always when OPEN)."""
    KEY, OPEN = "k-test-123", False

    def do_GET(self) -> None:  # noqa: N802
        ok = self.path == "/v1/models" and (self.OPEN or self.headers.get("Authorization") == f"Bearer {self.KEY}")
        self.send_response(200 if ok else 401)
        self.end_headers()

    def log_message(self, *a: object) -> None:
        pass


def serve_dir(tmp: Path, port: int) -> Path:
    d = tmp / "serve"
    write(d / "4242.ready", f"127.0.0.1 {port}\n", 0o644)
    write(d / "4242.key", _Models.KEY, 0o600)
    return d


def test_remote_vlm_used_and_key_checked(job: dict, tmp_path: Path) -> None:
    """vlm=<jobid>: endpoint from <serve>/<jobid>.ready, key from .key; /models must be 200 with the key, 401 without."""
    srv = HTTPServer(("127.0.0.1", 0), _Models)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        d = serve_dir(tmp_path, srv.server_port)
        rc, out = finish(start(job, args=("vlm=4242",), SIM_SERVE_DIR=str(d)))
        run, log = job["run"], (job["run"] / "job.log").read_text()
        assert rc == 0 and "remote vllm (job 4242) http://127.0.0.1:" in log and "401 without the key: checked" in log, out
        assert f"vlm http://127.0.0.1:{srv.server_port}/v1" in (run / "s0/cmd.log").read_text()
        assert _Models.KEY not in log and _Models.KEY not in (run / "s0/cmd.log").read_text()
        _Models.OPEN = True  # an endpoint that answers without the key is refused
        shutil.rmtree(run)
        rc, out = finish(start(job, args=("vlm=4242",), SIM_SERVE_DIR=str(d)))
        assert rc == 5 and "expected 401" in (job["run"] / "job.log").read_text(), out
    finally:
        _Models.OPEN = False
        srv.shutdown()


def test_params_mismatch_stops_before_the_plan(job: dict) -> None:
    rc, out = finish(start(job, FAKE_XY="0.25"))
    log = (job["run"] / "job.log").read_text()
    assert rc == 9 and "PARAMS MISMATCH" in log and "plan start" not in log, out
    assert "0.25 (want 0.10) False" in (job["run"] / "s0/params_check.txt").read_text()
    assert log.count("processes left: 0") == 2 and not partition_pids(f"j{job['jid']}s0")


def test_not_ready_stops_everything(job: dict) -> None:
    rc, out = finish(start(job, n=1, FAKE_DOCK_FAIL="1"))
    log = (job["run"] / "job.log").read_text()
    assert rc == 9 and log.count("sim 0 NOT READY") == 2 and "sim 0 restart" in log and "plan start" not in log, out
    assert "processes left: 0" in log and not partition_pids(f"j{job['jid']}s0")


def test_stuck_bringup_restarted_once(job: dict) -> None:
    rc, out = finish(start(job, FAKE_FAIL_ONCE=str(job["fake"] / "failed_once")))
    run, log = job["run"], (job["run"] / "job.log").read_text()
    assert rc == 0 and log.count("NOT READY") == 1 and log.count("restart (bringup stuck)") == 1, out
    i = re.search(r"sim (\d+) NOT READY", log).group(1)  # the sim that missed (whichever started first)
    assert log.index(f"sim {i} restart") < log.index(f"sim {i} ready") < log.index("plan start")
    assert "DOCK MOVE FAILED" in (run / f"s{i}/launch.try1.log").read_text()
    assert "DOCK MOVE FAILED" not in (run / f"s{i}/launch.log").read_text()
    assert "plan end rc=0" in log and log.count("processes left: 0") == 3  # 1 restart stop + 2 final stops


def test_signal_mid_plan_stops_the_sims(job: dict) -> None:
    p = start(job, FAKE_PLAN_SLEEP="60")
    log = job["run"] / "job.log"
    for _ in range(100):
        if log.exists() and "plan start" in log.read_text():
            break
        time.sleep(0.2)
    assert all(partition_pids(f"j{job['jid']}s{i}") for i in (0, 1))  # both sims alive mid-plan
    os.killpg(p.pid, signal.SIGTERM)  # as scancel: TERM to the job's processes (the sims are their own sessions)
    rc, out = finish(p)
    text = log.read_text()
    assert rc == 143 and "signal: stopping" in text and text.count("processes left: 0") == 2, out
    assert not partition_pids(f"j{job['jid']}s0") and not partition_pids(f"j{job['jid']}s1")


def test_dirty_repo_or_bad_args_refused(job: dict) -> None:
    rc, out = finish(subprocess.Popen(["bash", str(job["repo"] / "hpc/sim_job.sbatch"), "hpc/plans/t.sh", "0"],
                                      env=job["env"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    assert rc == 2 and "bad args" in out
    (job["repo"] / "stray").write_text("x")
    rc, out = finish(start(job))
    assert rc == 2 and "not a clean git checkout" in out and not job["run"].exists()
    shutil.rmtree(job["repo"] / ".git")
    (job["repo"] / "stray").unlink()
    rc, out = finish(start(job))
    assert rc == 2 and "not a clean git checkout" in out


def test_domain_blocks_of_consecutive_jobs_never_overlap() -> None:
    """PERUN_SIM.md section 2 (007, 143846 / 143847 on one node both used domain 40): the domain block is chosen by job id;
    6 consecutive job ids get disjoint, non-adjacent domain sets for N <= 8, all within 2..96."""
    lib = (ROOT / "hpc/sim_lib.sh").read_text()
    expr = re.search(r"sim_dom\(\) \{ echo \$\(\((.*)\)\); \}", lib).group(1)
    assert expr == "2 + 16 * (10#${JOB##*[!0-9]} % 6) + 2 * $1", expr
    doms = {j: {2 + 16 * (j % 6) + 2 * i for i in range(8)} for j in range(143846, 143852)}
    flat = sorted(d for s in doms.values() for d in s)
    assert len(flat) == 48 and min(flat) == 2 and max(flat) == 96
    assert all(abs(a - b) >= 2 for j1, a1 in doms.items() for j2, a2 in doms.items() if j1 != j2 for a in a1 for b in a2)
