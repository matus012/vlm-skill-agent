# hpc/PERUN_SIM.md — contract for running the clinic sim on PERUN (task 006)

Frozen at the start of 006 so the image, the job plumbing and the gate probes can be built in parallel. Change only by
editing this file first, in its own commit. Nothing here changes the agent, skills, prompts, world, robot params or
gate code (006 non-goals); only where the code runs and the launch plumbing.

## 1. Paths
`PROJ=$PROJ` (PERUN).
- **Image:** `$PROJ/images/sim.sif`, built from `hpc/sim.def` by `hpc/build_sim.sh` on the login node (`apptainer build
  --fakeroot`, tmp on node-local `/tmp`, 114 pattern). Base `ros:jazzy` (Ubuntu 24.04, Python 3.12) + apt: Gazebo
  Harmonic / `ros-jazzy-ros-gz`, navigation2, nav2-bringup, turtlebot4 simulator + navigation, irobot create sim, the
  same versions as the local machine where apt allows. glvnd EGL vendor file for `libEGL_nvidia.so.0` (`--nv` binds the driver).
  uv + a venv `/opt/venv` (python3.12, `--system-site-packages` so rclpy imports) with the project deps from
  `uv.lock` (`uv sync --frozen --no-install-project`, numpy < 2). Versions: `hpc/SIM_VERSIONS.md` (the local machine vs image);
  sha256 in `hpc/JOBS.md`.
- **Code:** `$PROJ/repo` = a clean git clone at one commit, made by `hpc/sync_code.sh` on the local machine (refuses a dirty
  tree; prints the commit). Never edited on PERUN.
- **`WS_LOCAL`** = the per-machine build / run dir. Default (local machine, unchanged): `$BUILD_DIR` (default `~/vlm_skill_agent_build`).
  PERUN: `$PROJ/build`. `sim/env.sh`, `sim/launch_clinic.sh`, `sim/build.sh` read `$WS_LOCAL` (default as before).
  - `$WS_LOCAL/ros_ws/{build,install,log}`: the colcon overlay (`sim/build.sh`, run inside the image).
  - `$WS_LOCAL/run/`: launch_clinic.sh's params copies (file names keyed by `GZ_PARTITION` + `ROS_DOMAIN_ID`: unique per
    job and sim, so N sims and concurrent jobs sharing `$PROJ/build` don't clash; keyed by domain only until jobs
    143819-143821 ran at once).
- **Python** on PERUN: `/opt/venv/bin/python` inside the image, cwd `$PROJ/repo`.
- **Runs:** `$PROJ/runs/sim/<SLURM_JOB_ID>/`: `job.log`, `vllm.log`, `s<i>/launch.log`, `s<i>/params_check.txt`,
  `s<i>/cmd.log`, plus the out dirs the plan names. `hpc/fetch_runs.sh <jobid> <local dir>` rsyncs it to the local machine.

## 2. One job = optional vLLM + N sims (gpu_short, 1 H200)
- `sbatch hpc/sim_job.sbatch <plan.sh> <N> [vlm]` (wall at submit, as `vlm_up.sh`; never `--export`). It starts vLLM
  first when `vlm` is given, then sims 0..N-1, each waited until ready, then sources the plan, then stops everything.
- **Sim i env:** `ROS_DOMAIN_ID = base + 2i`, `base = 2 + 16 * (SLURM_JOB_ID mod 6)`, N ≤ 8 (domains 2..96; 007: jobs
  143846 / 143847 shared node gpu25 and both used domain 40 → the two jobs' Nav2 stacks discovered each other, map_server
  lifecycle stuck, params check failed; ROS discovery on one node is keyed by domain only, GZ_PARTITION already was
  per job; consecutive job ids never share a block; until 007: 40 + 2i) (never adjacent IDs: `sim/fastdds_localhost.xml` announces to participant
  IDs 0..127 = 256 ports, but domains are only 250 ports apart, so domain d's discovery reaches domain d+1's first
  participants and they share /clock — job 143783: "jump back in time" floods, Nav2 never active; spacing 2 = 500
  ports, no overlap), `GZ_PARTITION=j${SLURM_JOB_ID}s$i`, `GZ_IP=127.0.0.1` (env.sh),
  `WS_LOCAL=$PROJ/build`. Started with `setsid bash sim/launch_clinic.sh localization:=true nav2:=true [x:= y:=
  yaw:=]` inside `apptainer exec --nv --cleanenv` of the sim image. Ready = 2× "Managed nodes are active" + "dock
  moved" in `s<i>/launch.log` (as on the local machine). Params check per sim as on the local machine (xy 0.10, planner 0.0, AMCL 0.05 / 0.1),
  written to `s<i>/params_check.txt`. Stop = `kill -- -<pgid>` of that sim's group (pgid read from the `gz sim`
  process whose `/proc/<pid>/environ` holds that sim's `GZ_PARTITION`; never `pgrep -f`, never `$!`), then `-9`,
  `fastdds shm clean`.
- **CPUs** (006 M5 rounds 2-3, `hpc/cpu_split.py`): sim i's launch runs pinned to its own whole L3 groups of the job's
  cpus (up to 16 cpus, wholly held NUMA nodes first; at least one group always left for the client, or the job stops);
  everything else the job starts (vLLM, params checks, every `sim_run` command: gate probes, agents, coordinators, their
  `gz` CLI reads) runs on the job's cpus no sim holds. Computed once per job before vLLM; the job's own shell is moved
  onto the client cpus, so what it starts inherits them. A process could still re-pin itself: `pin_snapshot` checks
  every job process and thread live (`$RUN/pinning.txt`).
- **Plan file API** (bash, sourced after all sims are ready): `sim_run <i> <cmd ...>` runs `cmd` in sim i's env
  inside the image (cwd `$PROJ/repo`, stdout+stderr to `s<i>/cmd.log`, returns its rc; `&` + `wait` for parallel);
  `$RUN` = the job's run dir; `$VLM_BASE_URL` / `$VLM_API_KEY` / `$VLM_MODEL` exported when vLLM is up.
- **vLLM:** same image, model and flags as `hpc/serve_vllm.sbatch`, plus `--gpu-memory-utilization 0.75` (leaves
  the Gazebo renderers room), bound to `127.0.0.1` on a free port; key file as now; `VLM_BASE_URL=http://127.0.0.1:<port>/v1`.
- **Time left in-job:** `probes/phase3_run.py --min-job-left-min` reads `squeue -h -j $SLURM_JOB_ID -o %L` locally
  when `SLURM_JOB_ID` is set (no ssh from a compute node); else as now (ssh $LOGIN_NODE, the JOBID in vlm_link.env).

## 3. Sharded counted runs (one invocation, one ledger, each entry once)
`probes/phase3_run.py --list L --out D --shards N --shard-env <i>=<ROS_DOMAIN_ID>:<GZ_PARTITION> ...` (or an
equivalent single coordinator): one process owns the invocation (invocation.json, ledger start / end / abort lines,
the B1 refusals), assigns each list entry to exactly one sim (fixed rule, written into invocation.json before the
first run), and runs the N shards in parallel, each in its sim's env, into the same `D`. All run.json carry the one
invocation_id. `--resume` keeps its meaning (entries without a run dir). G1 / G2 / G3 code is unchanged and must pass
on a sharded run exactly as on a serial one (test with a fake adapter).

## 4. Equivalence probes (006 gates M2, M5)
- **M2 seg agreement:** reference = the local machine frames from `.cache/vlm_skill_agent/runs/2026-11-04/002-counted/photos/`
  (world v1; v1 → v2 changed one visual's colour only, no geometry, no labels, so seg labels are identical). On
  PERUN the robot is teleported (Gazebo set_pose) so that its camera's GT pose equals the recorded one; the GT camera
  pose is read back and must match (≤ 5 mm, ≤ 0.2°) before the frame counts. Agreement = equal label pixels in the
  cropped seg frame, ≥ 98 % per frame. Plus live `camera_info` 640×480, fx 320 ± 0.5. Truth / world sha256 =
  `phase1-world-frozen-v2`.
- **M5 cross-talk:** per sim, its run log's GT poses come from its own Gazebo partition, and its ROS graph holds one
  copy of each node; a planted publish on sim A's domain is not seen on sim B (negative), the same publish on B's
  domain is (positive control).

## 5. 007 contract: cross-job counted lists, world variants, remote vLLM (frozen before 007's build)
- **Cross-job counted list (007 Step 0, 2026-11-05 (a)-(d)).** One counted list = one invocation over k jobs × N sims.
  - **Split** (before any run, on the login node inside the sim image, no GPU):
    `phase3_run.py --list L --out D --split K:N [--shard-world j:i=<variant> ...]` writes `D/invocation.json` (uuid, list
    sha256, commit, counted, `"split": {run_id: [job j, shard i]}`), with the same refusals as a counted `--out` today.
    Rule: list order, each entry to the next shard (job-major: j0s0, j0s1, j1s0, ...) whose world equals the entry's
    `world`; a world with no shard → refuse. No run happens here. The split's own ledger line goes to the plan ledger
    file (below).
  - **Job part** (inside job j): `phase3_run.py --list L --job-part D --job j --shards N --shard-env i=DOM:PART ...`
    runs only job j's entries, each on its split shard, through today's worker protocol. Job j's segments go to
    `D/invocation.j<j>.json`, the only file it writes besides run dirs. `--job-part D --job j --resume` runs job j's
    entries with no run dir.
  - **Ledgers:** per invocation, `<ledger dir>/<invocation_id>.plan.jsonl` and `<invocation_id>.j<j>.jsonl`. Each has
    exactly one writer. The ledger dir is the counted ledger's dir.
  - **Merge** (once, after every job ended): `phase3_run.py --merge D` checks each split entry has exactly one run dir
    whose run.json carries D's invocation_id and the split's (job, shard). It then appends the plan and job ledgers in
    time order to the counted ledger (single writer) and writes `D/merged.json` (per-entry status, per-job ends).
    Missing, duplicated or misplaced entries → merged.json `ok: false`, and every gate refuses D.
  - **Every run.json** adds `slurm_job_id`, `node` (hostname), `job` (j), `shard` (i) and `world`. Gates report
    per-run sim_s / wall_s per node; counted runs below 0.6 are flagged (no verdict change).
- **World variants.** `closed` = `sim/worlds/clinic/generated/` (tag `phase1-world-frozen-v2`). `cabinet_open` =
  `sim/worlds/clinic/generated_cabinet_open/`: the same layout with `cabinet_blue_1` state open, generated
  deterministically by `generate.py` from a committed layout variant, and frozen by a git tag before 007's dev.
  - A list entry may say `world: cabinet_open` (default `closed`).
  - Sim i's world comes from a `sim_job.sbatch` arg `w<i>=<variant>`. It is exported to sim i's env as
    `SIM_WORLD=<variant>`, and the launch passes `worlddir:=` that dir.
  - A shard worker refuses an entry whose world differs from its `SIM_WORLD`.
  - run_task uses that variant's truth.json (truth hook, start) and records `world`. Scorer and gates pick the truth
    by run.json `world` and still check its layout sha256.
- **Remote vLLM (007 Step 0b).** `sim_job.sbatch <plan> <N> vlm=<serve jobid>` uses a running `hpc/serve_vllm.sbatch` job:
  - `VLM_BASE_URL=http://<node>:<port>/v1` comes from `$PROJ/runs/serve/<jobid>.ready`; the key comes from
    `$PROJ/runs/serve/<jobid>.key`;
  - `GET /models` must answer with the key and 401 without it, or the job stops (exit 5).
  - `vlm` alone = local vLLM as before.
