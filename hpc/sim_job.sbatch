#!/bin/bash
#SBATCH --job-name=sim115
#SBATCH --partition=gpu_short
#SBATCH --account=<PERUN_ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=<PROJECT_DIR>/logs/%x_%j.out
#SBATCH --error=<PROJECT_DIR>/logs/%x_%j.err
# One job = optional vLLM + N clinic sims on localhost (hpc/PERUN_SIM.md section 2). Submit from the clean clone:
#   cd $PROJ/repo && sbatch --time=HH:MM:SS hpc/sim_job.sbatch <plan.sh> <N> [vlm]       (never --export)
# 32 CPUs / 256G: an assumption for N = 4 (a gpu_short node: 128 CPUs, 2.2 TB, 8 GPUs); override at submit
# (sbatch --cpus-per-task / --mem) once N = 4 is measured. The repo used is $SLURM_SUBMIT_DIR; it must be clean.
# Args after N (PERUN_SIM.md section 5): `vlm` = local vLLM in this job; `vlm=<jobid>` = a running hpc/serve_vllm.sbatch
# job's endpoint (its runs/serve/<jobid>.ready + .key); `w<i>=<variant>` = sim i's world (default closed;
# sim/worlds/clinic/generated_<variant>/ must exist); `map=<name>` = every sim navigates on sim/maps/<name>.yaml (default
# the frozen clinic.yaml; 008: clinic_static = no movables); `slam` = SLAM Toolbox instead of AMCL (map tours: 1 lifecycle
# manager to wait for, no AMCL params); `p:KEY=VALUE` = PLAN_KEY=VALUE for the plan (KEY [A-Z_]+, VALUE
# [A-Za-z0-9_./:,-]+), e.g. `sbatch ... hpc/plans/p4_part.sh 2 vlm w1=cabinet_open p:D=$PROJ/runs/p4/dev1 p:J=1`.
# Order: vLLM (if `vlm`; exports VLM_BASE_URL / VLM_API_KEY / VLM_MODEL) -> sims 0..N-1 started together, each waited
# until ready + params checked -> the plan sourced in a subshell (API: sim_run <i> <cmd ...>, $RUN, $PROJ,
# $SLURM_JOB_ID) -> stop all (trap EXIT). Output: $PROJ/runs/sim/<jobid>/ (job.log, vllm.log, s<i>/...).
# Exit: plan's rc; 9 a sim not ready twice (one restart) / params wrong; 5 vLLM not up; 2 usage / repo not clean /
# overlay stale / no cpus for a sim or the client.
#
# One-time overlay build on the login node (not via sbatch; no GPU): from the clean clone,
#   bash hpc/sim_job.sbatch --build-overlay
# runs sim/build.sh inside $PROJ/images/sim.sif with WS_LOCAL=$PROJ/build and stamps the ros_ws/src tree hash it
# built from; a job refuses to start when the clone's ros_ws/src differs from the stamp (rebuild then).
set -uo pipefail  # no -e: every step's rc is handled; the trap must always run
PROJ=${PROJ:-$HOME/vlm_agent_project}  # your cluster project dir
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
STAMP=$PROJ/build/ros_ws/BUILT_FROM_SRC_TREE
[ -f "$REPO/hpc/sim_lib.sh" ] || { echo "run from the repo root ($REPO has no hpc/sim_lib.sh)"; exit 2; }
# GPU nodes have no host git (job 143781 failed on it): git runs from the sim image, which has it (hpc/sim.def).
git() { if command -v /usr/bin/git > /dev/null; then /usr/bin/git "$@"; else apptainer exec -B "$PROJ" "$PROJ/images/sim.sif" git "$@"; fi; }
ST=$(git -C "$REPO" status --porcelain) && [ -z "$ST" ] || { echo "repo $REPO not a clean git checkout"; exit 2; }
SRC_TREE=$(git -C "$REPO" rev-parse HEAD:ros_ws/src) || exit 2

if [ "${1:-}" = --build-overlay ]; then
    SIF=$PROJ/images/sim.sif
    mkdir -p "$PROJ/build"
    apptainer exec --cleanenv -B "$PROJ" --pwd "$REPO" --env "WS_LOCAL=$PROJ/build" "$SIF" bash sim/build.sh || exit 1
    [ -z "$(git -C "$REPO" status --porcelain)" ] || { echo "build left $REPO dirty"; exit 1; }
    echo "$SRC_TREE" > "$STAMP"
    echo "overlay built in $PROJ/build/ros_ws from ros_ws/src tree $SRC_TREE (commit $(git -C "$REPO" rev-parse HEAD))"
    exit 0
fi

PLAN=${1:?usage: sbatch hpc/sim_job.sbatch <plan.sh> <N> [vlm | vlm=<jobid>] [w<i>=<variant> ...]}
NSIM=${2:?usage: sbatch hpc/sim_job.sbatch <plan.sh> <N> [vlm | vlm=<jobid>] [w<i>=<variant> ...]}
VLM=""; VLM_JOB=""; SIM_WORLDS=(); SIM_MAP=""; SIM_SLAM=""; ARGS_OK=1
for a in "${@:3}"; do
    case $a in
        vlm) VLM=vlm ;;
        vlm=*) VLM_JOB=${a#vlm=}; [[ $VLM_JOB =~ ^[0-9]+$ ]] || ARGS_OK=0 ;;
        w[0-9]*=*) k=${a%%=*}; k=${k#w}; [[ $k =~ ^[0-9]+$ && ${a#*=} =~ ^[a-z_]+$ ]] && SIM_WORLDS[k]=${a#*=} || ARGS_OK=0 ;;
        map=*) SIM_MAP=${a#map=}; [[ $SIM_MAP =~ ^[a-z_]+$ ]] || ARGS_OK=0 ;;
        slam) SIM_SLAM=1 ;;
        p:*=*) k=${a%%=*}; k=${k#p:}; v=${a#*=}
               if [[ $k =~ ^[A-Z_]+$ && $v =~ ^[A-Za-z0-9_./:,-]+$ ]]; then export "PLAN_$k=$v"; else ARGS_OK=0; fi ;;
        *) ARGS_OK=0 ;;
    esac
done
JOB=${SIM_JOB_ID:-${SLURM_JOB_ID:?run under sbatch}}   # SIM_JOB_ID: the fake-launch plumbing test only
case $PLAN in /*) ;; *) PLAN=$REPO/$PLAN ;; esac
[ -f "$PLAN" ] && [[ $NSIM =~ ^[1-8]$ ]] && [ "$ARGS_OK" = 1 ] && [[ -z $VLM || -z $VLM_JOB ]] ||
    { echo "bad args: $*"; exit 2; }
RUN=${SIM_RUN_DIR:-$PROJ/runs/sim/$JOB}
mkdir -p "$RUN"
export PROJ RUN SLURM_JOB_ID=$JOB
CVD=${CUDA_VISIBLE_DEVICES:-0}
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CVD%%,*}  # the first GPU; sims spread over all (sim_lib sim_gpu)
# shellcheck source=hpc/sim_lib.sh
source "$REPO/hpc/sim_lib.sh"
[ -n "${SIM_FAKE:-}" ] || [ "$(cat "$STAMP" 2>/dev/null)" = "$SRC_TREE" ] ||
    { echo "overlay stale or missing: $STAMP != ros_ws/src tree $SRC_TREE; run hpc/sim_job.sbatch --build-overlay"; exit 2; }
VDESC=${VLM:-${VLM_JOB:+remote $VLM_JOB}}${SIM_MAP:+ map $SIM_MAP}${SIM_SLAM:+ slam}
jlog "=== sim115 job $JOB node $(hostname) N=$NSIM vlm=${VDESC:-no} worlds ${SIM_WORLDS[*]:-closed} plan $PLAN $(env | grep '^PLAN_' | sort | tr '\n' ' ')repo $REPO commit $(git -C "$REPO" rev-parse HEAD)"
[ -n "${SIM_FAKE:-}" ] || jlog "sif $SIF sha256 $(sha256sum "$SIF" | cut -c1-64); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
trap stop_all EXIT
trap 'jlog "signal: stopping"; exit 143' TERM INT   # scancel / wall: SLURM sends TERM first, then KILL

# before vLLM: every sim and the client must get cpus (job 143814: cpu_split crashed on the host python -> empty
# taskset lists); from here on this shell and all it starts run on the client cpus, the sims on their own (sim_lib)
cpus_init || exit 2
[ -z "$SIM_MAP" ] || [ -f "$REPO/sim/maps/$SIM_MAP.yaml" ] || { jlog "no map sim/maps/$SIM_MAP.yaml"; exit 2; }
for i in $(seq 0 $((NSIM - 1))); do  # each sim's world dir must exist before anything starts
    d=$(sim_worlddir "$i") || { jlog "sim $i: no world dir for variant ${SIM_WORLDS[$i]}"; exit 2; }
done
if [ -n "$VLM" ]; then vllm_start || exit 5; fi
if [ -n "$VLM_JOB" ]; then vllm_remote "$VLM_JOB" || exit 5; fi
for i in $(seq 0 $((NSIM - 1))); do sim_start "$i"; done
for i in $(seq 0 $((NSIM - 1))); do
    sim_wait_ready "$i" && continue
    # a stuck bringup (job 143804 sim 3: lifecycle_manager_localization never got past "Configuring map_server", the
    # other 3 sims ready in 32 s) is restarted once, its first log kept; a second miss stops the job
    jlog "sim $i restart (bringup stuck)"
    sim_stop "$i"; mv "$RUN/s$i/launch.log" "$RUN/s$i/launch.try1.log"; sim_start "$i"
    sim_wait_ready "$i" || exit 9
done
for i in $(seq 0 $((NSIM - 1))); do sim_params_check "$i" || exit 9; done
[ -n "${SIM_FAKE:-}" ] || nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/gpu mem with sims: /' >> "$RUN/job.log"

jlog "plan start $PLAN"
# shellcheck disable=SC1090  # the plan is an argument
( source "$PLAN" ); PRC=$?
jlog "plan end rc=$PRC"
exit $PRC
