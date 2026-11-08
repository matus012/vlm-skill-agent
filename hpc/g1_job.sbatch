#!/bin/bash
#SBATCH --job-name=g1sim115
#SBATCH --partition=gpu_short
#SBATCH --account=<PERUN_ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=<PROJECT_DIR>/logs/%x_%j.out
#SBATCH --error=<PROJECT_DIR>/logs/%x_%j.err
# One job = optional vLLM + N MuJoCo G1 clinic sims (task 012 design section 3). Same args and flow as
# hpc/sim_job.sbatch, image $PROJ/images/g1.sif, sims from hpc/g1_lib.sh (g1_node per sim, ready.py, params, watchdog):
#   cd $PROJ/repo && sbatch --time=HH:MM:SS hpc/g1_job.sbatch <plan.sh> <N> [vlm | vlm=<jobid>] [pitch] [qpos] [w<i>=<world>] [p:K=V]
# `w<i>=<world>`: sim i's world (closed default, cabinet_open, blocked_<room> dev only); its node builds the scene into
# $RUN/scenes/j<JOB>s<i>/. `qpos` (task 013): every node also logs the robot's full qpos to $RUN/s<i>/qpos.jsonl (video replay). No map= / slam (no Nav2 on the G1). hpc/plans/p4_part.sh runs unchanged (its env is sim_vars).
# Order: vLLM -> sims started together -> each ready (one restart) -> params -> g1_watch in the background -> plan in a
# subshell (API as sim_job.sbatch: sim_run <i> <cmd>, in_sim, $RUN, $PROJ, $PY_RUN, $PY_G1) -> stop all (trap EXIT).
# Exit: plan's rc; 9 a sim not ready twice / params wrong; 5 vLLM not up; 2 usage / repo not clean / overlay stale / cpus.
# The overlay (sim/env.sh sources it) is sim_job.sbatch's: build it with `bash hpc/sim_job.sbatch --build-overlay`.
set -uo pipefail  # no -e: every step's rc is handled; the trap must always run
PROJ=${PROJ:-$HOME/vlm_agent_project}  # your cluster project dir
REPO=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
STAMP=$PROJ/build/ros_ws/BUILT_FROM_SRC_TREE
[ -f "$REPO/hpc/g1_lib.sh" ] || { echo "run from the repo root ($REPO has no hpc/g1_lib.sh)"; exit 2; }
git() { if command -v /usr/bin/git > /dev/null; then /usr/bin/git "$@"; else apptainer exec -B "$PROJ" "$PROJ/images/sim.sif" git "$@"; fi; }
ST=$(git -C "$REPO" status --porcelain) && [ -z "$ST" ] || { echo "repo $REPO not a clean git checkout"; exit 2; }
SRC_TREE=$(git -C "$REPO" rev-parse HEAD:ros_ws/src) || exit 2

USAGE="usage: sbatch hpc/g1_job.sbatch <plan.sh> <N> [vlm | vlm=<jobid>] [pitch] [qpos] [w<i>=<world> ...] [p:KEY=VALUE ...]"
PLAN=${1:?$USAGE}
NSIM=${2:?$USAGE}
VLM=""; VLM_JOB=""; SIM_WORLDS=(); ARGS_OK=1
export G1_NODE_EXTRA=""   # set only by the `pitch` arg below, never inherited from the submit shell (no env leaks into jobs)
export G1_QPOS_LOG=""    # likewise: set only by the `qpos` arg
for a in "${@:3}"; do
    case $a in
        vlm) VLM=vlm ;;
        pitch) G1_NODE_EXTRA=--pitch-control ;;   # a recorded deviation addendum: every node also renders the pitched G7 control camera
        qpos) G1_QPOS_LOG=1 ;;
        vlm=*) VLM_JOB=${a#vlm=}; [[ $VLM_JOB =~ ^[0-9]+$ ]] || ARGS_OK=0 ;;
        w[0-9]*=*) k=${a%%=*}; k=${k#w}; [[ $k =~ ^[0-9]+$ && ${a#*=} =~ ^[a-z_]+$ ]] && SIM_WORLDS[k]=${a#*=} || ARGS_OK=0 ;;
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
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CVD%%,*}
SIF=$PROJ/images/g1.sif
# shellcheck source=hpc/sim_lib.sh
source "$REPO/hpc/sim_lib.sh"
# shellcheck source=hpc/g1_lib.sh
source "$REPO/hpc/g1_lib.sh"
[ -n "${SIM_FAKE:-}" ] || [ "$(cat "$STAMP" 2>/dev/null)" = "$SRC_TREE" ] ||
    { echo "overlay stale or missing: $STAMP != ros_ws/src tree $SRC_TREE; run hpc/sim_job.sbatch --build-overlay"; exit 2; }
VDESC=${VLM:-${VLM_JOB:+remote $VLM_JOB}}
jlog "=== g1sim115 job $JOB node $(hostname) N=$NSIM vlm=${VDESC:-no} worlds ${SIM_WORLDS[*]:-closed} plan $PLAN $(env | grep '^PLAN_' | sort | tr '\n' ' ')repo $REPO commit $(git -C "$REPO" rev-parse HEAD)"
[ -n "${SIM_FAKE:-}" ] || jlog "sif $SIF sha256 $(sha256sum "$SIF" | cut -c1-64); $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
WATCH=""
g1_stop_all() { [ -n "$WATCH" ] && kill "$WATCH" 2>/dev/null; stop_all; }
trap g1_stop_all EXIT
trap 'jlog "signal: stopping"; exit 143' TERM INT

cpus_init || exit 2
for i in $(seq 0 $((NSIM - 1))); do
    d=$(sim_worlddir "$i") || { jlog "sim $i: no world dir for variant ${SIM_WORLDS[$i]}"; exit 2; }
done
if [ -n "$VLM" ]; then vllm_start || exit 5; fi
if [ -n "$VLM_JOB" ]; then vllm_remote "$VLM_JOB" || exit 5; fi
for i in $(seq 0 $((NSIM - 1))); do sim_start "$i"; done
for i in $(seq 0 $((NSIM - 1))); do
    sim_wait_ready "$i" && continue
    jlog "sim $i restart (not ready)"
    sim_stop "$i"; mv "$RUN/s$i/launch.log" "$RUN/s$i/launch.try1.log"; sim_start "$i"
    sim_wait_ready "$i" || exit 9
done
for i in $(seq 0 $((NSIM - 1))); do sim_params_check "$i" || exit 9; done
[ -n "${SIM_FAKE:-}" ] || nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/gpu mem with sims: /' >> "$RUN/job.log"
g1_watch & WATCH=$!

jlog "plan start $PLAN"
# shellcheck disable=SC1090  # the plan is an argument
( source "$PLAN" ); PRC=$?
jlog "plan end rc=$PRC; node restarts: $(for i in $(seq 0 $((NSIM - 1))); do printf 's%s=%s ' "$i" "$(cat "$RUN/s$i/restarts.txt" 2>/dev/null | wc -l)"; done)"
exit $PRC
