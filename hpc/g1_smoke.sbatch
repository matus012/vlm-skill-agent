#!/bin/bash
#SBATCH --job-name=g1smoke115
#SBATCH --partition=gpu_short
#SBATCH --account=<PERUN_ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=00:20:00
#SBATCH --output=<PROJECT_DIR>/logs/%x_%j.out
#SBATCH --error=<PROJECT_DIR>/logs/%x_%j.err
# 012 early check (no bar): g1.sif on a GPU node -- MuJoCo EGL under --nv, and the G1 clinic node's unpaced real-time
# factor with N = 1, 2, 4, 6 nodes side by side (camera on, AGILE policy on CPU). Each node: own ROS_DOMAIN_ID, own
# fake HOME (side-H paths: $BUILD_DIR/{ext,.venv-g1,g1_scene}). Submit from the clean clone:
#   cd $PROJ/repo && sbatch hpc/g1_smoke.sbatch            Out: $PROJ/runs/g1smoke/<jobid>/ (stats per node, logs)
set -uo pipefail
PROJ=${PROJ:-$HOME/vlm_agent_project}  # your cluster project dir
REPO=${SLURM_SUBMIT_DIR:?}
OUT=$PROJ/runs/g1smoke/$SLURM_JOB_ID; mkdir -p "$OUT"
SIF=$PROJ/images/g1.sif
log() { echo "$(date -Is) $*" | tee -a "$OUT/job.log"; }
log "node $(hostname) cpus $(nproc) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
home_for() {  # $1 = tag
  local h=$OUT/fakehome/$1 b; b=$h/vlm_skill_agent_build
  mkdir -p "$b/g1_scene"; ln -sfn /opt/g1ext "$b/ext"; ln -sfn /opt/venv-g1 "$b/.venv-g1"; echo "$h"
}
for N in 1 2 4 6; do
  pids=()
  for i in $(seq 0 $((N-1))); do
    tag=n${N}_$i; h=$(home_for "$tag")
    setsid apptainer exec --nv --cleanenv --home "$h" -B "$PROJ" --pwd "$REPO" \
      --env WS_LOCAL=$PROJ/build --env ROS_DOMAIN_ID=$((40+2*i)) "$SIF" \
      bash humanoid/run_g1.sh humanoid/g1_node.py --clinic sim/worlds/clinic/layout.yaml --rate 0 \
      --stats "$OUT/stats-$tag.json" > "$OUT/node-$tag.log" 2>&1 &
    pids+=($!)
  done
  sleep 120
  for p in "${pids[@]}"; do kill -TERM -- "-$p" 2>/dev/null; done
  for p in "${pids[@]}"; do wait "$p" 2>/dev/null; done
  sleep 3
  log "N=$N: $(for f in "$OUT"/stats-n${N}_*.json; do python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[1].split('stats-')[1], {k: d[k] for k in d if 'rtf' in k.lower() or 'min' in k.lower()})" "$f" 2>/dev/null || echo "$f missing"; done | tr '\n' ' ')"
done
log "done"
