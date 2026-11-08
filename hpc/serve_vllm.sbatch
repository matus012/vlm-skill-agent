#!/bin/bash
#SBATCH --job-name=vlm115
#SBATCH --partition=gpu_short
#SBATCH --account=<PERUN_ACCOUNT>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=<PROJECT_DIR>/logs/%x_%j.out
#SBATCH --error=<PROJECT_DIR>/logs/%x_%j.err
# 8 CPUs per job (2026-11-04 ~20:11 CET; project ~21:30 CET: GPU hours only, bundled CPUs not counted; 16 up to job 143731).
# Serve Qwen3.8-27B with vLLM on 1 x H200. Wall is set at submit time (hpc/vlm_up.sh passes --time; default here 2 h).
# Never `sbatch --export`. One positional arg: the path of a 0600 file holding the API key (written by hpc/vlm_up.sh
# before submit; the path is not secret, the key never is on any command line). The job moves it to
# $PROJ/runs/serve/<jobid>.key (so `vlm_up.sh --job` can attach) and hands it to vLLM as env VLLM_API_KEY, which
# `vllm serve` uses exactly like --api-key (v0.32.0 middleware/register.py) but without putting it in `ps`.
# /v1/* then needs `Authorization: Bearer <key>`; /health stays open (v0.32.0 authenticate.py guards only
# /v1 /v2 /inference /cohere). When /health is 200 it writes $PROJ/runs/serve/<jobid>.ready containing
# "<node> <port>". Stop it with scancel (hpc/vlm_down.sh) -- an idle job still burns H200 hours.
set -euo pipefail
PROJ=${PROJ:-$HOME/vlm_agent_project}  # your cluster project dir
SIF=$PROJ/images/vllm.sif
MODEL=$PROJ/models/Qwen3.8-27B
JOB=${SLURM_JOB_ID:?run under sbatch}
KEYIN=${1:?usage: sbatch serve_vllm.sbatch <keyfile>}
T0=$(date +%s)
mkdir -p "$PROJ/runs/serve" "$PROJ/cache/vllm" "$PROJ/cache/hf" "$PROJ/cache/tmp/vllm_$JOB"
[ -f "$SIF" ] && [ -d "$MODEL" ] || { echo "missing $SIF or $MODEL"; exit 1; }
KEYF=$PROJ/runs/serve/$JOB.key
[ -f "$KEYF" ] || mv "$KEYIN" "$KEYF"   # mv keeps the 0600 mode; skipped on a requeue
[ -s "$KEYF" ] && [ "$(stat -c %a "$KEYF")" = 600 ] || { echo "key file $KEYF missing, empty or not 0600"; exit 1; }
export APPTAINERENV_VLLM_API_KEY=$(cat "$KEYF")   # passed through --cleanenv; /proc/<pid>/environ is owner-only
NGPU=$(nvidia-smi -L | wc -l); [ "$NGPU" -eq 1 ] || { echo "expected 1 GPU, got $NGPU"; exit 1; }
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])')
NODE=$(hostname)
echo "=== vlm115 job $JOB node $NODE port $PORT $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) sif $(basename "$SIF") start $(date -Is)"

export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
apptainer exec --nv --cleanenv -B "$PROJ" \
    --env HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_HOME=$PROJ/cache/hf,VLLM_CACHE_ROOT=$PROJ/cache/vllm,TMPDIR=$PROJ/cache/tmp/vllm_$JOB,VLLM_NO_USAGE_STATS=1,DO_NOT_TRACK=1,PYTHONUNBUFFERED=1 \
    "$SIF" vllm serve "$MODEL" \
    --served-model-name qwen3.8-27b \
    --host 0.0.0.0 --port "$PORT" \
    --tensor-parallel-size 1 --max-model-len 32768 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{"image": 4}' &
VPID=$!

# health watcher: bounded 30 min, aborts if vllm dies
for _ in $(seq 360); do
    kill -0 "$VPID" 2>/dev/null || { echo "vllm exited before healthy"; wait "$VPID"; exit 1; }
    if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" = 200 ]; then break; fi
    sleep 5
done
[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" = 200 ] || { echo "not healthy after 30 min"; kill "$VPID"; exit 1; }
echo "=== healthy: load_time_s $(( $(date +%s) - T0 )) (job start -> /health 200)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
echo "$NODE $PORT" > "$PROJ/runs/serve/$JOB.ready"
echo "=== ready file written $(date -Is)"
wait "$VPID"
