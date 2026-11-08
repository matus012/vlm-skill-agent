#!/bin/bash
# Run on the login node (idempotent, resumable). Usage: nohup bash fetch_weights.sh > $P/logs/fetch_weights.log 2>&1 &
set -euo pipefail
P=${PROJ:-$HOME/vlm_agent_project}  # your cluster project dir
REPO=${1:-Qwen/Qwen3.8-27B}
mkdir -p "$P"/{logs,models,cache/hf,cache/uv,tools}
export HF_HOME=$P/cache/hf UV_CACHE_DIR=$P/cache/uv
[ -x "$P/tools/venv/bin/hf" ] || { uv venv --python 3.11 "$P/tools/venv"; uv pip install -p "$P/tools/venv/bin/python" "huggingface_hub[hf_xet]"; }
"$P/tools/venv/bin/hf" download "$REPO" --local-dir "$P/models/$(basename "$REPO")"
echo "fetch done $(date -Is)"
