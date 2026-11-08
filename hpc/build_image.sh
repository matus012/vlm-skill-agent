#!/bin/bash
# Run on the login node (idempotent). Usage: nohup bash build_image.sh [vllm-version] > $P/logs/build_image.log 2>&1 &
set -euo pipefail
VER=${1:-0.32.0}
P=${PROJ:-$HOME/vlm_agent_project}  # your cluster project dir
mkdir -p "$P"/{images,logs,cache/apptainer,cache/tmp}
export APPTAINER_CACHEDIR=$P/cache/apptainer APPTAINER_TMPDIR=$P/cache/tmp
SIF=$P/images/vllm-$VER.sif
if [ ! -s "$SIF" ]; then
  rm -f "$SIF.part"
  apptainer build "$SIF.part" "docker://vllm/vllm-openai:v$VER"
  mv "$SIF.part" "$SIF"
fi
ln -sfn "vllm-$VER.sif" "$P/images/vllm.sif"
ls -lh "$SIF" "$P/images/vllm.sif"
echo "build done $(date -Is)"
