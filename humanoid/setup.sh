#!/usr/bin/env bash
# Reproduce the side-H G1 environment on the local machine (Ubuntu 24.04, ROS 2 Jazzy installed): pinned clones + the g1 venv.
# Everything lands in $BUILD_DIR/ (ext4), nothing in the repo. ~1.9 GB. Idempotent.
# Never `uv sync` / `pip install -e` AGILE's pyproject: it pins isaaclab[isaacsim] + torch cu128 (Isaac Sim).
set -euo pipefail
BASE="${BUILD_DIR:-$HOME/vlm_skill_agent_build}"
EXT="$BASE/ext"
VENV="$BASE/.venv-g1"
AGILE_COMMIT=f5c388ebc7c1b056de891e3edb8f47693a50d5d3          # nvidia-isaac/WBC-AGILE main, 2026-10-23
UNITREE_MUJOCO_COMMIT=1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d # unitreerobotics/unitree_mujoco main, 2026-09-07
mkdir -p "$EXT"

if [ ! -d "$EXT/agile" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/nvidia-isaac/WBC-AGILE "$EXT/agile"
fi
git -C "$EXT/agile" -c advice.detachedHead=false checkout -q "$AGILE_COMMIT"
# Only the Velocity-G1-v0 LEAPP bundle (ONNX + yaml + initial values), not every LFS blob.
git -C "$EXT/agile" lfs pull --include="agile/data/policy/velocity_g1/leapp/**"

if [ ! -d "$EXT/unitree_mujoco" ]; then
  git clone -q https://github.com/unitreerobotics/unitree_mujoco "$EXT/unitree_mujoco"
fi
git -C "$EXT/unitree_mujoco" fetch -q --depth 1 origin "$UNITREE_MUJOCO_COMMIT" 2>/dev/null || true
git -C "$EXT/unitree_mujoco" -c advice.detachedHead=false checkout -q "$UNITREE_MUJOCO_COMMIT"

# Python 3.12 = ROS Jazzy's system Python; --system-site-packages so ROS python deps resolve (rclpy itself comes
# from /opt/ros via PYTHONPATH when sim/env.sh is sourced).
[ -d "$VENV" ] || uv venv --python /usr/bin/python3.12 --system-site-packages "$VENV"
export VIRTUAL_ENV="$VENV"
# CPU torch first and pinned, so leapp's `torch>=2.6` does not pull a CUDA wheel from PyPI.
uv pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.11.0+cpu"
# leapp pin from the bundle's yaml (leapp version 0.5.2). pandas/seaborn/plotly/jinja2/tqdm/scipy/matplotlib only
# because agile.sim2mujoco's package __init__ imports agile.algorithms.evaluation (plotting) — and the venv copies of
# scipy/matplotlib keep Ubuntu's numpy-1.x builds out of the numpy-2 venv.
uv pip install "leapp==0.5.2" "mujoco==3.17.0" "onnxruntime==1.31.1" "pyyaml==6.0.3" "imageio==2.38.1" \
  "pandas==3.1.0" "seaborn==0.13.2" "plotly==7.1.0" "jinja2==3.1.6" "tqdm==4.70.1" "scipy==1.18.1" \
  "matplotlib==3.11.2" "pytest==9.2.0"
"$VENV/bin/python" -c "import torch, mujoco, leapp; assert torch.version.cuda is None; print('ok', torch.__version__, mujoco.__version__)"
