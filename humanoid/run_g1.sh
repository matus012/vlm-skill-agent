#!/usr/bin/env bash
# Run a G1-lane python entry point with ROS Jazzy + the project's localhost DDS profile + the g1 venv.
#   humanoid/run_g1.sh humanoid/g1_node.py [args]       # the sim node
#   humanoid/run_g1.sh humanoid/probes/h1_walk.py ...   # a probe
# ROS_DOMAIN_ID defaults to 18 (side lane H), so Stage 1 sims on 15 never see it.
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-18}
source "$REPO/sim/env.sh"
export MUJOCO_GL=egl
export PYTHONPATH="$REPO:${BUILD_DIR:-$HOME/vlm_skill_agent_build}/ext/agile${PYTHONPATH:+:$PYTHONPATH}"
exec "${BUILD_DIR:-$HOME/vlm_skill_agent_build}/.venv-g1/bin/python" "$@"
