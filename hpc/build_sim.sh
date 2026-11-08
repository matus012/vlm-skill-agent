#!/usr/bin/env bash
# Build the 115 clinic-sim image on the PERUN login node (task 006 M1; 114 pattern: --fakeroot, tmp on node-local /tmp).
# Idempotent: skips the build when sim.sif exists and was built from the same staged inputs (sim.def, pyproject.toml,
# uv.lock); the versions file is rewritten either way.
# Launch from the local machine (repo root) -- stages the inputs, then starts this script under nohup on the login node:
#   P=$PROJ; ssh $LOGIN_NODE mkdir -p $P/build/sim $P/logs && rsync hpc/sim.def hpc/build_sim.sh pyproject.toml uv.lock $LOGIN_NODE:$P/build/sim/ && ssh $LOGIN_NODE "nohup bash $P/build/sim/build_sim.sh > $P/logs/build_sim.log 2>&1 < /dev/null &"
# Out: $P/images/sim.sif, sim.sif.sha256, sim.inputs.sha256, sim.versions.txt; log $P/logs/build_sim.log (BUILD_OK).
set -euo pipefail
P=${PROJ:-$HOME/vlm_agent_project}  # your cluster project dir
B=$P/build/sim SIF=$P/images/sim.sif
mkdir -p "$P/images" "/tmp/$USER-sim"
export APPTAINER_TMPDIR=/tmp/$USER-sim APPTAINER_CACHEDIR=/tmp/$USER-sim/cache
cd "$B"
inputs=$(sha256sum sim.def pyproject.toml uv.lock)
if [[ -s $SIF && "$(cat "$SIF.inputs.sha256" 2>/dev/null)" == "$inputs" ]]; then
  echo "sim.sif up to date with the staged inputs, no rebuild"
else
  echo "build start $(date -Is)"
  apptainer build --fakeroot --force "$SIF.tmp" sim.def
  mv "$SIF.tmp" "$SIF"
  sha256sum "$SIF" > "$SIF.sha256"
  echo "$inputs" > "$SIF.inputs.sha256"
  echo "build end $(date -Is)"
fi
{ echo "# $(cat "$SIF.sha256")  built $(date -Is -r "$SIF")"
  apptainer exec "$SIF" dpkg-query -W -f='${Package} ${Version}\n' 'ros-jazzy-ros-gz*' ros-jazzy-navigation2 \
    ros-jazzy-nav2-bringup 'ros-jazzy-turtlebot4*' 'ros-jazzy-irobot-create*' ros-jazzy-ros-base ros-jazzy-ros-core \
    ros-jazzy-xacro ros-jazzy-fastrtps ros-jazzy-rmw-fastrtps-cpp ros-jazzy-cv-bridge ros-jazzy-slam-toolbox \
    'ros-jazzy-gz-*-vendor' python3-colcon-common-extensions python3-yaml python3-opencv python3-numpy procps
  echo "# /opt/venv"
  apptainer exec "$SIF" /usr/local/bin/uv pip list --python /opt/venv/bin/python
  echo "# gz"
  apptainer exec "$SIF" bash -c 'source /opt/ros/jazzy/setup.bash && gz sim --version | head -1'
} > "$P/images/sim.versions.txt"
ls -l "$SIF"; cat "$SIF.sha256"
echo BUILD_OK
