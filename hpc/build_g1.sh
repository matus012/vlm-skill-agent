#!/usr/bin/env bash
# Build the 115 G1 image on the PERUN login node (task 012), on top of images/sim.sif (build_sim.sh pattern).
# Launch from the local machine (repo root): stages g1.def + requirements + the pinned clones, then runs this under nohup on the login node:
#   P=$PROJ; E=$BUILD_DIR/ext
#   ssh $LOGIN_NODE mkdir -p $P/build/g1/ext/unitree_mujoco/unitree_robots $P/logs && rsync -a hpc/g1.def hpc/build_g1.sh hpc/g1_requirements.txt $LOGIN_NODE:$P/build/g1/ && rsync -a --exclude .git $E/agile $LOGIN_NODE:$P/build/g1/ext/ && rsync -a $E/unitree_mujoco/unitree_robots/g1 $LOGIN_NODE:$P/build/g1/ext/unitree_mujoco/unitree_robots/ && ssh $LOGIN_NODE "nohup bash $P/build/g1/build_g1.sh > $P/logs/build_g1.log 2>&1 < /dev/null &"
# Out: $P/images/g1.sif, g1.sif.sha256, g1.inputs.sha256, g1.versions.txt; log $P/logs/build_g1.log (BUILD_OK).
set -euo pipefail
P=${PROJ:-$HOME/vlm_agent_project}  # your cluster project dir
B=$P/build/g1 SIF=$P/images/g1.sif
mkdir -p "$P/images" "/tmp/$USER-g1"
export APPTAINER_TMPDIR=/tmp/$USER-g1 APPTAINER_CACHEDIR=/tmp/$USER-g1/cache
cd "$B"
inputs=$( { sha256sum g1.def g1_requirements.txt "$P/images/sim.sif.sha256"; find ext -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum; } )
if [[ -s $SIF && "$(cat "$SIF.inputs.sha256" 2>/dev/null)" == "$inputs" ]]; then
  echo "g1.sif up to date with the staged inputs, no rebuild"
else
  echo "build start $(date -Is)"
  apptainer build --fakeroot --force "$SIF.tmp" g1.def
  mv "$SIF.tmp" "$SIF"
  sha256sum "$SIF" > "$SIF.sha256"
  echo "$inputs" > "$SIF.inputs.sha256"
  echo "build end $(date -Is)"
fi
{ echo "# $(cat "$SIF.sha256")  built $(date -Is -r "$SIF")  from sim.sif $(cut -c1-16 "$P/images/sim.sif.sha256")"
  echo "# /opt/venv-g1"
  apptainer exec "$SIF" /usr/local/bin/uv pip list --python /opt/venv-g1/bin/python
  echo "# clones"
  echo "agile f5c388ebc7c1b056de891e3edb8f47693a50d5d3; unitree_mujoco 1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d (unitree_robots/g1)"
} > "$P/images/g1.versions.txt"
ls -l "$SIF"; cat "$SIF.sha256"
echo BUILD_OK
