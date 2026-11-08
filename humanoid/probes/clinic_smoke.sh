#!/usr/bin/env bash
# Clinic smoke under the shared gz.lockdir: a g1 node on the MuJoCo clinic, then clinic_smoke.py through G1Adapter.
#   humanoid/probes/clinic_smoke.sh [rooms ...]      (default: storage)
# Hard cap 18 min of holding the lock (shared rule: <= 20 min per session). Stops only its own process group.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="$REPO/humanoid/run_g1.sh"
export SIDEH_EVIDENCE="${SIDEH_EVIDENCE:-.cache/vlm_skill_agent/runs/2026-11-04/003-H}"
LOCK="${BUILD_DIR:-$HOME/vlm_skill_agent_build}/gz.lockdir"
LOGS="${BUILD_DIR:-$HOME/vlm_skill_agent_build}/logs/sideH-003"
mkdir -p "$SIDEH_EVIDENCE" "$LOGS"
NODE_PID=""

until mkdir "$LOCK" 2>/dev/null; do sleep 15; done
echo "H $(date -u +%T)" > "$LOCK/owner"
cleanup() {
  if [ -n "$NODE_PID" ]; then kill -TERM -- "-$NODE_PID" 2>/dev/null || true; wait "$NODE_PID" 2>/dev/null || true; fi
  rm -rf "$LOCK"
}
trap cleanup EXIT
echo "lock taken $(date -u +%T)"
setsid "$RUN" "$REPO/humanoid/g1_node.py" --clinic "$REPO/sim/worlds/clinic/layout.yaml" \
  --stats "$SIDEH_EVIDENCE/sideH-003-node.json" > "$LOGS/node.log" 2>&1 &
NODE_PID=$!
sleep 10
timeout 1020 "$RUN" -m humanoid.probes.clinic_smoke "${@:-storage}" 2>&1 | tee "$LOGS/smoke.log"
echo "smoke done $(date -u +%T)"
