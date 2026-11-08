#!/usr/bin/env bash
# Side-H rehearsal: a fresh g1 node per probe (spawn pose each time), H1 (+ load sampling), H2 then H3 on the same
# node, H4 with the truth log, self-view; then one unpaced run for the max real-time factor.
# Takes the shared gz.lockdir (GPU timing measurement) for the whole run, ~8 min; releases it on exit.
# Stops only its own process groups. Evidence: $SIDEH_EVIDENCE (default .cache/vlm_skill_agent/runs/2026-11-04).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="$REPO/humanoid/run_g1.sh"
EV="${SIDEH_EVIDENCE:-.cache/vlm_skill_agent/runs/2026-11-04}"
LOCK="${BUILD_DIR:-$HOME/vlm_skill_agent_build}/gz.lockdir"
LOGS="${BUILD_DIR:-$HOME/vlm_skill_agent_build}/logs/sideH"
mkdir -p "$EV" "$LOGS"
NODE_PID=""

until mkdir "$LOCK" 2>/dev/null; do sleep 15; done
echo "g1-115 $(date -u +%T)" > "$LOCK/owner"
cleanup() { [ -n "$NODE_PID" ] && kill -TERM -- "-$NODE_PID" 2>/dev/null || true; rm -rf "$LOCK"; }
trap cleanup EXIT

start_node() {  # $1 = tag, rest = node args
  local tag=$1; shift
  setsid "$RUN" "$REPO/humanoid/g1_node.py" --stats "$EV/sideH-node-$tag.json" "$@" > "$LOGS/node-$tag.log" 2>&1 &
  NODE_PID=$!
  sleep 10
}
stop_node() { kill -TERM -- "-$NODE_PID"; wait "$NODE_PID" 2>/dev/null || true; NODE_PID=""; }

start_node H1
"$RUN" -m humanoid.probes.load "$NODE_PID" H1 & LOAD_PID=$!
"$RUN" -m humanoid.probes.h1_walk > "$LOGS/H1.log" 2>&1 || echo "H1 probe exited $?"
kill -TERM "$LOAD_PID"; wait "$LOAD_PID" || true
stop_node

start_node H2
"$RUN" -m humanoid.probes.h2_turn > "$LOGS/H2.log" 2>&1 || echo "H2 probe exited $?"
"$RUN" -m humanoid.probes.h3_camera > "$LOGS/H3.log" 2>&1 || echo "H3 probe exited $?"
stop_node

start_node H4 --truth-log "$LOGS/truth-H4.jsonl"
"$RUN" -m humanoid.probes.h4_pose --truth-log "$LOGS/truth-H4.jsonl" > "$LOGS/H4.log" 2>&1 || echo "H4 probe exited $?"
stop_node

start_node selfview
"$RUN" -m humanoid.probes.selfview > "$LOGS/selfview.log" 2>&1 || echo "selfview probe exited $?"
stop_node

start_node unpaced --rate 0   # standing, camera on: max achievable RTF
sleep 30
stop_node
grep -h '"verdict"' "$EV"/sideH-H*.json
