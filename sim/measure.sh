#!/usr/bin/env bash
# Sizing sampler: gz real-time factor, GPU util/memory, CPU load for SECONDS (default 90) while the sim runs.
# Usage: sim/measure.sh <out_prefix> [seconds]   -> <out_prefix>-{rtf,gpu,cpu,top}.txt
set -eo pipefail
source "$(dirname "$0")/env.sh"
OUT=$1; SECS=${2:-90}
gz topic -e -t /stats -d "$SECS" | grep real_time_factor > "$OUT-rtf.txt" &
RTF=$!
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw --format=csv -l 1 > "$OUT-gpu.txt" &
GPU=$!
mpstat 2 $((SECS / 2)) > "$OUT-cpu.txt" &
CPU=$!
{ sleep $((SECS / 3)); uptime; top -bn1 -o %CPU | head -30; nvidia-smi; } > "$OUT-top.txt"
wait $RTF $CPU
kill $GPU
{ echo; uptime; } >> "$OUT-top.txt"
