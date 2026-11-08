#!/bin/bash
# Copy a sim job's run dir from PERUN to the local machine (hpc/PERUN_SIM.md section 1):  hpc/fetch_runs.sh <jobid> <local dir>
# rsync of $PROJ/runs/sim/<jobid>/ into <local dir>/ (created; existing files are updated, nothing deleted).
set -euo pipefail
JOB=${1:?usage: fetch_runs.sh <jobid> <local dir>}
DST=${2:?usage: fetch_runs.sh <jobid> <local dir>}
[[ $JOB =~ ^[0-9]+$ ]] || { echo "jobid must be a number"; exit 2; }
mkdir -p "$DST"
rsync -a -e "ssh -o BatchMode=yes -o LogLevel=ERROR" \
    "$LOGIN_NODE:$PROJ/runs/sim/$JOB/" "$DST/"
echo "fetched job $JOB -> $DST ($(find "$DST" -type f | wc -l) files)"
