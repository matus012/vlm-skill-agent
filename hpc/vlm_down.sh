#!/bin/bash
# Kill the ssh forward and scancel the vLLM job recorded in $BUILD_DIR/vlm_link.env. Usage: hpc/vlm_down.sh [JOBID]
# The env file is removed only after scancel succeeded, so a failed ssh (e.g. Pritunl down) never loses the job id.
# Also removes the job's API key files on PERUN ($PROJ/runs/serve/<jobid>.key and the pre-submit KEYTMP).
set -euo pipefail
ENVF=${BUILD_DIR:-$HOME/vlm_skill_agent_build}/vlm_link.env
JOB=${1:-}
PROJ=${PROJ:-$HOME/vlm_agent_project}  # your cluster project dir
FWD_PID=""; KEYTMP=""
if [ -f "$ENVF" ]; then
    source "$ENVF"
    JOB=${JOB:-$JOBID}
fi
[ -n "$JOB" ] || { echo "no $ENVF and no JOBID arg"; exit 1; }
if [ -n "$FWD_PID" ]; then
    kill "$FWD_PID" 2>/dev/null && echo "forward $FWD_PID killed" || echo "forward $FWD_PID already gone"
fi
ssh -o BatchMode=yes -o LogLevel=ERROR $LOGIN_NODE "scancel $JOB && rm -f $PROJ/runs/serve/$JOB.key $KEYTMP" ||
    { echo "scancel $JOB FAILED (ssh/Pritunl?) -- $ENVF kept; re-run hpc/vlm_down.sh"; exit 1; }
rm -f "$ENVF"
echo "scancel $JOB sent (check: ssh $LOGIN_NODE squeue -h --me -- absent = done)"
