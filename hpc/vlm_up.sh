#!/bin/bash
# Bring the PERUN vLLM up and forward it to the local machine.
# Usage: hpc/vlm_up.sh [--job JOBID] [--wall HH:MM:SS] [--port LOCALPORT]   (defaults: submit new, 02:00:00, 8000)
#   --job attaches to an already running serve job instead of submitting.
# Needs Pritunl up. Writes $BUILD_DIR/vlm_link.env (0600: BASE_URL, JOBID, FWD_PID, API_KEY, KEYTMP)
# -- JOBID right after submit, so hpc/vlm_down.sh can always cancel the job, even if this script fails later.
# API key: a fresh random key per job (openssl rand -hex 16), sent over ssh stdin into a 0600 file under
# $PROJ/runs/serve/ before submit; only that file's path goes to sbatch (as the script's positional arg). The key is
# never on a command line (ps/squeue) and never in --export. The local check sends it as a header read from stdin.
set -euo pipefail
LOGIN=$LOGIN_NODE
PROJ=${PROJ:-$HOME/vlm_agent_project}  # your cluster project dir
HERE=$(cd "$(dirname "$0")" && pwd)
ENVF=${BUILD_DIR:-$HOME/vlm_skill_agent_build}/vlm_link.env
SSH=(ssh -o BatchMode=yes -o LogLevel=ERROR "$LOGIN")
JOB=""; WALL=02:00:00; LPORT=8000; SUBMITTED=0
while [ $# -gt 0 ]; do
    case $1 in
        --job) JOB=$2; shift 2 ;;
        --wall) WALL=$2; shift 2 ;;
        --port) LPORT=$2; shift 2 ;;
        *) echo "unknown arg $1 (usage: --job ID --wall HH:MM:SS --port N)"; exit 2 ;;
    esac
done
mkdir -p "$(dirname "$ENVF")"
[ -f "$ENVF" ] && { echo "$ENVF exists: run hpc/vlm_down.sh first"; exit 1; }
umask 077   # the env file holds the key
KEY=""; KEYTMP=""

if [ -z "$JOB" ]; then
    # copy the script to $PROJ so sbatch reads a file; no --export
    scp -q -o BatchMode=yes "$HERE/serve_vllm.sbatch" "$LOGIN:$PROJ/serve_vllm.sbatch"
    KEY=$(openssl rand -hex 16)
    KEYTMP=$PROJ/runs/serve/pending_$(date -u +%Y%m%dT%H%M%S)_$$.key
    "${SSH[@]}" "umask 077 && mkdir -p $PROJ/runs/serve && cat > $KEYTMP" <<<"$KEY"
    JOB=$("${SSH[@]}" "cd $PROJ && sbatch --parsable --time=$WALL serve_vllm.sbatch $KEYTMP") ||
        { "${SSH[@]}" "rm -f $KEYTMP"; echo "sbatch failed"; exit 1; }
    JOB=${JOB%%;*}
    SUBMITTED=1
    echo "submitted job $JOB (wall $WALL)"
fi
printf 'BASE_URL=\nJOBID=%s\nFWD_PID=\nAPI_KEY=\nKEYTMP=%s\n' "$JOB" "$KEYTMP" > "$ENVF"
# On any failure after submit: cancel the job this script submitted (never an attached --job), so a pending or
# running job can't burn H200 hours unnoticed. If scancel itself fails, the env file keeps the id for vlm_down.sh.
fail() {
    echo "$1"
    if [ "$SUBMITTED" = 1 ]; then
        "${SSH[@]}" "scancel $JOB && rm -f $PROJ/runs/serve/$JOB.key $KEYTMP" &&
            { rm -f "$ENVF"; echo "job $JOB cancelled, key files removed"; } ||
            echo "scancel $JOB FAILED -- run hpc/vlm_down.sh (job id kept in $ENVF)"
    else
        echo "attached job $JOB left running: hpc/vlm_down.sh cancels it"
    fi
    exit 1
}

# wait (poll 30 s, max 40 min) for RUNNING, then for the ready file
READY=""; ERRS=0
for _ in $(seq 80); do
    # squeue --me (not -j: -j errors once the job is gone); an ssh/squeue failure is never read as "gone"
    if ! Q=$("${SSH[@]}" "squeue -h --me -o '%i %T'"); then
        ERRS=$((ERRS + 1)); [ "$ERRS" -lt 3 ] || fail "squeue failed 3 times"; sleep 30; continue
    fi
    ERRS=0
    ST=$(awk -v j="$JOB" '$1 == j {print $2}' <<<"$Q")
    [ -n "$ST" ] || { echo "job $JOB left the queue before ready; see $PROJ/logs/vlm115_$JOB.out";
                      "${SSH[@]}" "rm -f $PROJ/runs/serve/$JOB.key $KEYTMP" && rm -f "$ENVF"; exit 1; }
    READY=$("${SSH[@]}" "cat $PROJ/runs/serve/$JOB.ready 2>/dev/null" || true)
    [ -n "$READY" ] && break
    echo "job $JOB: $ST, not ready yet"; sleep 30
done
[ -n "$READY" ] || fail "timeout waiting for ready"
read -r NODE PORT <<<"$READY"
# --job attach: the running job moved its key to <jobid>.key (0600); read it over ssh stdout, not argv
[ -n "$KEY" ] || KEY=$("${SSH[@]}" "cat $PROJ/runs/serve/$JOB.key") || fail "cannot read key of job $JOB"

ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -N -L "$LPORT:$NODE:$PORT" "$LOGIN" &
FWD=$!
BASE=http://127.0.0.1:$LPORT/v1
# the key goes in as a header read from stdin (curl -H @-), so it is not in curl's argv
models() { curl -sf -H @- "$BASE/models" <<<"Authorization: Bearer $KEY" >/dev/null; }
for _ in $(seq 20); do
    models && break
    kill -0 "$FWD" 2>/dev/null || fail "forward died"
    sleep 3
done
models || { kill "$FWD"; fail "no answer on $BASE/models with the key"; }
NOKEY=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/models")
[ "$NOKEY" = 401 ] || { kill "$FWD"; fail "$BASE/models without the key answered $NOKEY, expected 401"; }
printf 'BASE_URL=%s\nJOBID=%s\nFWD_PID=%s\nAPI_KEY=%s\nKEYTMP=%s\n' "$BASE" "$JOB" "$FWD" "$KEY" "$KEYTMP" > "$ENVF"
echo "up: BASE_URL=$BASE JOBID=$JOB FWD_PID=$FWD (node $NODE:$PORT), key in $ENVF (401 without it: checked)"
