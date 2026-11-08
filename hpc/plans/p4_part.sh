# 007: this job's part (PLAN_J) of a cross-job invocation PLAN_D of list PLAN_LIST (hpc/PERUN_SIM.md section 5). The
# split was written before on the login node (`phase3_run.py --split`); --merge and the gates run on the login node after every job
# ended. Each sim's world goes to its worker as the :<world> field of --shard-env. PLAN_RESUME=1: --resume (only this
# job's entries with no run dir). pinning.txt: live snapshots.
#   sbatch --time=.. --cpus-per-task=64 --mem=400G hpc/sim_job.sbatch hpc/plans/p4_part.sh 2 vlm [w1=cabinet_open] \
#       p:LIST=tasks/phase4_dev.yaml p:D=$PROJ/runs/p4/<name> p:J=<j> [p:RESUME=1]
PY=/opt/venv/bin/python
[ -n "${PLAN_LIST:-}" ] && [ -n "${PLAN_D:-}" ] && [ -n "${PLAN_J:-}" ] || { echo "p4_part: need p:LIST, p:D, p:J"; exit 2; }
ENVS=(); for i in $(seq 0 $((NSIM - 1))); do ENVS+=(--shard-env "$i=$(sim_dom "$i"):$(sim_part "$i"):$(sim_world "$i")"); done
pin_snapshot "sims ready, idle"
( sleep 600; pin_snapshot "job part, +600 s"; sleep 1200; pin_snapshot "job part, +1800 s" ) & SNAP=$!
sim_run 0 $PY probes/phase3_run.py --list "$PLAN_LIST" --job-part "$PLAN_D" --job "$PLAN_J" --shards "$NSIM" \
    "${ENVS[@]}" ${PLAN_RESUME:+--resume} || echo "job part rc=$?"
kill "$SNAP" 2>/dev/null || true
