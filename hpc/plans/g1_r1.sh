# 012 R1 (design section 8): G1 instrument checks, no vLLM, no bar beyond the probes' own; 3 sims (closed, cabinet_open,
# blocked_storage: the mocap read-back needs both counted worlds, the blocked check its own world).
#   sbatch --time=00:55:00 --cpus-per-task=64 --mem=256G hpc/g1_job.sbatch hpc/plans/g1_r1.sh 3 w1=cabinet_open w2=blocked_storage
# Ready + params: g1_job.sbatch itself (s<i>/ready.json, params_check.txt). Then in parallel:
#   sim 0 closed:        mocap read-back (synthetic dev + regress specs), drift probe hall/ward/office, smoke find-toolbox
#   sim 1 cabinet_open:  mocap read-back, drift probe reception/storage, smoke count-chairs
#   sim 2 blocked_storage: 3 x navigate_to storage -> failed: blocked (phase6_pre P1 rule)
# then the nodes stay up until LOG_HOLD_S of plan time (the 45 min FastDDS log-advance check; g1_watch restarts a
# stalled node and logs it) and r1_checks logs summarizes every node log. Out: $RUN/r1/.
# Also checked here (design section 9): the runner python imports the adapter without mujoco / torch.
LOG_HOLD_S=2700  # design section 8 R1: FastDDS log-advance over 45 min
R1=$RUN/r1
[ "$(sim_world 0) $(sim_world 1) $(sim_world 2)" = "closed cabinet_open blocked_storage" ] ||
    { echo "g1_r1: need w1=cabinet_open w2=blocked_storage"; exit 2; }
P0=$(date +%s)
sim_run 0 "$PY_RUN" -c 'import sys, humanoid.adapter_g1; bad = {"mujoco", "torch"} & set(sys.modules); print("runner imports adapter_g1, heavy:", sorted(bad)); sys.exit(bool(bad))' ||
    echo "g1_r1: adapter import pulled mujoco / torch into the runner"
pin_snapshot "sims ready, idle"
( sim_run 0 "$PY_RUN" humanoid/probes/r1_checks.py mocap --out "$R1"
  sim_run 0 "$PY_RUN" humanoid/probes/r1_checks.py drift --out "$R1" --places hall,ward,office
  sim_run 0 "$PY_RUN" adapters/run_task.py --task find-toolbox --out "$R1/smoke/find-toolbox" ) & A=$!
( sim_run 1 "$PY_RUN" humanoid/probes/r1_checks.py mocap --out "$R1"
  sim_run 1 "$PY_RUN" humanoid/probes/r1_checks.py drift --out "$R1" --places reception,storage
  sim_run 1 "$PY_RUN" adapters/run_task.py --task count-chairs --out "$R1/smoke/count-chairs" ) & B=$!
sim_run 2 "$PY_RUN" humanoid/probes/r1_checks.py blocked --out "$R1" & C=$!
RC=0
for p in $A $B $C; do wait "$p" || RC=1; done
jlog "g1_r1 probes done rc=$RC; holding the nodes to +$LOG_HOLD_S s of plan time"
while [ $(( $(date +%s) - P0 )) -lt "$LOG_HOLD_S" ]; do sleep 30; done
pin_snapshot "after hold"
sim_run 0 "$PY_RUN" humanoid/probes/r1_checks.py logs --run "$RUN" || RC=1
exit $RC
