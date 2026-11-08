# 009 preconditions P1 + P2 (INTERFACES section 11), live, no vLLM: every sim of the job runs a blocked_<room> world on
# the movables-free map and runs probes/phase6_pre.py all (P1: 10 runtime runs navigate_to(<blocked room>) -> failed:
# blocked, then hall; P2: 10 Phase 1 G1 navigations over the reachable places) in parallel, out dir $RUN/p6pre/s<i>.
#   sbatch --time=01:45:00 --cpus-per-task=64 --mem=400G hpc/sim_job.sbatch hpc/plans/p6_pre.sh 2 w0=blocked_storage w1=blocked_reception map=clinic_static
# Refuses (rc 2, before any probe) a sim whose world is not blocked_* or a map other than clinic_static.
# Estimate per sim: P1 10 tries x (setup move + blocked nav <= 120 s + hall nav + photos) ~ 60-200 s sim each, P2 11
# navigations x ~25-40 s sim -> ~15-40 min sim, ~25-60 min wall at RTF ~0.7, + ~5 min startup.
PY=/opt/venv/bin/python
[ "$SIM_MAP" = clinic_static ] || { echo "p6_pre: need map=clinic_static (got '${SIM_MAP}')"; exit 2; }
for i in $(seq 0 $((NSIM - 1))); do
    [[ $(sim_world "$i") == blocked_* ]] || { echo "p6_pre: sim $i world $(sim_world "$i") is not blocked_*"; exit 2; }
done
PIDS=()
for i in $(seq 0 $((NSIM - 1))); do
    # p1 and p2 in separate processes: P1's adapter shuts rclpy down on close; a fresh process avoids a re-init in one
    ( sim_run "$i" $PY probes/phase6_pre.py p1 --out "$RUN/p6pre/s$i"; r1=$?
      sim_run "$i" $PY probes/phase6_pre.py p2 --out "$RUN/p6pre/s$i"; r2=$?
      [ $r1 = 0 ] && [ $r2 = 0 ] ) & PIDS[i]=$!
done
RC=0
for i in $(seq 0 $((NSIM - 1))); do
    wait "${PIDS[i]}" || { echo "p6_pre sim $i ($(sim_world "$i")) rc=$?"; RC=1; }
done
exit $RC
