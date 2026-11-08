# 006 throughput measure (no gate; 2026-11-05: runs per hour at N = 2 vs 4 vs 8 for the owner's choice). N sims
# (the job's N) + vLLM, each sim pinned to its own cpus (hpc/cpu_split.py); isolation + idle RTF over all sims, then the
# 005 dev list twice, each sharded over the N sims (2 x 8 runs); the coordinator, its shard runtimes and vLLM on the
# job's cpus no sim holds (sim_lib cpus_init, M5 round 3 layout). Runs per hour = 16 / wall of the two invocations
# (t0 / t1 below); RTF under load = run.json sim_s / wall_s per run; pinning.txt = live snapshots during each list.
# 80 cpus = 20 whole 4-core L3 groups: N = 4 -> 4 x 16 + 16 client; N = 8 -> 8 x 8 + 16 client:
#   sbatch --time=01:15:00 --cpus-per-task=80 --mem=400G hpc/sim_job.sbatch hpc/plans/measure_n.sh 4 vlm
#   sbatch --time=01:15:00 --cpus-per-task=80 --mem=400G hpc/sim_job.sbatch hpc/plans/measure_n.sh 8 vlm
PY=/opt/venv/bin/python
SIMS=(); ENVS=()
for i in $(seq 0 $((NSIM - 1))); do
    SIMS+=(--sim "$(sim_dom "$i"):$(sim_part "$i")"); ENVS+=(--shard-env "$i=$(sim_dom "$i"):$(sim_part "$i")")
done
sim_run 0 $PY probes/crosstalk.py --out "$RUN/m5n$NSIM" "${SIMS[@]}" || echo "M5 n$NSIM rc=$?"
date -Is > "$RUN/throughput_t0.txt"
for k in a b; do
    ( sleep 300; pin_snapshot "N=$NSIM dev list $k, +300 s" ) & SNAP=$!
    sim_run 0 $PY probes/phase3_run.py --list tasks/phase3_dev.yaml --out "$RUN/n$NSIM$k/runs" \
        --rehearsal --shards "$NSIM" "${ENVS[@]}" || echo "N$NSIM$k rc=$?"
    kill "$SNAP" 2>/dev/null
done
date -Is > "$RUN/throughput_t1.txt"
for k in a b; do
    sim_run 0 $PY probes/phase3_gates.py measure --out "$RUN/n$NSIM$k/runs" --list tasks/phase3_dev.yaml --rehearsal \
        || echo "N$NSIM$k measure rc=$?"
done
