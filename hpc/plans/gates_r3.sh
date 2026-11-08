# 006 M5 round 3 (the last, 2026-11-05): N = 2, each sim on its own 16 cpus as in round 2; vLLM, the gate probes,
# agents, coordinator and their gz CLI reads on the job's cpus no sim holds (sim_lib cpus_init). M5 verdict = each sim's
# median per-run sim_s / wall_s over its M3 / M4 runs here (M4 on sim 0, M3 on sim 1, as in 143815), >= 0.7 per sim; the
# M2 / M3 / M4 verdicts stand from 143815, a gate failure here is reported as a new finding. Then the dev list sharded
# over the 2 sims (N = 2 throughput, t0 / t1). pinning.txt: live snapshots of every process's cpus + 5 s cpu use.
#   sbatch --time=01:45:00 --cpus-per-task=64 --mem=400G hpc/sim_job.sbatch hpc/plans/gates_r3.sh 2 vlm
SIMS=(); ENVS=()
for i in 0 1; do SIMS+=(--sim "$(sim_dom $i):$(sim_part $i)"); ENVS+=(--shard-env "$i=$(sim_dom $i):$(sim_part $i)"); done
PY=/opt/venv/bin/python
pin_snapshot "sims ready, idle"
sim_run 0 $PY probes/crosstalk.py --out "$RUN/m5" "${SIMS[@]}" || echo "M5 rc=$?"
( sim_run 0 $PY probes/phase2_gates.py g1 --out "$RUN/m4" || echo "M4 rc=$?" ) & P4=$!
( sim_run 1 $PY probes/phase1_gates.py g1 --out "$RUN/m3" || echo "M3 rc=$?" ) & P3=$!
( sleep 300; pin_snapshot "M4 on sim 0 + M3 on sim 1, +300 s"; sleep 1200; pin_snapshot "M4 + M3, +1500 s" ) & SNAP=$!
wait "$P4" "$P3"
kill "$SNAP" 2>/dev/null
date -Is > "$RUN/throughput_t0.txt"
( sleep 600; pin_snapshot "N=2 dev list, +600 s" ) & SNAP=$!
sim_run 0 $PY probes/phase3_run.py --list tasks/phase3_dev.yaml --out "$RUN/n2/runs" --rehearsal --shards 2 "${ENVS[@]}" || echo "N2 rc=$?"
date -Is > "$RUN/throughput_t1.txt"
kill "$SNAP" 2>/dev/null
sim_run 0 $PY probes/phase3_gates.py measure --out "$RUN/n2/runs" --list tasks/phase3_dev.yaml --rehearsal || echo "N2 measure rc=$?"
