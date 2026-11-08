# NOTE (e7f62ee, 006 M5 round 3): since then sim_run commands run on the client cpus (the job's cpus no sim holds) and
# /proc/self Cpus_allowed_list in a plan = the client cpus; results recorded for this plan used the older layout.
# 006 gate job B (M5 round 2: N = 2, each sim pinned to its own whole L3 groups, hpc/cpu_split.py; M2 round 2:
# face() final turns; a recorded deviation: gz CLI retry). Re-counts M5, M2 (seg + photos + G2/G3/G4/G7), M3 (Phase 1 G1
# 40/40), M4 (Phase 2 G1 12/12), then the dev list sharded over the 2 sims (N = 2 throughput). Same gate code.
#   sbatch --time=02:15:00 --cpus-per-task=64 --mem=400G hpc/sim_job.sbatch hpc/plans/gates_n2.sh 2 vlm
SIMS=(); ENVS=()
for i in 0 1; do SIMS+=(--sim "$(sim_dom $i):$(sim_part $i)"); ENVS+=(--shard-env "$i=$(sim_dom $i):$(sim_part $i)"); done
PY=/opt/venv/bin/python
sim_run 0 $PY probes/crosstalk.py --out "$RUN/m5" "${SIMS[@]}" || echo "M5 rc=$?"
( sim_run 0 $PY probes/seg_compare.py --out "$RUN/m2" --ref "$PROJ/ref/002-counted-photos" || echo "M2 seg rc=$?"
  sim_run 0 $PY probes/phase1_gates.py photos --out "$RUN/m2p1" || echo "M2 photos rc=$?"
  sim_run 0 $PY probes/phase1_gates.py checks --out "$RUN/m2p1" || echo "M2 checks rc=$?"
  sim_run 0 $PY probes/phase2_gates.py g1 --out "$RUN/m4" || echo "M4 rc=$?" ) &
( sim_run 1 $PY probes/phase1_gates.py g1 --out "$RUN/m3" || echo "M3 rc=$?" ) &
wait
sim_run 0 $PY probes/phase3_run.py --list tasks/phase3_dev.yaml --out "$RUN/n2/runs" --rehearsal --shards 2 "${ENVS[@]}" || echo "N2 rc=$?"
sim_run 0 $PY probes/phase3_gates.py measure --out "$RUN/n2/runs" --list tasks/phase3_dev.yaml --rehearsal || echo "N2 measure rc=$?"
