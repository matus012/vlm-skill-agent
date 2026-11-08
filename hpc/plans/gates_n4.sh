# NOTE (e7f62ee, 006 M5 round 3): since then sim_run commands run on the client cpus (the job's cpus no sim holds) and
# /proc/self Cpus_allowed_list in a plan = the client cpus; results recorded for this plan used the older layout.
# 006 gates M2-M6 in one job: N = 4 sims + vLLM on localhost. From the clean clone ($PROJ/repo):
#   sbatch --time=01:45:00 hpc/sim_job.sbatch hpc/plans/gates_n4.sh 4 vlm
# 1. M5 with all 4 sims idle (cross-talk, node lists, RTF).
# 2. In parallel: s0 M2 (seg agreement by teleport, then Phase 1 photos + G2/G3/G4/G7 checks on PERUN frames);
#    s1 M3 (Phase 1 G1, 40 tries); s2 M4 (Phase 2 G1, 12 scripted runs); s3 M6 (005 dev list, 8 VLM runs, local vLLM).
# 3. Throughput at N = 4: the dev list once more, sharded over the 4 sims (each entry once).
SIMS4=(); ENVS4=()
for i in 0 1 2 3; do SIMS4+=(--sim "$(sim_dom $i):$(sim_part $i)"); ENVS4+=(--shard-env "$i=$(sim_dom $i):$(sim_part $i)"); done
PY=/opt/venv/bin/python
sim_run 0 $PY probes/crosstalk.py --out "$RUN/m5" "${SIMS4[@]}" || echo "M5 rc=$?"
( sim_run 0 $PY probes/seg_compare.py --out "$RUN/m2" --ref "$PROJ/ref/002-counted-photos" || echo "M2 seg rc=$?"
  sim_run 0 $PY probes/phase1_gates.py photos --out "$RUN/m2p1" || echo "M2 photos rc=$?"
  sim_run 0 $PY probes/phase1_gates.py checks --out "$RUN/m2p1" || echo "M2 checks rc=$?" ) &
( sim_run 1 $PY probes/phase1_gates.py g1 --out "$RUN/m3" || echo "M3 rc=$?" ) &
( sim_run 2 $PY probes/phase2_gates.py g1 --out "$RUN/m4" || echo "M4 rc=$?" ) &
( sim_run 3 $PY probes/phase3_run.py --list tasks/phase3_dev.yaml --out "$RUN/m6/runs" --rehearsal || echo "M6 rc=$?"
  sim_run 3 $PY probes/phase3_gates.py all --out "$RUN/m6/runs" --list tasks/phase3_dev.yaml --rehearsal || echo "M6 gates rc=$?" ) &
wait
echo "parallel block done $(date -Is)"
sim_run 0 $PY probes/phase3_run.py --list tasks/phase3_dev.yaml --out "$RUN/n4/runs" --rehearsal --shards 4 "${ENVS4[@]}" || echo "N4 rc=$?"
sim_run 0 $PY probes/phase3_gates.py measure --out "$RUN/n4/runs" --list tasks/phase3_dev.yaml --rehearsal || echo "N4 measure rc=$?"
