# NOTE (e7f62ee, 006 M5 round 3): since then sim_run commands run on the client cpus (the job's cpus no sim holds) and
# /proc/self Cpus_allowed_list in a plan = the client cpus; results recorded for this plan used the older layout.
# Superseded by hpc/plans/measure_n.sh (pinned sims, unpinned coordinator, 2 x 8 runs); kept for the 006 record.
# 006 measure at N = 8 sims + vLLM: M5 across all 8 (cross-talk + RTF), then the 005 dev list sharded over the 8 sims
# (one entry each) for runs per hour. From the clean clone ($PROJ/repo):
#   sbatch --time=01:00:00 --cpus-per-task=64 --mem=400G hpc/sim_job.sbatch hpc/plans/measure_n8.sh 8 vlm
PY=/opt/venv/bin/python
SIMS=(); ENVS=()
for i in 0 1 2 3 4 5 6 7; do SIMS+=(--sim "$(sim_dom $i):$(sim_part $i)"); ENVS+=(--shard-env "$i=$(sim_dom $i):$(sim_part $i)"); done
sim_run 0 $PY probes/crosstalk.py --out "$RUN/m5n8" "${SIMS[@]}" || echo "M5 n8 rc=$?"
sim_run 0 $PY probes/phase3_run.py --list tasks/phase3_dev.yaml --out "$RUN/n8/runs" --rehearsal --shards 8 "${ENVS[@]}" || echo "N8 rc=$?"
sim_run 0 $PY probes/phase3_gates.py all --out "$RUN/n8/runs" --list tasks/phase3_dev.yaml --rehearsal || echo "N8 gates rc=$?"
