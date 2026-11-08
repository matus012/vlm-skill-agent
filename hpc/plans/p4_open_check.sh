# 007 precondition (2026-11-05, before any counted run; not a phase gate): world cabinet_open (tag
# phase4-world-cabinet-open-v1, doors swung out into storage) was never navigated under Phase 1 G1. One sim in that world
# (sim_job ... 1 w0=cabinet_open, no vLLM):
# - reach: Phase 1 G1 for storage only (10 tries ending at storage, starts cycling through the other places, hall
#   first); the uncounted moves out of storage between tries are judged by the same G1 rule (SUCCEEDED, GT <= 0.30 m,
#   <= 120 s sim) and the first 5 must pass;
# - G7 (and G2-G4) on photos at every place, checked against the open world's truth.
#   sbatch --time=00:50:00 --cpus-per-task=32 --mem=200G hpc/sim_job.sbatch hpc/plans/p4_open_check.sh 1 w0=cabinet_open
PY=/opt/venv/bin/python
T=sim/worlds/clinic/generated_cabinet_open/truth.json
sim_run 0 $PY probes/phase1_gates.py g1 --rooms storage --out "$RUN/open_g1" || echo "open g1 rc=$?"
sim_run 0 $PY probes/phase1_gates.py photos --out "$RUN/open_p1" || echo "open photos rc=$?"
sim_run 0 $PY probes/phase1_gates.py checks --out "$RUN/open_p1" --truth "$T" --variant cabinet_open || echo "open checks rc=$?"
