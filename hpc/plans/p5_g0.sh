# 008 G0 (nav base, before any VLM run): Phase 1 G1 40/40 + G8 on the movables-free SLAM map sim/maps/clinic_static.yaml
# (tag phase5-map-static-v1, job 143995), in the frozen v2 world with the movables at their frozen poses: they are not in
# the map, Nav2's obstacle layer sees them live (sim_job ... map=clinic_static).
#   sbatch --time=01:20:00 --cpus-per-task=32 --mem=200G hpc/sim_job.sbatch hpc/plans/p5_g0.sh 1 map=clinic_static
PY=/opt/venv/bin/python
sim_run 0 $PY probes/phase1_gates.py g8 --out "$RUN/g0" --map sim/maps/clinic_static.yaml || echo "G8 rc=$?"
sim_run 0 $PY probes/phase1_gates.py g1 --out "$RUN/g0" || echo "G1 rc=$?"
