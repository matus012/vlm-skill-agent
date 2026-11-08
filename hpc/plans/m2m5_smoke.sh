# 006 M2 + M5 smoke, N = 2. Sourced by hpc/sim_job.sbatch once both sims are ready (API: hpc/PERUN_SIM.md section 2):
#   sbatch hpc/sim_job.sbatch hpc/plans/m2m5_smoke.sh 2
# Needs the local machine reference photos (runs/2026-11-04/002-counted/photos/) copied to $PROJ/ref/002-counted-photos.
# M2 first (sim 0 alone), then M5 across sims 0 and 1 (sim env per contract: ROS_DOMAIN_ID $(sim_dom i), GZ_PARTITION $(sim_part i)).
sim_run 0 /opt/venv/bin/python probes/seg_compare.py --out "$RUN/m2" --ref "$PROJ/ref/002-counted-photos" \
  || echo "M2 rc=$?"
sim_run 0 /opt/venv/bin/python probes/crosstalk.py --out "$RUN/m5" \
  --sim "$(sim_dom 0):$(sim_part 0)" --sim "$(sim_dom 1):$(sim_part 1)" \
  || echo "M5 rc=$?"
