# 008 G9 for new placements: can the VLM name the toolbox in office / reception (a recorded deviation: never in the ward) and a chair in ward / office /
# storage? probes/g9_readable.py --selection on the 3 photos per pair picked by p5_spot.sh (PLAN_SPOT = that job's
# p5_spot dir); bar per pair = the 002/005 G9 rule: named in >= 2 of 3 photos ("List the objects you see, with colour.",
# thinking on, frozen synonym matcher). Needs only photos + the VLM: sim_job allows N >= 1, so one sim idles (no
# sim_run on it); if sim_job ever allows N=0, use 0. Local vLLM (`vlm`), or `vlm=<serve jobid>` for a running server.
#   sbatch --time=00:30:00 --cpus-per-task=32 --mem=200G hpc/sim_job.sbatch hpc/plans/p5_g9.sh 1 vlm p:SPOT=$PROJ/runs/sim/<p5_spot jobid>/p5_spot
# Estimate: vLLM load ~2 min + sim bringup ~1 min + <= 15 list calls (~1-2 min) -> ~8 min wall, ~0.15 H200-h
# (002 job 142964: 44 calls in 69 s, load 110 s).
PY=/opt/venv/bin/python
[ -n "${PLAN_SPOT:-}" ] && [ -f "$PLAN_SPOT/g9_selection.json" ] || { echo "p5_g9: need p:SPOT=<dir with g9_selection.json>"; exit 2; }
sim_run 0 $PY probes/g9_readable.py --selection "$PLAN_SPOT/g9_selection.json" --out "$RUN/p5_g9" \
    --base-url "$VLM_BASE_URL" --model "$VLM_MODEL" || echo "g9 rc=$?"
