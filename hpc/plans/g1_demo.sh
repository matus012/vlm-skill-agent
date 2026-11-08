# 012 R6 (design sections 6, 8): after the FINAL verdict only. Side table and the side-by-side demo, no sim work (the
# one sim idles), no vLLM. p:D = the merged G1 run dir, p:T = the TB4 pass-2 run dir (hpc/JOBS.md 144735-7), both on
# $PROJ; p:DROPS = optional hosting-drop file (one run_id per line). Out: $RUN/demo/.
#   sbatch --time=00:30:00 hpc/g1_job.sbatch hpc/plans/g1_demo.sh 1 p:D=$PROJ/runs/p10/<name> p:T=$PROJ/runs/p8/<pass2>
# Look at side_by_side.mp4 before sending (look-at-the-artifact). The gallery is not built here.
# 145379: the gates ran (rc 0); the demo step failed (demo_recorder needs mujoco, $PY_RUN has none). The 012 demo was
# rendered on the local machine instead (demo_side_by_side.py --recorder-python $BUILD_DIR/.venv-g1/bin/python).
[ -n "${PLAN_D:-}" ] && [ -n "${PLAN_T:-}" ] || { echo "g1_demo: need p:D, p:T"; exit 2; }
OUT=$RUN/demo
mkdir -p "$OUT"
RC=0
sim_run 0 "$PY_RUN" probes/phase10_gates.py all --out "$PLAN_D" --tb4-out "$PLAN_T" ${PLAN_DROPS:+--drops "$PLAN_DROPS"} || RC=1
sim_run 0 "$PY_RUN" probes/demo_side_by_side.py --out "$OUT/side_by_side.mp4" --tb4-out "$PLAN_T" --g1-out "$PLAN_D" || RC=1
jlog "g1_demo: gates + demo rc=$RC; demo $(ls -l "$OUT/side_by_side.mp4" 2>/dev/null | awk '{print $5}') bytes"
exit $RC
