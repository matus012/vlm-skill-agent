# 008 G3 layout fitness, live part: spot check of 7 seeded layouts in 2 sims (sim 0 closed, sim 1 cabinet_open), no
# vLLM, on the movables-free map. Per seed (probes/spot_check.py): run_task's teleport, photos at all 5 places, Phase 1
# G2/G3/G4/G7 against seeded.truth_for(seed, world), comparison with the ray-caster; then --merge -> p5_spot/spot.json
# and --g9-select -> p5_spot/g9_selection.json (input of hpc/plans/p5_g9.sh).
#   sbatch --time=01:20:00 --cpus-per-task=64 --mem=400G hpc/sim_job.sbatch hpc/plans/p5_spot.sh 2 w1=cabinet_open map=clinic_static
# Seeds: ok seeds of the re-vetted tasks/phase5_seeds.json (a recorded deviation: stricter toolbox rule, never in the ward; 306 ok
# over 0..699), highest indices first, skipping every layout_seed of tasks/phase5_counted.yaml (683, 669, 668 are counted).
# Dev seeds only.
#   seed  world         toolbox    chairs ward/office/reception/storage  a room's chairs over >= 2 photos
#   694   closed        storage    1/0/1/2                              storage
#   693   closed        reception  0/2/2/0                              office, reception
#   676   closed        office     0/0/1/3                              storage
#   666   closed        reception  1/0/1/2                              storage
#   690   cabinet_open  reception  1/1/0/2                              storage
#   675   cabinet_open  office     0/0/2/2                              reception, storage
#   660   cabinet_open  office     1/1/1/1                              -
# Why: G9 needs 3 photos per new (class, room) pair and a movable is wholly in ONE photo of its room, so >= 3 seeds with
# the toolbox in the office (676, 675, 660) and in reception (693, 666, 690); 694 (the top ok seed) adds the toolbox in
# storage live. 691 / 677 / 673 / 665 / 662 (toolbox in storage) are skipped: a 3-photo toolbox@storage pair would cost
# 2 more seeds and storage is not a new room for it. Chairs: ward 4, office 4, reception 8, storage 12 placements; 6 of
# 7 seeds have a room whose chairs span >= 2 photos.
# Estimate: ~10 min wall per seed (5 navigations + 20 photos at RTF ~0.7) -> ~40-45 min (4 seeds on sim 0), + ~5 min
# startup; ~0.8 H200-h. Tripwire: sim 0 below 2 seeds done at +40 min of plan time is an anomaly (check RTF / device).
PY=/opt/venv/bin/python
D=$RUN/p5_spot
[ "$(sim_world 0)" = closed ] && [ "$(sim_world 1)" = cabinet_open ] || { echo "p5_spot: need sim 0 closed, sim 1 cabinet_open"; exit 2; }
sim_run 0 $PY probes/spot_check.py --seeds 694,693,676,666 --out "$D" & P0=$!
sim_run 1 $PY probes/spot_check.py --seeds 690,675,660 --out "$D" & P1=$!
wait $P0 || echo "spot sim 0 rc=$?"
wait $P1 || echo "spot sim 1 rc=$?"
sim_run 0 $PY probes/spot_check.py --merge "$D" || echo "spot merge rc=$? (agree_all false)"
sim_run 0 $PY probes/spot_check.py --g9-select "$D" || echo "g9 select rc=$? (a pair short of 3 photos)"
