# 012 H-P1 alone (design ruling: collision = a MuJoCo robot-clinic contact, hp1_reach --collision contact): the contact
# sanity control, then H-P1 with one layout per sim: sim 0 the closed world, sims 1-3 the dev seeds HP1_SEEDS, each sim
# all 4 rooms in turn; then the merge. No vLLM. Out: $RUN/hp/hp1/ (same part names as g1_pre.sh).
#   sbatch --time=01:30:00 --cpus-per-task=80 --mem=320G hpc/g1_job.sbatch hpc/plans/g1_hp1.sh 4
# Control (sim 1, before its tries): seed 474 placed, raw /cmd_vel straight into a chair; pass = the chair in
# /g1/contacts (contact_control_seed474.json). Estimate (the local machine rehearsal, RTF 1.0, ~13 min sim per room-layout):
# 4 rooms ~55 min per sim + control ~3 min + startup. Tripwire: no H-P1 part by +20 min of plan time is an anomaly.
HP=$RUN/hp
ROOMS=(ward office reception storage)
HP1_SEEDS=(474 434 196)
for i in 0 1 2 3; do [ "$(sim_world "$i")" = closed ] || { echo "g1_hp1: sim $i must be closed"; exit 2; }; done
mkdir -p "$HP/hp1"
pin_snapshot "sims ready, idle"
( for r in "${ROOMS[@]}"; do sim_run 0 "$PY_RUN" humanoid/probes/hp1_reach.py --room "$r" --out "$HP/hp1"; done ) & P[0]=$!
for k in 0 1 2; do
    i=$((k + 1)); s=${HP1_SEEDS[k]}
    ( [ "$s" = 474 ] && sim_run "$i" "$PY_RUN" humanoid/probes/contact_control.py --out "$HP/hp1" --layout-seed 474
      for r in "${ROOMS[@]}"; do
          sim_run "$i" "$PY_RUN" humanoid/probes/hp1_reach.py --room "$r" --out "$HP/hp1" --layout-seed "$s"
      done ) & P[i]=$!
done
( sleep 600; pin_snapshot "+600 s" ) & SNAP=$!
for p in "${P[@]}"; do wait "$p"; done
kill "$SNAP" 2>/dev/null
sim_run 0 "$PY_RUN" humanoid/probes/hp1_reach.py --merge "$HP/hp1" --seeds "$(IFS=,; echo "${HP1_SEEDS[*]}")"
RC=0
for f in hp1/hp1 hp1/contact_control_seed474; do
    v=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get('pass'))" "$HP/$f.json" 2>/dev/null)
    jlog "g1_hp1: $f pass=${v:-missing}"; [ "$v" = True ] || RC=1
done
exit $RC
