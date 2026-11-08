# 012 R2 (design section 5): G1 preconditions on a clean committed tree (the JSONs record the commit). vLLM for H-P3
# (g9_readable) only. Out: $RUN/hp/. Two modes (p:MODE=..., g1_job.sbatch exports PLAN_MODE):
#   hp34 (H-P3 + H-P4 + the vet list; H-P1 has its own plan hpc/plans/g1_hp1.sh, H-P2 passed in 145142):
#   sbatch --time=02:00:00 --cpus-per-task=96 --mem=400G hpc/g1_job.sbatch hpc/plans/g1_pre.sh 5 vlm w4=cabinet_open p:MODE=hp34
#   all (default; H-P1..H-P4 + the vet list):
#   sbatch --time=02:30:00 --cpus-per-task=96 --mem=400G hpc/g1_job.sbatch hpc/plans/g1_pre.sh 5 vlm pitch w4=cabinet_open
# Sims 0-3 closed, 4 cabinet_open (H-P3's open-cabinet photos). `all` needs the `pitch` arg (g1_job.sbatch sets
# G1_NODE_EXTRA, read by g1_lib.sh sim_start): H-P2's G7 negative control (a recorded deviation addendum) reads
# /head_rgb_pitch/segmentation on sim 0.
# hp34 flow:
#   t0     g1_pairs (test list -> aggregate pairs.json); vet list render + count in the background (aggregates only,
#          ~2 min in 145142)
#   t0     sim 0: H-P4 (phase2_gates g1, 12 scripted frozen-world runs) FIRST on its sim: no layout ran there before it
#          (and run_one restores the frozen world on the G1 before every frozen run anyway, task 012 fix 1)
#   t0     H-P3 rounds (hp3_rounds): g1_spot pick closed (sims 1-3, round robin) + cabinet_open (sim 4, seeds disjoint
#          from the closed ones), live, again until pick is empty: every group needs 3 photos from 3 different dev seeds
#          (g1_spot module doc). A round where every live failed stops the rounds (an anomaly, not a grind).
#   end    H-P3 select (D/g9_seeds.json = the seeds per group) + g9_readable --selection (movable + bed + cabinet
#          states, one call) + verdict
# all flow: H-P1 sims 0-3 (closed + HP1_SEEDS, sim 0 first the contact control), then sim 0: H-P2, sim 1: H-P4, and the
#   H-P3 rounds on sims 2, 3 (closed, after their H-P1) + 4 (cabinet_open).
# HP1_SEEDS (vetted dev seeds, hp1_reach refuses others): a movable in every room, the toolbox in three different rooms
# (474: office, 434: reception, 196: storage; chairs 1/1/1/1 each).
# Estimate (the local machine rehearsal 2026-11-06, RTF 1.0; 145142): H-P4 ~25 min; H-P3 ~12 min sim per seed (4 room places),
# round 1 = 8 closed seeds on 3 sims ~ 36 min, top-up rounds ~12 min each; g9_readable ~5 min; + startup: hp34 ~60 min.
# all: H-P1 4 layouts x ~13 min ~ 55 min per sim first: ~110 min. Tripwire: no H-P3 seed dir by +20 min of plan time
# (or H-P1 part in `all`) is an anomaly (check the node logs / RTF), not a wait.
HP=$RUN/hp
MODE=${PLAN_MODE:-all}
ROOMS=(ward office reception storage)
HP1_SEEDS=(474 434 196)
case $MODE in all | hp34) ;; *) echo "g1_pre: p:MODE=$MODE is not all or hp34"; exit 2 ;; esac
for i in 0 1 2 3; do [ "$(sim_world "$i")" = closed ] || { echo "g1_pre: sim $i must be closed"; exit 2; }; done
[ "$(sim_world 4)" = cabinet_open ] || { echo "g1_pre: need w4=cabinet_open"; exit 2; }
[ "$MODE" = hp34 ] || [[ " ${G1_NODE_EXTRA:-} " == *" --pitch-control "* ]] ||
    { echo "g1_pre: submit with the pitch arg (H-P2 G7 control)"; exit 2; }
mkdir -p "$HP/hp3"
sim_run 0 "$PY_RUN" humanoid/probes/g1_pairs.py --out "$HP/pairs.json" || { echo "g1_pre: g1_pairs failed"; exit 2; }
BASE_Z=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['base_z'])" "$RUN/s0/ready.json")
( sim_run 0 "$PY_G1" humanoid/probes/g1_vet_list.py render --scene "closed=$(sim_scene_dir 0)" \
      --scene "cabinet_open=$(sim_scene_dir 4)" --base-z "$BASE_Z" --out "$HP/vet_render.json" &&
  sim_run 0 "$PY_RUN" humanoid/probes/g1_vet_list.py count --render "$HP/vet_render.json" --out "$HP/vet_list.json" ) & VET=$!
pin_snapshot "sims ready, idle"

hp3_rounds() {  # <open sim> <closed sim>...: pick / live until g1_spot pick has nothing left (module doc above)
    local open=$1 r=0 c o k j p n fail exc=""; shift
    local sims=("$@")
    while :; do
        c=$(in_sim "${sims[0]}" "$PY_RUN" humanoid/probes/g1_spot.py pick --pairs "$HP/pairs.json" --out "$HP/hp3" \
            --exclude "$exc" | tail -1)
        o=$(in_sim "$open" "$PY_RUN" humanoid/probes/g1_spot.py pick --pairs "$HP/pairs.json" --out "$HP/hp3" \
            --exclude "$exc${exc:+,}$c" | tail -1)
        [ -n "$c$o" ] || break
        r=$((r + 1)); exc=$exc${exc:+,}$c${c:+${o:+,}}$o
        jlog "g1_pre: H-P3 round $r seeds closed ${c:-none} | cabinet_open ${o:-none}"
        local C=() Q=() W=()
        IFS=, read -ra C <<< "$c"
        for k in "${!C[@]}"; do j=$((k % ${#sims[@]})); Q[j]=${Q[j]:+${Q[j]},}${C[k]}; done
        for j in "${!Q[@]}"; do sim_run "${sims[j]}" "$PY_RUN" humanoid/probes/g1_spot.py live --seeds "${Q[j]}" --out "$HP/hp3" & W+=($!); done
        [ -z "$o" ] || { sim_run "$open" "$PY_RUN" humanoid/probes/g1_spot.py live --seeds "$o" --out "$HP/hp3" & W+=($!); }
        n=0; fail=0
        for p in "${W[@]}"; do n=$((n + 1)); wait "$p" || fail=$((fail + 1)); done
        [ "$fail" = 0 ] || jlog "g1_pre: H-P3 round $r: $fail of $n live calls failed (see the sims' cmd.log)"
        [ "$fail" -lt "$n" ] || { jlog "g1_pre: H-P3 rounds stopped: every live call of round $r failed"; break; }
    done
}

( sleep 600; pin_snapshot "+600 s" ) & SNAP=$!
P=()
if [ "$MODE" = hp34 ]; then
    sim_run 0 "$PY_RUN" probes/phase2_gates.py g1 --out "$HP/hp4" & P[0]=$!   # first on sim 0: no layout before it
    hp3_rounds 4 1 2 3 & P[1]=$!
    for p in "${P[@]}"; do wait "$p"; done
    CHECK=(hp3/hp3 hp4/g1)
else
    for i in 0 1 2 3; do
        ( [ "$i" = 0 ] && sim_run 0 "$PY_RUN" humanoid/probes/contact_control.py --out "$HP/hp1" --layout-seed 474
          sim_run "$i" "$PY_RUN" humanoid/probes/hp1_reach.py --room "${ROOMS[i]}" --out "$HP/hp1"
          for s in "${HP1_SEEDS[@]}"; do
              sim_run "$i" "$PY_RUN" humanoid/probes/hp1_reach.py --room "${ROOMS[i]}" --out "$HP/hp1" --layout-seed "$s"
          done ) & P[i]=$!
    done
    wait "${P[0]}"; sim_run 0 "$PY_RUN" humanoid/probes/hp2_photos.py --out "$HP/hp2" & P[0]=$!
    wait "${P[1]}"; sim_run 1 "$PY_RUN" probes/phase2_gates.py g1 --out "$HP/hp4" & P[1]=$!
    wait "${P[2]}"; wait "${P[3]}"; unset 'P[3]'; hp3_rounds 4 2 3 & P[2]=$!
    for p in "${P[@]}"; do wait "$p"; done
    sim_run 0 "$PY_RUN" humanoid/probes/hp1_reach.py --merge "$HP/hp1" --seeds "$(IFS=,; echo "${HP1_SEEDS[*]}")"
    CHECK=(hp1/hp1 hp1/contact_control_seed474 hp2/hp2 hp3/hp3 hp4/g1)
fi
wait "$VET" || echo "g1_pre: vet list rc=$?"
kill "$SNAP" 2>/dev/null
sim_run 0 "$PY_RUN" humanoid/probes/g1_spot.py select --pairs "$HP/pairs.json" --out "$HP/hp3" ||
    jlog "g1_pre: H-P3 selection short (a group under 3 photos from 3 seeds)"
jlog "g1_pre: H-P3 seeds per group $(tr -d '\n ' < "$HP/hp3/g9_seeds.json" 2>/dev/null | head -c 600)"
sim_run 0 "$PY_RUN" probes/g9_readable.py --selection "$HP/hp3/g9_selection.json" --out "$HP/hp3/g9_sel" \
    --base-url "$VLM_BASE_URL" --model "$VLM_MODEL"
sim_run 0 "$PY_RUN" humanoid/probes/g1_spot.py verdict --pairs "$HP/pairs.json" --out "$HP/hp3"
RC=0
for f in "${CHECK[@]}"; do
    v=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1])).get('pass'))" "$HP/$f.json" 2>/dev/null)
    jlog "g1_pre: $f pass=${v:-missing}"; [ "$v" = True ] || RC=1
done
jlog "g1_pre: vet list $(tr -d '\n ' < "$HP/vet_list.json" 2>/dev/null | head -c 400)"
exit $RC
