# hpc/g1_lib.sh -- source after hpc/sim_lib.sh, do not execute (task 012 design section 3). The G1 form of a sim: one
# MuJoCo g1_node per sim instead of the Gazebo launch; everything else (cpus, domains, partitions, in_sim, sim_run,
# sim_stop, vLLM, pin_snapshot, stop_all) is sim_lib.sh's own code, unchanged. Redefined here: sim_vars, sim_start,
# sim_pgid, sim_wait_ready, sim_params_check; added: g1_watch (the FastDDS-hang watchdog, humanoid/README.md).
# Runner + adapter + probes run in /opt/venv (numpy 1.x); g1_node and the MuJoCo probes in /opt/venv-g1 (numpy 2).
# SIM_FAKE=<dir> (tests/test_g1_plumbing.py): <dir>/python_g1 and <dir>/python_run stand in for the two pythons.
PY_RUN=/opt/venv/bin/python
PY_G1=/opt/venv-g1/bin/python
[ -n "${SIM_FAKE:-}" ] && { PY_RUN=$SIM_FAKE/python_run; PY_G1=$SIM_FAKE/python_g1; }
G1_STALL_S=${G1_STALL_S:-120}    # design section 3: a node whose "sim t" log line stalls > 120 s wall is restarted
WATCH_POLL_S=${WATCH_POLL_S:-10}  # sampling of the node logs (the node logs every 10 s sim); not a gate number

eval "tb4_$(declare -f sim_vars)"  # sim_lib's sim_vars as tb4_sim_vars: the G1 env is that plus the G1 keys
sim_vars() {  # G1_SCENES/<GZ_PARTITION>/scene.json is each sim's scene (adapters/robots.make; shard workers set their own partition)
    echo "$(tb4_sim_vars "$1") ROBOT=g1 G1_EXT=/opt/g1ext G1_SCENES=$RUN/scenes PYTHONPATH=$REPO:/opt/g1ext/agile"
}
sim_scene_dir() { echo "$RUN/scenes/$(sim_part "$1")"; }

sim_start() {  # G1_QPOS_LOG=1 (job arg `qpos`, task 013): + --qpos-log s<i>/qpos.jsonl. setsid: the node is its own process group leader; an earlier truth log / stats are kept (restart)
    local i=$1 f n
    mkdir -p "$RUN/s$i" "$(sim_scene_dir "$i")"
    n=$(find "$RUN/s$i" -maxdepth 1 -name 'truth_qpos*.jsonl' | wc -l)
    for f in truth_qpos.jsonl qpos.jsonl stats.json; do [ -e "$RUN/s$i/$f" ] && mv "$RUN/s$i/$f" "$RUN/s$i/${f%.*}.$n.${f##*.}"; done
    in_sim "$i" --sim-cpus setsid "$PY_G1" humanoid/g1_node.py --world "$(sim_world "$i")" \
        --scene-dir "$(sim_scene_dir "$i")" --rate 1.0 --torch-threads 2 ${G1_NODE_EXTRA:-} \
        ${G1_QPOS_LOG:+--qpos-log "$RUN/s$i/qpos.jsonl"} --truth-log "$RUN/s$i/truth_qpos.jsonl" --stats "$RUN/s$i/stats.json" > "$RUN/s$i/launch.log" 2>&1 < /dev/null &
    jlog "sim $i start (g1, world $(sim_world "$i"), ROS_DOMAIN_ID $(sim_dom "$i"), GZ_PARTITION $(sim_part "$i")$(g=$(sim_gpu "$i") && echo ", GPU $g"), cpus $(sim_cpus "$i"))"
}

sim_pgid() {  # pgid of sim i's g1_node (the setsid'd group leader whose args hold humanoid/g1_node.py); empty if none
    local p pg cmd want
    want="GZ_PARTITION=$(sim_part "$1")"
    while read -r p pg cmd; do
        [ "$p" = "$pg" ] || continue
        case $cmd in *" humanoid/g1_node.py "*) ;; *) continue ;; esac
        tr '\0' '\n' 2>/dev/null < "/proc/$p/environ" | grep -qx -- "$want" && { echo "$pg"; return; }
    done < <(ps -u "$(id -u)" -o pid=,pgid=,args=)
}

sim_wait_ready() {  # humanoid/probes/ready.py (bounded 180 s) -> s<i>/ready.json; then the node's pgid
    local i=$1
    in_sim "$i" "$PY_RUN" humanoid/probes/ready.py wait --out "$RUN/s$i/ready.json" > "$RUN/s$i/ready.log" 2>&1 ||
        { jlog "sim $i NOT READY: $(tail -1 "$RUN/s$i/ready.log")"; return 1; }
    sim_pgid "$i" > "$RUN/s$i/pgid.txt"
    [ -s "$RUN/s$i/pgid.txt" ] || { jlog "sim $i ready but no g1_node process with its GZ_PARTITION"; return 1; }
    jlog "sim $i ready (pgid $(cat "$RUN/s$i/pgid.txt")): $(tail -1 "$RUN/s$i/ready.log")"
}

# camera_info, scene labels vs a fresh build in the image, the node's --rate / --torch-threads (its /proc cmdline).
# Writes s<i>/params_check.txt; returns 1 on any mismatch.
sim_params_check() {
    local i=$1 f=$RUN/s$1/params_check.txt pg
    pg=$(cat "$RUN/s$i/pgid.txt")
    in_sim "$i" "$PY_G1" humanoid/probes/ready.py params --ready "$RUN/s$i/ready.json" \
        --cmdline "$(tr '\0' ' ' < "/proc/$pg/cmdline" 2>/dev/null)" > "$f" 2>&1
    local rc=$?
    jlog "sim $i params check rc=$rc: $(tail -1 "$f")"; return $rc
}

# g1_watch: background loop for the whole plan. Per sim, the last "sim <t> s, RTF" line of launch.log; when it has not
# changed for > G1_STALL_S wall the node is restarted (stop, log kept as launch.stall<n>.log, start, ready). Each
# restart is a job.log line and a line in s<i>/restarts.txt (measure counts them). A run in flight on that sim dies by
# drive.py's ODOM_HANG_S (crash, origin adapter -> label sim); nothing is hidden.
g1_watch() {
    local i now line last=() seen=() n
    for i in $(seq 0 $((NSIM - 1))); do last[i]=$(date +%s); seen[i]=""; done
    while sleep "$WATCH_POLL_S"; do
        now=$(date +%s)
        for i in $(seq 0 $((NSIM - 1))); do
            line=$(grep -o 'sim [0-9.]* s, RTF [0-9.]*' "$RUN/s$i/launch.log" 2>/dev/null | tail -1)
            if [ "$line" != "${seen[i]}" ]; then seen[i]=$line; last[i]=$now; continue; fi
            [ $((now - last[i])) -gt "$G1_STALL_S" ] || continue
            n=$(( $(cat "$RUN/s$i/restarts.txt" 2>/dev/null | wc -l) + 1 ))
            echo "$(date -Is) restart $n: last line '${seen[i]}' unchanged for $((now - last[i])) s" >> "$RUN/s$i/restarts.txt"
            jlog "g1_watch: sim $i log stalled $((now - last[i])) s (last '${seen[i]}'): restart $n"
            sim_stop "$i"; mv "$RUN/s$i/launch.log" "$RUN/s$i/launch.stall$n.log"; sim_start "$i"
            sim_wait_ready "$i" || jlog "g1_watch: sim $i not ready after restart $n"
            seen[i]=""; last[i]=$(date +%s)
        done
    done
}
