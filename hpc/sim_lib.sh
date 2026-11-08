# hpc/sim_lib.sh -- source, do not execute. Sims and vLLM inside one PERUN job (hpc/PERUN_SIM.md section 2).
# Needs set: PROJ, REPO (clean clone, cwd of every sim command), RUN (job run dir), JOB (SLURM job id).
# Sim i: ROS_DOMAIN_ID base+2i, base per job (never adjacent, never another job's on the node; PERUN_SIM.md §2), GZ_PARTITION j<JOB>s<i>, WS_LOCAL $PROJ/build, inside `apptainer exec --nv --cleanenv`
# of $SIF. Functions: cpus_init, sim_start / sim_wait_ready / sim_params_check / sim_run / sim_stop <i>, vllm_start /
# vllm_stop, pin_snapshot, stop_all, jlog. Never pgrep -f: a sim's processes are found by GZ_PARTITION=<its partition> in /proc/<pid>/environ.
# SIM_FAKE=<dir> (plumbing test without apptainer / ROS / GPU): <dir>/launch.sh replaces sim/launch_clinic.sh and <dir>
# goes first on PATH (fake ros2, fastdds); commands run with `env` on the host instead of the image.
SIF=${SIF:-$PROJ/images/sim.sif}
VLLM_SIF=$PROJ/images/vllm.sif
MODEL=$PROJ/models/Qwen3.8-27B
WS_LOCAL_PERUN=$PROJ/build
# squeue inside the image (phase3_run --min-job-left-min in-job): the host's binary + libs + config + munge socket
# (checked on the login node 2026-11-04 with vllm.sif, Ubuntu 24.04: without passwd/group squeue rejects SlurmUser).
SLURM_BINDS=(-B /usr/bin/squeue -B /usr/lib64/slurm -B /etc/slurm -B /run/munge -B /etc/passwd -B /etc/group
             -B /usr/lib64/libmunge.so.2:/usr/lib/x86_64-linux-gnu/libmunge.so.2)
READY_POLLS=${READY_POLLS:-90}   # x 2 s = 180 s: PERUN sims are ready ~30 s after start (143786, 143804); a stuck
                                  # bringup is restarted once by sim_job.sbatch (the local machine block scripts wait 480 s)
T0=${T0:-$(date +%s)}
# The job's GPUs (SLURM CUDA_VISIBLE_DEVICES): vLLM on the first; with > 1, sim i on GPU i mod n (006 M5 round 2: sims
# sharing one H200 slow each other, 0.78 alone -> 0.60-0.69 at 2 -> 0.55-0.62 at 4 on dedicated cpus, job 143808).
CVDS=${CUDA_VISIBLE_DEVICES:-}
read -r -a GPUS <<< "${CVDS//,/ }"
sim_gpu() { [ "${#GPUS[@]}" -gt 1 ] && echo "${GPUS[$(($1 % ${#GPUS[@]}))]}"; }
# Ogre's EGL takes the first EGL device it can open; CUDA_VISIBLE_DEVICES does not steer it (job 143811: all 4 sims
# rendered on one of the 2 GPUs). So sim i's container gets the job's OTHER GPUs' /dev/nvidia<minor> bound over by an
# unreadable file (as a GPU outside the job's cgroup looks); its only openable GPU is sim_gpu i.
sim_hide() {
    [ "${#GPUS[@]}" -gt 1 ] && [ -z "${SIM_FAKE:-}" ] || return 0
    local nodev=$PROJ/cache/nodev j k=$(($1 % ${#GPUS[@]})) minors
    [ -e "$nodev" ] || { mkdir -p "$PROJ/cache" && : > "$nodev" && chmod 000 "$nodev"; }
    mapfile -t minors < <(nvidia-smi -q | awk '/Minor Number/ {print $NF}')  # the job's GPUs, in index order
    for j in "${!minors[@]}"; do [ "$j" != "$k" ] && printf -- '-B %s:/dev/nvidia%s ' "$nodev" "${minors[$j]}"; done
}

# Sim i's launch runs pinned to its own whole L3 groups of the job's cpus (hpc/cpu_split.py, 006 M5 round 2: 143808
# pinning +0.05-0.10 RTF; a sim on L3 groups shared with other jobs was the slowest in 143808 / 143811 / 143812).
# Everything else -- vLLM, params checks, every sim_run command (gate probes, agents, coordinators, their gz CLI reads)
# -- on the client cpus = the job's cpus no sim holds (M5 round 3: in 143815 the gate client ran on sim 0's 16 cpus,
# per-run sim/wall 0.75 idle -> 0.683 under M4). cpus_init computes both once and moves this shell onto the client
# cpus, so whatever the job starts inherits them; only sim_start pins to a sim's cpus.
cpus_list() {  # cpus_list <i|client>: hpc/cpu_split.py's comma list; 1 on anything else (stderr in $RUN/cpu_split.err)
    local out
    out=$(python3 "$REPO/hpc/cpu_split.py" "$1" "$NSIM" 2> "$RUN/cpu_split.err")
    [[ $out =~ ^[0-9]+(,[0-9]+)*$ ]] && echo "$out"
}
cpus_init() {  # sets SIM_CPUS[i], CLIENT_CPUS; 1 + a log line when hpc/cpu_split.py fails (job 143814: host python 3.9)
    local i
    SIM_CPUS=()
    for i in $(seq 0 $((NSIM - 1))); do
        SIM_CPUS[i]=$(cpus_list "$i") || { jlog "no cpus for sim $i: hpc/cpu_split.py failed ($(tail -1 "$RUN/cpu_split.err"))"; return 1; }
    done
    CLIENT_CPUS=$(cpus_list client) || { jlog "no client cpus: hpc/cpu_split.py failed ($(tail -1 "$RUN/cpu_split.err"))"; return 1; }
    jlog "cpus: job $(grep Cpus_allowed_list /proc/self/status | awk '{print $2}'), client $CLIENT_CPUS"
    taskset -cp "$CLIENT_CPUS" $$ > /dev/null || { jlog "taskset of the job shell to the client cpus failed"; return 1; }
}
sim_cpus() { echo "${SIM_CPUS[$1]}"; }

jlog() { echo "$(date -Is) +$(( $(date +%s) - T0 ))s $*" | tee -a "$RUN/job.log"; }
sim_part() { echo "j${JOB}s$1"; }
# spacing 2: adjacent domains overlap under sim/fastdds_localhost.xml (job 143783); a 16-wide block per job, chosen by
# job id: two jobs on one node with the same domain discover each other's nodes (143846 / 143847 both on 40, gpu25)
sim_dom() { echo $((2 + 16 * (10#${JOB##*[!0-9]} % 6) + 2 * $1)); }
sim_vars() {  # the env of sim i, as NAME=value words (WS=$PROJ: phase3_run's default ledger goes to
             # $PROJ/.cache/vlm_skill_agent/runs/, outside the repo, as on the local machine)
    local g; g=$(sim_gpu "$1")
    echo "ROS_DOMAIN_ID=$(sim_dom "$1") GZ_PARTITION=$(sim_part "$1") WS_LOCAL=$WS_LOCAL_PERUN WS=$PROJ SLURM_JOB_ID=$JOB" \
         "SIM_WORLD=$(sim_world "$1")" \
         "PYTHONUNBUFFERED=1${VLM_BASE_URL:+ VLM_BASE_URL=$VLM_BASE_URL VLM_MODEL=$VLM_MODEL}${g:+ SIM_GPU=$g}"
}

# in_sim <i> [--sim-cpus] <cmd...>: cmd in sim i's env, cwd $REPO, ROS + overlay sourced (sim/env.sh), /opt/venv/bin
# first on PATH; on the client cpus (inherited), or on sim i's own cpus with --sim-cpus (sim_start only).
# VLM_API_KEY reaches the image as APPTAINERENV_VLM_API_KEY (vllm_start), never on a command line.
in_sim() {
    local i=$1 pin=(); shift
    [ "${1:-}" = --sim-cpus ] && { pin=(taskset -c "$(sim_cpus "$i")"); shift; }
    # shellcheck disable=SC2046  # sim_vars is NAME=value words without spaces
    if [ -n "${SIM_FAKE:-}" ]; then
        (cd "$REPO" && "${pin[@]}" env $(sim_vars "$i") VLM_API_KEY="${VLM_API_KEY:-}" PATH="$SIM_FAKE:$PATH" "$@")
    else
        # shellcheck disable=SC2046  # sim_hide is -B src:dst words without spaces
        "${pin[@]}" apptainer exec --nv --cleanenv $(sim_hide "$i") -B "$PROJ" "${SLURM_BINDS[@]}" --pwd "$REPO" \
            --env "$(sim_vars "$i" | tr ' ' ',')" "$SIF" \
            bash -c 'source sim/env.sh && export PATH=/opt/venv/bin:$PATH && exec "$@"' in_sim "$@"
    fi
}

# Sim i's world (PERUN_SIM.md section 5): SIM_WORLDS[i] from sim_job.sbatch's w<i>=<variant>, default closed; the
# variant's dir is sim/worlds/clinic/generated_<variant>/ (closed = generated/, the frozen v2 world).
sim_world() { echo "${SIM_WORLDS[$1]:-closed}"; }
sim_worlddir() {  # repo-relative world dir of sim i; 1 if it does not exist
    local w d; w=$(sim_world "$1")
    if [ "$w" = closed ]; then d=sim/worlds/clinic/generated; else d=sim/worlds/clinic/generated_$w; fi
    [ -f "$REPO/$d/clinic.sdf" ] && echo "$d"
}

sim_start() {  # setsid: the sim is its own process group; the pgid is read later from its gz sim process
    local i=$1 launch=sim/launch_clinic.sh wd=()
    [ -n "${SIM_FAKE:-}" ] && launch=$SIM_FAKE/launch.sh
    [ "$(sim_world "$i")" != closed ] && wd=("worlddir:=$REPO/$(sim_worlddir "$i")")
    [ -n "${SIM_MAP:-}" ] && wd+=("map:=$REPO/sim/maps/$SIM_MAP.yaml")  # 008: clinic_static (no movables)
    mkdir -p "$RUN/s$i"
    # SIM_SLAM (sim_job.sbatch arg `slam`, 008 map tour): SLAM Toolbox instead of AMCL on a map, as 002's map step
    local loc=(localization:=true); [ -n "${SIM_SLAM:-}" ] && loc=(slam:=true)
    in_sim "$i" --sim-cpus setsid bash "$launch" "${loc[@]}" nav2:=true "${wd[@]}" > "$RUN/s$i/launch.log" 2>&1 < /dev/null &
    jlog "sim $i start (world $(sim_world "$i"), ROS_DOMAIN_ID $(sim_dom "$i"), GZ_PARTITION $(sim_part "$i")$(g=$(sim_gpu "$i") && echo ", GPU $g"), cpus $(sim_cpus "$i"))"
}

sim_pids() {  # own processes whose environ holds GZ_PARTITION=<sim i's partition>
    local p want
    want="GZ_PARTITION=$(sim_part "$1")"
    for p in $(ps -u "$(id -u)" -o pid=); do
        tr '\0' '\n' 2>/dev/null < "/proc/$p/environ" | grep -qx -- "$want" && echo "$p"
    done
}

sim_pgid() {  # pgid of sim i's `gz sim` process (empty if none)
    local p pg cmd want
    want="GZ_PARTITION=$(sim_part "$1")"
    while read -r p pg cmd; do
        case $cmd in "gz sim"*) ;; *) continue ;; esac
        tr '\0' '\n' 2>/dev/null < "/proc/$p/environ" | grep -qx -- "$want" && { echo "$pg"; return; }
    done < <(ps -u "$(id -u)" -o pid=,pgid=,args=)
}

sim_wait_ready() {  # ready = 2x "Managed nodes are active" + "dock moved" in launch.log; bounded
    local i=$1 log=$RUN/s$1/launch.log n d need=2
    [ -n "${SIM_SLAM:-}" ] && need=1  # SLAM mode: no localization lifecycle manager
    for _ in $(seq "$READY_POLLS"); do
        n=$(grep -c "Managed nodes are active" "$log"); d=$(grep -c "dock moved" "$log")
        if [ "$n" -ge "$need" ] && [ "$d" -ge 1 ]; then
            sim_pgid "$i" > "$RUN/s$i/pgid.txt"
            [ -s "$RUN/s$i/pgid.txt" ] || { jlog "sim $i ready lines but no gz sim process with its GZ_PARTITION"; return 1; }
            jlog "sim $i ready (pgid $(cat "$RUN/s$i/pgid.txt"))"; return 0
        fi
        grep -q "DOCK MOVE FAILED" "$log" && break
        sleep 2
    done
    jlog "sim $i NOT READY after $((READY_POLLS * 2)) s"; return 1
}

# Params as launch_clinic.sh / config/robots/turtlebot4_amcl.yaml set them (PERUN_SIM.md section 2: xy 0.10,
# planner 0.0, AMCL 0.05 / 0.1). Writes s<i>/params_check.txt; returns 1 on any mismatch.
sim_params_check() {
    local i=$1 f=$RUN/s$1/params_check.txt
    in_sim "$i" timeout 300 env SKIP_AMCL="${SIM_SLAM:-}" bash -c '
        ros2 node list > /dev/null 2>&1; sleep 3; bad=0
        while read -r node par want; do
            [ -n "$SKIP_AMCL" ] && [ "$node" = /amcl ] && continue  # SLAM mode: no AMCL node
            # "Node not found" = CLI discovery race (local run 003 too, job 143785), "Wait for service timed out" = the
            # node did not answer within --timeout (default 1 s; jobs 143820 / 143821, /amcl): retry; a wrong value is not
            for _ in 1 2 3 4 5; do
                got=$(timeout 60 ros2 param get "$node" "$par" --timeout 10 2>&1 | grep -v RTPS_TRANSPORT_SHM | tail -1)
                case $got in *"Node not found"*|*"Wait for service timed out"*) sleep 3 ;; *) break ;; esac
            done
            v=${got##* }
            ok=$(python3 -c "import sys; print(abs(float(sys.argv[1]) - float(sys.argv[2])) < 1e-9)" "$v" "$want" 2>/dev/null)
            [ "$ok" = True ] || bad=1
            echo "$node $par: $got (want $want) ${ok:-False}"
        done <<EOF
/controller_server general_goal_checker.xy_goal_tolerance 0.10
/planner_server GridBased.tolerance 0.0
/amcl update_min_d 0.05
/amcl update_min_a 0.1
EOF
        [ "$bad" = 0 ] && echo PARAMS OK || { echo PARAMS MISMATCH; exit 1; }' > "$f" 2>&1
    local rc=$?
    jlog "sim $i params check rc=$rc: $(tail -1 "$f")"; return $rc
}

# Plan API: sim_run <i> <cmd ...> -- cmd in sim i's env inside the image, cwd $REPO, output appended to s<i>/cmd.log,
# returns cmd's rc. Run several with & + wait for parallel.
sim_run() {
    local i=$1; shift
    echo "=== $(date -Is) sim_run $i: $*" >> "$RUN/s$i/cmd.log"
    in_sim "$i" "$@" >> "$RUN/s$i/cmd.log" 2>&1
    local rc=$?
    echo "=== $(date -Is) rc=$rc" >> "$RUN/s$i/cmd.log"
    return $rc
}

sim_stop() {  # kill -- -pgid, 8 s, -9 on the group, then -9 on anything still holding the partition
    local i=$1 pg left
    pg=$(cat "$RUN/s$i/pgid.txt" 2>/dev/null); [ -n "$pg" ] || pg=$(sim_pgid "$i")
    if [ -n "$pg" ]; then
        kill -- -"$pg" 2>/dev/null
        for _ in $(seq 8); do [ -z "$(sim_pids "$i")" ] && break; sleep 1; done
        kill -9 -- -"$pg" 2>/dev/null
    fi
    left=$(sim_pids "$i")
    # shellcheck disable=SC2086  # one pid per word
    [ -n "$left" ] && { kill -9 $left 2>/dev/null; sleep 1; }
    jlog "sim $i stopped (pgid ${pg:-none}), processes left: $(sim_pids "$i" | wc -l)"
}

vllm_start() {  # same model + flags as hpc/serve_vllm.sbatch, 127.0.0.1, free port, --gpu-memory-utilization 0.75
    local keyf=$PROJ/runs/serve/$JOB.key port
    mkdir -p "$PROJ/runs/serve" "$PROJ/cache/vllm" "$PROJ/cache/hf" "$PROJ/cache/tmp/vllm_$JOB"
    [ -f "$VLLM_SIF" ] && [ -d "$MODEL" ] || { jlog "missing $VLLM_SIF or $MODEL"; return 1; }
    (umask 077 && python3 -c "import secrets; print(secrets.token_hex(16))" > "$keyf")  # per-job key, 0600, never on a command line; python3 (no openssl on GPU nodes assumed)
    [ "$(stat -c %a "$keyf")" = 600 ] || { jlog "key file $keyf not 0600"; return 1; }
    VLM_API_KEY=$(cat "$keyf")
    export VLM_API_KEY APPTAINERENV_VLM_API_KEY=$VLM_API_KEY APPTAINERENV_VLLM_API_KEY=$VLM_API_KEY
    export APPTAINERENV_CUDA_VISIBLE_DEVICES=${GPUS[0]:-0}
    port=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
    apptainer exec --nv --cleanenv -B "$PROJ" \
        --env "HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_HOME=$PROJ/cache/hf,VLLM_CACHE_ROOT=$PROJ/cache/vllm,TMPDIR=$PROJ/cache/tmp/vllm_$JOB,VLLM_NO_USAGE_STATS=1,DO_NOT_TRACK=1,PYTHONUNBUFFERED=1" \
        "$VLLM_SIF" vllm serve "$MODEL" \
        --served-model-name qwen3.8-27b \
        --host 127.0.0.1 --port "$port" \
        --tensor-parallel-size 1 --max-model-len 32768 \
        --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
        --limit-mm-per-prompt '{"image": 4}' --gpu-memory-utilization 0.75 > "$RUN/vllm.log" 2>&1 < /dev/null &
    VPID=$!
    jlog "vllm start pid $VPID port $port"
    for _ in $(seq 360); do  # bounded 30 min, as serve_vllm.sbatch
        kill -0 "$VPID" 2>/dev/null || { jlog "vllm exited before healthy"; return 1; }
        [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$port/health")" = 200 ] && break
        sleep 5
    done
    export VLM_BASE_URL=http://127.0.0.1:$port/v1 VLM_MODEL=qwen3.8-27b
    curl -sf -H @- "$VLM_BASE_URL/models" <<<"Authorization: Bearer $VLM_API_KEY" > /dev/null ||
        { jlog "vllm not answering $VLM_BASE_URL/models with the key after 30 min"; return 1; }
    local nokey; nokey=$(curl -s -o /dev/null -w '%{http_code}' "$VLM_BASE_URL/models")
    [ "$nokey" = 401 ] || { jlog "$VLM_BASE_URL/models without the key answered $nokey, expected 401"; return 1; }
    jlog "vllm healthy $VLM_BASE_URL (401 without the key: checked)"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/gpu mem after vllm: /' >> "$RUN/job.log"
}

# vllm_remote <jobid>: use a running hpc/serve_vllm.sbatch job (PERUN_SIM.md section 5): node + port from its
# runs/serve/<jobid>.ready, the key from runs/serve/<jobid>.key (0600, never on a command line); /models must answer
# with the key and 401 without it. Nothing to stop at the end: that job is scancelled on its own.
vllm_remote() {
    local serve=${SIM_SERVE_DIR:-$PROJ/runs/serve} node port code
    [ -s "$serve/$1.ready" ] && [ -s "$serve/$1.key" ] || { jlog "vllm job $1: no $serve/$1.ready / .key"; return 1; }
    read -r node port < "$serve/$1.ready"
    VLM_API_KEY=$(cat "$serve/$1.key")
    export VLM_API_KEY APPTAINERENV_VLM_API_KEY=$VLM_API_KEY VLM_BASE_URL=http://$node:$port/v1 VLM_MODEL=qwen3.8-27b
    code=$(curl -s -o /dev/null -w '%{http_code}' -H @- "$VLM_BASE_URL/models" <<<"Authorization: Bearer $VLM_API_KEY")
    [ "$code" = 200 ] || { jlog "remote vllm $VLM_BASE_URL/models with the key answered $code"; return 1; }
    code=$(curl -s -o /dev/null -w '%{http_code}' "$VLM_BASE_URL/models")
    [ "$code" = 401 ] || { jlog "remote vllm $VLM_BASE_URL/models without the key answered $code, expected 401"; return 1; }
    jlog "remote vllm (job $1) $VLM_BASE_URL healthy (401 without the key: checked)"
}

vllm_stop() {
    [ -n "${VPID:-}" ] || return 0
    kill "$VPID" 2>/dev/null
    for _ in $(seq 30); do kill -0 "$VPID" 2>/dev/null || break; sleep 1; done
    kill -9 "$VPID" 2>/dev/null
    rm -f "$PROJ/runs/serve/$JOB.key"
    jlog "vllm stopped"; VPID=""
}

# pin_snapshot <label>: every job process's (this shell's descendants) allowed cpus (all threads), current cpu and 5 s
# cpu use, sim i's group / starter chain / client, and a PASS / FAIL / VOID check (hpc/pin_snapshot.py), appended to
# $RUN/pinning.txt and logged (006 M5 round 3 evidence: inherited affinity is a default, not a guarantee; this checks
# it live). Never fails the plan.
pin_snapshot() {
    local out
    out=$(python3 "$REPO/hpc/pin_snapshot.py" "$RUN" "$1" $$ "$CLIENT_CPUS" "${SIM_CPUS[@]}" 2>&1)
    printf '%s\n' "$out" >> "$RUN/pinning.txt"
    jlog "pin snapshot '$1': $(grep '^pin check' <<< "$out" || echo "no check line: $(tail -1 <<< "$out")")"
}

stop_all() {  # trap EXIT: every sim, Fast DDS shm, vLLM; idempotent
    local i
    for i in $(seq 0 $((NSIM - 1))); do sim_stop "$i"; done
    in_sim 0 timeout 60 fastdds shm clean > /dev/null 2>&1
    vllm_stop
    jlog "all stopped"
}
