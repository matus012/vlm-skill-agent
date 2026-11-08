# NOTE (e7f62ee, 006 M5 round 3): since then sim_run commands run on the client cpus (the job's cpus no sim holds) and
# /proc/self Cpus_allowed_list in a plan = the client cpus; results recorded for this plan used the older layout.
# Diagnosis helpers for 006 plans (sourced; needs the sim_lib plan API + D, PY): snap <tag>, rtf_all <tag> <i...>,
# chunk <i> <n>, pin <i> <n>. Results append to $D/summary.txt.
snap() {  # 10 s: processes and threads by CPU, our CPU total, GPU per-second use
    top -b -n 2 -d 5 -w 250 -o %CPU 2>&1 | awk '/^top -/{n++} n==2' > "$D/top_$1.txt" &
    top -b -H -n 2 -d 5 -w 250 -o %CPU 2>&1 | awk '/^top -/{n++} n==2' | head -60 > "$D/topH_$1.txt" &
    nvidia-smi dmon -s pucm -c 10 > "$D/dmon_$1.txt" 2>&1
    wait
    awk -v u="$(id -un | cut -c1-7)" '$2 ~ "^"u {s += $9} END {printf "CPU %s ours %.0f %%\n", "'"$1"'", s}' "$D/top_$1.txt" | tee -a "$D/summary.txt"
}
rtf_all() {  # crosstalk.rtf: Gazebo sim time over 10 wall s, per sim, all at once
    local tag=$1 i; shift
    for i in "$@"; do
        sim_run "$i" $PY -c "import sys; from probes.crosstalk import rtf; print('RTF', sys.argv[1], sys.argv[2], round(rtf('clinic', sys.argv[2]), 3))" "$tag" "$(sim_part "$i")" &
    done
    wait
    grep -h "^RTF $tag " "$RUN"/s*/cmd.log >> "$D/summary.txt"
}
chunk() {  # chunk <i> <n>: the i-th of n equal contiguous slices of this job's cpus
    python3 - "$1" "$2" <<'PY'
import re, sys
i, n = int(sys.argv[1]), int(sys.argv[2])
a = re.search(r"Cpus_allowed_list:\s*(\S+)", open("/proc/self/status").read()).group(1)
c = [x for r in a.split(",") for x in range(int(r.split("-")[0]), int(r.split("-")[-1]) + 1)]
k = len(c) // n
print(",".join(map(str, c[i * k:(i + 1) * k])))
PY
}
pin() {  # pin <i> <n>: every process (all threads) of sim i to chunk i of n
    local c p k=0
    c=$(chunk "$1" "$2")
    for p in $(sim_pids "$1"); do taskset -a -p -c "$c" "$p" > /dev/null 2>&1 && k=$((k + 1)); done
    echo "PIN sim $1 -> $c ($k processes)" | tee -a "$D/summary.txt"
}
