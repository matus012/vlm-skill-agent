# NOTE (e7f62ee, 006 M5 round 3): since then sim_run commands run on the client cpus (the job's cpus no sim holds) and
# /proc/self Cpus_allowed_list in a plan = the client cpus; results recorded for this plan used the older layout.
# 006 diagnosis, no gate verdicts. Job 143786: RTF 0.41-0.46 per sim at N = 4 (the local machine 0.85-0.94) and `gz topic -e` CLI
# deaths (SIGSEGV / SIGABRT) in 15 of 20 M4 / M6 runs (never on the local machine). Records host CPU / GPU facts, which GL / EGL
# libraries sim 0's processes load, RTF + per-process / per-thread CPU + GPU use with 4 sims, the gz CLI death rate,
# then the same with sim 0 alone. No vLLM: separates the sims' own cost.
#   sbatch --time=00:20:00 hpc/sim_job.sbatch hpc/plans/diag_rtf.sh 4
D=$RUN/diag; mkdir -p "$D"
PY=/opt/venv/bin/python
{ lscpu; echo "nproc $(nproc)"; grep Cpus_allowed_list /proc/self/status; cat /sys/fs/cgroup/cpu.max 2>/dev/null
  free -g; nvidia-smi; nvidia-smi -q -d CLOCK,PERFORMANCE; } > "$D/host.txt" 2>&1
for p in $(sim_pids 0); do  # GL / EGL vendor actually mapped: NVIDIA (libEGL_nvidia / libnvidia-*) or Mesa (llvmpipe)
    m=$(grep -oE "lib(EGL|GLX)_(nvidia|mesa)[^ ]*|libnvidia-(egl|gl)core[^ ]*|libgallium[^ ]*|[a-z_]*swrast[^ ]*|libLLVM[^ ]*" \
        "/proc/$p/maps" 2>/dev/null | sort -u | tr '\n' ' ')
    [ -n "$m" ] && echo "$p $(tr '\0' ' ' < "/proc/$p/cmdline" | cut -c1-100) :: $m"
done > "$D/render_libs.txt"
snap() {  # 10 s: processes and threads by CPU (2nd top sample = the interval), GPU per-second use
    top -b -n 2 -d 5 -w 250 -o %CPU 2>&1 | awk '/^top -/{n++} n==2' | head -50 > "$D/top_$1.txt" &
    top -b -H -n 2 -d 5 -w 250 -o %CPU 2>&1 | awk '/^top -/{n++} n==2' | head -60 > "$D/topH_$1.txt" &
    nvidia-smi dmon -s pucm -c 10 > "$D/dmon_$1.txt" 2>&1
    wait; nvidia-smi > "$D/smi_$1.txt" 2>&1
}
rtf_all() {  # crosstalk.rtf: Gazebo sim time over 10 wall s, per sim, all at once
    local tag=$1 i; shift
    for i in "$@"; do
        sim_run "$i" $PY -c "import sys; from probes.crosstalk import rtf; print('RTF', sys.argv[1], sys.argv[2], round(rtf('clinic', sys.argv[2]), 3))" "$tag" "$(sim_part "$i")" &
    done
    wait
}
gz_cli() {  # 40 x the GT read adapters use; rc + last stderr line of each death
    sim_run 0 bash -c 'ok=0; bad=0; t0=$(date +%s.%N)
        for k in $(seq 40); do
            e=$(gz topic -e -n 1 -t /world/clinic/pose/info --json-output 2>&1 > /dev/null); rc=$?
            if [ $rc = 0 ]; then ok=$((ok + 1)); else bad=$((bad + 1)); echo "GZCLI death rc=$rc: $(echo "$e" | tail -2 | tr "\n" " " | cut -c1-300)"; fi
        done
        echo "GZCLI $0 ok $ok bad $bad mean_s $(awk -v a="$t0" -v b="$(date +%s.%N)" "BEGIN { printf \"%.2f\", (b - a) / 40 }")"' "$1"
}
snap n4 & rtf_all n4 0 1 2 3; wait
gz_cli n4
sim_stop 1; sim_stop 2; sim_stop 3; sleep 20
snap n1 & rtf_all n1 0; wait
gz_cli n1
grep -h "^RTF\|^GZCLI" "$RUN"/s*/cmd.log | tee "$D/summary.txt"
