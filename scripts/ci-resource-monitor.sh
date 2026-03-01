#!/usr/bin/env bash
# CI resource monitor for runner crash diagnosis
# Prints system metrics every MONITOR_INTERVAL seconds to stderr, prefixed with [MONITOR].
# Uses stderr to avoid interleaving with pytest stdout mid-line.
# GitHub Actions captures both fds, so metrics printed before a crash
# will be visible even if the runner dies (no logs survive a hard kill).
#
# Usage: ./scripts/ci-resource-monitor.sh &
# Environment: MONITOR_INTERVAL=30 (default)

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

INTERVAL="${MONITOR_INTERVAL:-30}"
OS="$(uname -s)"

# ============================================================================
# Cleanup
# ============================================================================

cleanup() {
    exit 0
}
trap cleanup SIGTERM SIGINT

# ============================================================================
# Shared helpers (use dynamic scoping — caller declares locals)
# ============================================================================

# QEMU process metrics shared between macOS and Linux.
# Sets: qemu_pids, qemu_procs, qemu_rss_kb, qemu_rss_mb
# Caller must compute qemu_threads separately (OS-specific).
collect_qemu_base() {
    qemu_pids="$(pgrep -f 'qemu-system-' 2>/dev/null || true)"
    qemu_procs=0
    qemu_rss_kb=0
    if [[ -n "$qemu_pids" ]]; then
        qemu_procs="$(echo "$qemu_pids" | wc -l | tr -d ' ')"
        qemu_rss_kb="$(echo "$qemu_pids" | xargs -I{} ps -o rss= -p {} 2>/dev/null | awk '{s+=$1} END {print s+0}')" || qemu_rss_kb=0
    fi
    qemu_rss_mb=$(( qemu_rss_kb / 1024 ))
}

# Emit the monitoring line to stderr.
# Reads all metric variables from caller's scope.
emit_metrics() {
    echo "[MONITOR] $(date +%H:%M:%S) qemu_procs=$qemu_procs qemu_threads=$qemu_threads qemu_rss_mb=$qemu_rss_mb mem_used_mb=$mem_used_mb mem_avail_mb=$mem_avail_mb mem_total_mb=$mem_total_mb mem_pressure=$mem_pressure swap_used_mb=$swap_used_mb load_1m=$load_1m sys_procs=$sys_procs sys_threads=$sys_threads" >&2
}

# ============================================================================
# macOS metric collection
# ============================================================================

collect_macos() {
    # QEMU process metrics (shared base + macOS thread counting)
    local qemu_pids qemu_procs qemu_threads=0 qemu_rss_kb qemu_rss_mb
    collect_qemu_base
    if [[ -n "$qemu_pids" ]]; then
        # Thread count from ps -M (one line per thread, skip header per PID)
        qemu_threads="$(echo "$qemu_pids" | while read -r pid; do
            ps -M -p "$pid" 2>/dev/null | tail -n +2
        done | wc -l | tr -d ' ')" || qemu_threads=0
    fi

    # System memory via vm_stat
    local page_size mem_free_pages=0 mem_active_pages=0 mem_inactive_pages=0
    local mem_speculative_pages=0 mem_wired_pages=0
    local vmstat_output
    vmstat_output="$(vm_stat 2>/dev/null || true)"
    if [[ -n "$vmstat_output" ]]; then
        page_size="$(echo "$vmstat_output" | head -1 | grep -o '[0-9]*' || echo 4096)"
        mem_free_pages="$(echo "$vmstat_output" | awk '/Pages free:/ {gsub(/\./,"",$3); print $3+0}')" || mem_free_pages=0
        mem_active_pages="$(echo "$vmstat_output" | awk '/Pages active:/ {gsub(/\./,"",$3); print $3+0}')" || mem_active_pages=0
        mem_inactive_pages="$(echo "$vmstat_output" | awk '/Pages inactive:/ {gsub(/\./,"",$3); print $3+0}')" || mem_inactive_pages=0
        mem_speculative_pages="$(echo "$vmstat_output" | awk '/Pages speculative:/ {gsub(/\./,"",$3); print $3+0}')" || mem_speculative_pages=0
        mem_wired_pages="$(echo "$vmstat_output" | awk '/Pages wired down:/ {gsub(/\./,"",$4); print $4+0}')" || mem_wired_pages=0
    else
        page_size=4096
    fi

    local mem_total_bytes mem_total_mb mem_used_mb mem_avail_mb
    mem_total_bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
    mem_total_mb=$(( mem_total_bytes / 1024 / 1024 ))
    mem_used_mb=$(( (mem_active_pages + mem_wired_pages) * page_size / 1024 / 1024 ))
    mem_avail_mb=$(( (mem_free_pages + mem_inactive_pages + mem_speculative_pages) * page_size / 1024 / 1024 ))

    # Memory pressure level
    local mem_pressure_raw mem_pressure
    mem_pressure_raw="$(sysctl -n kern.memorystatus_vm_pressure_level 2>/dev/null || echo 1)"
    case "$mem_pressure_raw" in
        1) mem_pressure="normal" ;;
        2) mem_pressure="WARN" ;;
        4) mem_pressure="CRITICAL" ;;
        *) mem_pressure="unknown($mem_pressure_raw)" ;;
    esac

    # Swap usage
    local swap_used_mb=0
    local swap_output
    swap_output="$(sysctl vm.swapusage 2>/dev/null || true)"
    if [[ -n "$swap_output" ]]; then
        swap_used_mb="$(echo "$swap_output" | awk '{for(i=1;i<=NF;i++) if($i=="used") {gsub(/M/,"",$(i+2)); printf "%.0f", $(i+2)+0}}' || echo 0)"
    fi

    # Load average (1-minute)
    local load_1m
    load_1m="$(sysctl -n vm.loadavg 2>/dev/null | awk '{print $2}' || echo 0)"

    # System-wide process and thread counts
    local sys_procs sys_threads
    sys_procs="$(ps -Ao pid= 2>/dev/null | wc -l | tr -d ' ')" || sys_procs=0
    sys_threads="$(ps -eM 2>/dev/null | tail -n +2 | wc -l | tr -d ' ')" || sys_threads=0

    emit_metrics
}

# ============================================================================
# Linux metric collection
# ============================================================================

collect_linux() {
    # QEMU process metrics (shared base + Linux thread counting)
    local qemu_pids qemu_procs qemu_threads=0 qemu_rss_kb qemu_rss_mb
    collect_qemu_base
    if [[ -n "$qemu_pids" ]]; then
        # Thread count from /proc/<pid>/status
        qemu_threads="$(echo "$qemu_pids" | while read -r pid; do
            awk '/^Threads:/ {print $2}' "/proc/$pid/status" 2>/dev/null || echo 0
        done | awk '{s+=$1} END {print s+0}')" || qemu_threads=0
    fi

    # System memory from /proc/meminfo
    local mem_total_kb=0 mem_avail_kb=0 swap_total_kb=0 swap_free_kb=0
    if [[ -r /proc/meminfo ]]; then
        mem_total_kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null)" || mem_total_kb=0
        mem_avail_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null)" || mem_avail_kb=0
        swap_total_kb="$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo 2>/dev/null)" || swap_total_kb=0
        swap_free_kb="$(awk '/^SwapFree:/ {print $2}' /proc/meminfo 2>/dev/null)" || swap_free_kb=0
    fi
    local mem_total_mb=$(( mem_total_kb / 1024 ))
    local mem_avail_mb=$(( mem_avail_kb / 1024 ))
    local mem_used_mb=$(( (mem_total_kb - mem_avail_kb) / 1024 ))
    local swap_used_mb=$(( (swap_total_kb - swap_free_kb) / 1024 ))

    # Memory pressure from PSI
    local mem_pressure="n/a"
    if [[ -r /proc/pressure/memory ]]; then
        local psi_avg10
        psi_avg10="$(awk '/^some/ {for(i=1;i<=NF;i++) if($i ~ /^avg10=/) {split($i,a,"="); printf "%.1f", a[2]}}' /proc/pressure/memory 2>/dev/null || echo "0.0")"
        if awk "BEGIN {exit !($psi_avg10 > 10)}" 2>/dev/null; then
            mem_pressure="HIGH(${psi_avg10}%)"
        else
            mem_pressure="normal(${psi_avg10}%)"
        fi
    fi

    # Load average (1-minute)
    local load_1m
    load_1m="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0)"

    # System-wide process and thread counts
    local sys_procs sys_threads
    sys_procs="$(ps -e --no-headers 2>/dev/null | wc -l | tr -d ' ')" || sys_procs=0
    sys_threads="$(awk '/^Threads:/ {s+=$2} END {print s+0}' /proc/[0-9]*/status 2>/dev/null)" || sys_threads=0

    emit_metrics
}

# ============================================================================
# Main
# ============================================================================

echo "[MONITOR] started pid=$$ interval=${INTERVAL}s os=$OS" >&2

collect() {
    case "$OS" in
        Darwin) collect_macos || true ;;
        Linux)  collect_linux || true ;;
        *)
            echo "[MONITOR] unsupported os=$OS" >&2
            exit 1
            ;;
    esac
}

# First sample immediately, then every INTERVAL seconds
collect
while sleep "$INTERVAL"; do
    collect
done
