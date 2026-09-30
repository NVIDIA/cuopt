#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Temporary CI-hang diagnostics for #1976: conda-cpp-build / wheel-build-libcuopt-mathopt
# have hung for hours on amd64 runners with no error, no OOM kill, and no reproduction
# outside CI. This prints resource state up front and a periodic snapshot for the life
# of the wrapped build, streamed to stdout so it is visible in the live job log even if
# the job is later cancelled. Remove once the root cause is confirmed and fixed.

debug_print_resource_snapshot() {
    echo "=== debug: resource snapshot ==="
    echo "nproc: $(nproc 2>/dev/null || echo n/a)"
    echo "nproc --all: $(nproc --all 2>/dev/null || echo n/a)"
    [ -f /sys/fs/cgroup/cpu.max ] && echo "cgroup cpu.max: $(cat /sys/fs/cgroup/cpu.max)"
    [ -f /sys/fs/cgroup/memory.max ] && echo "cgroup memory.max: $(cat /sys/fs/cgroup/memory.max)"
    echo "PARALLEL_LEVEL: ${PARALLEL_LEVEL:-unset}"
    echo "CMAKE_BUILD_PARALLEL_LEVEL: ${CMAKE_BUILD_PARALLEL_LEVEL:-unset}"
    echo "RUNNER_NAME: ${RUNNER_NAME:-unset}"
    echo "=================================="
}

debug_start_watchdog() {
    local interval="${1:-20}"
    (
        while true; do
            sleep "${interval}"
            ts=$(date -u +%H:%M:%S)
            n_cc=0
            for p in /proc/[0-9]*; do
                comm=$(cat "${p}/comm" 2>/dev/null) || continue
                case "${comm}" in
                    cicc|ptxas|cc1plus|nvcc|cc1|collect2|ld) n_cc=$((n_cc + 1)) ;;
                esac
            done
            mem=$( [ -f /sys/fs/cgroup/memory.current ] && cat /sys/fs/cgroup/memory.current || echo n/a )
            peak=$( [ -f /sys/fs/cgroup/memory.peak ] && cat /sys/fs/cgroup/memory.peak || echo n/a )
            events=$( [ -f /sys/fs/cgroup/memory.events ] && tr '\n' ' ' < /sys/fs/cgroup/memory.events || echo n/a )
            cpustat=$( [ -f /sys/fs/cgroup/cpu.stat ] && tr '\n' ' ' < /sys/fs/cgroup/cpu.stat || echo n/a )
            disk=$(df -h /tmp 2>/dev/null | tail -1)
            echo "[debug-watchdog ${ts}] compiler_procs=${n_cc} mem=${mem} peak=${peak}"
            echo "[debug-watchdog ${ts}] memory.events: ${events}"
            echo "[debug-watchdog ${ts}] cpu.stat: ${cpustat}"
            echo "[debug-watchdog ${ts}] disk(/tmp): ${disk}"
            echo "[debug-watchdog ${ts}] oldest processes (pid etimes comm):"
            ps -eo pid,etimes,comm --sort=-etimes 2>/dev/null | head -6 | tail -5
        done
    ) &
    DEBUG_WATCHDOG_PID=$!
    trap 'kill "${DEBUG_WATCHDOG_PID}" 2>/dev/null || true' EXIT
}
