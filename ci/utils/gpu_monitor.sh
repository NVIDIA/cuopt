#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Lightweight background GPU-memory sampler for diagnosing OOM during
# parallel (pytest-xdist) test runs. Samples device memory and per-process
# (per xdist worker) usage on an interval, then prints the peak on stop.
#
# Usage:
#   source ci/utils/gpu_monitor.sh
#   gpu_mon_start          # begin sampling in the background
#   <run tests>            # e.g. pytest -n 2 ...
#   gpu_mon_stop           # stop sampling, print peak summary
#
# Tunables (env): GPU_MON_INTERVAL (seconds between samples, default 2).

GPU_MON_PID=""
GPU_MON_PEAK_FILE=""

# Start the background sampler. No-op (with a note) when nvidia-smi is absent.
gpu_mon_start() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[gpu-mon] nvidia-smi not found; skipping GPU monitoring"
        return 0
    fi

    local interval="${GPU_MON_INTERVAL:-2}"
    GPU_MON_PEAK_FILE="$(mktemp)"
    echo 0 > "${GPU_MON_PEAK_FILE}"

    echo "[gpu-mon] monitoring GPU memory every ${interval}s (python PIDs below are the xdist workers)"
    echo "[gpu-mon] initial device state:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free \
        --format=csv,noheader | sed 's/^/[gpu-mon]   /'

    (
        # Sampling loop is best-effort; never let a transient nvidia-smi
        # hiccup kill it or leak set -e into the parent.
        set +e
        while true; do
            local ts total used free apps
            ts="$(date -u +%H:%M:%S)"
            read -r total used free < <(
                nvidia-smi --query-gpu=memory.total,memory.used,memory.free \
                    --format=csv,noheader,nounits 2>/dev/null | tr -d ',' | head -1
            )
            apps="$(
                nvidia-smi --query-compute-apps=pid,used_memory \
                    --format=csv,noheader,nounits 2>/dev/null \
                    | tr -d ',' | awk '{printf "pid%s=%sMiB ", $1, $2}'
            )"
            echo "[gpu-mon] ${ts} used=${used:-?}MiB free=${free:-?}MiB total=${total:-?}MiB | ${apps:-no compute apps}"

            local prev
            prev="$(cat "${GPU_MON_PEAK_FILE}" 2>/dev/null || echo 0)"
            if [ -n "${used}" ] && [ "${used}" -gt "${prev:-0}" ] 2>/dev/null; then
                echo "${used}" > "${GPU_MON_PEAK_FILE}"
            fi
            sleep "${interval}"
        done
    ) &
    GPU_MON_PID=$!
}

# Stop the sampler and print the peak device usage observed.
gpu_mon_stop() {
    [ -z "${GPU_MON_PID}" ] && return 0

    kill "${GPU_MON_PID}" 2>/dev/null || true
    wait "${GPU_MON_PID}" 2>/dev/null || true

    local peak total
    peak="$(cat "${GPU_MON_PEAK_FILE}" 2>/dev/null || echo '?')"
    total="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
    echo "[gpu-mon] >>> PEAK device memory used: ${peak}MiB / ${total:-?}MiB"

    rm -f "${GPU_MON_PEAK_FILE}"
    GPU_MON_PID=""
    GPU_MON_PEAK_FILE=""
}
