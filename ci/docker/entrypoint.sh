#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Entrypoint for the cuOpt container image.
#
# CUOPT_SERVER_TYPE takes precedence over the container command:
#   proxy   — HTTP proxy + cuopt_grpc_server
#   grpc    — cuopt_grpc_server only
#   legacy  — Python REST server (cuopt_server.cuopt_service)
# When it is unset, the command proxy, grpc, or legacy selects the same
# modes. No command means proxy. Any other command is executed as given.
#
# Combined-mode / gRPC-only env vars:
#   CUOPT_GRPC_PORT    — gRPC listen port when starting the sidecar or
#                        when CUOPT_SERVER_TYPE=grpc (default: 5001).
#                        CUOPT_SERVER_TYPE=grpc also still honors
#                        CUOPT_SERVER_PORT as the gRPC port for compatibility.
#   CUOPT_GPU_COUNT    — worker processes (default: omit, server default)
#   CUOPT_GRPC_ARGS    — additional CLI flags passed verbatim
#                        (e.g. "--tls --tls-cert server.crt --log-to-console")
#                        See docs/cuopt/source/cuopt-grpc/advanced.rst (flags/env);
#                        cpp/docs/grpc-server-architecture.md for contributor IPC details.
#
# Combined-mode HTTP (the proxy) uses CUOPT_SERVER_PORT (default: 5000)
# and talks to gRPC at 127.0.0.1:${CUOPT_GRPC_PORT:-5001}.

set -e

export HOME="/opt/cuopt"

build_grpc_cmd() {
    local port="$1"
    GRPC_CMD=(cuopt_grpc_server --port "${port}")

    if [ -n "${CUOPT_GPU_COUNT}" ]; then
        GRPC_CMD+=(--workers "${CUOPT_GPU_COUNT}")
    fi

    if [ -n "${CUOPT_GRPC_ARGS}" ]; then
        read -ra EXTRA <<< "${CUOPT_GRPC_ARGS}"
        GRPC_CMD+=("${EXTRA[@]}")
    fi
}

wait_for_tcp() {
    local port="$1"
    local pid="$2"
    for _ in $(seq 1 60); do
        if (echo >/dev/tcp/127.0.0.1/"${port}") 2>/dev/null; then
            return 0
        fi
        if ! kill -0 "${pid}" 2>/dev/null; then
            echo "cuopt_grpc_server exited before opening port ${port}" >&2
            return 1
        fi
        # A trapped SIGTERM interrupts sleep. Do not let set -e abort before
        # the caller can reap gRPC and return the signal status.
        sleep 0.5 || true
    done
    echo "timed out waiting for cuopt_grpc_server on port ${port}" >&2
    return 1
}

run_grpc_only() {
    local port="${CUOPT_GRPC_PORT:-${CUOPT_SERVER_PORT:-5001}}"
    build_grpc_cmd "${port}"
    exec "${GRPC_CMD[@]}"
}

grpc_pid=""
proxy_pid=""
shutdown_signal=""

stop_children() {
    local pid
    for pid in "${proxy_pid}" "${grpc_pid}"; do
        if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
            kill -TERM "${pid}" 2>/dev/null || true
        fi
    done
}

handle_shutdown() {
    shutdown_signal="$1"
    stop_children
}

run_proxy_and_grpc() {
    local grpc_port="${CUOPT_GRPC_PORT:-5001}"
    local child_status
    build_grpc_cmd "${grpc_port}"

    # This shell is PID 1 in the container. The kernel ignores SIGTERM and
    # SIGINT on PID 1 until a handler exists, so install the traps before
    # starting gRPC. Readiness can take up to 30 seconds. If either child
    # exits independently, stop the sibling so the container exits and its
    # restart policy can replace the whole stack.
    trap 'handle_shutdown TERM' TERM
    trap 'handle_shutdown INT' INT
    trap stop_children EXIT

    "${GRPC_CMD[@]}" &
    grpc_pid=$!
    if ! wait_for_tcp "${grpc_port}" "${grpc_pid}" || [ -n "${shutdown_signal}" ]; then
        kill -TERM "${grpc_pid}" 2>/dev/null || true
        wait "${grpc_pid}" 2>/dev/null || true
        trap - TERM INT EXIT
        case "${shutdown_signal}" in
            TERM) exit 143 ;;
            INT) exit 130 ;;
            *) exit 1 ;;
        esac
    fi

    python -m cuopt_server.cuopt_proxy &
    proxy_pid=$!

    set +e
    wait -n "${proxy_pid}" "${grpc_pid}"
    child_status=$?
    set -e

    stop_children
    wait || true
    trap - TERM INT EXIT

    if [ -n "${shutdown_signal}" ]; then
        # Preserve the conventional signal exit status (SIGTERM=143,
        # SIGINT=130) after both children have shut down.
        if [ "${shutdown_signal}" = "TERM" ]; then
            return 143
        fi
        return 130
    fi
    return "${child_status}"
}

# CUOPT_SERVER_TYPE selects the server and overrides the container command.
mode=""
case "${CUOPT_SERVER_TYPE-}" in
    "")
        ;;
    proxy | grpc | legacy)
        mode="${CUOPT_SERVER_TYPE}"
        ;;
    *)
        echo "CUOPT_SERVER_TYPE must be proxy, grpc, or legacy" >&2
        exit 1
        ;;
esac

# When the variable is unset, a single word selects the same modes.
# No arguments means proxy.
if [ -z "${mode}" ]; then
    if [ $# -eq 0 ] || { [ $# -eq 1 ] && [ "$1" = "proxy" ]; }; then
        mode="proxy"
    elif [ $# -eq 1 ] && [ "$1" = "grpc" ]; then
        mode="grpc"
    elif [ $# -eq 1 ] && [ "$1" = "legacy" ]; then
        mode="legacy"
    fi
fi

case "${mode}" in
    proxy)
        run_proxy_and_grpc
        exit $?
        ;;
    grpc)
        run_grpc_only
        ;;
    legacy)
        exec python -m cuopt_server.cuopt_service
        ;;
    *)
        exec "$@"
        ;;
esac
