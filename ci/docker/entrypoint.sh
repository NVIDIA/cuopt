#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Entrypoint for the cuOpt container image.
#
# Server selection (in order of precedence):
#   1. Explicit command: docker run <image> cuopt_grpc_server [args...]
#   2. Environment variable: CUOPT_SERVER_TYPE=grpc
#   3. Default: Python REST server (cuopt_server.cuopt_service)
#
# When CUOPT_SERVER_TYPE=grpc, the following env vars configure the gRPC server:
#   CUOPT_SERVER_PORT  — listen port       (default: 5001)
#   CUOPT_GPU_COUNT    — worker processes  (default: 1)
#   CUOPT_GRPC_ARGS    — additional CLI flags passed verbatim
#                        (e.g. "--tls --tls-cert server.crt --log-to-console")
#                        See docs/cuopt/source/cuopt-grpc/advanced.rst (flags/env);
#                        cpp/docs/grpc-server-architecture.md for contributor IPC details.
#                        for all available flags.

set -e

export HOME="/opt/cuopt"

# If CUOPT_SERVER_TYPE=grpc, build a command line from env vars and launch.
if [ "${CUOPT_SERVER_TYPE}" = "grpc" ]; then
    GRPC_CMD=(cuopt_grpc_server)

    GRPC_CMD+=(--port "${CUOPT_SERVER_PORT:-5001}")

    if [ -n "${CUOPT_GPU_COUNT}" ]; then
        GRPC_CMD+=(--workers "${CUOPT_GPU_COUNT}")
    fi

    # Allow arbitrary extra flags (e.g. --tls, --log-to-console)
    if [ -n "${CUOPT_GRPC_ARGS}" ]; then
        read -ra EXTRA <<< "${CUOPT_GRPC_ARGS}"
        GRPC_CMD+=("${EXTRA[@]}")
    fi

    exec "${GRPC_CMD[@]}"
fi

exec "$@"
