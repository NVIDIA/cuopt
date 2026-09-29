#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Smoke-test that a published cuOpt image starts its default REST server and
# the gRPC server via CUOPT_SERVER_TYPE=grpc. Runs on the host with docker so
# the real ENTRYPOINT/CMD path is exercised (unlike test_image.sh, which runs
# inside a GHA job container and never launches the servers).
#
# Usage (any published or locally built tag):
#   ./ci/docker/smoke_image.sh nvidia/cuopt:[TAG]
#   ./ci/docker/smoke_image.sh nvidia/cuopt:[TAG]-ubi10
#
# Env:
#   SMOKE_TIMEOUT_SECS  Max seconds to wait for listen (default: 90)
#   SMOKE_GPU_ARGS      Docker GPU flags (default: --gpus all)

set -euo pipefail

IMAGE="${1:?usage: $0 <image>}"
TIMEOUT_SECS="${SMOKE_TIMEOUT_SECS:-90}"
# shellcheck disable=SC2206
GPU_ARGS=(${SMOKE_GPU_ARGS:---gpus all})

pass() { printf 'PASS  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*" >&2; exit 1; }
info() { printf 'INFO  %s\n' "$*"; }

smoke_one() {
  local label="$1"
  local expect_re="$2"
  shift 2
  # Remaining args are extra docker run flags (e.g. -e CUOPT_SERVER_TYPE=grpc).

  local name log cid i
  name="cuopt-smoke-${label}-$$"
  log="$(mktemp)"
  cid=""

  smoke_fail() {
    printf 'FAIL  %s\n' "$*" >&2
  }

  cleanup() {
    if [[ -n "${cid}" ]]; then
      docker rm -f "${cid}" >/dev/null 2>&1 || true
    fi
    rm -f "${log}"
  }
  trap cleanup RETURN

  info "Starting ${label} server from ${IMAGE}"
  # Do not use --rm: a fast crash (e.g. missing libnccl.so.2) would delete the
  # container before we can collect logs.
  if ! cid="$(docker run -d --name "${name}" "${GPU_ARGS[@]}" "$@" "${IMAGE}")"; then
    smoke_fail "${label}: docker run failed"
    return 1
  fi

  for ((i = 1; i <= TIMEOUT_SECS; i++)); do
    docker logs "${cid}" >"${log}" 2>&1 || true

    if grep -qiE 'error while loading shared libraries|libnccl\.so|FATAL FIPS SELFTEST|OpenSSL internal error' "${log}"; then
      echo "----- ${label} logs -----"
      cat "${log}"
      smoke_fail "${label}: loader/crypto failure while starting"
      return 1
    fi

    if grep -qE "${expect_re}" "${log}"; then
      pass "${label}: matched /${expect_re}/"
      return 0
    fi

    # Container exited before listen — dump logs and fail.
    if ! docker inspect -f '{{.State.Running}}' "${cid}" 2>/dev/null | grep -qx true; then
      echo "----- ${label} logs -----"
      cat "${log}"
      smoke_fail "${label}: container exited before becoming ready"
      return 1
    fi

    sleep 1
  done

  echo "----- ${label} logs -----"
  cat "${log}"
  smoke_fail "${label}: timed out after ${TIMEOUT_SECS}s waiting for /${expect_re}/"
  return 1
}

info "Pulling ${IMAGE}"
if ! docker pull "${IMAGE}"; then
  if docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    info "Pull failed; using local image ${IMAGE}"
  else
    fail "Pull failed and no local image named ${IMAGE}"
  fi
fi

smoke_one rest 'Uvicorn running on'
smoke_one grpc 'Listening on' -e CUOPT_SERVER_TYPE=grpc

pass "Smoke OK for ${IMAGE}"
