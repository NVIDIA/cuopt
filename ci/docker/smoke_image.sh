#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Smoke-test that a published cuOpt image starts:
#   default  — HTTP proxy + cuopt_grpc_server (image command: proxy)
#   grpc     — command grpc
#   legacy   — command legacy
# Runs on the host with docker so the real ENTRYPOINT/CMD path is exercised
# (unlike test_image.sh, which runs inside a GHA job container and never
# launches the servers).
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
  shift
  local patterns=()
  while [[ $# -gt 0 && "$1" != -* ]]; do
    patterns+=("$1")
    shift
  done
  # Remaining args are extra docker run flags, then an optional
  # `-- command` placed after the image name.

  local name log cid i pat
  local docker_args=()
  local cmd=()
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--" ]]; then
      shift
      cmd=("$@")
      break
    fi
    docker_args+=("$1")
    shift
  done
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
  local -a run_args=("${GPU_ARGS[@]}")
  if [[ ${#docker_args[@]} -gt 0 ]]; then
    run_args+=("${docker_args[@]}")
  fi
  run_args+=("${IMAGE}")
  if [[ ${#cmd[@]} -gt 0 ]]; then
    run_args+=("${cmd[@]}")
  fi
  if ! cid="$(docker run -d --name "${name}" "${run_args[@]}")"; then
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

    local all_matched=1
    for pat in "${patterns[@]}"; do
      if ! grep -qE "${pat}" "${log}"; then
        all_matched=0
        break
      fi
    done
    if [[ "${all_matched}" -eq 1 ]]; then
      if ! docker inspect -f '{{.State.Running}}' "${cid}" 2>/dev/null | grep -qx true; then
        echo "----- ${label} logs -----"
        cat "${log}"
        smoke_fail "${label}: container exited after becoming ready"
        return 1
      fi
      pass "${label}: matched ${patterns[*]}"
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
  smoke_fail "${label}: timed out after ${TIMEOUT_SECS}s waiting for ${patterns[*]}"
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

smoke_one default 'Listening on' 'cuOpt HTTP proxy'
smoke_one grpc 'Listening on' -- grpc
smoke_one legacy 'cuopt server version' -- legacy

pass "Smoke OK for ${IMAGE}"
