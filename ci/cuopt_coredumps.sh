#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Enable core dumps during CI test scripts and collect files into
# ${RAPIDS_ARTIFACTS_DIR}/coredumps so rapids-upload-artifacts-dir uploads them (S3).
#
# Shells: source this file from repo ci/ scripts, then call cuopt_enable_coredumps
# and trap cuopt_collect_coredumps on EXIT.

cuopt_enable_coredumps() {
  local ws base pattern
  ws="${GITHUB_WORKSPACE:-${PWD}}"
  base="${RAPIDS_ARTIFACTS_DIR:-${ws}/artifacts}"
  export CUOPT_COREDUMP_DIR="${base}/coredumps"
  mkdir -p "${CUOPT_COREDUMP_DIR}"

  ulimit -c unlimited 2>/dev/null || true

  if [[ -w /proc/sys/kernel/core_pattern ]]; then
    echo "${CUOPT_COREDUMP_DIR}/core.%e.%p.%t" >/proc/sys/kernel/core_pattern 2>/dev/null || true
  fi

  pattern="$(cat /proc/sys/kernel/core_pattern 2>/dev/null || echo n/a)"
  if declare -F rapids-logger &>/dev/null; then
    rapids-logger "Core dumps: dir=${CUOPT_COREDUMP_DIR} ulimit -c=$(ulimit -c) core_pattern=${pattern}"
  else
    echo "Core dumps: dir=${CUOPT_COREDUMP_DIR} ulimit -c=$(ulimit -c) core_pattern=${pattern}"
  fi
}

cuopt_collect_coredumps() {
  local ws base dest n_before n_after f rel dest_name dest_path
  ws="${GITHUB_WORKSPACE:-${PWD}}"
  base="${RAPIDS_ARTIFACTS_DIR:-${ws}/artifacts}"
  dest="${base}/coredumps"
  mkdir -p "${dest}"

  n_before="$(find "${dest}" -type f 2>/dev/null | wc -l | tr -d '[:space:]')"

  while IFS= read -r -d '' f; do
    [[ -f "${f}" ]] || continue
    case "${f}" in
      "${dest}/"*) continue ;;
    esac
    rel="${f#"${ws}"/}"
    if [[ "${rel}" == "${f}" ]]; then
      rel="$(basename "${f}")"
    fi
    dest_name="${rel//\//_}"
    dest_path="${dest}/${dest_name}"
    if [[ -e "${dest_path}" ]]; then
      dest_path="${dest}/${dest_name}.${RANDOM}"
    fi
    cp -a "${f}" "${dest_path}" 2>/dev/null || true
  done < <(
    find "${ws}" \
      \( -path '*/.git/*' -o -path '*/opt/conda/*' -o -path '*/conda_pkgs/*' -o -path '*/artifacts/coredumps/*' \) -prune -o \
      \( -name 'core' -o -name 'core.*' \) -type f -print0 2>/dev/null
  )

  n_after="$(find "${dest}" -type f 2>/dev/null | wc -l | tr -d '[:space:]')"
  if [[ "${n_after}" -gt "${n_before}" ]]; then
    if declare -F rapids-logger &>/dev/null; then
      rapids-logger "Wrote $((n_after - n_before)) core file(s) into ${dest} (${n_after} total)"
    else
      echo "cuOpt coredumps: ${n_after} file(s) in ${dest}"
    fi
  fi
}
