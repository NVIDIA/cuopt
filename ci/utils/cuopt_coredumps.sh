#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Enable core dumps during CI test scripts and collect files into
# ${RAPIDS_ARTIFACTS_DIR}/${CUOPT_GDB_CORE_ARTIFACT_DIR} so rapids-upload-artifacts-dir
# uploads them to S3 as:
#   {rapids-matrix-prefix}.{cuopt-gdb-cores_JOB_cudaVER_pyVER_arch_BUILDTYPE}
# RAPIDS rapids-upload-to-s3 tgz-streams each directory (gzip-compressed tar); the object
# name often has no .tar.gz suffix in listings, but downloads are still archives. Very small
# sizes (~100 B) usually mean an almost-empty archive (no core files landed on disk). The
# trailing segment includes a job label, resolved in order:
#   1) CUOPT_CI_JOB_LABEL if set (workflow/setter can export the real GitHub job id).
#   2) GITHUB_JOB if it looks like a caller job id (not generic RAPIDS callee ids such as tests/test).
#   3) The ci/test_*.sh that sourced this file: label derived by naming rules (new drivers need
#      no edits here — e.g. test_foo.sh → conda-foo-tests, test_wheel_bar.sh → wheel-bar-tests).
# Then CUDA / Python / arch / build_type from RAPIDS CI env.
#
# Test drivers: source ci/utils/cuopt_coredumps.sh from a sibling ci/test_*.sh, then call
# cuopt_coredumps_ci_setup (enable + EXIT trap).

# Set in cuopt_enable_coredumps; collect reuses when non-empty.
CUOPT_GDB_CORE_ARTIFACT_DIR=

# Reusable RAPIDS workflows often use non-unique job ids ("tests", "test", "build", …).
# GITHUB_JOB=test (singular) is common; treating it as meaningful produced labels like
# "test" and hid script-based names (wheel-cuopt-tests, conda-cpp-tests).
cuopt__github_job_is_generic() {
  case "${1:-}" in
    "" | test | tests | build | compute-matrix | prepare | package) return 0 ;;
    *) return 1 ;;
  esac
}

# test_cpp.sh → conda-cpp-tests; test_wheel_cuopt_server.sh → wheel-cuopt-server-tests; etc.
cuopt__job_label_from_entry_script_basename() {
  local b="$1"
  b="${b%.sh}"
  case "$b" in
    test_wheel_*)
      b="${b#test_wheel_}"
      echo "wheel-${b//_/-}-tests"
      ;;
    test_self_hosted_*)
      b="${b#test_self_hosted_}"
      echo "self-hosted-${b//_/-}-tests"
      ;;
    test_skills_*)
      b="${b#test_skills_}"
      echo "conda-skills-${b//_/-}"
      ;;
    test_*memcheck)
      b="${b#test_}"
      echo "conda-${b//_/-}"
      ;;
    test_*)
      b="${b#test_}"
      echo "conda-${b//_/-}-tests"
      ;;
    *)
      echo "unknown-job"
      ;;
  esac
}

cuopt__find_ci_entry_test_script_basename() {
  local i f base
  for ((i = 0; i < ${#BASH_SOURCE[@]}; i++)); do
    f="${BASH_SOURCE[$i]}"
    base="$(basename "${f}")"
    [[ "${base}" == "cuopt_coredumps.sh" ]] && continue
    case "${base}" in
      test_*.sh) echo "${base}"; return ;;
    esac
  done
  echo ""
}

cuopt__infer_ci_job_label_from_call_stack() {
  local nb
  nb="$(cuopt__find_ci_entry_test_script_basename)"
  if [[ -n "${nb}" ]]; then
    cuopt__job_label_from_entry_script_basename "${nb}"
    return
  fi
  echo "unknown-job"
}

cuopt__resolve_ci_job_label() {
  if [[ -n "${CUOPT_CI_JOB_LABEL:-}" ]]; then
    echo "${CUOPT_CI_JOB_LABEL}"
    return
  fi
  if [[ -n "${GITHUB_JOB:-}" ]] && ! cuopt__github_job_is_generic "${GITHUB_JOB}"; then
    echo "${GITHUB_JOB}"
    return
  fi
  cuopt__infer_ci_job_label_from_call_stack
}

cuopt__gdb_core_artifact_basename() {
  local job cuda_ver py_ver arch_ bt
  job="$(cuopt__resolve_ci_job_label)"
  job="${job//[^a-zA-Z0-9_-]/_}"
  cuda_ver="${RAPIDS_CUDA_VERSION:-unknown}"
  cuda_ver="${cuda_ver//[^a-zA-Z0-9._-]/_}"
  py_ver="${RAPIDS_PY_VERSION:-na}"
  py_ver="${py_ver//[^a-zA-Z0-9._-]/_}"
  arch_="$(arch 2>/dev/null || true)"
  [[ -z "${arch_}" ]] && arch_="$(uname -m)"
  arch_="${arch_//[^a-zA-Z0-9_-]/_}"
  bt="${RAPIDS_BUILD_TYPE:-na}"
  bt="${bt//[^a-zA-Z0-9_-]/_}"
  echo "cuopt-gdb-cores_${job}_cuda${cuda_ver}_py${py_ver}_${arch_}_${bt}"
}

cuopt_enable_coredumps() {
  local ws base pattern
  ws="${GITHUB_WORKSPACE:-${PWD}}"
  base="${RAPIDS_ARTIFACTS_DIR:-${ws}/artifacts}"
  CUOPT_CI_JOB_LABEL="$(cuopt__resolve_ci_job_label)"
  export CUOPT_CI_JOB_LABEL
  CUOPT_GDB_CORE_ARTIFACT_DIR="$(cuopt__gdb_core_artifact_basename)"
  export CUOPT_GDB_CORE_ARTIFACT_DIR
  export CUOPT_COREDUMP_DIR="${base}/${CUOPT_GDB_CORE_ARTIFACT_DIR}"
  # Record startup time so coredumpctl collection can filter to this session only.
  export CUOPT_COREDUMP_SINCE
  CUOPT_COREDUMP_SINCE="$(date '+%Y-%m-%d %H:%M:%S' 2>/dev/null || echo '')"
  mkdir -p "${CUOPT_COREDUMP_DIR}"

  local pattern_target="${CUOPT_COREDUMP_DIR}/core.%e.%p.%t"

  # Raise soft limit to match hard limit when possible (some shells default to 0).
  ulimit -c unlimited 2>/dev/null || true
  ulimit -H -c unlimited 2>/dev/null || true

  # Write the coredump filter to the kernel's per-process file (env var alone has no effect).
  # 0xff = dump all memory segments (shared, private, huge, DAX — Linux 4.6+).
  local filter="${COREDUMP_FILTER:-0xff}"
  if [[ -w /proc/self/coredump_filter ]]; then
    echo "${filter}" >/proc/self/coredump_filter 2>/dev/null || true
  fi

  # Prefer writing cores as files under CUOPT_COREDUMP_DIR (often fails in unprivileged Docker).
  if [[ -w /proc/sys/kernel/core_pattern ]]; then
    echo "${pattern_target}" >/proc/sys/kernel/core_pattern 2>/dev/null || true
  fi
  if command -v sysctl >/dev/null 2>&1; then
    sysctl -q -w "kernel.core_pattern=${pattern_target}" 2>/dev/null || true
  fi

  pattern="$(cat /proc/sys/kernel/core_pattern 2>/dev/null || echo n/a)"

  # Track whether core_pattern points to our directory (file-based) or a pipe/collector.
  export CUOPT_COREDUMP_PATTERN_IS_PIPE=0
  if [[ "${pattern}" == \|* ]]; then
    CUOPT_COREDUMP_PATTERN_IS_PIPE=1
  fi

  local coredump_filter_val="n/a"
  if [[ -r /proc/self/coredump_filter ]]; then
    coredump_filter_val="$(cat /proc/self/coredump_filter 2>/dev/null || echo n/a)"
  fi

  local _log_msg="Core dumps: dir=${CUOPT_COREDUMP_DIR} ulimit=$(ulimit -c) core_pattern=${pattern} coredump_filter=${coredump_filter_val}"
  if declare -F rapids-logger &>/dev/null; then
    rapids-logger "${_log_msg}"
  else
    echo "${_log_msg}"
  fi

  if [[ "${CUOPT_COREDUMP_PATTERN_IS_PIPE}" == 1 ]]; then
    local _pipe_msg="WARNING: core_pattern pipes to a collector — cores will NOT appear as files. Fallback: coredumpctl (systemd-coredump) or /var/crash (apport) will be checked at collection time."
    if command -v coredumpctl &>/dev/null; then
      _pipe_msg+=" coredumpctl is available."
    else
      _pipe_msg+=" coredumpctl NOT found; if systemd-coredump is the handler, cores may be lost."
    fi
    if declare -F rapids-logger &>/dev/null; then
      rapids-logger "${_pipe_msg}"
    else
      echo "WARNING: ${_pipe_msg}" >&2
    fi
  fi
}

cuopt__log() {
  if declare -F rapids-logger &>/dev/null; then
    rapids-logger "$1"
  else
    echo "$1"
  fi
}

# Copy a single core file into the artifact directory with a sanitized name.
cuopt__copy_core_to_dest() {
  local f="$1" dest="$2" label="${3:-}"
  [[ -f "${f}" && -s "${f}" ]] || return 0
  local base_name
  base_name="$(basename "${f}")"
  if [[ -n "${label}" ]]; then
    base_name="${label}_${base_name}"
  fi
  base_name="${base_name//\//_}"
  local dest_path="${dest}/${base_name}"
  if [[ -e "${dest_path}" ]]; then
    dest_path="${dest}/${base_name}.${RANDOM}"
  fi
  cp -a "${f}" "${dest_path}" 2>/dev/null || true
}

# Collect cores written as files (core_pattern was file-based or we got lucky).
cuopt__collect_core_files() {
  local dest="$1"
  shift
  local search_dirs=("$@")
  local f
  for dir in "${search_dirs[@]}"; do
    [[ -d "${dir}" ]] || continue
    while IFS= read -r -d '' f; do
      [[ -f "${f}" ]] || continue
      # Skip files already in dest.
      case "${f}" in
        "${dest}/"*) continue ;;
      esac
      cuopt__copy_core_to_dest "${f}" "${dest}" ""
    done < <(
      find "${dir}" \
        \( -path '*/.git/*' -o -path '*/opt/conda/*' -o -path '*/conda_pkgs/*' -o -path "${dest}/*" \) -prune -o \
        \( -name 'core' -o -name 'core.*' \) -type f -print0 2>/dev/null
    )
  done
}

# Fallback: extract cores via coredumpctl (systemd-coredump handler).
cuopt__collect_via_coredumpctl() {
  local dest="$1"
  command -v coredumpctl &>/dev/null || return 0

  cuopt__log "Attempting coredumpctl extraction (core_pattern is piped to systemd-coredump)"

  # Build the coredumpctl list command — scope to this session if we have a start time.
  local -a list_cmd=(coredumpctl list --no-pager --no-legend)
  if [[ -n "${CUOPT_COREDUMP_SINCE:-}" ]]; then
    list_cmd+=(--since "${CUOPT_COREDUMP_SINCE}")
    cuopt__log "  Filtering coredumpctl to cores since ${CUOPT_COREDUMP_SINCE}"
  fi

  local line pid exe core_path
  # --no-legend output format: DAY DATE TIME TZ PID UID GID SIG COREFILE EXE...
  while IFS= read -r line; do
    # Skip header / empty lines.
    [[ "${line}" =~ ^[[:space:]]*[A-Z] ]] && continue
    [[ -z "${line}" ]] && continue
    # Parse PID (5th field) and EXE (last field).
    pid="$(echo "${line}" | awk '{print $5}')"
    exe="$(echo "${line}" | awk '{print $NF}')"
    [[ -n "${pid}" ]] || continue
    core_path="${dest}/coredumpctl_${pid}_$(basename "${exe:-unknown}").core"
    if [[ -e "${core_path}" ]]; then
      core_path="${core_path}.${RANDOM}"
    fi
    coredumpctl dump "${pid}" -o "${core_path}" 2>/dev/null || true
    if [[ -s "${core_path}" ]]; then
      cuopt__log "Extracted core for PID ${pid} (${exe}) → ${core_path} ($(du -h "${core_path}" | cut -f1))"
    else
      rm -f "${core_path}" 2>/dev/null || true
    fi
  done < <("${list_cmd[@]}" 2>/dev/null || true)
}

# Fallback: collect cores from apport crash reports (/var/crash).
cuopt__collect_from_apport() {
  local dest="$1"
  local crash_dir="/var/crash"
  [[ -d "${crash_dir}" ]] || return 0
  local f
  for f in "${crash_dir}"/*.crash "${crash_dir}"/core.* "${crash_dir}"/core; do
    [[ -f "${f}" && -s "${f}" ]] || continue
    cuopt__copy_core_to_dest "${f}" "${dest}" "apport"
  done
}

cuopt_collect_coredumps() {
  local ws base dest n_before n_after
  ws="${GITHUB_WORKSPACE:-${PWD}}"
  base="${RAPIDS_ARTIFACTS_DIR:-${ws}/artifacts}"
  if [[ -z "${CUOPT_GDB_CORE_ARTIFACT_DIR:-}" ]]; then
    CUOPT_GDB_CORE_ARTIFACT_DIR="$(cuopt__gdb_core_artifact_basename)"
  fi
  dest="${base}/${CUOPT_GDB_CORE_ARTIFACT_DIR}"
  mkdir -p "${dest}"

  n_before="$(find "${dest}" -type f 2>/dev/null | wc -l | tr -d '[:space:]')"

  # 1) Search for core files in workspace + common system locations.
  cuopt__collect_core_files "${dest}" \
    "${ws}" "/tmp" "/var/lib/systemd/coredump" "/var/crash"

  # 2) If core_pattern pipes to a collector, try extracting via coredumpctl / apport.
  if [[ "${CUOPT_COREDUMP_PATTERN_IS_PIPE:-0}" == 1 ]]; then
    cuopt__collect_via_coredumpctl "${dest}"
    cuopt__collect_from_apport "${dest}"
  fi

  n_after="$(find "${dest}" -type f 2>/dev/null | wc -l | tr -d '[:space:]')"
  if [[ "${n_after}" -gt "${n_before}" ]]; then
    cuopt__log "Collected $((n_after - n_before)) core file(s) into ${dest} (${n_after} total)"
    ls -lh "${dest}"/ 2>/dev/null || true
  else
    cuopt__log "WARNING: No core files found. Cores may have been discarded by the system collector."
    cuopt__log "  core_pattern=$(cat /proc/sys/kernel/core_pattern 2>/dev/null || echo n/a)"
    cuopt__log "  Searched: ${ws} /tmp /var/lib/systemd/coredump /var/crash"
    if [[ "${CUOPT_COREDUMP_PATTERN_IS_PIPE:-0}" == 1 ]]; then
      if command -v coredumpctl &>/dev/null; then
        cuopt__log "  coredumpctl list output:"
        coredumpctl list --no-pager 2>/dev/null || true
      else
        cuopt__log "  coredumpctl not available; cannot extract from systemd-coredump"
      fi
    fi
  fi
}

# Standard CI wiring for ci/test_*.sh: call once after sourcing this file.
cuopt_coredumps_ci_setup() {
  cuopt_enable_coredumps
  trap 'cuopt_collect_coredumps || true' EXIT
}
