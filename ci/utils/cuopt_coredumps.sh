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
  mkdir -p "${CUOPT_COREDUMP_DIR}"

  local pattern_target="${CUOPT_COREDUMP_DIR}/core.%e.%p.%t"

  # Raise soft limit to match hard limit when possible (some shells default to 0).
  ulimit -c unlimited 2>/dev/null || true
  ulimit -H -c unlimited 2>/dev/null || true

  # When unset, ask the kernel for broad core dump contents (Linux 4.6+; ignored elsewhere).
  if [[ -z "${COREDUMP_FILTER:-}" ]]; then
    export COREDUMP_FILTER=0xff
  fi

  # Prefer writing cores as files under CUOPT_COREDUMP_DIR (often fails in unprivileged Docker).
  if [[ -w /proc/sys/kernel/core_pattern ]]; then
    echo "${pattern_target}" >/proc/sys/kernel/core_pattern 2>/dev/null || true
  fi
  if command -v sysctl >/dev/null 2>&1; then
    sysctl -q -w "kernel.core_pattern=${pattern_target}" 2>/dev/null || true
  fi

  pattern="$(cat /proc/sys/kernel/core_pattern 2>/dev/null || echo n/a)"
  if declare -F rapids-logger &>/dev/null; then
    rapids-logger "Core dumps: dir=${CUOPT_COREDUMP_DIR} ulimit -c=$(ulimit -c) core_pattern=${pattern}"
    if [[ "${pattern}" == \|* ]]; then
      rapids-logger "WARNING: core_pattern pipes to a collector (e.g. apport); cores may not appear as files under ${CUOPT_COREDUMP_DIR}. Use a writable core_pattern or a privileged runner if needed."
    fi
  else
    echo "Core dumps: dir=${CUOPT_COREDUMP_DIR} ulimit -c=$(ulimit -c) core_pattern=${pattern}"
    if [[ "${pattern}" == \|* ]]; then
      echo "WARNING: core_pattern pipes to a collector; files may not land in ${CUOPT_COREDUMP_DIR}" >&2
    fi
  fi
}

cuopt_collect_coredumps() {
  local ws base dest n_before n_after f rel dest_name dest_path
  ws="${GITHUB_WORKSPACE:-${PWD}}"
  base="${RAPIDS_ARTIFACTS_DIR:-${ws}/artifacts}"
  if [[ -z "${CUOPT_GDB_CORE_ARTIFACT_DIR:-}" ]]; then
    CUOPT_GDB_CORE_ARTIFACT_DIR="$(cuopt__gdb_core_artifact_basename)"
  fi
  dest="${base}/${CUOPT_GDB_CORE_ARTIFACT_DIR}"
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
      \( -path '*/.git/*' -o -path '*/opt/conda/*' -o -path '*/conda_pkgs/*' -o -path "${dest}/*" \) -prune -o \
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

# Standard CI wiring for ci/test_*.sh: call once after sourcing this file.
cuopt_coredumps_ci_setup() {
  cuopt_enable_coredumps
  trap 'cuopt_collect_coredumps || true' EXIT
}
