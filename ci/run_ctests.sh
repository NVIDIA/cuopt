#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support customizing the gtests' install location
# First, try the installed location (CI/conda environments)
installed_test_location="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcuopt/"
# Fall back to the build directory (devcontainer environments)
devcontainers_test_location="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../cpp/build/latest/gtests/libcuopt/"

if [[ -d "${installed_test_location}" ]]; then
    GTEST_DIR="${installed_test_location}"
elif [[ -d "${devcontainers_test_location}" ]]; then
    GTEST_DIR="${devcontainers_test_location}"
else
    echo "Error: Test location not found. Searched:" >&2
    echo "  - ${installed_test_location}" >&2
    echo "  - ${devcontainers_test_location}" >&2
    exit 1
fi

# Run first: intentional SIGSEGV to validate core dump collection (ci/cuopt_coredumps.sh).
# Expect a non-zero exit (e.g. 139). The same binary is invoked again in the loop without
# CUOPT_TEST_COREDUMP and skips the fatal case.
if [[ -x "${GTEST_DIR}/COREDUMP_SANITY_TEST" ]]; then
  echo "Running COREDUMP_SANITY_TEST with CUOPT_TEST_COREDUMP=1 (expected fatal signal)"
  set +e
  CUOPT_TEST_COREDUMP=1 "${GTEST_DIR}/COREDUMP_SANITY_TEST" "$@"
  _coredump_ret=$?
  set -e
  if [[ "${_coredump_ret}" -eq 0 ]]; then
    echo "ERROR: COREDUMP_SANITY_TEST exited 0 with CUOPT_TEST_COREDUMP=1; expected crash" >&2
    exit 1
  fi
else
  echo "Skipping COREDUMP_SANITY_TEST (binary not found)"
fi

shopt -s nullglob
for gt in "${GTEST_DIR}"/*_TEST; do
    test_name=$(basename "${gt}")
    echo "Running gtest ${test_name}"
    if ! "${gt}" "$@"; then
      _g_rc=$?
      echo "ERROR: gtest ${test_name} failed (exit ${_g_rc}); stopping run_ctests.sh" >&2
      exit "${_g_rc}"
    fi
done
shopt -u nullglob

# Run C_API_TEST with CPU memory for local solves (excluding time limit tests)
if [ -x "${GTEST_DIR}/C_API_TEST" ]; then
  echo "Running gtest C_API_TEST with CUOPT_USE_CPU_MEM_FOR_LOCAL"
  CUOPT_USE_CPU_MEM_FOR_LOCAL=1 "${GTEST_DIR}/C_API_TEST" --gtest_filter=-c_api/TimeLimitTestFixture.* "$@"
else
  echo "Skipping C_API_TEST with CUOPT_USE_CPU_MEM_FOR_LOCAL (binary not found)"
fi
