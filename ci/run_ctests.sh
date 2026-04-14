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

# ---------------------------------------------------------------------------
# Label-based filtering
#
# Set CUOPT_TEST_LABELS to a comma-separated list of labels to run only tests
# matching ANY of those labels. For example:
#   CUOPT_TEST_LABELS=routing          — run only routing tests
#   CUOPT_TEST_LABELS=solver           — run all LP/MIP/QP tests
#   CUOPT_TEST_LABELS=tier1            — run only fast unit tests
#   CUOPT_TEST_LABELS=routing,solver   — run routing + solver tests
#
# When unset or empty, all tests run (backward-compatible default).
# ---------------------------------------------------------------------------
LABEL_MANIFEST="${GTEST_DIR}/test_labels.txt"

# should_run_test <test_name>
# Returns 0 (true) if the test should run given the current label filter.
should_run_test() {
    local test_name="$1"

    # No filter requested — run everything.
    if [[ -z "${CUOPT_TEST_LABELS:-}" ]]; then
        return 0
    fi

    # No manifest available — cannot filter, run everything with a warning.
    if [[ ! -f "${LABEL_MANIFEST}" ]]; then
        if [[ -z "${_label_warn_printed:-}" ]]; then
            echo "Warning: CUOPT_TEST_LABELS set but ${LABEL_MANIFEST} not found; running all tests." >&2
            _label_warn_printed=1
        fi
        return 0
    fi

    # Look up the test's labels in the manifest (format: TEST_NAME:label1,label2).
    local entry
    entry=$(grep "^${test_name}:" "${LABEL_MANIFEST}" 2>/dev/null || true)
    if [[ -z "${entry}" ]]; then
        # Test not in manifest — run it to be safe.
        return 0
    fi

    local test_labels="${entry#*:}"

    # Check if any requested label matches any of the test's labels.
    IFS=',' read -ra requested <<< "${CUOPT_TEST_LABELS}"
    IFS=',' read -ra actual <<< "${test_labels}"
    for req in "${requested[@]}"; do
        for act in "${actual[@]}"; do
            if [[ "${req}" == "${act}" ]]; then
                return 0
            fi
        done
    done

    # No label match — skip.
    return 1
}

for gt in "${GTEST_DIR}"/*_TEST; do
    test_name=$(basename "${gt}")
    if should_run_test "${test_name}"; then
        echo "Running gtest ${test_name}"
        "${gt}" "$@"
    else
        echo "Skipping gtest ${test_name} (labels do not match CUOPT_TEST_LABELS=${CUOPT_TEST_LABELS})"
    fi
done

# Run C_API_TEST with CPU memory for local solves (excluding time limit tests)
if [ -x "${GTEST_DIR}/C_API_TEST" ] && should_run_test "C_API_TEST"; then
  echo "Running gtest C_API_TEST with CUOPT_USE_CPU_MEM_FOR_LOCAL"
  CUOPT_USE_CPU_MEM_FOR_LOCAL=1 "${GTEST_DIR}/C_API_TEST" --gtest_filter=-c_api/TimeLimitTestFixture.* "$@"
else
  echo "Skipping C_API_TEST with CUOPT_USE_CPU_MEM_FOR_LOCAL (binary not found or filtered out)"
fi
