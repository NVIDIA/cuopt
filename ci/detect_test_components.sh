#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Helper script to detect which test components should run based on changed files.
# Source this script to set the CUOPT_TEST_COMPONENTS variable.
#
# Usage: source ci/detect_test_components.sh
#
# Sets CUOPT_TEST_COMPONENTS to a comma-separated list of components (routing,lp,mip)
# or "all" if broad/shared changes are detected or if not in a pull-request build.

set -euo pipefail

detect_test_components() {
    # If CUOPT_TEST_COMPONENTS is already set externally, respect it
    if [[ -n "${CUOPT_TEST_COMPONENTS:-}" ]]; then
        export CUOPT_TEST_COMPONENTS
        return
    fi

    # Only apply selective testing for pull-request builds
    if [[ "${RAPIDS_BUILD_TYPE:-}" != "pull-request" ]]; then
        export CUOPT_TEST_COMPONENTS="all"
        return
    fi

    # In RAPIDS CI pull-request builds, the branch is a merge commit so
    # HEAD~1..HEAD gives the full PR diff.  If the parent isn't reachable
    # (e.g. shallow clone or non-merge workflow), fall back to running all.
    local changed_files
    if ! git rev-parse --verify HEAD~1 &>/dev/null; then
        export CUOPT_TEST_COMPONENTS="all"
        return
    fi
    if ! changed_files=$(git diff --name-only HEAD~1..HEAD 2>/dev/null); then
        # If git diff fails, run all tests to be safe
        export CUOPT_TEST_COMPONENTS="all"
        return
    fi

    if [[ -z "${changed_files}" ]]; then
        export CUOPT_TEST_COMPONENTS="all"
        return
    fi

    local components=""
    local run_all=false

    # Check for shared/infrastructure changes that should trigger all tests
    while IFS= read -r file; do
        case "${file}" in
            cpp/include/*|cpp/cmake/*|cpp/CMakeLists.txt|cpp/src/utilities/*)
                run_all=true
                break
                ;;
            # Changes to test infrastructure
            cpp/tests/CMakeLists.txt|cpp/tests/utilities/*)
                run_all=true
                break
                ;;
            # Changes to CI infrastructure
            ci/test_cpp.sh|ci/test_python.sh|ci/test_wheel_cuopt.sh|ci/run_ctests.sh|ci/run_cuopt_pytests.sh|ci/detect_test_components.sh)
                run_all=true
                break
                ;;
            # Changes to conda/build config
            conda/*|dependencies.yaml)
                run_all=true
                break
                ;;
        esac
    done <<< "${changed_files}"

    if ${run_all}; then
        export CUOPT_TEST_COMPONENTS="all"
        return
    fi

    # Detect individual components
    local has_routing=false
    local has_lp=false
    local has_mip=false

    while IFS= read -r file; do
        case "${file}" in
            # Routing component
            cpp/src/routing/*|cpp/src/distance/*|cpp/tests/routing/*|cpp/tests/distance_engine/*|cpp/tests/examples/routing/*)
                has_routing=true
                ;;
            python/cuopt/cuopt/routing/*|python/cuopt/cuopt/tests/routing/*)
                has_routing=true
                ;;
            regression/routing*)
                has_routing=true
                ;;
            # LP component
            cpp/src/dual_simplex/*|cpp/src/barrier/*|cpp/src/pdlp/*|cpp/src/math_optimization/*)
                has_lp=true
                ;;
            cpp/tests/linear_programming/*|cpp/tests/dual_simplex/*|cpp/tests/qp/*)
                has_lp=true
                ;;
            python/cuopt/cuopt/tests/linear_programming/*|python/cuopt/cuopt/tests/quadratic_programming/*)
                has_lp=true
                ;;
            # LP regression
            regression/lp*)
                has_lp=true
                ;;
            # MIP component
            cpp/src/branch_and_bound/*|cpp/src/cuts/*|cpp/src/mip_heuristics/*)
                has_mip=true
                ;;
            cpp/tests/mip/*)
                has_mip=true
                ;;
            regression/mip*)
                has_mip=true
                ;;
            # Python source changes that could affect any component
            python/cuopt/cuopt/*.py|python/cuopt/cuopt/distance_engine/*|python/cuopt/cuopt/utils/*)
                has_routing=true
                has_lp=true
                ;;
            python/libcuopt/*)
                has_routing=true
                has_lp=true
                has_mip=true
                ;;
        esac
    done <<< "${changed_files}"

    # Build the components string
    if ${has_routing}; then
        components="${components:+${components},}routing"
    fi
    if ${has_lp}; then
        components="${components:+${components},}lp"
    fi
    if ${has_mip}; then
        components="${components:+${components},}mip"
    fi

    # If no specific component was detected, default to all
    if [[ -z "${components}" ]]; then
        components="all"
    fi

    export CUOPT_TEST_COMPONENTS="${components}"
}

detect_test_components
echo "CUOPT_TEST_COMPONENTS=${CUOPT_TEST_COMPONENTS}"
