#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Directory setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOCS_ROOT="${PROJECT_ROOT}/docs/cuopt/source"
RESULTS_DIR="${PROJECT_ROOT}/test-results"
SERVER_PID=""

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_failure() {
    echo -e "${RED}[✗]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
}

# Server management functions
check_server() {
    if curl -s http://localhost:5000/docs > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

start_server() {
    log_info "Starting cuOpt server..."
    python -m cuopt_server.cuopt_service --ip localhost --port 5000 > "${RESULTS_DIR}/cuopt-server.log" 2>&1 &
    SERVER_PID=$!

    # Wait for server to start (max 30 seconds)
    for _ in {1..30}; do
        sleep 1
        if check_server; then
            log_success "Server started (PID: ${SERVER_PID})"
            return 0
        fi
    done

    log_failure "Server failed to start after 30 seconds"
    return 1
}

stop_server() {
    if [ -n "${SERVER_PID}" ] && ps -p "${SERVER_PID}" > /dev/null 2>&1; then
        log_info "Stopping server (PID: ${SERVER_PID})..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        log_success "Server stopped"
        SERVER_PID=""
    fi
}

# Cleanup on exit
cleanup() {
    stop_server
    log_info "Test Summary:"
    log_info "  Total:   ${TOTAL_TESTS}"
    log_success "  Passed:  ${PASSED_TESTS}"
    log_failure "  Failed:  ${FAILED_TESTS}"
    log_skip "  Skipped: ${SKIPPED_TESTS}"

    if [ ${FAILED_TESTS} -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
}

trap cleanup EXIT INT TERM

# Test Python examples
test_python_examples() {
    local module=$1  # cuopt-python or cuopt-server
    log_info "Testing Python examples in ${module}..."

    local base_dir="${DOCS_ROOT}/${module}"

    # Find all Python files in examples directories
    while IFS= read -r -d '' py_file; do
        TOTAL_TESTS=$((TOTAL_TESTS + 1))

        local example_dir
        local example_name
        local relative_path
        example_dir=$(dirname "${py_file}")
        example_name=$(basename "${py_file}")
        relative_path="${py_file#${DOCS_ROOT}/}"

        log_info "Running: ${relative_path}"

        # Change to example directory
        pushd "${example_dir}" > /dev/null || return

        # Check if example requires server
        local requires_server=false
        if grep -q "cuopt_sh_client\|CuOptServiceSelfHostClient" "${example_name}"; then
            requires_server=true
        fi

        # Skip if server is required but not running
        if [ "${requires_server}" = true ] && ! check_server; then
            log_skip "${example_name} (requires cuOpt server)"
            SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
            popd > /dev/null || return
            continue
        fi

        # Run the example
        if timeout 60 python "${example_name}" > "${RESULTS_DIR}/${example_name}.log" 2>&1; then
            log_success "${example_name}"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            log_failure "${example_name}"
            echo "  Last 10 lines of output:"
            tail -n 10 "${RESULTS_DIR}/${example_name}.log" | sed 's/^/    /'
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi

        popd > /dev/null || return
    done < <(find "${base_dir}" -type f -path "*/examples/*.py" -print0 2>/dev/null)
}

# Test C examples
test_c_examples() {
    log_info "Testing C examples..."

    local c_examples_dir="${DOCS_ROOT}/cuopt-c/lp-milp/examples"

    if [ ! -d "${c_examples_dir}" ]; then
        log_warning "C examples directory not found"
        return
    fi

    pushd "${c_examples_dir}" > /dev/null || return

    # Count C source files
    local c_file_count
    c_file_count=$(find . -maxdepth 1 -name "*.c" -type f | wc -l)
    if [ ${c_file_count} -eq 0 ]; then
        log_warning "No C source files found"
        popd > /dev/null || return
        return
    fi

    log_info "Found ${c_file_count} C source files"

    # Find cuOpt paths (search in common locations, not entire filesystem)
    log_info "Searching for cuOpt libraries..."
    local include_path=""
    local lib_path=""

    for search_dir in "${HOME}" "${CONDA_PREFIX}" "/usr" "/opt"; do
        if [ -z "${include_path}" ] && [ -d "${search_dir}" ]; then
            include_path=$(find "${search_dir}" -name "cuopt_c.h" -path "*/linear_programming/*" -printf "%h\n" 2>/dev/null | sed 's/\/linear_programming//' | head -1)
        fi
        if [ -z "${lib_path}" ] && [ -d "${search_dir}" ]; then
            lib_path=$(find "${search_dir}" -name "libcuopt.so" 2>/dev/null | head -1 | xargs dirname)
        fi
        # Break early if both found
        if [ -n "${include_path}" ] && [ -n "${lib_path}" ]; then
            break
        fi
    done

    if [ -z "${include_path}" ] || [ -z "${lib_path}" ]; then
        log_warning "Could not find cuOpt headers or libraries - C examples may fail to build"
    else
        log_info "Found: INCLUDE_PATH=${include_path}"
        log_info "Found: LIBCUOPT_LIBRARY_PATH=${lib_path}"
    fi

    # Clean and build
    if [ -f "Makefile" ]; then
        log_info "Building C examples..."
        local make_vars=""
        [ -n "${include_path}" ] && make_vars="${make_vars} INCLUDE_PATH=${include_path}"
        [ -n "${lib_path}" ] && make_vars="${make_vars} LIBCUOPT_LIBRARY_PATH=${lib_path}"

        if make clean > "${RESULTS_DIR}/c-clean.log" 2>&1 && \
           make ${make_vars} all > "${RESULTS_DIR}/c-build.log" 2>&1; then
            log_success "C examples built successfully"
        else
            log_failure "Failed to build C examples"
            tail -n 20 "${RESULTS_DIR}/c-build.log" | sed 's/^/    /'
            popd > /dev/null || return
            return
        fi
    else
        log_warning "No Makefile found in C examples directory"
        popd > /dev/null || return
        return
    fi

    # Run each compiled example
    for c_file in *.c; do
        local executable="${c_file%.c}"

        if [ ! -f "${executable}" ]; then
            log_warning "Executable not found for ${c_file}"
            continue
        fi

        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        log_info "Running: ${executable}"

        # Check if it needs an MPS file
        if grep -q "argc.*2\|Usage.*mps" "${c_file}"; then
            # Needs MPS file argument
            local mps_file=""
            if echo "${executable}" | grep -q "milp"; then
                mps_file="mip_sample.mps"
            else
                mps_file="sample.mps"
            fi

            if [ ! -f "${mps_file}" ]; then
                log_failure "${executable} (MPS file ${mps_file} not found)"
                FAILED_TESTS=$((FAILED_TESTS + 1))
                continue
            fi

            if timeout 60 ./"${executable}" "${mps_file}" > "${RESULTS_DIR}/${executable}.log" 2>&1; then
                log_success "${executable}"
                PASSED_TESTS=$((PASSED_TESTS + 1))
            else
                log_failure "${executable}"
                echo "  Last 10 lines of output:"
                tail -n 10 "${RESULTS_DIR}/${executable}.log" | sed 's/^/    /'
                FAILED_TESTS=$((FAILED_TESTS + 1))
            fi
        else
            # No MPS file needed
            if timeout 60 ./"${executable}" > "${RESULTS_DIR}/${executable}.log" 2>&1; then
                log_success "${executable}"
                PASSED_TESTS=$((PASSED_TESTS + 1))
            else
                log_failure "${executable}"
                echo "  Last 10 lines of output:"
                tail -n 10 "${RESULTS_DIR}/${executable}.log" | sed 's/^/    /'
                FAILED_TESTS=$((FAILED_TESTS + 1))
            fi
        fi
    done

    popd > /dev/null || return
}

# Test CLI examples
test_cli_examples() {
    log_info "Testing CLI examples..."

    local cli_examples_dir="${DOCS_ROOT}/cuopt-cli/examples"

    if [ ! -d "${cli_examples_dir}" ]; then
        log_warning "CLI examples directory not found"
        return
    fi

    # Check if server is running (required for CLI)
    if ! check_server; then
        log_warning "CLI examples require cuOpt server - skipping all CLI tests"
        return
    fi

    # Find all shell scripts in examples directories
    while IFS= read -r -d '' sh_file; do
        TOTAL_TESTS=$((TOTAL_TESTS + 1))

        local example_dir
        local example_name
        local relative_path
        example_dir=$(dirname "${sh_file}")
        example_name=$(basename "${sh_file}")
        relative_path="${sh_file#${DOCS_ROOT}/}"

        log_info "Running: ${relative_path}"

        # Change to example directory
        pushd "${example_dir}" > /dev/null || return

        # Make executable
        chmod +x "${example_name}"

        # Run the example
        if timeout 60 bash "${example_name}" > "${RESULTS_DIR}/${example_name}.log" 2>&1; then
            log_success "${example_name}"
            PASSED_TESTS=$((PASSED_TESTS + 1))
        else
            log_failure "${example_name}"
            echo "  Last 10 lines of output:"
            tail -n 10 "${RESULTS_DIR}/${example_name}.log" | sed 's/^/    /'
            FAILED_TESTS=$((FAILED_TESTS + 1))
        fi

        popd > /dev/null || return
    done < <(find "${cli_examples_dir}" -type f -path "*/examples/*.sh" -print0 2>/dev/null)
}

# Main execution
main() {
    log_info "============================================"
    log_info "  cuOpt Examples Test Suite"
    log_info "============================================"
    echo ""

    # Check if we should start server
    local server_needed=false
    if [ -d "${DOCS_ROOT}/cuopt-server/examples" ] || [ -d "${DOCS_ROOT}/cuopt-cli/examples" ]; then
        server_needed=true
    fi

    # Start server if needed
    if [ "${server_needed}" = true ]; then
        if ! check_server; then
            log_info "Server not running, attempting to start..."
            if start_server; then
                log_success "Server is ready"
            else
                log_warning "Could not start server - server-dependent tests will be skipped"
            fi
        else
            log_success "Server is already running"
        fi
    fi

    echo ""

    # Run tests for each module
    test_python_examples "cuopt-python"
    echo ""

    test_python_examples "cuopt-server"
    echo ""

    test_c_examples
    echo ""

    test_cli_examples
    echo ""

    log_info "============================================"
    log_info "All tests completed!"
    log_info "Logs saved to: ${RESULTS_DIR}"
    log_info "============================================"
}

# Run main
main "$@"
