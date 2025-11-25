#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Bash strict mode
set -o pipefail  # Catch errors in pipes
set -o nounset   # Exit on undefined variables
set -o errtrace  # Inherit ERR trap in functions
set -e           # Exit on error

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

# C library paths (set by find_cuopt_libraries)
include_path=""
lib_path=""

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

# Check if example file should be skipped based on TEST_SKIP marker
#
# To skip an example from automated testing, add this comment line near the top
# of the example file (works for .py, .sh, .c files):
#   # TEST_SKIP: <reason for skipping>
#
# Example:
#   # TEST_SKIP: This is a template file requiring manual configuration
#
# Returns: 0 if should skip, 1 if should run
# Sets global variable: SKIP_REASON with the skip reason
should_skip_example() {
    local file=$1

    if grep -q "^#.*TEST_SKIP:" "${file}"; then
        SKIP_REASON=$(grep "^#.*TEST_SKIP:" "${file}" | head -1 | sed 's/^#.*TEST_SKIP: *//')
        return 0
    fi

    return 1
}

# Run a shell script with proper error detection
#
# This wrapper ensures that shell scripts fail properly on errors by running
# them in bash strict mode. This catches:
#   - Command failures (set -e)
#   - Pipeline failures (set -o pipefail)
#   - Undefined variable usage (set -u)
#
# Example authors: Your shell scripts will be run with these flags, so ensure:
#   1. Commands that may fail are handled explicitly (|| true, or if checks)
#   2. Variables are defined before use
#   3. Pipelines are constructed correctly
#
# To allow a command to fail without stopping the script, use:
#   command_that_may_fail || true
#   # OR
#   if command_that_may_fail; then
#       echo "Success"
#   else
#       echo "Failed but continuing"
#   fi
run_shell_script() {
    local script=$1
    local log_file=$2
    local timeout_seconds=${3:-60}

    # Run the script with bash in strict mode to catch errors
    # We use a subshell to avoid affecting the parent shell
    timeout "${timeout_seconds}" bash -c "
        set -e           # Exit on error
        set -o pipefail  # Catch errors in pipes
        set -u           # Exit on undefined variables
        source '${script}'
    " > "${log_file}" 2>&1

    return $?
}

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

        # Skip files with TEST_SKIP marker
        if should_skip_example "${example_name}"; then
            log_skip "${example_name} (${SKIP_REASON})"
            SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
            popd > /dev/null || return
            continue
        fi

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

    # Also test shell scripts in cuopt-python and cuopt-server modules
    if [ "${module}" = "cuopt-python" ] || [ "${module}" = "cuopt-server" ]; then
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

            # Skip files with TEST_SKIP marker
            if should_skip_example "${example_name}"; then
                log_skip "${example_name} (${SKIP_REASON})"
                SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
                popd > /dev/null || return
                continue
            fi

            # Check if example requires server
            local requires_server=false
            if grep -q "cuopt_sh\|localhost.*5000" "${example_name}"; then
                requires_server=true
            fi

            # Skip if server is required but not running
            if [ "${requires_server}" = true ] && ! check_server; then
                log_skip "${example_name} (requires cuOpt server)"
                SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
                popd > /dev/null || return
                continue
            fi

            # Run the shell script with error detection
            if run_shell_script "${example_name}" "${RESULTS_DIR}/${example_name}.log" 60; then
                log_success "${example_name}"
                PASSED_TESTS=$((PASSED_TESTS + 1))
            else
                log_failure "${example_name}"
                echo "  Last 10 lines of output:"
                tail -n 10 "${RESULTS_DIR}/${example_name}.log" | sed 's/^/    /'
                FAILED_TESTS=$((FAILED_TESTS + 1))
            fi

            popd > /dev/null || return
        done < <(find "${base_dir}" -type f -path "*/examples/*.sh" -print0 2>/dev/null)
    fi
}

# Test C examples
test_c_examples() {
    log_info "Testing C examples..."

    local cuopt_c_dir="${DOCS_ROOT}/cuopt-c"

    if [ ! -d "${cuopt_c_dir}" ]; then
        log_warning "cuopt-c directory not found"
        return
    fi

    # Find all examples directories under cuopt-c
    local examples_dirs=()
    while IFS= read -r -d '' dir; do
        examples_dirs+=("${dir}")
    done < <(find "${cuopt_c_dir}" -type d -name "examples" -print0 2>/dev/null)

    if [ ${#examples_dirs[@]} -eq 0 ]; then
        log_warning "No C examples directories found under cuopt-c"
        return
    fi

    log_info "Found ${#examples_dirs[@]} C examples director(y/ies)"

    # Find cuOpt libraries once for all C examples
    if ! find_cuopt_libraries; then
        # Count all C files across all directories as failed
        local total_c_files=0
        for dir in "${examples_dirs[@]}"; do
            local count
            count=$(find "${dir}" -maxdepth 1 -name "*.c" -type f 2>/dev/null | wc -l)
            total_c_files=$((total_c_files + count))
        done
        log_info "  Marking all ${total_c_files} C examples as failed"
        FAILED_TESTS=$((FAILED_TESTS + total_c_files))
        TOTAL_TESTS=$((TOTAL_TESTS + total_c_files))
        return
    fi

    # Process each examples directory
    for c_examples_dir in "${examples_dirs[@]}"; do
        local relative_path="${c_examples_dir#${DOCS_ROOT}/}"
        log_info "Processing: ${relative_path}"

        pushd "${c_examples_dir}" > /dev/null || return

        # Count C source files
        local c_file_count
        c_file_count=$(find . -maxdepth 1 -name "*.c" -type f | wc -l)
        if [ ${c_file_count} -eq 0 ]; then
            log_warning "  No C source files found in ${relative_path}"
            popd > /dev/null || return
            continue
        fi

        log_info "  Found ${c_file_count} C source files"

        # Clean and build
        if [ -f "Makefile" ]; then
            log_info "  Building C examples..."
            local make_vars=""
            [ -n "${include_path}" ] && make_vars="${make_vars} INCLUDE_PATH=${include_path}"
            [ -n "${lib_path}" ] && make_vars="${make_vars} LIBCUOPT_LIBRARY_PATH=${lib_path}"

            if make clean > "${RESULTS_DIR}/c-clean-${relative_path//\//_}.log" 2>&1 && \
               make ${make_vars} all > "${RESULTS_DIR}/c-build-${relative_path//\//_}.log" 2>&1; then
                log_success "  C examples built successfully"
            else
                log_failure "  Failed to build C examples"
                tail -n 20 "${RESULTS_DIR}/c-build-${relative_path//\//_}.log" | sed 's/^/    /'
                log_info "    Marking all C examples in this directory as failed"
                # Count all C files as failed
                FAILED_TESTS=$((FAILED_TESTS + c_file_count))
                TOTAL_TESTS=$((TOTAL_TESTS + c_file_count))
                popd > /dev/null || return
                continue
            fi
        else
            log_warning "  No Makefile found in ${relative_path}"
            log_info "    Marking all C examples in this directory as failed"
            # Count all C files as failed
            FAILED_TESTS=$((FAILED_TESTS + c_file_count))
            TOTAL_TESTS=$((TOTAL_TESTS + c_file_count))
            popd > /dev/null || return
            continue
        fi

        # Run each compiled example
        for c_file in *.c; do
            local executable="${c_file%.c}"

            if [ ! -f "${executable}" ]; then
                log_warning "  Executable not found for ${c_file}"
                continue
            fi

            TOTAL_TESTS=$((TOTAL_TESTS + 1))

            # Skip files with TEST_SKIP marker
            if should_skip_example "${c_file}"; then
                log_skip "  ${c_file} (${SKIP_REASON})"
                SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
                continue
            fi

            log_info "  Running: ${executable}"

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
                    log_failure "    ${executable} (MPS file ${mps_file} not found)"
                    FAILED_TESTS=$((FAILED_TESTS + 1))
                    continue
                fi

                if timeout 60 ./"${executable}" "${mps_file}" > "${RESULTS_DIR}/${executable}.log" 2>&1; then
                    log_success "    ${executable}"
                    PASSED_TESTS=$((PASSED_TESTS + 1))
                else
                    log_failure "    ${executable}"
                    echo "      Last 10 lines of output:"
                    tail -n 10 "${RESULTS_DIR}/${executable}.log" | sed 's/^/        /'
                    FAILED_TESTS=$((FAILED_TESTS + 1))
                fi
            else
                # No MPS file needed
                if timeout 60 ./"${executable}" > "${RESULTS_DIR}/${executable}.log" 2>&1; then
                    log_success "    ${executable}"
                    PASSED_TESTS=$((PASSED_TESTS + 1))
                else
                    log_failure "    ${executable}"
                    echo "      Last 10 lines of output:"
                    tail -n 10 "${RESULTS_DIR}/${executable}.log" | sed 's/^/        /'
                    FAILED_TESTS=$((FAILED_TESTS + 1))
                fi
            fi
        done

        popd > /dev/null || return
    done

    # Note: Library search is done once per invocation outside the loop above
}

# Find cuOpt libraries (shared function for C examples)
find_cuopt_libraries() {
    # Find cuOpt paths (search in common locations, not entire filesystem)
    log_info "Searching for cuOpt libraries..."

    # Reset global variables
    include_path=""
    lib_path=""

    # Get Python site-packages directory
    local site_packages=""
    if command -v python > /dev/null 2>&1; then
        site_packages=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
    fi

    # Search in common locations including Python site-packages
    local search_dirs=("${HOME}" "${CONDA_PREFIX}" "/usr" "/opt")
    if [ -n "${site_packages}" ] && [ -d "${site_packages}" ]; then
        search_dirs+=("${site_packages}")
    fi

    for search_dir in "${search_dirs[@]}"; do
        if [ -z "${search_dir}" ] || [ ! -d "${search_dir}" ]; then
            continue
        fi

        if [ -z "${include_path}" ]; then
            # Search for cuopt_c.h
            local found_header
            found_header=$(find "${search_dir}" -name "cuopt_c.h" -path "*/linear_programming/*" 2>/dev/null | head -1)

            if [ -n "${found_header}" ]; then
                # Check if this is a Python package installation (contains libcuopt/include)
                if echo "${found_header}" | grep -q "/libcuopt/include/"; then
                    # Python package structure: /path/to/libcuopt/include/cuopt/linear_programming/cuopt_c.h
                    # Extract the include directory by going up 3 directories from the header file
                    include_path=$(dirname "$(dirname "$(dirname "${found_header}")")")
                else
                    # Standard installation: /path/to/include/cuopt/linear_programming/cuopt_c.h
                    # Extract the include directory by going up 2 directories
                    include_path=$(dirname "$(dirname "${found_header}")")
                fi
            fi
        fi

        if [ -z "${lib_path}" ]; then
            # Search for libcuopt.so in both lib and lib64 directories
            local found_lib
            found_lib=$(find "${search_dir}" -name "libcuopt.so" \( -path "*/lib/*" -o -path "*/lib64/*" \) 2>/dev/null | head -1)
            if [ -n "${found_lib}" ]; then
                lib_path=$(dirname "${found_lib}")
            fi
        fi

        # Break early if both found
        if [ -n "${include_path}" ] && [ -n "${lib_path}" ]; then
            break
        fi
    done

    if [ -z "${include_path}" ] || [ -z "${lib_path}" ]; then
        log_failure "Could not find cuOpt headers or libraries"
        if [ -z "${include_path}" ]; then
            log_failure "  Missing: INCLUDE_PATH (searched for cuopt_c.h)"
        fi
        if [ -z "${lib_path}" ]; then
            log_failure "  Missing: LIBCUOPT_LIBRARY_PATH (searched for libcuopt.so)"
        fi
        return 1
    else
        log_info "Found: INCLUDE_PATH=${include_path}"
        log_info "Found: LIBCUOPT_LIBRARY_PATH=${lib_path}"
        return 0
    fi
}

# Test CLI examples
test_cli_examples() {
    log_info "Testing CLI examples..."

    local cli_examples_dir="${DOCS_ROOT}/cuopt-cli/examples"

    if [ ! -d "${cli_examples_dir}" ]; then
        log_warning "CLI examples directory not found"
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

        # Skip files with TEST_SKIP marker
        if should_skip_example "${example_name}"; then
            log_skip "${example_name} (${SKIP_REASON})"
            SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
            popd > /dev/null || return
            continue
        fi

        # Check if example uses cuopt_sh (requires server) or cuopt_cli (standalone)
        local requires_server=false
        if grep -q "cuopt_sh" "${example_name}"; then
            requires_server=true
        fi

        # Skip if server is required but not running
        if [ "${requires_server}" = true ] && ! check_server; then
            log_skip "${example_name} (requires cuOpt server)"
            SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
            popd > /dev/null || return
            continue
        fi

        # Run the example with error detection
        if run_shell_script "${example_name}" "${RESULTS_DIR}/${example_name}.log" 60; then
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
