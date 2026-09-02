#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuopt cuopt --cuda "$RAPIDS_CUDA_VERSION")")
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_python cuopt cuopt --stable --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
export RAPIDS_TESTS_DIR
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-print-env

RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
export RAPIDS_DATASET_ROOT_DIR

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
FAILED_STEPS=()
trap "EXITCODE=1" ERR
set +e

# shellcheck source=ci/utils/crash_helpers.sh
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/utils/crash_helpers.sh"

rapids-logger "Test cuopt_cli"
run_step_with_timeout "cuopt_cli" 10m "" \
  bash ./python/libcuopt/libcuopt/tests/test_cli.sh

rapids-logger "pytest cuopt"
run_step_with_timeout "pytest cuopt" 45m "${RAPIDS_TESTS_DIR}/junit-cuopt.xml" \
  ./ci/run_cuopt_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuopt.xml" \
  --cov-config=.coveragerc \
  --cov=cuopt \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuopt-coverage.xml" \
  --cov-report=term \
  --ignore=raft

rapids-logger "pytest cuopt-server"
run_step_with_timeout "pytest cuopt-server" 20m "${RAPIDS_TESTS_DIR}/junit-cuopt-server.xml" \
  ./ci/run_cuopt_server_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuopt-server.xml" \
  --cov-config=.coveragerc \
  --cov=cuopt_server \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuopt-server-coverage.xml" \
  --cov-report=term

rapids-logger "Test skills/ assets (Python, C, CLI)"
run_step_with_timeout "skills assets" 10m "" ./ci/test_skills_assets.sh

rapids-logger "Generate nightly test report"
source "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/utils/nightly_report_helper.sh"
generate_nightly_report "python" --with-python-version

if [ "${#FAILED_STEPS[@]}" -gt 0 ]; then
    EXITCODE=1
    echo ""
    echo "==================== FAILED TEST STEPS (${#FAILED_STEPS[@]}) ===================="
    for s in "${FAILED_STEPS[@]}"; do echo "  - ${s}"; done
    echo "================================================================"
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
