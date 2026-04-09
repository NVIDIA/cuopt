#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# shellcheck source=ci/utils/cuopt_coredumps.sh
source "$(dirname "${BASH_SOURCE[0]}")/utils/cuopt_coredumps.sh"
cuopt_coredumps_ci_setup

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

CPP_CHANNEL=$(rapids-download-conda-from-github cpp)

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --prepend-channel "${CPP_CHANNEL}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test --channel "${CPP_CHANNEL}"

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Download datasets"
./datasets/linear_programming/download_pdlp_test_dataset.sh
./datasets/mip/download_miplib_test_dataset.sh

RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
export RAPIDS_DATASET_ROOT_DIR
pushd "${RAPIDS_DATASET_ROOT_DIR}"
./get_test_data.sh
popd

EXITCODE=0
trap "EXITCODE=1" ERR
set +e
# Run gtests from libcuopt-tests package
export GTEST_OUTPUT=xml:${RAPIDS_TESTS_DIR}/

rapids-logger "Run gtests"
timeout 40m ./ci/run_ctests.sh

# Optional core-dump path check: no compiled binary — child bash sends itself SIGSEGV.
# Child exits 139; || true keeps this script running so the EXIT trap can collect cores.
# For normal CI, leave unset and set CUOPT_CI_COREDUMP_PROBE=1 only when probing artifacts.
CUOPT_CI_COREDUMP_PROBE=1
if [[ "${CUOPT_CI_COREDUMP_PROBE:-}" == 1 ]]; then
  rapids-logger "CUOPT_CI_COREDUMP_PROBE: child bash SIGSEGV (core dump artifact check)"
  # Count core files before the probe.
  _probe_n_before="$(find "${CUOPT_COREDUMP_DIR:-/dev/null}" -type f 2>/dev/null | wc -l | tr -d '[:space:]')"
  bash -c 'kill -SEGV $$' || true
  # Brief pause so the kernel can finish writing the core.
  sleep 1
  # Eagerly collect now so we can verify the probe worked.
  cuopt_collect_coredumps || true
  _probe_n_after="$(find "${CUOPT_COREDUMP_DIR:-/dev/null}" -type f 2>/dev/null | wc -l | tr -d '[:space:]')"
  if [[ "${_probe_n_after}" -gt "${_probe_n_before}" ]]; then
    rapids-logger "COREDUMP_PROBE: SUCCESS — $((_probe_n_after - _probe_n_before)) core file(s) collected"
  else
    rapids-logger "COREDUMP_PROBE: FAILED — no core file collected for SIGSEGV probe"
    rapids-logger "  core_pattern=$(cat /proc/sys/kernel/core_pattern 2>/dev/null || echo n/a)"
    rapids-logger "  ulimit -c=$(ulimit -c)"
    rapids-logger "  CUOPT_COREDUMP_DIR=${CUOPT_COREDUMP_DIR:-unset}"
    rapids-logger "  Hint: core_pattern may require a privileged container or --cap-add=SYS_PTRACE"
  fi
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
