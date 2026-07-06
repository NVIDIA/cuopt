#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run cuOpt multi-GPU (NCCL) C++ gtests on a >=2-GPU runner.
#
# Why a dedicated script/runner:
#   NCCL communication cannot be validated on a single GPU. Per
#   https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html
#   multiple communicators/ranks on one device are unsupported, and using the
#   same CUDA device as multiple ranks of one communicator may hang. Algorithmic
#   validity is covered by the single-GPU C++ tests (ci/test_cpp.sh); this script
#   only exercises the cross-GPU communication paths (e.g. distributed PDLP).
#
# Convention:
#   Multi-GPU gtest binaries are named with a "_MG_TEST" suffix so this script
#   can select them. They must GTEST_SKIP when fewer than two GPUs are visible,
#   so the single-GPU suite (ci/run_ctests.sh) runs them harmlessly. Until such
#   binaries exist in the libcuopt test package, this script exits 0 with a
#   notice so it can be wired into CI ahead of the tests landing.
#   (CI selection moves to a `multigpu` CTest label once tests run via ctest.)

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

# Bootstrap the C++ test conda environment from the libcuopt build artifact,
# mirroring ci/test_cpp.sh. RAPIDS_CUDA_VERSION is inferred from the ci-conda
# container image (its tag bakes the value into the environment), so no CUDA
# version needs to be pinned by the workflow. If it is unset we skip the
# bootstrap and fall back to any gtests already present in the container/build
# tree — keeping the job non-breaking until the multi-GPU tests land.
if [ -n "${RAPIDS_CUDA_VERSION:-}" ]; then
  rapids-logger "Using CUDA ${RAPIDS_CUDA_VERSION} (inferred from container image)"
  rapids-logger "Configuring conda strict channel priority"
  conda config --set channel_priority strict

  CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuopt cuopt --cuda "$RAPIDS_CUDA_VERSION")")

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
else
  rapids-logger "RAPIDS_CUDA_VERSION unset; skipping conda test-env bootstrap."
fi

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"
export RAPIDS_TESTS_DIR

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

# Multi-GPU tests are meaningless on a single device — fail loudly rather than
# passing a run that never exercised NCCL.
GPU_COUNT=$(nvidia-smi -L | wc -l)
rapids-logger "Detected ${GPU_COUNT} GPU(s)"
if [ "${GPU_COUNT}" -lt 2 ]; then
  echo "::error::Multi-GPU tests require at least 2 GPUs, found ${GPU_COUNT}." >&2
  echo "::error::This job must run on a 2-GPU runner (e.g. linux-amd64-gpu-rtxpro6000-latest-2)." >&2
  exit 1
fi

# Locate the installed gtest binaries (same search order as ci/run_ctests.sh).
installed_test_location="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcuopt/"
devcontainers_test_location="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../cpp/build/latest/gtests/libcuopt/"
if [[ -d "${installed_test_location}" ]]; then
  GTEST_DIR="${installed_test_location}"
elif [[ -d "${devcontainers_test_location}" ]]; then
  GTEST_DIR="${devcontainers_test_location}"
else
  echo "::error::gtest install location not found (searched '${installed_test_location}' and '${devcontainers_test_location}')." >&2
  exit 1
fi

# Select multi-GPU test binaries by the *_MG_TEST naming convention.
shopt -s nullglob
mg_tests=("${GTEST_DIR}"/*_MG_TEST)
shopt -u nullglob

if [ "${#mg_tests[@]}" -eq 0 ]; then
  rapids-logger "No multi-GPU gtest binaries (*_MG_TEST) found in ${GTEST_DIR}; nothing to run."
  echo "::notice::No multi-GPU tests present yet — skipping. This job lights up once *_MG_TEST binaries land (distributed PDLP)."
  exit 0
fi

EXITCODE=0
for gt in "${mg_tests[@]}"; do
  test_name=$(basename "${gt}")
  rapids-logger "Running multi-GPU gtest ${test_name}"
  "${gt}" --gtest_output="xml:${RAPIDS_TESTS_DIR}/${test_name}.xml" "$@" || EXITCODE=$?
done

rapids-logger "Multi-GPU test script exiting with value: ${EXITCODE}"
exit ${EXITCODE}
