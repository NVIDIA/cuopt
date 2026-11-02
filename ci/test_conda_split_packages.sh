#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Installs each libcuopt-* conda package on its own, in a fresh environment, and builds and runs a
# C++ binary against it, to catch a package that only works because another one happened to be
# installed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPONENTS=(client mathopt routing)

. "$(conda info --base)/etc/profile.d/conda.sh"

conda config --set channel_priority strict
CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuopt cuopt --cuda "$RAPIDS_CUDA_VERSION")")
RAPIDS_CHANNEL=rapidsai-nightly
rapids-is-release-build && RAPIDS_CHANNEL=rapidsai
CUDA_VERSION=${RAPIDS_CUDA_VERSION%.*}

EXITCODE=0
trap "EXITCODE=1" ERR

for component in "${COMPONENTS[@]}"; do
  echo "=== conda: libcuopt-${component} alone ==="
  package="libcuopt-${component}"
  expected=(libcuopt-client "${package}")
  if [[ "${component}" == "client" ]]; then
    expected=(libcuopt-client)
  fi

  env_dir="$(mktemp -d)/env"
  set +e
  rapids-mamba-retry create --yes --prefix "${env_dir}" \
    --channel "${CPP_CHANNEL}" --channel "${RAPIDS_CHANNEL}" --channel conda-forge --channel nvidia \
    "${package}" "cuda-version=${CUDA_VERSION}" \
    cmake ninja cxx-compiler cuda-nvcc cuda-cudart-dev

  python "${REPO_ROOT}/ci/split_package_smoke/env_check.py" conda-packages "${env_dir}" "${expected[@]}"

  set +u
  conda activate "${env_dir}"
  set -u
  bash "${REPO_ROOT}/ci/split_package_smoke/build_and_run.sh" "${component}" "${env_dir}"
  conda deactivate
  set -e
done

rapids-logger "Test script exiting with value: ${EXITCODE}"
exit "${EXITCODE}"
