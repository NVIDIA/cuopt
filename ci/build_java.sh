#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

RUN_TESTS=false
if [[ "${1:-}" == "--run-java-tests" ]]; then
  RUN_TESTS=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--run-java-tests]" >&2
  exit 2
fi

if [[ -e /opt/conda/etc/profile.d/conda.sh ]]; then
  . /opt/conda/etc/profile.d/conda.sh
fi

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading the C++ artifact"
CPP_CHANNEL=$(rapids-download-from-github \
  "$(rapids-artifact-name conda_cpp libcuopt cuopt --cuda "$RAPIDS_CUDA_VERSION")")

rapids-logger "Generating Java build dependencies"
ENV_YAML_DIR=$(mktemp -d)
rapids-dependency-file-generator \
  --output conda \
  --file-key java \
  --prepend-channel "${CPP_CHANNEL}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n java \
  --channel "${CPP_CHANNEL}"

# Temporarily allow unbound variables for conda activation.
set +u
conda activate java
set -u

rapids-print-env

export CUOPT_PREFIX="${CONDA_PREFIX}"

# libcuopt comes from the conda artifact here, not a local build tree, so build.sh's 'java'
# target uses the install prefix directly.
if [[ "${RUN_TESTS}" == true ]]; then
  rapids-logger "Building and testing the Java bindings"
  ./build.sh java --run-java-tests
else
  rapids-logger "Building the Java bindings"
  ./build.sh java
fi
