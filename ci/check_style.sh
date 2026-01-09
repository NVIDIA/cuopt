#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

echo "--- checking if expected secret 'MY_COOL_SECRET_NAME' is set ---"
if test -n "${MY_COOL_SECRET_NAME:-}"; then
  echo "it is: ${MY_COOL_SECRET_NAME}"
  exit 2
else
  echo "it is not"
  exit 1
fi

echo "--- checking if expected secret 'MY_COOL_PASSOWRD' is set ---"
if test -n "${MY_COOL_PASSWORD:-}"; then
  echo "it is: ${MY_COOL_PASSWORD}"
  exit 2
else
  echo "it is not"
  exit 1
fi

rapids-logger "Create checks conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-dependency-file-generator \
  --output conda \
  --file-key checks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n checks
conda activate checks

# Run pre-commit checks
pre-commit run --all-files --show-diff-on-failure
